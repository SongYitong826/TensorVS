"""Measure TensorVS and local Faiss Metal; save one CSV/JSON pair per method/index.

Run with conda run -n tensorvs python -u bench/compare_mps_faiss.py --dataset ...
Torch and Faiss run in separate processes because their OpenMP runtimes differ.
Temporary index exports are removed automatically. Results use MPS_ names in
result/; original source-index build metadata is saved separately in result/Build/.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time

try:
    from ._mps_results import index_stem, read_rows, save_source_index, write_result
except ImportError:
    from _mps_results import index_stem, read_rows, save_source_index, write_result

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["FAISS_METAL_IVF_ALLOW_CPU_FALLBACK"] = "0"
os.environ["FAISS_METAL_IVF_LOG_CPU_FALLBACK"] = "1"

ROOT = Path(__file__).resolve().parents[1]
SIFT_PROBES = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
GLOVE_PROBES = [50, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280,
                300, 320, 340, 360, 380, 400]
FIELDS = ["dataset", "index_type", "method", "execution_device", "q_batch", "K",
          "nprobes", "qps", "recall", "query_count", "nlist", "dim", "M", "nbits",
          "pq_working_dim", "use_compile", "scan", "lut_dtype", "omp_threads",
          "warmup_iterations", "warmup_scope", "repetitions", "sample1_seconds",
          "sample2_seconds", "sample3_seconds", "peak_gpu_memory", "peak_rss_memory",
          "memory_samples", "index_path", "index_size_bytes",
          "index_mtime_ns", "query_path", "groundtruth_path", "source_sha256",
          "library_version", "measured_at_utc"]
REMEASURE_FIELDS = [*(f"sample{i}_seconds" for i in range(4, 8)),
                    "sample_sweep_counts", "minimum_sample_seconds"]


def _maximum_rss_bytes():
    """Return the process high-water RSS in bytes on macOS."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class PeakMemorySampler:
    """Sample Metal-driver allocation and process RSS during one search sweep."""

    def __init__(self, driver_memory):
        self.driver_memory = driver_memory
        self.peak_driver = 0
        self.peak_rss = 0
        self.samples = 0
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        self.peak_driver = max(self.peak_driver, int(self.driver_memory()))
        self.peak_rss = max(self.peak_rss, _maximum_rss_bytes())
        self.samples += 1

    def _run(self):
        try:
            while not self.stop.wait(.01):
                self.sample()
        except BaseException as error:
            self.error = error
            self.stop.set()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()
        if self.error is not None:
            raise RuntimeError("Peak-memory sampling failed") from self.error


def faiss_metal_driver_memory(faiss_root):
    """Load the local Metal allocation helper without importing PyTorch."""
    source = faiss_root / "local_bench" / "metal_memory.mm"
    library_path = source.with_suffix(".dylib")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not library_path.exists() or library_path.stat().st_mtime_ns < source.stat().st_mtime_ns:
        subprocess.run([
            "xcrun", "clang++", "-arch", "arm64", "-std=c++17", "-O2",
            "-dynamiclib", "-fobjc-arc", "-framework", "Metal", "-framework",
            "Foundation", str(source), "-o", str(library_path),
        ], check=True)
    library = ctypes.CDLL(str(library_path))
    library.metal_driver_allocated_bytes.restype = ctypes.c_ulonglong
    library.metal_driver_allocated_bytes.argtypes = []
    return library, library.metal_driver_allocated_bytes


def read_vectors(path, dtype):
    import numpy as np
    if path.suffix == ".npy":
        return np.ascontiguousarray(np.load(path), dtype=dtype)
    raw = np.memmap(path, dtype=np.int32, mode="r")
    dim = int(raw[0])
    if dim <= 0 or len(raw) % (dim + 1):
        raise ValueError(f"Invalid vector file: {path}")
    rows = raw.reshape(-1, dim + 1)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"Inconsistent vector dimensions: {path}")
    values = rows[:, 1:].view(np.float32) if np.dtype(dtype).kind == "f" else rows[:, 1:]
    return np.ascontiguousarray(values, dtype=dtype)


def write_csv(path, rows, *, complete=False):
    fields = FIELDS + REMEASURE_FIELDS if any("sample4_seconds" in row for row in rows) else FIELDS
    write_result(path, [{field: row.get(field, "") for field in fields} for row in rows],
                 complete=complete)


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def import_faiss(args):
    import faiss
    import numpy as np
    folder = args.export_dir
    arrays = {name: np.load(folder / f"{name}.npy", mmap_mode="r")
              for name in ("centroids", "data", "ids", "counts", "offsets")}
    centers, data, ids = (arrays[name] for name in ("centroids", "data", "ids"))
    nlist, dim = centers.shape
    rotation = None
    if (folder / "rotation.npy").exists():
        rotation = np.load(folder / "rotation.npy")
        centers = np.load(folder / "pq_centroids.npy")
    working_dim = centers.shape[1]
    quantizer = faiss.IndexFlatL2(working_dim)
    quantizer.add(np.ascontiguousarray(centers))
    if args.kind == "ivfflat":
        cpu = faiss.IndexIVFFlat(quantizer, working_dim, nlist, faiss.METRIC_L2)
        m, nbits = 0, 0
    else:
        books = np.load(folder / "codebooks.npy")
        m, ksub, subdim = books.shape
        if ksub != 256 or working_dim != m * subdim:
            raise ValueError("PQ dimensions do not match exported codebooks")
        nbits = 8
        cpu = faiss.IndexIVFPQ(quantizer, working_dim, nlist, m, nbits, faiss.METRIC_L2)
        faiss.copy_array_to_vector(books.reshape(-1), cpu.pq.centroids)
        cpu.by_residual = True
        np.testing.assert_array_equal(faiss.vector_to_array(cpu.pq.centroids), books.reshape(-1))
    if not np.array_equal(np.sort(ids), np.arange(len(ids))):
        raise ValueError("Expected original contiguous vector IDs")
    cpu.is_trained = True
    for list_id, count in enumerate(arrays["counts"]):
        count, start = int(count), int(arrays["offsets"][list_id])
        if not count:
            continue
        list_ids = np.ascontiguousarray(ids[start:start + count])
        codes = np.ascontiguousarray(data[start:start + count]).view(np.uint8).reshape(-1)
        cpu.invlists.add_entries(list_id, count, faiss.swig_ptr(list_ids), faiss.swig_ptr(codes))
        np.testing.assert_array_equal(faiss.rev_swig_ptr(cpu.invlists.get_ids(list_id), count), list_ids)
        np.testing.assert_array_equal(faiss.rev_swig_ptr(cpu.invlists.get_codes(list_id), count * cpu.code_size), codes)
    cpu.ntotal = len(ids)
    np.testing.assert_array_equal(cpu.quantizer.reconstruct_n(0, nlist), centers)
    if args.kind == "ivfpq":
        cpu.precompute_table()
    # The Metal IVF-PQ wrapper rejects M > 64 even though search delegates to
    # CPU. Allow that same CPU search path without constructing the wrapper.
    resources = None if args.faiss_cpu else faiss.StandardGpuResources()
    metal = cpu if args.faiss_cpu else faiss.index_cpu_to_gpu(resources, 0, cpu)
    wrapper = metal
    transform = None
    if rotation is not None:
        transform = faiss.LinearTransform(dim, working_dim)
        faiss.copy_array_to_vector(rotation.reshape(-1), transform.A)
        transform.is_trained = True
        wrapper = faiss.IndexPreTransform(transform, metal)
    print(f"Imported identical lists/codes/IDs: dim={dim}, working_dim={working_dim}, M={m}", flush=True)
    # Faiss wrappers restrict new attributes; retain ownership in the caller.
    return wrapper, cpu, (resources, metal, transform), (nlist, dim, m, nbits, working_dim)


def worker(args):
    import numpy as np
    queries = read_vectors(args.queries, np.float32)
    truth = read_vectors(args.groundtruth, np.int64)
    if not len(queries) or len(queries) != len(truth) or truth.shape[1] < args.topk:
        raise ValueError("Insufficient or mismatched queries/ground truth")
    nq = min(args.nq, len(queries))
    queries, truth = queries[:nq], truth[:nq]
    print(f"Using {nq} queries (requested maximum: {args.nq})", flush=True)
    if not np.isfinite(queries).all():
        raise ValueError("Nonfinite query vectors")
    flat = args.kind == "ivfflat"
    source_paths = [Path(__file__), ROOT / "bench/export_comparison_index.py"]
    if args.worker == "tensorvs":
        import torch
        from tensorvs import load_ivfflat_index, load_ivfpq_index, search_ivfflat_index, search_ivfpq_index
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is unavailable")
        loader, search = ((load_ivfflat_index, search_ivfflat_index) if flat else
                          (load_ivfpq_index, search_ivfpq_index))
        index = loader(args.index, load_lists=True)
        index.reside_on_device("mps", backend="torch", persist_layout=False)
        nlist, dim = index.centroids.shape
        m = 0 if flat else int(index.metadata["m"])
        nbits = 0 if flat else int(index.metadata["nbits"])
        working_dim = dim if flat else m * index.pq_codebooks.shape[2]
        version, device = str(torch.__version__), "mps"
        memory_library = None
        driver_memory = torch.mps.driver_allocated_memory
        source_paths += [ROOT / "src/tensorvs" / name for name in
                         ("_torch_search.py", "_torch_pq_bucketed.py", "_layout_cache.py",
                          f"index_{args.kind}.py")]
        sync = torch.mps.synchronize
        def prepare_memory_sweep():
            sync()
            torch.mps.empty_cache()
        def prepare(batch_size):
            batches = [torch.from_numpy(queries[i:i + batch_size]).to("mps")
                       for i in range(0, len(queries), batch_size)]
            options = dict(device="mps", backend="torch", persist_layout=False,
                           torch_query_batch_size=batch_size, torch_memory_budget_mb=128,
                           use_compile=not flat)
            options.update({"torch_flat_scan": "grouped"} if flat else
                           {"torch_pq_scan": "bucketed", "lut_dtype": "fp16"})
            def call(batch, probe):
                return search(index, batch, topk=args.topk, nprobe=probe, **options)
            return batches, call
        context = torch.inference_mode()
        def retain(values, ids):
            if values.device.type != "mps" or ids.device.type != "mps":
                raise RuntimeError("TensorVS outputs must reside on MPS")
            return values.cpu().numpy(), ids.cpu().numpy()
    else:
        import contextlib
        import faiss
        faiss.omp_set_num_threads(args.threads)
        if args.faiss_cpu:
            memory_library, driver_memory = None, lambda: 0
        else:
            memory_library, driver_memory = faiss_metal_driver_memory(args.faiss_root)
        index, cpu, resources, configuration = import_faiss(args)
        nlist, dim, m, nbits, working_dim = configuration
        version, device = faiss.__version__, "metal" if flat and not args.faiss_cpu else "cpu"
        metal_source = args.faiss_root / "faiss/gpu_metal" / f"MetalIndex{'IVFFlat' if flat else 'IVFPQ'}.mm"
        source_paths.append(metal_source)
        if not flat and not args.faiss_cpu and "cpuIndex_->search(n, x, k, distances, labels)" not in metal_source.read_text():
            raise RuntimeError("Faiss PQ execution path changed; review device label")
        sync = lambda: None  # Faiss Metal's Python search API is synchronous.
        prepare_memory_sweep = lambda: None
        def prepare(batch_size):
            batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
            def call(batch, probe):
                params = faiss.SearchParametersIVF()
                params.nprobe = probe
                return index.search(batch, args.topk, params=params)
            return batches, call
        context = contextlib.nullcontext()
        retain = lambda values, ids: (values, ids)
        # Verify the actual Metal wrapper against the imported CPU index.
        for probe in (min(args.nprobes), max(args.nprobes)):
            params = faiss.SearchParametersIVF()
            params.nprobe = probe
            _, call = prepare(args.q_batches[0])
            values, ids = call(queries[:8], probe)
            cpu_queries = queries[:8]
            if (args.export_dir / "rotation.npy").exists():
                rotation = np.load(args.export_dir / "rotation.npy")
                cpu_queries = np.ascontiguousarray(cpu_queries @ rotation.T)
            expected, expected_ids = cpu.search(cpu_queries, args.topk, params=params)
            np.testing.assert_allclose(values, expected, rtol=2e-4, atol=.2 if flat else 2e-4)
            overlap = np.mean([len(set(a) & set(b)) / args.topk for a, b in zip(ids, expected_ids)])
            if overlap < .95:
                raise RuntimeError(f"Metal/CPU neighbor overlap too low: {overlap}")
            print(f"Verified Faiss/CPU overlap={overlap:.4f} nprobe={probe}", flush=True)
    if dim != queries.shape[1] or max(args.nprobes) > nlist:
        raise ValueError("Queries/nprobe do not match index")
    rows = []
    if args.output.exists():
        rows = read_rows(args.output)
    method = "TensorVS" if args.worker == "tensorvs" else "Faiss Metal"
    existing_keys = {
        (int(row["q_batch"]), int(row["nprobes"]))
        for row in rows if row["method"] == method
    }
    if not args.remeasure_points and existing_keys and not args.resume:
        raise FileExistsError(
            f"Existing {method} rows in {args.output}; use --resume to keep completed points"
        )
    stat = args.index.stat()
    base = dict(dataset=args.dataset.name, index_type=args.kind, method=method,
                execution_device=device, K=args.topk, query_count=len(queries), nlist=nlist,
                dim=dim, M=m, nbits=nbits, pq_working_dim=working_dim,
                use_compile=args.worker == "tensorvs" and not flat,
                scan=("grouped" if flat else "bucketed") if args.worker == "tensorvs" else "faiss",
                lut_dtype="none" if flat else "fp16" if args.worker == "tensorvs" else "fp32",
                omp_threads=args.threads if args.worker == "faiss" else "",
                warmup_iterations=args.warmup, warmup_scope="all-batches", repetitions=3,
                index_path=str(args.index.resolve()), index_size_bytes=stat.st_size,
                index_mtime_ns=stat.st_mtime_ns, query_path=str(args.queries.resolve()),
                groundtruth_path=str(args.groundtruth.resolve()), source_sha256=fingerprint(source_paths),
                library_version=version)
    if args.remeasure_points:
        remeasure(args, rows, base, prepare, sync, retain, context, queries, truth)
        return
    with context:
        for batch_size in args.q_batches:
            batches, call = prepare(batch_size)
            for probe in args.nprobes:
                if (batch_size, probe) in existing_keys:
                    print(f"Keeping completed {method} QBatch={batch_size} nprobe={probe}", flush=True)
                    continue
                for _ in range(args.warmup):
                    for batch in batches:
                        values, ids = call(batch, probe)
                        sync()
                        del values, ids
                samples = []
                saved = []
                for repetition in range(3):
                    elapsed = 0.0
                    for batch in batches:
                        sync()
                        begin = time.perf_counter()
                        values, ids = call(batch, probe)
                        sync()
                        elapsed += time.perf_counter() - begin
                        if repetition == 2:
                            values, ids = retain(values, ids)
                            if not np.isfinite(values).all() or (ids < 0).any():
                                raise RuntimeError("Invalid search results")
                            if any(len(set(row)) != args.topk for row in ids):
                                raise RuntimeError("Duplicate result IDs")
                            saved.append(ids.copy())
                        del values, ids
                    samples.append(elapsed)
                found = np.concatenate(saved)
                recall = sum(len(set(row) & set(expected[:args.topk])) / args.topk
                             for row, expected in zip(found, truth)) / len(queries)
                qps = len(queries) / statistics.median(samples)
                if not math.isfinite(qps) or qps <= 0 or not 0 <= recall <= 1:
                    raise RuntimeError("Invalid QPS/recall")
                prepare_memory_sweep()
                with PeakMemorySampler(driver_memory) as memory:
                    for batch in batches:
                        values, ids = call(batch, probe)
                        sync()
                        memory.sample()
                        del values, ids
                row = dict(base, q_batch=batch_size, nprobes=probe, qps=qps, recall=recall,
                           peak_gpu_memory=memory.peak_driver / (1 << 20),
                           peak_rss_memory=memory.peak_rss / (1 << 20),
                           memory_samples=memory.samples,
                           measured_at_utc=datetime.now(timezone.utc).isoformat(),
                           **{f"sample{i + 1}_seconds": value for i, value in enumerate(samples)})
                rows.append(row)
                write_csv(args.output, rows)
                print(f"{args.dataset.name} {args.kind} {method} QBatch={batch_size} "
                      f"nprobe={probe} QPS={qps:.2f} Recall@{args.topk}={recall:.6f} "
                      f"PeakMetal={row['peak_gpu_memory']:.1f}MiB "
                      f"PeakRSS={row['peak_rss_memory']:.1f}MiB", flush=True)
    write_csv(args.output, rows, complete=True)


def remeasure(args, rows, base, prepare, sync, retain, context, queries, truth):
    """Replace selected points using long, interleaved actual timing samples."""
    import numpy as np
    with args.remeasure_points.open(newline="") as file:
        keys = {(int(r["q_batch"]), int(r["nprobes"])) for r in csv.DictReader(file)}
    references = {(int(r["q_batch"]), int(r["nprobes"])): (i, r)
                  for i, r in enumerate(rows) if r["method"] == base["method"]}
    if not keys or not keys <= references.keys():
        raise ValueError("Remeasurement requires existing, identified points")
    for key in keys:
        reference = references[key][1]
        for field in ("index_path", "index_size_bytes", "index_mtime_ns", "query_path",
                      "groundtruth_path", "query_count", "K", "dim", "M", "nbits"):
            if str(reference[field]) != str(base[field]):
                raise ValueError(f"Remeasurement configuration changed: {field}")
    prepared = {batch: prepare(batch) for batch in sorted({batch for batch, _ in keys})}
    measurements = {key: dict(samples=[], sweeps=[]) for key in keys}

    def sweep(key, save=False):
        batch_size, probe = key
        batches, call = prepared[batch_size]
        elapsed, saved = 0.0, []
        for batch in batches:
            sync()
            begin = time.perf_counter()
            values, ids = call(batch, probe)
            sync()
            elapsed += time.perf_counter() - begin
            if save:
                values, ids = retain(values, ids)
                if not np.isfinite(values).all() or (ids < 0).any():
                    raise RuntimeError("Invalid search results")
                if any(len(set(row)) != args.topk for row in ids):
                    raise RuntimeError("Duplicate result IDs")
                saved.append(ids.copy())
            del values, ids
        return elapsed, saved

    ordered = sorted(keys)
    with context:
        for key in ordered:
            for _ in range(args.warmup):
                sweep(key)
            _, saved = sweep(key, save=True)
            found = np.concatenate(saved)
            recall = sum(len(set(row) & set(expected[:args.topk])) / args.topk
                         for row, expected in zip(found, truth)) / len(queries)
            original_recall = float(references[key][1]["recall"])
            if abs(recall - original_recall) > 2 / (len(queries) * args.topk) + 1e-10:
                raise ValueError("Recall changed during timing-only remeasurement; verify inputs/implementation")
            measurements[key]["recall"] = recall
        # Every configuration is warmed before alternating the round order.
        for repetition in range(args.repetitions):
            for key in ordered if repetition % 2 == 0 else reversed(ordered):
                elapsed, sweeps = 0.0, 0
                while sweeps == 0 or elapsed < args.min_sample_seconds:
                    duration, _ = sweep(key)
                    elapsed += duration
                    sweeps += 1
                measurements[key]["samples"].append(elapsed / sweeps)
                measurements[key]["sweeps"].append(sweeps)
            print(f"{args.dataset.name} {args.kind} {base['method']} "
                  f"remeasure round {repetition + 1}/{args.repetitions}, {len(keys)} configurations", flush=True)
    for key in ordered:
        measurement = measurements[key]
        samples = measurement["samples"]
        qps = len(queries) / statistics.median(samples)
        position, old = references[key]
        row = dict(old)
        row.update(base)
        row.update(q_batch=key[0], nprobes=key[1], qps=qps,
                   recall=measurement["recall"], repetitions=args.repetitions,
                   minimum_sample_seconds=args.min_sample_seconds,
                   sample_sweep_counts=";".join(map(str, measurement["sweeps"])),
                   measured_at_utc=datetime.now(timezone.utc).isoformat(),
                   **{f"sample{i + 1}_seconds": value for i, value in enumerate(samples)})
        rows[position] = row
        print(f"Replaced {base['method']} QBatch={key[0]} nprobe={key[1]}: "
              f"QPS {float(old['qps']):.2f} -> {qps:.2f}, recall={measurement['recall']:.6f}", flush=True)
    write_csv(args.output, rows, complete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--groundtruth", type=Path)
    parser.add_argument("--kind", choices=("both", "ivfflat", "ivfpq"), default="both")
    parser.add_argument("--q-batches", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--nprobes", type=int, nargs="+")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--faiss-cpu", action="store_true",
                        help="Run Faiss CPU directly, including PQ configurations beyond Metal's M limit.")
    parser.add_argument("--remeasure-points", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--repetitions", type=int, choices=(3, 7), default=3)
    parser.add_argument("--min-sample-seconds", type=float, default=.4)
    parser.add_argument("--resume", action="store_true",
                        help="Keep valid completed method/QBatch/nprobe rows and run missing points.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--faiss-root", type=Path, default=ROOT.parent / "faiss-metal")
    parser.add_argument("--worker", choices=("tensorvs", "faiss"), help=argparse.SUPPRESS)
    parser.add_argument("--index", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--export-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.topk, args.nq, args.warmup, args.threads, *args.q_batches) <= 0:
        parser.error("All sizes/counts must be positive")
    if not math.isfinite(args.min_sample_seconds) or args.min_sample_seconds <= 0:
        parser.error("min-sample-seconds must be positive")
    if args.remeasure_points and (not args.worker or args.repetitions != 7):
        parser.error("Remeasurement requires a worker and seven repetitions")
    prefix = "sift" if args.dataset.name.startswith("sift") else args.dataset.name
    query_candidate = args.dataset / f"{args.dataset.name}_query.fvecs"
    truth_candidate = args.dataset / f"{args.dataset.name}_groundtruth.ivecs"
    args.queries = args.queries or (query_candidate if query_candidate.is_file() else
                                    args.dataset / f"{prefix}_query.fvecs")
    args.groundtruth = args.groundtruth or (truth_candidate if truth_candidate.is_file() else
                                          args.dataset / f"{prefix}_groundtruth.ivecs")
    args.nprobes = args.nprobes or (GLOVE_PROBES if prefix.startswith("glove") else SIFT_PROBES)
    if min(args.nprobes) <= 0 or len(set(args.nprobes)) != len(args.nprobes):
        parser.error("nprobes must be positive and unique")
    if not args.queries.is_file() or not args.groundtruth.is_file():
        parser.error(f"Missing queries/ground truth: {args.queries}, {args.groundtruth}")
    if args.worker:
        worker(args)
        return
    for kind in (("ivfflat", "ivfpq") if args.kind == "both" else (args.kind,)):
        pattern = "TenVS_IVFFlat_*.bin" if kind == "ivfflat" else "TenVS_IVFPQResidual_*.bin"
        indexes = list((args.dataset / "GPUIndex").glob(pattern))
        if len(indexes) != 1:
            raise ValueError(f"Expected one {kind} index: {indexes}")
        outputs = {
            backend: args.output_dir / (
                index_stem(indexes[0], args.dataset.name, kind, method) + ".csv")
            for backend, method in (("faiss", "Faiss Metal"), ("tensorvs", "TensorVS"))
        }
        for output in outputs.values():
            if (output.exists() or output.with_suffix(".json").exists()) and not args.resume:
                raise FileExistsError(output)
        save_source_index(indexes[0], args.dataset.name, kind, args.output_dir)
        with tempfile.TemporaryDirectory(prefix="tensorvs-comparison-") as directory:
            export = Path(directory)
            subprocess.run([sys.executable, str(ROOT / "bench/export_comparison_index.py"),
                            kind, str(indexes[0]), str(export)], check=True)
            for backend in ("faiss", "tensorvs"):
                subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:],
                                "--kind", kind, "--worker", backend, "--index", str(indexes[0]),
                                "--export-dir", str(export), "--output", str(outputs[backend])], check=True)


if __name__ == "__main__":
    main()
