"""Faiss Ascend residual inner-product IVF-PQ baseline; native NPU build.

The complete public-API pipeline is timed: CPU query padding, NPU coarse
quantizer search, and NPU search_preassigned. This exposes the actual scanned
lists for exact visit accounting. Host/device transfers are included, unlike
the resident Tensor API. Training and encoding use Faiss Ascend; no TensorVS index is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = "Ascend_Faiss_IVFPQ_glove1.2m_nlist1024_M32_nbits8_train512"
PROBES_PER_SCAN = 64


def save_build_summary(path, build):
    fields = ("dataset", "nlist", "M", "nbits", "build_time_seconds", "train_seconds",
              "add_seconds", "index_size_bytes", "index_size_mb", "preparation_seconds", "serialize_seconds")
    row = dict(dataset="glove1.2m", nlist=1024, M=32, nbits=8,
               **{key: build[key] for key in fields[4:]})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(path)


def search_batch(faiss, index, coarse, resources, device, batch, working_dim, topk):
    """Both retrieval stages run through Faiss NPU; return actual scanned lists."""
    try:
        from ._ascend_faiss_build import pad_vectors
    except ImportError:
        from _ascend_faiss_build import pad_vectors
    transformed = pad_vectors(batch, working_dim) if working_dim is not None else batch
    distances, lists = coarse.search(transformed, int(index.nprobe))
    distances = np.ascontiguousarray(distances, dtype=np.float32)
    lists = np.ascontiguousarray(lists, dtype=np.int64)
    # NpuIndexIVFPQ inherits IndexIVFInterface, not IndexIVF: use its raw
    # SWIG entry point instead of assuming the CPU array wrapper is installed.
    raw = getattr(index, "search_preassigned_c", index.search_preassigned)
    # The release's multi-block residual-IP scan disagrees with its CPU export
    # at large nprobe. Scan disjoint groups through the supported single-block
    # path and merge their top-K on the host, all inside the timed API pipeline.
    original_probe = int(index.nprobe)
    parts = []
    try:
        for start in range(0, original_probe, PROBES_PER_SCAN):
            assigned = np.ascontiguousarray(lists[:, start:start + PROBES_PER_SCAN])
            coarse_d = np.ascontiguousarray(distances[:, start:start + PROBES_PER_SCAN])
            index.nprobe = assigned.shape[1]
            out_d = np.empty((len(batch), topk), dtype=np.float32)
            out_i = np.empty((len(batch), topk), dtype=np.int64)
            raw(len(batch), faiss.swig_ptr(transformed), topk,
                faiss.swig_ptr(assigned), faiss.swig_ptr(coarse_d),
                faiss.swig_ptr(out_d), faiss.swig_ptr(out_i), False)
            resources.getResources().syncDefaultStream(device)
            parts.append((out_d, out_i))
    finally:
        index.nprobe = original_probe
    if len(parts) > 1:
        values = np.concatenate([part[0] for part in parts], axis=1)
        labels = np.concatenate([part[1] for part in parts], axis=1)
        scores = np.where(labels >= 0, values, -np.inf)
        order = np.argsort(-scores, axis=1, kind="stable")[:, :topk]
        out_d = np.take_along_axis(values, order, axis=1)
        out_i = np.take_along_axis(labels, order, axis=1)
    return out_d, out_i, lists


def run(args):
    import faiss
    from tensorvs import load_dataset
    try:
        from .amd_tensor_bench import FIELDS, NPROBES, atomic_csv, atomic_json
        from ._ascend_process_memory import NpuProcessMemorySampler
        from ._ascend_faiss_build import build_or_load, digest, exact_ip_ground_truth
    except ImportError:
        from amd_tensor_bench import FIELDS, NPROBES, atomic_csv, atomic_json
        from _ascend_process_memory import NpuProcessMemorySampler
        from _ascend_faiss_build import build_or_load, digest, exact_ip_ground_truth

    required = ("NpuIndexIVFPQ", "NpuIndexIVFPQConfig", "StandardNpuResources")
    if not all(hasattr(faiss, name) for name in required):
        raise RuntimeError(f"Faiss Ascend NPU bindings missing: {faiss.__file__}")
    faiss.omp_set_num_threads(args.cpu_threads)
    dataset = args.data_root / "glove1.2m"
    source = dataset / "glove1.2m_base.fvecs"
    query = dataset / "glove1.2m_query.fvecs"
    csv_path, json_path = args.output_dir / (STEM + ".csv"), args.output_dir / (STEM + ".json")
    if csv_path.exists() or json_path.exists():
        if args.rebuild_index and not args.overwrite:
            raise FileExistsError("Rebuilding requires --overwrite when search results exist")
        if not args.resume and not args.overwrite:
            raise FileExistsError(f"Use --resume or --overwrite: {csv_path}")
        if args.resume and json_path.exists():
            if "build" not in json.loads(json_path.read_text()):
                raise ValueError("Old converted-index results cannot be resumed; use --overwrite")
    resources = faiss.StandardNpuResources()
    options = faiss.NpuIndexIVFPQConfig()
    options.device = args.npu_id
    options.useNpuTrain = True
    # Bookkeeping quantizer stays on CPU; timed retrieval uses the explicit
    # NPU coarse index below in both fresh-build and cache-load runs.
    options.allowCpuCoarseQuantizer = True
    options.useFloat16LookupTables = args.lut_dtype == "fp16"
    options.cp.niter = args.coarse_niter
    options.cp.seed = args.seed
    options.cp.nredo = 1
    options.cp.spherical = True
    options.cp.max_points_per_centroid = 512
    index, cpu, counts, build, cache_loaded = build_or_load(faiss, resources, options, args, source)
    save_build_summary(args.output_dir / "Build" / (STEM + "_build.csv"), build)
    working_dim = 256  # independent zero padding, no TensorVS transform
    original_meta = dict(dim=200, pq_working_dim=256, nlist=1024, m=32, nbits=8,
                         pq_mode="residual", metric="inner_product", normalization="none",
                         num_vectors=int(index.ntotal), preprocessing="zero padding")
    queries = np.ascontiguousarray(load_dataset(query).numpy()[:args.nq], dtype=np.float32)
    if not len(queries):
        raise ValueError("Empty queries")
    if queries.shape[1] != 200:
        raise ValueError("Expected 200-dimensional GloVe queries")
    truth, truth_info = exact_ip_ground_truth(faiss, source, queries, args.topk,
        args.index_dir / (STEM + ".groundtruth.npz"))
    probes = args.nprobes or NPROBES["glove1.2m"]
    if max(probes) > cpu.nlist:
        raise ValueError("nprobe exceeds nlist")
    paths = [source, query]
    config = dict(dataset="glove1.2m", kind="ivfpq", metric="inner_product", normalization="none",
                  ground_truth_signature=truth_info["signature"], nq=len(queries), topk=args.topk,
                  q_batches=args.q_batches, nprobes=list(probes), npu_id=args.npu_id,
                  warmup=args.warmup, repetitions=args.repetitions, lut_dtype=args.lut_dtype,
                  cpu_threads=args.cpu_threads, seed=args.seed, coarse_niter=args.coarse_niter,
                  pq_niter=args.pq_niter, train_size=args.train_size, add_batch_size=args.add_batch_size,
                  index_sha256=digest(build["index_path"]), faiss_version=faiss.__version__,
                  faiss_source_revision=os.environ.get("FAISS_ASCEND_SOURCE_REVISION"),
                  faiss_source_patch_sha256=os.environ.get("FAISS_ASCEND_SOURCE_PATCH_SHA256"),
                  faiss_path=faiss.__file__, faiss_library_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path(faiss.__file__).parent.glob("_swigfaiss*.so")},
                  inputs={str(p): dict(size_bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in paths},
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__), ROOT / "bench/_ascend_process_memory.py", ROOT / "bench/_ascend_faiss_build.py", ROOT / "bench/amd_tensor_bench.py",
                                *(ROOT / "src/tensorvs").glob("*.py"))},
                  cann_home=os.environ.get("ASCEND_HOME_PATH"),
                  probes_per_scan=PROBES_PER_SCAN,
                  pipeline="NPU coarse.search + NPU IVFPQ.search_preassigned in disjoint probe groups + CPU top-K merge")
    ordered = [(b, p) for b in args.q_batches for p in probes]
    csv_path, json_path = args.output_dir / (STEM + ".csv"), args.output_dir / (STEM + ".json")
    completed, details = {}, {}
    if csv_path.exists() or json_path.exists():
        if not args.overwrite:
            if not args.resume or not csv_path.exists() or not json_path.exists():
                raise FileExistsError(f"Use --resume or --overwrite: {csv_path}")
            old = json.loads(json_path.read_text())
            if old.get("config") != json.loads(json.dumps(config)):
                raise ValueError("Resume settings, source files or inputs differ")
            with csv_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != list(FIELDS):
                    raise ValueError("CSV schema differs")
                for row in reader:
                    key = int(row["q_batch"]), int(row["nprobes"])
                    if (key not in ordered or key in completed or int(row["K"]) != args.topk
                            or not all(math.isfinite(float(row[f])) for f in FIELDS)):
                        raise ValueError("Invalid resume CSV row")
                    completed[key] = row
            details = {(int(r["q_batch"]), int(r["nprobes"])): r for r in old["results"]}
            if details.keys() != completed.keys():
                raise ValueError("CSV/JSON results differ")
    if len(completed) == len(ordered):
        print(f"Already complete: {csv_path}")
        return
    # Native training keeps a CPU bookkeeping quantizer. Use the library's
    # supported default FP16 Flat search path for the explicit NPU coarse stage.
    coarse_options = faiss.NpuIndexFlatConfig()
    coarse_options.device = args.npu_id
    coarse_options.storageType = faiss.NpuFlatStorageFloat16
    coarse = faiss.NpuIndexFlatIP(resources, faiss.downcast_index(cpu.quantizer), coarse_options)
    if coarse.getDevice() != args.npu_id:
        raise RuntimeError("Unexpected NPU coarse device")
    resources.getResources().syncDefaultStream(args.npu_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(status="running", config=config, backend="faiss_ascend", device=f"npu:{args.npu_id}",
        implementation="Faiss Ascend NPU coarse search + NpuIndexIVFPQ.search_preassigned (at most 64 probes/call), host top-K merge; no CPU vector scan",
        index_metadata=original_meta, expected_rows=len(ordered), build=build, index_loaded_from_cache=cache_loaded,
        index_construction="Faiss Ascend trains coarse centroids and residual PQ, then encodes/adds raw GloVe vectors; no TensorVS index dependency",
        timing="Median synchronized full-query sweeps; sum of per-batch CPU query padding + NPU coarse.search + "
               "NPU search_preassigned and host top-K merge, including API transfers and CPU control; excludes recall/visit accounting/HBM sampling.",
        timing_comparison_limit="Faiss accepts/returns host arrays; Tensor uses resident NPU query batches. "
                                "Coarse centroid storage and query representation use Faiss's FP16 Flat path. "
                                "The index is independently trained with zero padding, unlike the TensorVS rotated PQ index.",
        recall="Mean top-K ID overlap with exhaustive raw-IP ground truth divided by K; dataset L2 truth is not used",
        ground_truth=truth_info,
        visited_vectors_definition="Sum of Faiss-built list sizes for the exact assignments passed to search_preassigned, excluding padded queries",
        memory_unit="MiB", memory_metric="Sampled npu-smi current-process HBM peak in a separate sweep; includes index/runtime/workspace/cache",
        results=[])

    def save():
        metadata.update(completed_rows=len(completed), results=[details[k] for k in ordered if k in details])
        atomic_json(json_path, metadata)
        atomic_csv(csv_path, [completed[k] for k in ordered if k in completed])

    save()
    started = time.perf_counter()
    try:
        for batch_size, probe in ordered:
            key = batch_size, probe
            if key in completed:
                continue
            index.nprobe = probe
            batches = []
            for offset in range(0, len(queries), batch_size):
                valid = min(batch_size, len(queries) - offset)
                batch = np.empty((batch_size, queries.shape[1]), dtype=np.float32)
                batch[:valid] = queries[offset:offset + valid]
                batch[valid:] = batch[valid - 1]
                batches.append((batch, valid))

            def sweep(retain=False, sampler=None):
                elapsed, ids, visits = 0.0, [], []
                for batch, valid in batches:
                    resources.getResources().syncDefaultStream(args.npu_id)
                    begin = time.perf_counter()
                    d, i, selected = search_batch(faiss, index, coarse, resources, args.npu_id, batch, working_dim, args.topk)
                    elapsed += time.perf_counter() - begin
                    if retain:
                        if (not np.isfinite(d[:valid][i[:valid] >= 0]).all() or (i[:valid] < -1).any()
                                or (i[:valid] >= int(index.ntotal)).any()
                                or (selected[:valid] < 0).any() or (selected[:valid] >= len(counts)).any()):
                            raise RuntimeError("Invalid search results or coarse assignments")
                        ids.append(i[:valid].copy())
                        visits.append(counts[selected[:valid]].sum(axis=1))
                    if sampler is not None:
                        sampler.sample()
                return elapsed, ids, visits

            print(f"glove1.2m Faiss Ascend ivfpq QBatch={batch_size} K={args.topk} nprobe={probe}", flush=True)
            for _ in range(args.warmup):
                sweep()
            samples = []
            for repetition in range(args.repetitions):
                elapsed, found, visits = sweep(retain=repetition == args.repetitions - 1)
                samples.append(elapsed)
            found, visits = np.concatenate(found), np.concatenate(visits)
            recall = float(np.mean([len(set(row) & set(gt[:args.topk])) / args.topk for row, gt in zip(found, truth)]))
            with NpuProcessMemorySampler() as sampler:
                sweep(sampler=sampler)
            row = dict(nprobes=probe, q_batch=batch_size, K=args.topk,
                       qps=len(queries) / statistics.median(samples), recall=recall,
                       visited_vectors_per_query=float(visits.mean()), peak_gpu_memory=sampler.peak_bytes / (1 << 20))
            if not all(math.isfinite(float(v)) for v in row.values()):
                raise RuntimeError("Nonfinite metric")
            completed[key] = row
            total = int(visits.sum())
            visit_statistics = dict(visited_vectors_total=total,
                visited_vectors_per_query=float(visits.mean()),
                visited_vectors_min_per_query=int(visits.min()), visited_vectors_max_per_query=int(visits.max()),
                visited_data_bytes_total=total * 32, visited_data_mb_total=total * 32 / (1 << 20),
                visited_data_mb_per_query=total * 32 / len(queries) / (1 << 20))
            details[key] = dict(row, samples_seconds=samples, memory_samples=sampler.samples,
                                missing_neighbor_slots=int((found == -1).sum()),
                                physical_npu=sampler.physical_device, visit_statistics=visit_statistics)
            save()
            print(f"  QPS={row['qps']:.2f} recall={recall:.6f} HBM={row['peak_gpu_memory']:.1f} MiB", flush=True)
            if (found == -1).any():
                print(f"  Faiss returned {(found == -1).sum()} missing-neighbor slots (-1); counted as recall misses", flush=True)
        metadata.update(status="complete", wall_seconds=time.perf_counter() - started)
        save()
    except BaseException as error:
        metadata.update(status="failed", error=repr(error), wall_seconds=time.perf_counter() - started)
        save()
        raise
    print(f"Saved {csv_path} and {json_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("glove1.2m",), default="glove1.2m")
    p.add_argument("--data-root", type=Path, default=Path("/home/ma-user/work/data"))
    p.add_argument("--output-dir", type=Path, default=ROOT / "result")
    p.add_argument("--npu-id", type=int, default=0)
    p.add_argument("--nq", type=int, default=1024)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--q-batches", type=int, nargs="+", default=[32, 128, 512])
    p.add_argument("--nprobes", type=int, nargs="+")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--lut-dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--index-dir", type=Path, default=ROOT / "indexes/faiss_ascend")
    p.add_argument("--rebuild-index", action="store_true", help="Rebuild instead of loading a compatible cache")
    p.add_argument("--train-size", type=int, default=0, help="0: supply all base vectors; Faiss samples internally")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--coarse-niter", type=int, default=25)
    p.add_argument("--pq-niter", type=int, default=50)
    p.add_argument("--add-batch-size", type=int, default=65536)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if min(*args.q_batches, *(args.nprobes or [1]), args.nq, args.topk, args.repetitions, args.cpu_threads) <= 0 or args.warmup < 0 or args.npu_id < 0:
        p.error("Positive sizes/repetitions/threads required; warmup/device must be nonnegative")
    if len(set(args.q_batches)) != len(args.q_batches) or (args.nprobes and len(set(args.nprobes)) != len(args.nprobes)):
        p.error("Duplicate batch size or nprobe")
    if args.train_size < 0 or min(args.coarse_niter, args.pq_niter, args.add_batch_size) <= 0:
        p.error("Training size must be nonnegative; iterations and add batch size must be positive")
    if args.rebuild_index and args.resume:
        p.error("--rebuild-index and --resume cannot be combined; use --overwrite for a fresh experiment")
    run(args)


if __name__ == "__main__":
    main()
