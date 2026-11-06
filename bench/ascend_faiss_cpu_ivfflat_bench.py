"""CPU IVF-Flat baseline using the installed Faiss Ascend package.

Builds an independent squared-L2 index from raw base vectors. No NPU resources
are created. Retrieval CSV keeps the Tensor benchmark's seven columns; CPU
RSS is recorded separately in JSON and peak_gpu_memory is zero.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MIB = 1 << 20
# Previous builders differ only in logging/query defaults; retain their caches and
# original build provenance instead of forcing a costly retraining operation.
PROGRESS_COMPATIBLE_BUILDERS = {
    "76fb94875e9a6272e468350662b040089f6439bdba35e149e2b1e1c9a229c0f4",
    "b7625bd65b611f3077bb97bdeb9471305763f17e723c52e00b765e77f1e5f21b",
    "c1769c0039e59f0bf3d74be64eaf6bc9363733e9a42c4c68b93fb9256347eaea",
    "03a1b83336218570dc2bb2c4a9f102db5c23ddcdaaf7cc470f69f8e90a75db22",
}
# Kept independent of the Torch runners to avoid loading another OpenMP runtime.
FIELDS = ("nprobes", "q_batch", "K", "qps", "recall", "visited_vectors_per_query", "peak_gpu_memory")
NPROBES = {
    "glove1.2m": (50, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400),
    "hunyuan": (350, 380, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000),
}


def default_cpu_threads():
    limits = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        limits.append(len(os.sched_getaffinity(0)))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limits.append(max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        try:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                limits.append(max(1, math.ceil(quota / period)))
        except (OSError, ValueError):
            pass
    return min(limits)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def cpu_information():
    details = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, separator, value = line.partition(":")
            key = key.strip().lower()
            if separator and key in {"model name", "hardware", "cpu implementer", "cpu part", "cpu architecture"}:
                details.setdefault(key, value.strip())
    except OSError:
        pass
    return dict(architecture=platform.machine(), processor=platform.processor(),
                logical_cpu_count=os.cpu_count(), available_cpu_threads=default_cpu_threads(), cpuinfo=details)


def file_signature(path):
    path = Path(path)
    return dict(path=str(path.resolve()), size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def build_progress(args, message):
    if not args.quiet_build:
        print(message, flush=True)


def compatible_build_signature(cached, current):
    if cached == current:
        return True
    return (cached.get("builder_sha256") in PROGRESS_COMPATIBLE_BUILDERS
            and {k: v for k, v in cached.items() if k != "builder_sha256"}
            == {k: v for k, v in current.items() if k != "builder_sha256"})


def read_vectors(path, *, integer=False, limit=None):
    raw = np.memmap(path, dtype="<i4", mode="r")
    dim = int(raw[0]) if raw.size else 0
    if dim <= 0 or raw.size % (dim + 1):
        raise ValueError(f"Invalid vector file: {path}")
    records = raw.reshape(-1, dim + 1)
    if not np.all(records[:, 0] == dim):
        raise ValueError(f"Inconsistent record dimensions: {path}")
    records = records[:limit, 1:]
    if integer:
        return np.ascontiguousarray(records, dtype=np.int64)
    result = np.ascontiguousarray(records.view("<f4"), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite vectors: {path}")
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def write_csv(path, rows, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def process_rss_bytes():
    if Path("/proc/self/statm").exists():
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             check=True, capture_output=True, text=True).stdout.strip()) * 1024


class RssSampler:
    def __init__(self):
        self.peak_bytes, self.samples, self.error = 0, 0, None
        self.stop = threading.Event()

    def sample(self):
        self.peak_bytes = max(self.peak_bytes, process_rss_bytes())
        self.samples += 1

    def poll(self):
        try:
            while not self.stop.wait(.01):
                self.sample()
        except BaseException as error:
            self.error = error

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()
        if self.error is not None:
            raise RuntimeError("CPU RSS sampling failed") from self.error


def build_or_load(faiss, args, base_path, stem, library_hash):
    path = args.index_dir / (stem + ".faiss")
    manifest_path = path.with_suffix(".build.json")
    signature = dict(base=file_signature(base_path), nlist=args.nlist, metric="squared_l2",
        cpu_threads=args.cpu_threads, seed=args.seed, coarse_niter=args.coarse_niter,
        train_fraction=args.train_fraction, add_batch_size=args.add_batch_size,
        faiss_version=faiss.__version__, library_sha256=library_hash,
        builder_sha256=digest(__file__), numpy_version=np.__version__, architecture=platform.machine())
    if (path.exists() or manifest_path.exists()) and not args.rebuild_index:
        build_progress(args, f"[Cache] Checking saved CPU index: {path}")
        if not path.exists() or not manifest_path.exists():
            raise ValueError("Incomplete CPU index cache; use --rebuild-index")
        manifest = json.loads(manifest_path.read_text())
        if not compatible_build_signature(manifest["signature"], signature) or manifest["index_sha256"] != digest(path):
            raise ValueError("Cached CPU index/settings differ; use --rebuild-index")
        signature = manifest["signature"]  # Preserve the original builder and resume settings.
        index, build, loaded = faiss.read_index(str(path)), manifest["build"], True
        print(f"Loaded {path}; retaining original build times", flush=True)
    else:
        build_progress(args, f"[1/4] Loading base vectors: {base_path} ({base_path.stat().st_size / MIB:.1f} MiB)")
        begin = time.perf_counter()
        base = read_vectors(base_path)
        if len(base) < args.nlist:
            raise ValueError("Training requires at least nlist base vectors")
        preparation = time.perf_counter() - begin
        build_progress(args, f"[1/4] Loaded {len(base):,} vectors, dim={base.shape[1]}, elapsed={preparation:.1f}s")
        max_points = max(1, math.ceil(len(base) * args.train_fraction / args.nlist))
        print(f"CPU IVF-Flat train/add: {base.shape}, nlist={args.nlist}, threads={args.cpu_threads}", flush=True)
        build_progress(args, f"[2/4] Training k-means: {args.coarse_niter} iterations, "
                       f"up to {min(len(base), max_points * args.nlist):,} samples; "
                       "Faiss iteration numbers start at 0")
        begin = time.perf_counter()
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(base.shape[1]), base.shape[1], args.nlist, faiss.METRIC_L2)
        index.cp.niter, index.cp.nredo, index.cp.seed = args.coarse_niter, 1, args.seed
        index.cp.max_points_per_centroid = max_points
        index.cp.verbose = not args.quiet_build
        train_start = time.perf_counter()
        index.train(base)
        train_seconds = time.perf_counter() - train_start
        index.cp.verbose = False
        build_progress(args, f"[2/4] Training complete: {train_seconds:.1f}s")
        build_progress(args, f"[3/4] Adding {len(base):,} vectors in batches of {args.add_batch_size:,}")
        add_start = time.perf_counter()
        for offset in range(0, len(base), args.add_batch_size):
            index.add(base[offset:offset + args.add_batch_size])
            added = min(offset + args.add_batch_size, len(base))
            elapsed = time.perf_counter() - add_start
            rate = added / max(elapsed, 1e-9)
            eta = (len(base) - added) / rate
            build_progress(args, f"[3/4] Added {added:,}/{len(base):,} ({added / len(base):.1%}) "
                           f"| elapsed={elapsed:.1f}s | {rate:,.0f} vectors/s | ETA~{eta:.1f}s")
        add_seconds = time.perf_counter() - add_start
        build_seconds = time.perf_counter() - begin
        if not index.is_trained or index.ntotal != len(base):
            raise RuntimeError("CPU index build is incomplete")
        path.parent.mkdir(parents=True, exist_ok=True)
        build_progress(args, f"[4/4] Saving index: {path}")
        begin = time.perf_counter()
        temporary = path.with_suffix(".faiss.tmp")
        faiss.write_index(index, str(temporary))
        temporary.replace(path)
        save_seconds = time.perf_counter() - begin
        build_progress(args, f"[4/4] Saved {path.stat().st_size / MIB:.3f} MiB in {save_seconds:.1f}s")
        build = dict(dataset=args.dataset, method="Faiss-CPU", execution_device="cpu", nlist=args.nlist,
            n_vectors=int(index.ntotal), dim=int(index.d), metric="squared_l2", cpu_threads=args.cpu_threads,
            cpu_information=cpu_information(),
            build_time_seconds=build_seconds, train_seconds=train_seconds, add_seconds=add_seconds,
            preparation_seconds=preparation, serialize_seconds=save_seconds,
            index_size_bytes=path.stat().st_size, index_size_mb=path.stat().st_size / MIB,
            index_size_unit="MiB", index_path=str(path.resolve()), training_input_rows=len(base),
            max_points_per_centroid=max_points, max_training_rows=min(len(base), max_points * args.nlist),
            timing_definition="CPU constructor + train + full add; excludes data loading and serialization",
            progress_logging=not args.quiet_build,
            progress_timing_note="Progress output during train/add is included in wall-clock build time",
            size_definition="actual serialized IVF-Flat file, including centroids, FP32 vectors and IDs")
        write_json(manifest_path, dict(signature=signature, build=build, index_sha256=digest(path)))
        loaded = False
        print(f"Build {build_seconds:.3f} s; index {build['index_size_mb']:.3f} MiB", flush=True)
    if type(index).__name__ != "IndexIVFFlat" or index.metric_type != faiss.METRIC_L2 or index.nlist != args.nlist:
        raise ValueError("Expected a CPU squared-L2 IndexIVFFlat")
    counts = np.array([index.invlists.list_size(i) for i in range(index.nlist)], dtype=np.int64)
    if int(counts.sum()) != int(index.ntotal):
        raise RuntimeError("CPU list sizes do not sum to ntotal")
    return index, counts, build, loaded, signature


def make_batches(queries, size):
    batches = []
    for start in range(0, len(queries), size):
        valid = min(size, len(queries) - start)
        batch = np.empty((size, queries.shape[1]), dtype=np.float32)
        batch[:valid] = queries[start:start + valid]
        batch[valid:] = batch[valid - 1]
        batches.append((batch, valid))
    return batches


def account_visits(faiss, index, counts, batch, valid, native_ndis):
    # IndexIVF::search splits queries BEFORE calling the coarse quantizer in
    # parallel_mode=0. Match those exact slice sizes: a whole-batch search can
    # select a different BLAS kernel and round near-tied coarse distances
    # differently, changing both list assignments and visited-vector counts.
    threads = (min(faiss.omp_get_max_threads(), len(batch))
               if (int(index.parallel_mode) & ~1024) == 0 else 1)
    selected = np.concatenate([
        index.quantizer.search(batch[len(batch) * i // threads:
                                     len(batch) * (i + 1) // threads], int(index.nprobe))[1]
        for i in range(threads)
    ])
    if (selected < 0).any() or (selected >= len(counts)).any():
        raise RuntimeError("Invalid coarse assignments")
    visits = counts[selected].sum(axis=1)
    if int(visits.sum()) != native_ndis:
        raise RuntimeError(f"Native Faiss ndis and list-size accounting disagree: "
                           f"native={native_ndis}, accounted={int(visits.sum())}, "
                           f"batch={len(batch)}, slices={threads}")
    return visits[:valid]


def run(args):
    import faiss
    if args.require_ascend_package and not hasattr(faiss, "NpuIndexIVFPQ"):
        raise RuntimeError(f"Expected installed Faiss Ascend package, got {faiss.__file__}")
    faiss.omp_set_num_threads(args.cpu_threads)
    library_hash = {p.name: digest(p) for p in Path(faiss.__file__).parent.glob("_swigfaiss*.so")}
    stem = f"Ascend_Faiss_IVFFLAT_{args.dataset}_nlist{args.nlist}"
    csv_path, json_path = args.output_dir / (stem + ".csv"), args.output_dir / (stem + ".json")
    if (csv_path.exists() or json_path.exists()) and not args.resume and not args.overwrite and not args.append:
        raise FileExistsError(f"Use --resume or --overwrite: {csv_path}")
    if args.rebuild_index and (csv_path.exists() or json_path.exists()) and not args.overwrite:
        raise FileExistsError("Rebuilding with existing results requires --overwrite")
    folder = args.data_root / args.dataset
    base_path = folder / f"{args.dataset}_base.fvecs"
    query_path = folder / f"{args.dataset}_query.fvecs"
    truth_path = folder / f"{args.dataset}_groundtruth.ivecs"
    queries = read_vectors(query_path, limit=args.nq)
    truth = read_vectors(truth_path, integer=True, limit=args.nq)
    probes = args.nprobes or NPROBES[args.dataset]
    if not len(queries) or len(queries) != len(truth) or truth.shape[1] < args.topk or max(probes) > args.nlist:
        raise ValueError("Invalid query/truth counts, K or nprobe")
    index, counts, build, loaded, signature = build_or_load(faiss, args, base_path, stem, library_hash)
    if queries.shape[1] != index.d or args.topk > index.ntotal:
        raise ValueError("Query dimension or K differs from index")
    build_fields = ("dataset", "method", "nlist", "n_vectors", "dim", "cpu_threads", "build_time_seconds",
                    "train_seconds", "add_seconds", "serialize_seconds", "index_size_bytes", "index_size_mb", "preparation_seconds")
    write_csv(args.output_dir / "Build" / (stem + "_build.csv"),
              [{key: build[key] for key in build_fields}], build_fields)
    write_json(args.output_dir / "Build" / (stem + "_build.json"),
               dict(build=build, signature=signature, index_loaded_from_cache=loaded))
    config = dict(dataset=args.dataset, kind="ivfflat", execution_device="cpu", metric="squared_l2",
        nq=len(queries), topk=args.topk, nlist=args.nlist, q_batches=args.q_batches, nprobes=list(probes),
        cpu_threads=args.cpu_threads, warmup=args.warmup, repetitions=args.repetitions,
        index_sha256=digest(build["index_path"]), build_signature=signature,
        inputs=[file_signature(query_path), file_signature(truth_path)],
        faiss_version=faiss.__version__, faiss_path=faiss.__file__,
        faiss_has_npu_bindings=hasattr(faiss, "NpuIndexIVFPQ"),
        cpu_information=cpu_information(),
        openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS"), mkl_threads=os.environ.get("MKL_NUM_THREADS"))
    keys = [(size, probe) for size in args.q_batches for probe in probes]
    completed, details = {}, {}
    if (csv_path.exists() or json_path.exists()) and not args.overwrite:
        if not (args.resume or args.append) or not csv_path.exists() or not json_path.exists():
            raise ValueError("Resume requires both CSV and JSON files")
        old = json.loads(json_path.read_text())
        old_config = old["config"]
        ignored = {"q_batches", "nprobes"} if args.append else set()
        if {k:v for k,v in old_config.items() if k not in ignored} != {k:v for k,v in config.items() if k not in ignored}:
            raise ValueError("Resume configuration, source or input files differ")
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FIELDS):
                raise ValueError("Resume CSV schema differs")
            for row in reader:
                key = int(row["q_batch"]), int(row["nprobes"])
                if (not args.append and key not in keys) or key in completed or int(row["K"]) != args.topk or not all(math.isfinite(float(row[k])) for k in FIELDS):
                    raise ValueError("Invalid resume row")
                completed[key] = row
        details = {(r["q_batch"], r["nprobes"]): r for r in old["results"]}
        if details.keys() != completed.keys():
            raise ValueError("Resume CSV and JSON rows differ")
        if args.append:
            keys = list(dict.fromkeys([*completed, *keys]))
            config["q_batches"] = sorted({k[0] for k in keys})
            config["nprobes"] = sorted({k[1] for k in keys})
    meta = dict(status="running", backend="faiss_cpu", device="cpu", config=config, build=build,
        index_loaded_from_cache=loaded, planned_keys=[list(k) for k in keys], cpu_architecture=platform.machine(), cpu_processor=platform.processor(),
        cpu_threads_available=default_cpu_threads(), expected_rows=len(keys),
        index_construction="Independent CPU train/add from raw vectors; no TensorVS index import; training may yield different IVF lists",
        timing="median full-query sweep; sum of per-batch CPU IndexIVFFlat.search wall time; resident padded queries; excludes recall, visits and memory sampling",
        recall="mean top-K ID overlap with dataset squared-L2 ground truth divided by K",
        memory_metric="peak_gpu_memory=0 (no GPU/NPU allocation); CPU sampled process RSS in JSON peak_rss_memory_mib, not comparable to HBM",
        visited_vectors_definition="sum of actual selected list sizes, validated against native indexIVF_stats.ndis; padding queries excluded")

    def save():
        meta.update(completed_rows=len(completed), results=[details[k] for k in keys if k in details])
        write_json(json_path, meta)
        write_csv(csv_path, [completed[k] for k in keys if k in completed])

    save()
    begin_run = time.perf_counter()
    try:
        for size, probe in keys:
            if (size, probe) in completed:
                continue
            index.nprobe = probe
            batches = make_batches(queries, size)

            def sweep(account=False, sampler=None):
                elapsed, found, scanned = 0., [], []
                for batch, valid in batches:
                    if account:
                        faiss.cvar.indexIVF_stats.reset()
                    begin = time.perf_counter()
                    distances, ids = index.search(batch, args.topk)
                    elapsed += time.perf_counter() - begin
                    if account:
                        ndis = int(faiss.cvar.indexIVF_stats.ndis)
                        if (ids[:valid] < -1).any() or (ids[:valid] >= index.ntotal).any() or not np.isfinite(distances[:valid][ids[:valid] >= 0]).all():
                            raise RuntimeError("Invalid CPU search output")
                        found.append(ids[:valid].copy())
                        scanned.append(account_visits(faiss, index, counts, batch, valid, ndis))
                    if sampler is not None:
                        sampler.sample()
                return elapsed, found, scanned

            print(f"{args.dataset} Faiss-CPU ivfflat QBatch={size} K={args.topk} nprobe={probe} threads={args.cpu_threads}", flush=True)
            for _ in range(args.warmup):
                sweep()
            times = [sweep()[0] for _ in range(args.repetitions)]
            with RssSampler() as memory:
                _, found, scanned = sweep(account=True, sampler=memory)
            found, scanned = np.concatenate(found), np.concatenate(scanned)
            recall = float(np.mean([len(set(row) & set(gt[:args.topk])) / args.topk for row, gt in zip(found, truth)]))
            row = dict(nprobes=probe, q_batch=size, K=args.topk, qps=len(queries) / statistics.median(times),
                       recall=recall, visited_vectors_per_query=float(scanned.mean()), peak_gpu_memory=0.)
            if not all(math.isfinite(float(value)) for value in row.values()):
                raise RuntimeError("Nonfinite CPU metric")
            completed[size, probe] = row
            details[size, probe] = dict(row, samples_seconds=times, peak_rss_memory_mib=memory.peak_bytes / MIB,
                memory_samples=memory.samples, missing_neighbor_slots=int((found == -1).sum()),
                visit_statistics=dict(visited_vectors_total=int(scanned.sum()), visited_vectors_min_per_query=int(scanned.min()),
                    visited_vectors_max_per_query=int(scanned.max()), visited_data_mb_per_query=float(scanned.mean()) * index.d * 4 / MIB))
            save()
            print(f"  QPS={row['qps']:.2f} recall={recall:.6f} CPU RSS={memory.peak_bytes / MIB:.1f} MiB", flush=True)
        meta.update(status="complete", wall_seconds=time.perf_counter() - begin_run)
        save()
    except BaseException as error:
        meta.update(status="failed", error=repr(error))
        save()
        raise
    print(f"Saved {csv_path}; build statistics: {args.output_dir / 'Build'}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=tuple(NPROBES), required=True)
    p.add_argument("--data-root", type=Path, default=Path("/home/ma-user/work/data"))
    p.add_argument("--output-dir", type=Path, default=ROOT / "result")
    p.add_argument("--index-dir", type=Path, default=ROOT / "indexes/faiss_ascend_cpu")
    p.add_argument("--cpu-threads", type=int, default=default_cpu_threads())
    p.add_argument("--nlist", type=int, default=1024)
    p.add_argument("--coarse-niter", type=int, default=20)
    p.add_argument("--train-fraction", type=float, default=.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--add-batch-size", type=int, default=65536)
    p.add_argument("--nq", type=int, default=512, help="Total query count (default: 512)")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--q-batches", type=int, nargs="+", default=[32, 128, 512])
    p.add_argument("--nprobes", type=int, nargs="+")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--require-ascend-package", action="store_true")
    p.add_argument("--quiet-build", action="store_true", help="Disable build-stage, iteration and add-batch progress")
    p.add_argument("--rebuild-index", action="store_true")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--append", action="store_true", help="Preserve existing results and add selected query configurations")
    group.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if min(args.cpu_threads, args.nlist, args.coarse_niter, args.add_batch_size, args.nq, args.topk,
           args.repetitions, *args.q_batches, *(args.nprobes or [1])) <= 0 or args.warmup < 0:
        p.error("Sizes, threads and repetitions must be positive; warmup must be nonnegative")
    if not 0 < args.train_fraction <= 1:
        p.error("train-fraction must be in (0, 1]")
    if len(set(args.q_batches)) != len(args.q_batches) or (args.nprobes and len(set(args.nprobes)) != len(args.nprobes)):
        p.error("Duplicate query batch or nprobe")
    if args.rebuild_index and (args.resume or args.append):
        p.error("Use --rebuild-index --overwrite for a fresh experiment, not --resume")
    return args


if __name__ == "__main__":
    run(parse_args())
