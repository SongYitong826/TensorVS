"""Build hipVS IVF indexes from original vectors and benchmark on AMD GPU.

Run one dataset and index type per process. The seven CSV columns match
bench/amd_tensor_bench.py. Indexes and result JSON are retained for resume.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

import cupy as cp
import numpy as np
from cuvs.neighbors import ivf_flat, ivf_pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("nprobes", "q_batch", "K", "qps", "recall",
          "visited_vectors_per_query", "peak_gpu_memory")
NPROBES = {
    "glove1.2m": (50, 80, 100, 120, 140, 160, 180, 200, 220, 240,
                   260, 280, 300, 320, 340, 360, 380, 400),
    "hunyuan": (350, 380, 400, 450, 500, 550, 600, 650, 700, 750,
                800, 850, 900, 950, 1000),
}
TRAIN_LABEL = {"glove1.2m": 512, "hunyuan": 256}
MEBIBYTE = 1 << 20


def read_vecs(path: Path, dtype: np.dtype) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"Empty vector file: {path}")
    dim = int(raw[0])
    if dim <= 0 or raw.size % (dim + 1):
        raise ValueError(f"Invalid vector records: {path}")
    rows = raw.reshape(-1, dim + 1)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"Inconsistent vector dimensions: {path}")
    return np.ascontiguousarray(rows[:, 1:].view(dtype))


def read_base(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        value = np.load(path, mmap_mode="r")
        if value.ndim != 2 or value.dtype != np.float32:
            raise ValueError("Base .npy must be a 2D float32 array")
        return value
    if path.suffix == ".fvecs":
        return read_vecs(path, np.float32)
    raise ValueError(f"Expected .fvecs or .npy base: {path}")


def find_base(directory: Path, dataset: str, supplied: Path | None) -> Path:
    if supplied is not None:
        if not supplied.is_file():
            raise FileNotFoundError(supplied)
        return supplied
    candidates = [directory / f"{dataset}_base{ext}" for ext in (".fvecs", ".npy")]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one original base file: {candidates}; found {found}")
    return found[0]


def index_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    directory = args.data_root / args.dataset / "GPUIndex" / "hipVS"
    stem = f"{args.kind}_nlist{args.nlist}"
    if args.kind == "ivfpq":
        stem += f"_M{args.pq_dim}_nbits{args.pq_bits}"
    path = directory / f"{stem}.bin"
    return path, path.with_suffix(".json")


def output_path(args: argparse.Namespace) -> Path:
    prefix = "IVFFLAT" if args.kind == "ivfflat" else "IVFPQ"
    stem = f"hipVS_{prefix}_{args.dataset}_nlist{args.nlist}"
    if args.kind == "ivfpq":
        stem += f"_M{args.pq_dim}_nbits{args.pq_bits}_train{TRAIN_LABEL[args.dataset]}"
    return args.output_dir / f"AMD_{stem}.csv"


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


class AmdProcessVramSampler:
    def __init__(self, gpu_id: int, interval: float = 0.05):
        self.gpu_id = gpu_id
        self.interval = interval
        self.pid = os.getpid()
        self.peak_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def sample(self) -> None:
        output = subprocess.run(["amd-smi", "process", "-g", str(self.gpu_id),
                                 "-p", str(self.pid), "--json"], check=True,
                                capture_output=True, text=True, timeout=15).stdout
        for gpu in json.loads(output):
            for entry in gpu.get("process_list", []):
                info = entry.get("process_info")
                if isinstance(info, dict) and int(info.get("pid", -1)) == self.pid:
                    memory = info["memory_usage"]["vram_mem"]
                    if memory.get("unit") != "B":
                        raise ValueError(f"Unexpected VRAM unit: {memory}")
                    self.peak_bytes = max(self.peak_bytes, int(memory["value"]))
                    self.samples += 1
                    return
        raise RuntimeError(f"AMD-SMI did not report PID {self.pid}")

    def __enter__(self):
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except BaseException as error:
                self._error = error
                self._stop.set()

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()
        if self._error is not None:
            raise RuntimeError("AMD-SMI sampling failed") from self._error


def squared_distances(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    return cp.sum(a * a, axis=1)[:, None] + cp.sum(b * b, axis=1)[None, :] - 2 * a @ b.T


def build(args: argparse.Namespace) -> None:
    index_path, meta_path = index_paths(args)
    if index_path.exists() or meta_path.exists():
        if args.overwrite_index:
            pass
        elif index_path.is_file() and meta_path.is_file():
            print(f"Index already built: {index_path}", flush=True)
            return
        else:
            raise FileExistsError(f"Incomplete index files: {index_path}; use --overwrite-index")
    directory = args.data_root / args.dataset
    base_path = find_base(directory, args.dataset, args.base)
    base = read_base(base_path)
    if len(base) < args.nlist or base.shape[1] <= 0:
        raise ValueError("Base is too small for nlist or has zero dimensions")
    if args.kind == "ivfpq" and args.pq_dim > base.shape[1]:
        raise ValueError("pq-dim exceeds base dimension")
    print(f"Building hipVS {args.kind} from {base_path}: {base.shape}", flush=True)
    gpu_base = cp.asarray(base)
    module = ivf_flat if args.kind == "ivfflat" else ivf_pq
    common = dict(n_lists=args.nlist, metric="sqeuclidean",
                  kmeans_n_iters=args.kmeans_iters,
                  kmeans_trainset_fraction=args.train_fraction)
    if args.kind == "ivfpq":
        common.update(pq_dim=args.pq_dim, pq_bits=args.pq_bits,
                      max_train_points_per_pq_code=args.pq_train_points)
    params = module.IndexParams(**common)
    start = time.perf_counter()
    index = module.build(params, gpu_base)
    cp.cuda.get_current_stream().synchronize()
    build_seconds = time.perf_counter() - start
    # DeviceTensorView reports element strides through its CUDA interface;
    # CuPy expects byte strides. The supported host copy preserves layout.
    centers = cp.asarray(index.centers.copy_to_host())
    counts = cp.zeros(args.nlist, dtype=cp.int64)
    # Count vectors assigned to hipVS coarse centers. These counts allow
    # visited-vector estimates at benchmark time without retaining the base.
    for start_row in range(0, len(gpu_base), args.assignment_batch):
        assigned = cp.argmin(squared_distances(
            gpu_base[start_row:start_row + args.assignment_batch], centers), axis=1)
        counts += cp.bincount(assigned, minlength=args.nlist)
    if int(counts.sum()) != len(base):
        raise RuntimeError("Coarse assignment count mismatch")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp = index_path.with_suffix(".bin.tmp")
    module.save(str(temp), index)
    temp.replace(index_path)
    meta = dict(dataset=args.dataset, kind=args.kind, nlist=args.nlist,
                pq_dim=args.pq_dim if args.kind == "ivfpq" else None,
                pq_bits=args.pq_bits if args.kind == "ivfpq" else None,
                n_vectors=len(base), dim=base.shape[1], base_path=str(base_path.resolve()),
                base_bytes=base_path.stat().st_size, build_seconds=build_seconds,
                index_size_bytes=index_path.stat().st_size,
                list_counts=cp.asnumpy(counts).tolist(),
                list_counts_method="nearest hipVS coarse center by squared L2; ties may differ")
    atomic_json(meta_path, meta)
    print(f"Saved {index_path} ({index_path.stat().st_size / MEBIBYTE:.1f} MiB) "
          f"in {build_seconds:.1f}s", flush=True)


def recall_at_k(ids: np.ndarray, truth: np.ndarray, k: int) -> float:
    return float(sum(len(set(found) & set(actual[:k])) for found, actual in zip(ids, truth))
                 / (len(ids) * k))


def benchmark(args: argparse.Namespace) -> None:
    index_path, meta_path = index_paths(args)
    if not index_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Build hipVS index first: {index_path}")
    index_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (index_meta["dataset"], index_meta["kind"], index_meta["nlist"]) != (args.dataset, args.kind, args.nlist):
        raise ValueError("Index metadata does not match requested experiment")
    module = ivf_flat if args.kind == "ivfflat" else ivf_pq
    index = module.load(str(index_path))
    directory = args.data_root / args.dataset
    query_path = directory / f"{args.dataset}_query.fvecs"
    truth_path = directory / f"{args.dataset}_groundtruth.ivecs"
    queries = read_vecs(query_path, np.float32)
    truth = read_vecs(truth_path, np.int32)
    if len(queries) != len(truth) or queries.shape[1] != index_meta["dim"]:
        raise ValueError("Query, truth and index dimensions/counts differ")
    nq = min(args.nq, len(queries))
    if truth.shape[1] < args.topk or args.topk > index_meta["n_vectors"]:
        raise ValueError("K exceeds ground truth or base vector count")
    queries, truth = queries[:nq], truth[:nq]
    gpu_queries = cp.asarray(queries)
    centers = cp.asarray(index.centers.copy_to_host())
    counts = cp.asarray(index_meta["list_counts"], dtype=cp.int64)
    nprobes = tuple(args.nprobes or NPROBES[args.dataset])
    if max(nprobes) > args.nlist:
        raise ValueError("nprobe exceeds nlist")
    order = cp.argsort(squared_distances(gpu_queries, centers), axis=1)
    visits = {probe: float(cp.mean(cp.sum(counts[order[:, :probe]], axis=1)).get())
              for probe in nprobes}
    csv_path = output_path(args)
    json_path = csv_path.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    config = dict(dataset=args.dataset, kind=args.kind, nq=nq, topk=args.topk,
                  q_batches=list(args.q_batches), nprobes=list(nprobes),
                  gpu_id=args.gpu_id, warmup=args.warmup, repetitions=args.repetitions,
                  lut_dtype=args.lut_dtype, index=str(index_path.resolve()),
                  index_size_bytes=index_path.stat().st_size,
                  index_mtime_ns=index_path.stat().st_mtime_ns,
                  query=str(query_path.resolve()), query_size_bytes=query_path.stat().st_size,
                  truth=str(truth_path.resolve()), truth_size_bytes=truth_path.stat().st_size)
    completed: dict[tuple[int, int], dict] = {}
    details: dict[tuple[int, int], dict] = {}
    if csv_path.exists() or json_path.exists():
        if args.overwrite:
            pass
        elif not args.resume or not csv_path.exists() or not json_path.exists():
            raise FileExistsError(f"Result exists; use --resume or --overwrite: {csv_path}")
        else:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
            if previous.get("config") != config:
                raise ValueError("Existing result configuration differs")
            with csv_path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != list(FIELDS):
                    raise ValueError("Existing CSV columns differ")
                for row in reader:
                    key = (int(row["q_batch"]), int(row["nprobes"]))
                    if key in completed or key[0] not in args.q_batches or key[1] not in nprobes:
                        raise ValueError(f"Invalid existing result row: {key}")
                    if int(row["K"]) != args.topk or not all(math.isfinite(float(row[f])) for f in FIELDS):
                        raise ValueError(f"Invalid result values: {row}")
                    completed[key] = row
            details = {(int(row["q_batch"]), int(row["nprobes"])): row
                       for row in previous.get("results", [])
                       if (int(row["q_batch"]), int(row["nprobes"])) in completed}
    keys = [(batch, probe) for batch in args.q_batches for probe in nprobes]
    if len(completed) == len(keys):
        print(f"Already complete: {csv_path}", flush=True)
        return
    metadata = dict(status="running", config=config, backend="hipVS",
                    implementation="hipVS IVF GPU search on AMD ROCm",
                    index_metadata=index_meta, expected_rows=len(keys),
                    memory_unit="MiB", memory_metric="sampled AMD-SMI current-process VRAM peak; "
                    "includes resident index, queries and GPU runtime; separate search sweep",
                    timing="median synchronized full-query sweeps; excludes recall and memory sweep",
                    recall="mean overlap with ground-truth top-K divided by K",
                    visits="mean sum of coarse-list counts across selected nprobe centers; "
                    "list counts reconstructed from base during build",
                    started_utc=datetime.now(timezone.utc).isoformat(), results=[])

    def save() -> None:
        metadata["results"] = [details[key] for key in keys if key in details]
        metadata["completed_rows"] = len(completed)
        atomic_json(json_path, metadata)
        atomic_csv(csv_path, [completed[key] for key in keys if key in completed])

    save()
    started = time.perf_counter()
    try:
        for q_batch, probe in keys:
            if (q_batch, probe) in completed:
                continue
            batches = []
            for first in range(0, nq, q_batch):
                valid = min(q_batch, nq - first)
                batch = gpu_queries[first:first + valid]
                if valid < q_batch:
                    batch = cp.concatenate((batch, cp.repeat(batch[-1:], q_batch - valid, axis=0)))
                batches.append((batch, valid))
            params = (ivf_flat.SearchParams(n_probes=probe) if args.kind == "ivfflat"
                      else ivf_pq.SearchParams(n_probes=probe,
                                               lut_dtype=np.float16 if args.lut_dtype == "fp16" else np.float32))

            def sweep(retain: bool = False, sampler: AmdProcessVramSampler | None = None):
                elapsed, saved = 0.0, []
                for batch, valid in batches:
                    cp.cuda.get_current_stream().synchronize()
                    start = time.perf_counter()
                    distances, ids = module.search(params, index, batch, args.topk)
                    distances, ids = cp.asarray(distances), cp.asarray(ids)
                    cp.cuda.get_current_stream().synchronize()
                    elapsed += time.perf_counter() - start
                    if sampler is not None:
                        sampler.sample()
                    if retain:
                        if not bool(cp.all(cp.isfinite(distances[:valid])).get()):
                            raise RuntimeError("Non-finite search distance")
                        saved.append(cp.asnumpy(ids[:valid]))
                    del distances, ids
                return elapsed, saved

            print(f"{args.dataset} {args.kind} QBatch={q_batch} K={args.topk} nprobe={probe}", flush=True)
            for _ in range(args.warmup):
                sweep()
            samples = []
            for repetition in range(args.repetitions):
                elapsed, saved = sweep(retain=repetition == args.repetitions - 1)
                samples.append(elapsed)
            ids = np.concatenate(saved)
            if ids.shape != (nq, args.topk) or np.any(ids < 0) or np.any(ids >= index_meta["n_vectors"]):
                raise RuntimeError("Invalid or missing search IDs")
            recall = recall_at_k(ids, truth, args.topk)
            with AmdProcessVramSampler(args.gpu_id) as sampler:
                sweep(sampler=sampler)
            row = dict(nprobes=probe, q_batch=q_batch, K=args.topk,
                       qps=nq / statistics.median(samples), recall=recall,
                       visited_vectors_per_query=visits[probe],
                       peak_gpu_memory=sampler.peak_bytes / MEBIBYTE)
            if not all(math.isfinite(float(value)) for value in row.values()):
                raise RuntimeError(f"Non-finite metric: {row}")
            key = (q_batch, probe)
            completed[key] = row
            details[key] = dict(row, samples_seconds=samples, memory_samples=sampler.samples)
            save()
            print(f"  QPS={row['qps']:.2f} recall={recall:.6f} VRAM={row['peak_gpu_memory']:.1f} MiB", flush=True)
        metadata.update(status="complete", wall_seconds=time.perf_counter() - started)
        save()
    except BaseException as error:
        metadata.update(status="failed", error=repr(error),
                        wall_seconds=time.perf_counter() - started)
        save()
        raise
    print(f"Saved {csv_path} and {json_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(NPROBES), required=True)
    parser.add_argument("--kind", choices=("ivfflat", "ivfpq"), required=True)
    parser.add_argument("--stage", choices=("build", "bench", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=Path("/root/vector_data"))
    parser.add_argument("--base", type=Path, help="Original base vectors, .fvecs or 2D float32 .npy")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "result")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--pq-dim", type=int, help="Default: 25 for GloVe, 384 for Hunyuan")
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--pq-train-points", type=int, default=256)
    parser.add_argument("--assignment-batch", type=int, default=8192)
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--q-batches", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--nprobes", type=int, nargs="+")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--lut-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-index", action="store_true")
    args = parser.parse_args()
    args.pq_dim = args.pq_dim or (25 if args.dataset == "glove1.2m" else 384)
    values = (*args.q_batches, *(args.nprobes or NPROBES[args.dataset]), args.nq,
              args.topk, args.repetitions, args.assignment_batch, args.nlist,
              args.pq_dim, args.pq_bits, args.kmeans_iters, args.pq_train_points)
    if min(values) <= 0 or args.warmup < 0 or not (0 < args.train_fraction <= 1):
        parser.error("Counts must be positive, warmup nonnegative, training fraction in (0, 1]")
    if len(set(args.q_batches)) != len(args.q_batches) or len(set(args.nprobes or ())) != len(args.nprobes or ()):
        parser.error("Duplicate batch size or nprobe")
    if args.resume and args.overwrite:
        parser.error("Choose --resume or --overwrite")
    cp.cuda.Device(args.gpu_id).use()
    if args.stage in ("build", "all"):
        build(args)
    if args.stage in ("bench", "all"):
        benchmark(args)


if __name__ == "__main__":
    main()
