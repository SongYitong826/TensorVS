"""Benchmark Faiss ROCm IVF-Flat search on the AMD GPU.

The CSV schema matches the existing AMD TensorVS and hipVS search benchmarks.
Query vectors remain on the GPU while timed searches run. Build metrics are
recorded separately by amd_ivfflat_build.py.
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
import time

import cupy as cp
import faiss
import numpy as np

from amd_ivfflat_build import VramSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "nprobes", "q_batch", "K", "qps", "recall",
    "visited_vectors_per_query", "peak_gpu_memory",
)
NPROBES = {
    "glove1.2m": (50, 80, 100, 120, 140, 160, 180, 200, 220, 240,
                   260, 280, 300, 320, 340, 360, 380, 400),
    "hunyuan": (350, 380, 400, 450, 500, 550, 600, 650, 700, 750,
                800, 850, 900, 950, 1000),
}
MEBIBYTE = 1 << 20


def read_vecs(path: Path, dtype: np.dtype) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i4")
    if raw.size == 0:
        raise ValueError(f"Empty vector file: {path}")
    dim = int(raw[0])
    if dim <= 0 or raw.size % (dim + 1):
        raise ValueError(f"Invalid vector records: {path}")
    records = raw.reshape(-1, dim + 1)
    if not np.all(records[:, 0] == dim):
        raise ValueError(f"Inconsistent dimensions: {path}")
    return np.ascontiguousarray(records[:, 1:].view(dtype))


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    return float(sum(len(set(row) & set(target[:k]))
                     for row, target in zip(found, truth)) / (len(found) * k))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in FIELDS} for row in rows])
    temporary.replace(path)


def gpu_search(index, batch: cp.ndarray, topk: int) -> tuple[cp.ndarray, cp.ndarray]:
    """Use Faiss's SWIG GPU pointer API without importing PyTorch's LLVM stack."""
    distances = cp.empty((len(batch), topk), dtype=cp.float32)
    ids = cp.empty((len(batch), topk), dtype=cp.int64)
    index.search_ex(
        len(batch), faiss.cast_integer_to_float_ptr(batch.data.ptr),
        faiss.Float32, topk,
        faiss.cast_integer_to_float_ptr(distances.data.ptr),
        faiss.cast_integer_to_idx_t_ptr(ids.data.ptr),
    )
    faiss.gpu_sync_all_devices()
    return distances, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(NPROBES), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/vector_data"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "result")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--q-batches", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--nprobes", type=int, nargs="+")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    probes = tuple(args.nprobes or NPROBES[args.dataset])
    if min(args.nlist, args.nq, args.topk, args.repetitions, *args.q_batches, *probes) <= 0:
        parser.error("nlist, NQ, K, repetitions, batch sizes and probes must be positive")
    if args.warmup < 0 or args.gpu_id < 0:
        parser.error("warmup and GPU ID must be nonnegative")
    if max(probes) > args.nlist:
        parser.error("nprobe exceeds nlist")
    if len(set(args.q_batches)) != len(args.q_batches) or len(set(probes)) != len(probes):
        parser.error("Duplicate batch size or nprobe")
    if args.resume and args.overwrite:
        parser.error("Choose either --resume or --overwrite")
    if args.gpu_id >= faiss.get_num_gpus():
        parser.error(f"Faiss ROCm cannot see GPU {args.gpu_id}")
    cp.cuda.Device(args.gpu_id).use()
    directory = args.data_root / args.dataset
    index_path = directory / "GPUIndex" / "FaissROCm" / f"ivfflat_nlist{args.nlist}_build.index"
    query_path = directory / f"{args.dataset}_query.fvecs"
    truth_path = directory / f"{args.dataset}_groundtruth.ivecs"
    if not index_path.is_file():
        raise FileNotFoundError(f"Build the Faiss ROCm IVF-Flat index first: {index_path}")
    cpu_index = faiss.read_index(str(index_path))
    if cpu_index.d <= 0 or cpu_index.ntotal <= 0 or cpu_index.nlist != args.nlist:
        raise ValueError("Index dimensions, size or nlist are invalid")
    queries = read_vecs(query_path, np.float32)
    truth = read_vecs(truth_path, np.int32)
    if len(queries) != len(truth) or queries.shape[1] != cpu_index.d:
        raise ValueError("Index, queries and ground truth have inconsistent dimensions/counts")
    if truth.shape[1] < args.topk or args.topk > cpu_index.ntotal:
        raise ValueError("K exceeds ground truth width or index size")
    nq = min(args.nq, len(queries))
    queries, truth = queries[:nq], truth[:nq]
    # These exact list sizes and assignments come from the serialized Faiss
    # index. The CPU computation is outside all timed GPU search sweeps.
    counts = np.array([cpu_index.invlists.list_size(i) for i in range(args.nlist)], dtype=np.int64)
    if int(counts.sum()) != cpu_index.ntotal:
        raise RuntimeError("Faiss inverted-list sizes do not sum to ntotal")
    _, coarse = cpu_index.quantizer.search(queries, max(probes))
    if coarse.shape != (nq, max(probes)) or np.any(coarse < 0):
        raise RuntimeError("Invalid coarse quantizer assignments")
    visits = {probe: float(counts[coarse[:, :probe]].sum(axis=1).mean()) for probe in probes}
    resources = faiss.StandardGpuResources()
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    index = faiss.index_cpu_to_gpu(resources, args.gpu_id, cpu_index, options)
    gpu_queries = cp.asarray(queries)
    cp.cuda.get_current_stream().synchronize()
    batches = {}
    for q_batch in args.q_batches:
        parts = []
        for first in range(0, nq, q_batch):
            valid = min(q_batch, nq - first)
            batch = gpu_queries[first:first + valid]
            if valid < q_batch:
                batch = cp.concatenate((batch, cp.repeat(batch[-1:], q_batch - valid, axis=0)))
            parts.append((batch, valid))
        batches[q_batch] = parts
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"AMD_FaissROCm_IVFFLAT_{args.dataset}_nlist{args.nlist}.csv"
    json_path = csv_path.with_suffix(".json")
    config = dict(
        dataset=args.dataset, kind="ivfflat", nlist=args.nlist, nq=nq, topk=args.topk,
        q_batches=list(args.q_batches), nprobes=list(probes), gpu_id=args.gpu_id,
        warmup=args.warmup, repetitions=args.repetitions,
        index=str(index_path.resolve()), index_size_bytes=index_path.stat().st_size,
        index_mtime_ns=index_path.stat().st_mtime_ns,
        query=str(query_path.resolve()), query_size_bytes=query_path.stat().st_size,
        truth=str(truth_path.resolve()), truth_size_bytes=truth_path.stat().st_size,
        faiss_version=faiss.__version__, cupy_version=cp.__version__,
    )
    ordered = [(batch, probe) for batch in args.q_batches for probe in probes]
    completed: dict[tuple[int, int], dict] = {}
    details: dict[tuple[int, int], dict] = {}
    if csv_path.exists() or json_path.exists():
        if args.overwrite:
            pass
        elif not args.resume or not csv_path.is_file() or not json_path.is_file():
            raise FileExistsError(f"Results exist; use --resume or --overwrite: {csv_path}")
        else:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
            if previous.get("config") != config:
                raise ValueError("Existing result configuration differs")
            with csv_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != list(FIELDS):
                    raise ValueError("Existing CSV columns differ")
                for row in reader:
                    key = (int(row["q_batch"]), int(row["nprobes"]))
                    if key in completed or key not in ordered or int(row["K"]) != args.topk:
                        raise ValueError(f"Invalid existing result row: {key}")
                    if not all(math.isfinite(float(row[field])) for field in FIELDS):
                        raise ValueError(f"Non-finite existing result row: {key}")
                    completed[key] = row
            details = {(int(row["q_batch"]), int(row["nprobes"])): row
                       for row in previous.get("results", [])
                       if (int(row["q_batch"]), int(row["nprobes"])) in completed}
    metadata = dict(
        status="running", config=config, backend="Faiss ROCm GPU IVF-Flat",
        expected_rows=len(ordered), memory_unit="MiB",
        memory_metric="sampled AMD-SMI current-process VRAM peak, including resident index and queries; separate sweep",
        timing="median synchronized full-query sweeps of GPU-resident queries; includes output allocation, excludes recall and memory sweep",
        recall="mean overlap with ground-truth top-K divided by K",
        visits="mean sum of exact Faiss coarse-list sizes for the selected nprobe lists",
        started_utc=datetime.now(timezone.utc).isoformat(), results=[],
    )

    def save() -> None:
        metadata["results"] = [details[key] for key in ordered if key in details]
        metadata["completed_rows"] = len(completed)
        atomic_json(json_path, metadata)
        atomic_csv(csv_path, [completed[key] for key in ordered if key in completed])

    save()
    if len(completed) == len(ordered):
        print(f"Already complete: {csv_path}", flush=True)
        return
    started = time.perf_counter()
    try:
        for q_batch, probe in ordered:
            if (q_batch, probe) in completed:
                continue
            index.nprobe = probe

            def sweep(retain: bool = False, sampler: VramSampler | None = None):
                elapsed, saved = 0.0, []
                for batch, valid in batches[q_batch]:
                    cp.cuda.get_current_stream().synchronize()
                    tick = time.perf_counter()
                    distances, ids = gpu_search(index, batch, args.topk)
                    elapsed += time.perf_counter() - tick
                    if sampler is not None:
                        sampler.sample()
                    if retain:
                        if not bool(cp.all(cp.isfinite(distances[:valid])).get()):
                            raise RuntimeError("Non-finite search distance")
                        saved.append(cp.asnumpy(ids[:valid]))
                    del distances, ids
                return elapsed, saved

            print(f"{args.dataset} Faiss ROCm QBatch={q_batch} K={args.topk} nprobe={probe}", flush=True)
            for _ in range(args.warmup):
                sweep()
            samples = []
            for repeat in range(args.repetitions):
                elapsed, saved = sweep(retain=repeat == args.repetitions - 1)
                samples.append(elapsed)
            ids = np.concatenate(saved)
            if ids.shape != (nq, args.topk) or np.any(ids < 0) or np.any(ids >= cpu_index.ntotal):
                raise RuntimeError("Invalid or missing search IDs")
            recall = recall_at_k(ids, truth, args.topk)
            with VramSampler(args.gpu_id) as sampler:
                sweep(sampler=sampler)
            row = dict(
                nprobes=probe, q_batch=q_batch, K=args.topk,
                qps=nq / statistics.median(samples), recall=recall,
                visited_vectors_per_query=visits[probe],
                peak_gpu_memory=sampler.peak_bytes / MEBIBYTE,
            )
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


if __name__ == "__main__":
    main()
