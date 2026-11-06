"""Benchmark Faiss ROCm GPU IVF-PQ search on zero-padded GloVe1.2M.

Use the index produced by faiss_rocm_ivfpq_build.py. Its 200-dimensional
GloVe vectors were padded with 24 zeros; the same padding is applied to each
query before search. The seven CSV columns and measurement method match the
existing AMD TensorVS, hipVS, and Faiss IVF-Flat search benchmarks.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

import cupy as cp
import faiss
import numpy as np

from amd_ivfflat_build import VramSampler
from faiss_rocm_ivfflat_bench import (
    FIELDS, MEBIBYTE, NPROBES, atomic_csv, atomic_json, gpu_search,
    read_vecs, recall_at_k,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "glove1.2m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(DATASET,), default=DATASET)
    parser.add_argument("--data-root", type=Path, default=Path("/root/vector_data"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "result")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nq", type=int, default=1000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--q-batches", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--nprobes", type=int, nargs="+")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    probes = tuple(args.nprobes or NPROBES[DATASET])
    if min(args.nlist, args.nq, args.topk, args.repetitions,
           *args.q_batches, *probes) <= 0:
        parser.error("nlist, NQ, K, repetitions, query batches and probes must be positive")
    if args.warmup < 0 or args.gpu_id < 0:
        parser.error("warmup and GPU ID must be nonnegative")
    if max(probes) > args.nlist:
        parser.error("nprobe exceeds nlist")
    if len(set(args.q_batches)) != len(args.q_batches) or len(set(probes)) != len(probes):
        parser.error("Duplicate query batch size or nprobe")
    if args.resume and args.overwrite:
        parser.error("Choose either --resume or --overwrite")
    if args.gpu_id >= faiss.get_num_gpus():
        parser.error(f"Faiss ROCm cannot see GPU {args.gpu_id}")
    cp.cuda.Device(args.gpu_id).use()

    directory = args.data_root / DATASET
    index_stem = f"ivfpq_nlist{args.nlist}_m32_nbits8_pad224_build"
    index_path = directory / "GPUIndex" / "FaissROCm" / f"{index_stem}.index"
    index_meta_path = index_path.with_suffix(".json")
    query_path = directory / f"{DATASET}_query.fvecs"
    truth_path = directory / f"{DATASET}_groundtruth.ivecs"
    for path in (index_path, index_meta_path, query_path, truth_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    build_meta = json.loads(index_meta_path.read_text(encoding="utf-8"))
    if (build_meta.get("status") != "complete" or
        build_meta.get("nlist") != args.nlist or
        build_meta.get("m") != 32 or build_meta.get("nbits") != 8 or
        build_meta.get("original_dim") != 200 or
        build_meta.get("padded_dim") != 224):
        raise ValueError("Faiss IVF-PQ build metadata does not match GloVe M32/8-bit pad224")
    train_points = int(build_meta["pq_train_points_per_centroid"])
    cpu_index = faiss.read_index(str(index_path))
    if (cpu_index.d != 224 or cpu_index.ntotal != build_meta["n_vectors"] or
        cpu_index.nlist != args.nlist or cpu_index.pq.M != 32 or
        cpu_index.pq.nbits != 8 or cpu_index.code_size != 32):
        raise ValueError("Serialized Faiss IVF-PQ index differs from build metadata")

    original_queries = read_vecs(query_path, np.float32)
    truth = read_vecs(truth_path, np.int32)
    if len(original_queries) != len(truth) or original_queries.shape[1] != 200:
        raise ValueError("GloVe queries and ground truth have inconsistent dimensions/counts")
    if truth.shape[1] < args.topk or args.topk > cpu_index.ntotal:
        raise ValueError("K exceeds ground-truth width or index size")
    nq = min(args.nq, len(original_queries))
    queries = np.ascontiguousarray(
        np.pad(original_queries[:nq], ((0, 0), (0, 24)), mode="constant"),
        dtype=np.float32,
    )
    truth = truth[:nq]
    if queries.shape != (nq, cpu_index.d):
        raise ValueError("Padded query dimension differs from the Faiss index")

    # Compute exact visited-list counts from the saved Faiss index, outside
    # timed search sweeps. The coarse assignments use padded queries too.
    counts = np.array([cpu_index.invlists.list_size(i)
                       for i in range(args.nlist)], dtype=np.int64)
    if int(counts.sum()) != cpu_index.ntotal:
        raise RuntimeError("Faiss inverted-list sizes do not sum to ntotal")
    _, coarse = cpu_index.quantizer.search(queries, max(probes))
    if coarse.shape != (nq, max(probes)) or np.any(coarse < 0):
        raise RuntimeError("Invalid coarse-quantizer assignments")
    visits = {probe: float(counts[coarse[:, :probe]].sum(axis=1).mean())
              for probe in probes}

    resources = faiss.StandardGpuResources()
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    index = faiss.index_cpu_to_gpu(resources, args.gpu_id, cpu_index, options)
    if index.d != 224 or index.ntotal != cpu_index.ntotal:
        raise RuntimeError("GPU index has wrong dimensions or vector count")
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
    name = (f"AMD_FaissROCm_IVFPQ_{DATASET}_nlist{args.nlist}_M32_nbits8_"
            f"train{train_points}_pad224.csv")
    csv_path = args.output_dir / name
    json_path = csv_path.with_suffix(".json")
    config = dict(
        dataset=DATASET, kind="ivfpq", nlist=args.nlist, m=32, nbits=8,
        original_dim=200, padded_dim=224, pq_train_points_per_centroid=train_points,
        nq=nq, topk=args.topk, q_batches=list(args.q_batches), nprobes=list(probes),
        gpu_id=args.gpu_id, warmup=args.warmup, repetitions=args.repetitions,
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
        status="running", config=config, backend="Faiss ROCm GPU IVF-PQ",
        expected_rows=len(ordered), memory_unit="MiB",
        memory_metric="sampled AMD-SMI current-process VRAM peak, including resident index and queries; separate sweep",
        timing="median synchronized full-query sweeps of GPU-resident padded queries; includes output allocation, excludes recall and memory sweep",
        recall="mean overlap with ground-truth top-K divided by K",
        visits="mean sum of exact Faiss coarse-list sizes for the selected nprobe lists",
        padding="24 zero coordinates appended to every 200-dimensional query before upload",
        started_utc=datetime.now(timezone.utc).isoformat(), results=[],
    )

    def save() -> None:
        metadata["results"] = [details[key] for key in ordered if key in details]
        metadata["completed_rows"] = len(completed)
        atomic_json(json_path, metadata)
        atomic_csv(csv_path, [completed[key] for key in ordered if key in completed])

    if len(completed) == len(ordered):
        print(f"Already complete: {csv_path}", flush=True)
        return
    save()
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

            print(f"{DATASET} Faiss ROCm IVF-PQ QBatch={q_batch} K={args.topk} "
                  f"nprobe={probe}", flush=True)
            for _ in range(args.warmup):
                sweep()
            samples = []
            for repeat in range(args.repetitions):
                elapsed, saved = sweep(retain=repeat == args.repetitions - 1)
                samples.append(elapsed)
            ids = np.concatenate(saved)
            if (ids.shape != (nq, args.topk) or np.any(ids < 0) or
                np.any(ids >= cpu_index.ntotal)):
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
            details[key] = dict(row, samples_seconds=samples,
                                memory_samples=sampler.samples)
            save()
            print(f"  QPS={row['qps']:.2f} recall={recall:.6f} "
                  f"VRAM={row['peak_gpu_memory']:.1f} MiB", flush=True)
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
