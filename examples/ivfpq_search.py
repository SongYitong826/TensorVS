from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch

from tensorvs import (
    build_ivfpq_index,
    compute_recall_at_k,
    load_dataset,
    load_ivecs,
    load_ivfpq_index,
    save_ivfpq_index,
    search_ivfpq_index,
)

DATASET_PATH = Path("/storage/vector_data/sift1m/sift1m_base.fvecs")
QUERY_PATH = Path("/storage/vector_data/sift1m/sift1m_query.fvecs")
GROUNDTRUTH_PATH = Path("/storage/vector_data/sift1m/sift1m_groundtruth.ivecs")
OUTPUT_DIR = Path("/storage/vector_data/sift1m/GPUIndex")
NLIST = 1024
NITER = 25
USE_TRITON = True
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
M = 16
NBITS = 8
TOPK = 10
NPROBE = 50
SELECTION_MODE = "triton_topk"  # triton_topk / tensor_topk
# Enable only when inspecting per-stage timing. For ``tensor_topk`` this also
# splits its normal fused graph, so leave it False for QPS measurements.
ENABLE_DETAILED_TIMING = False
# LUT precision switch: ``"fp32"`` uses TF32 Tensor Core BMM while retaining
# FP32 LUT storage/ADC accumulation; ``"fp16"`` uses fused residual/LUT
# construction with FP16 Tensor Core dot operands.
LUT_DTYPE = "fp32"
# The sole Triton IVF-PQ path: query-major static-vector correction. Both
# FP32 and FP16 use this scheme; FP16 stores G compactly but accumulates ADC
# values together with gamma in FP32.
LUT_STRATEGY = "decomposed_vector_correction"
REFINE_FACTOR = 1
NQ = 1000
Q_BATCH = 64
# 聚类参数
SEED = 0
N_INIT = 2


def print_timing_table(title: str, timings: dict[str, float], divisor: int) -> None:
    """Pretty-print aggregate and per-query timing statistics."""
    if not timings:
        return

    print(title)
    for name, total_seconds in sorted(timings.items(), key=lambda item: item[1], reverse=True):
        total_ms = total_seconds * 1000.0
        avg_us = (total_seconds / max(divisor, 1)) * 1_000_000.0
        print(f"  {name:<28} total={total_ms:10.4f} ms  avg={avg_us:10.4f} us/query")


def append_statistics_row(output_root: str | Path, metadata: dict[str, object]) -> Path:
    """Append build statistics to statistics_ivfpq.csv, creating it if needed."""
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    statistics_path = output_root_path / "statistics_ivfpq.csv"
    fieldnames = [
        "dataset_path",
        "output_dir",
        "nlist",
        "m",
        "nbits",
        "num_vectors",
        "dim",
        "device",
        "build_time_seconds",
        "index_size_bytes",
        "index_size_mb",
    ]

    write_header = not statistics_path.exists()
    with statistics_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: metadata.get(name, "") for name in fieldnames})

    return statistics_path


def main() -> None:
    index_dir = Path(OUTPUT_DIR) / (
        f"TenVS_IVFPQResidual_nlist{NLIST}_m{M}_nbits{NBITS}"
        f".bin"
    )

    if index_dir.exists():
        print(f"Found existing index at: {index_dir}")
    else:
        print(f"Index not found at: {index_dir}")
        print("Building IVF-PQ index first...")
        base_vectors = load_dataset(DATASET_PATH)
        print(
            f"Successfully prepared dataset from {DATASET_PATH} "
            f"with shape {tuple(base_vectors.shape)} and dtype {base_vectors.dtype}"
        )
        print(
            f"Starting clustering with nlist={NLIST}, niter={NITER}, "
            f"m={M}, nbits={NBITS}, device={DEVICE}..."
        )
        build_start_time = time.perf_counter()
        build_result = build_ivfpq_index(
            vectors=base_vectors,
            nlist=NLIST,
            niter=NITER,
            use_triton=USE_TRITON,
            device=DEVICE,
            m=M,
            nbits=NBITS,
            seed=SEED,
            n_init=N_INIT,
        )
        build_time_seconds = round(time.perf_counter() - build_start_time, 4)
        print("Clustering finished.")
        print("Saving IVF-PQ index files...")
        index_dir = save_ivfpq_index(
            build_result=build_result,
            output_dir=index_dir,
            dataset_path=DATASET_PATH,
            build_time_seconds=build_time_seconds,
        )
        if index_dir.is_file():
            metadata = torch.load(index_dir, map_location="cpu")["metadata"]
        else:
            metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        statistics_path = append_statistics_row(OUTPUT_DIR, metadata)
        print(f"Statistics written to: {statistics_path}")

    index = load_ivfpq_index(index_dir, load_lists=True)
    print(f"Loaded IVF-PQ index from: {index.index_dir}")
    print(f"Moving IVF-PQ index to resident device: {DEVICE}")
    index.reside_on_device(DEVICE)

    base_vectors = load_dataset(DATASET_PATH)
    queries = load_dataset(QUERY_PATH)
    groundtruth = load_ivecs(GROUNDTRUTH_PATH)
    if queries.shape[0] != groundtruth.shape[0]:
        raise ValueError(
            f"Query count {queries.shape[0]} does not match groundtruth count {groundtruth.shape[0]}"
        )

    nq = min(int(NQ), queries.shape[0])
    q_batch = max(1, int(Q_BATCH))
    queries = queries[:nq]
    groundtruth = groundtruth[:nq]
    print(f"Loaded queries from: {QUERY_PATH}")
    print(f"Loaded groundtruth from: {GROUNDTRUTH_PATH}")
    print(
        f"Running ANN search for {queries.shape[0]} queries with "
        f"topk={TOPK}, nprobe={NPROBE}, q_batch={q_batch}..."
    )

    all_distances = []
    all_ids = []
    batch_timings = {
        "query_to_device": 0.0,
        "pre_search_sync": 0.0,
        "warmup": 0.0,
        "search_core": 0.0,
        "post_search_sync": 0.0,
        "result_to_cpu": 0.0,
    }
    search_step_timings: dict[str, float] = {}
    warmup_step_timings: dict[str, float] = {}
    warmup_done = False
    for start in range(0, nq, q_batch):
        end = min(start + q_batch, nq)
        transfer_start_time = time.perf_counter()
        query_batch = queries[start:end].to(DEVICE)
        batch_timings["query_to_device"] += time.perf_counter() - transfer_start_time

        if DEVICE.type == "cuda":
            sync_start_time = time.perf_counter()
            torch.cuda.synchronize(DEVICE)
            batch_timings["pre_search_sync"] += time.perf_counter() - sync_start_time

        if not warmup_done:
            warmup_profile: dict[str, float] | None = {} if ENABLE_DETAILED_TIMING else None
            warmup_start_time = time.perf_counter()
            search_ivfpq_index(
                index=index,
                queries=query_batch,
                topk=TOPK,
                nprobe=NPROBE,
                selection_mode=SELECTION_MODE,
                refine_vectors=base_vectors if REFINE_FACTOR > 1 else None,
                refine_factor=REFINE_FACTOR,
                profile=warmup_profile,
                detailed_tensor_topk_profile=ENABLE_DETAILED_TIMING,
                lut_dtype=LUT_DTYPE,
                lut_strategy=LUT_STRATEGY,
            )
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            batch_timings["warmup"] += time.perf_counter() - warmup_start_time
            warmup_done = True
            if warmup_profile is not None:
                warmup_step_timings = warmup_profile

        batch_profile: dict[str, float] | None = {} if ENABLE_DETAILED_TIMING else None
        batch_search_start_time = time.perf_counter()
        batch_distances, batch_ids = search_ivfpq_index(
            index=index,
            queries=query_batch,
            topk=TOPK,
            nprobe=NPROBE,
            selection_mode=SELECTION_MODE,
            refine_vectors=base_vectors if REFINE_FACTOR > 1 else None,
            refine_factor=REFINE_FACTOR,
            profile=batch_profile,
            detailed_tensor_topk_profile=ENABLE_DETAILED_TIMING,
            lut_dtype=LUT_DTYPE,
            lut_strategy=LUT_STRATEGY,
        )
        batch_search_elapsed = time.perf_counter() - batch_search_start_time
        batch_timings["search_core"] += batch_search_elapsed
        if DEVICE.type == "cuda":
            sync_start_time = time.perf_counter()
            torch.cuda.synchronize(DEVICE)
            batch_timings["post_search_sync"] += time.perf_counter() - sync_start_time
        if batch_profile is not None:
            for name, value in batch_profile.items():
                search_step_timings[name] = search_step_timings.get(name, 0.0) + value

        cpu_copy_start_time = time.perf_counter()
        all_distances.append(batch_distances.cpu())
        all_ids.append(batch_ids.cpu())
        batch_timings["result_to_cpu"] += time.perf_counter() - cpu_copy_start_time

    distances = torch.cat(all_distances, dim=0)
    ids = torch.cat(all_ids, dim=0)

    search_time_seconds = max(batch_timings["search_core"] + batch_timings["post_search_sync"], 1e-12)
    average_search_time_ms = (search_time_seconds / queries.shape[0]) * 1000
    qps = queries.shape[0] / search_time_seconds
    strict_recall_at_topk = compute_recall_at_k(ids, groundtruth, TOPK)

    print(f"Average search time (with sync): {average_search_time_ms:.4f} ms/query")
    print(f"QPS (with sync): {qps:.4f}")
    print(f"Strict Recall@{TOPK}: {strict_recall_at_topk:.4f}")
    if ENABLE_DETAILED_TIMING:
        print_timing_table("Batch-level timing breakdown:", batch_timings, queries.shape[0])
        print_timing_table(
            "Warmup search kernel timing breakdown (excluded from steady-state):",
            warmup_step_timings,
            min(q_batch, queries.shape[0]),
        )
        print_timing_table("Steady-state search kernel timing breakdown:", search_step_timings, queries.shape[0])


if __name__ == "__main__":
    main()
