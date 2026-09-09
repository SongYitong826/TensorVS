from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from bench._benchmark_utils import probe_visit_statistics
from tensorvs import (
    build_ivfpq_index,
    compute_recall_at_k,
    load_dataset,
    load_ivecs,
    load_ivfpq_index,
    save_ivfpq_index,
    search_ivfpq_index,
)

DIR_PATH = Path("/storage/vector_data/glove1.2m")
DATASET_NAME = DIR_PATH.name
DATASET_PATH = DIR_PATH / f"{DATASET_NAME}_base.fvecs"
QUERY_PATH = DIR_PATH / f"{DATASET_NAME}_query.fvecs"
GROUNDTRUTH_PATH = DIR_PATH / f"{DATASET_NAME}_groundtruth.ivecs"
OUTPUT_DIR = DIR_PATH / "GPUIndex"
NLIST = 1024
IVF_NITER = 25
IVF_N_INIT = 1
IVF_MAX_POINTS_PER_CENTROID = 256
PQ_NITER = 25
PQ_N_INIT = 1
PQ_MAX_POINTS_PER_CENTROID = 256
KMEANS_INIT_METHOD = "kmeans++"
USE_TRITON = True
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# ``M`` need not divide the input dimension. For example GloVe D=200,M=32
# uses a cuVS-style 224-dimensional orthogonal PQ working space (32 x 7).
M = 32
NBITS = 8
TOPK = 10
NPROBE = 50
SELECTION_MODE = "triton_topk"
# The public IVFPQ search API only supports this Triton TopK path.
ENABLE_DETAILED_TIMING = False
# Set this to False to use the unpacked subspace-major code loads.
USE_PACKED_16B_CODES = True
# Measure a small CTA scan sweep and persist the fastest configuration under
# ~/.cache/tensorvs. Later runs reuse it automatically, so enable this only
# when a new GPU/workload shape needs calibration.
ENABLE_TRITON_AUTOTUNE = True
# LUT precision switch: ``"fp32"`` uses TF32 Tensor Core BMM while retaining
# FP32 LUT storage/ADC accumulation; ``"fp16"`` uses fused residual/LUT
# construction with FP16 Tensor Core dot operands.
LUT_DTYPE = "fp32"  # fp32 / fp16
# Triton TopK uses a static-vector correction. FP16 stores G compactly but
# accumulates ADC values together with gamma in FP32.
REFINE_FACTOR = 1
NQ = 1024
Q_BATCH = 256
WARMUP_ITERATIONS = 3
# 聚类参数
SEED = 0
SEARCH_RESULT_PATH = OUTPUT_DIR / (
    f"IVFPQ_{DATASET_NAME}_nlist{NLIST}_m{M}_nbits{NBITS}"
    f"_K{TOPK}_nprobe{NPROBE}.csv"
)


def print_timing_table(title: str, timings: dict[str, float], divisor: int) -> None:
    """Pretty-print aggregate and per-query timing statistics."""
    if not timings:
        return

    print(title)
    for name, total_seconds in sorted(timings.items(), key=lambda item: item[1], reverse=True):
        total_ms = total_seconds * 1000.0
        avg_us = (total_seconds / max(divisor, 1)) * 1_000_000.0
        print(f"  {name:<28} total={total_ms:10.4f} ms  avg={avg_us:10.4f} us/query")


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
            f"Starting clustering with nlist={NLIST}, "
            f"ivf_niter={IVF_NITER}, pq_niter={PQ_NITER}, "
            f"m={M}, nbits={NBITS}, device={DEVICE}..."
        )
        build_start_time = time.perf_counter()
        build_result = build_ivfpq_index(
            vectors=base_vectors,
            nlist=NLIST,
            niter=IVF_NITER,
            use_triton=USE_TRITON,
            device=DEVICE,
            m=M,
            nbits=NBITS,
            seed=SEED,
            n_init=IVF_N_INIT,
            coarse_niter=IVF_NITER,
            pq_niter=PQ_NITER,
            coarse_n_init=IVF_N_INIT,
            pq_n_init=PQ_N_INIT,
            coarse_max_points_per_centroid=IVF_MAX_POINTS_PER_CENTROID,
            pq_max_points_per_centroid=PQ_MAX_POINTS_PER_CENTROID,
            coarse_init_method=KMEANS_INIT_METHOD,
            coarse_kmeans_method="balanced",
            pq_init_method=KMEANS_INIT_METHOD,
            pq_kmeans_method="balanced",
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
    code_bytes_per_vector = int(index.metadata["m"]) * index.persisted_list_codes_interleaved_by_subspace.element_size()
    visit_stats = probe_visit_statistics(
        index=index,
        queries=queries,
        nprobe=NPROBE,
        device=DEVICE,
        data_bytes_per_vector=code_bytes_per_vector,
    )
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
    peak_gpu_memory: float | None = None
    for start in range(0, nq, q_batch):
        end = min(start + q_batch, nq)
        batch_size = end - start
        transfer_start_time = time.perf_counter()
        query_batch = queries[start:end].to(DEVICE)
        # Keep one compiled/search workspace shape throughout the benchmark.
        # The GloVe query set has 1,000 rows, so an unpadded final 232-query
        # batch otherwise exercises a cold shape after warming only Q_BATCH.
        if batch_size < q_batch:
            query_batch = torch.cat(
                (query_batch, query_batch[-1:].expand(q_batch - batch_size, -1)),
                dim=0,
            )
        batch_timings["query_to_device"] += time.perf_counter() - transfer_start_time

        if DEVICE.type == "cuda":
            sync_start_time = time.perf_counter()
            torch.cuda.synchronize(DEVICE)
            batch_timings["pre_search_sync"] += time.perf_counter() - sync_start_time

        if not warmup_done:
            warmup_start_time = time.perf_counter()
            warmup_profile: dict[str, float] | None = None
            for _ in range(max(1, int(WARMUP_ITERATIONS))):
                warmup_profile = {} if ENABLE_DETAILED_TIMING else None
                search_ivfpq_index(
                    index=index,
                    queries=query_batch,
                    topk=TOPK,
                    nprobe=NPROBE,
                    selection_mode=SELECTION_MODE,
                    refine_vectors=base_vectors if REFINE_FACTOR > 1 else None,
                    refine_factor=REFINE_FACTOR,
                    profile=warmup_profile,
                    use_packed_16b_codes=USE_PACKED_16B_CODES,
                    triton_autotune=ENABLE_TRITON_AUTOTUNE,
                    lut_dtype=LUT_DTYPE,
                )
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
                torch.cuda.reset_peak_memory_stats(DEVICE)
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
            use_packed_16b_codes=USE_PACKED_16B_CODES,
            triton_autotune=ENABLE_TRITON_AUTOTUNE,
            lut_dtype=LUT_DTYPE,
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
        all_distances.append(batch_distances[:batch_size].cpu())
        all_ids.append(batch_ids[:batch_size].cpu())
        batch_timings["result_to_cpu"] += time.perf_counter() - cpu_copy_start_time

    distances = torch.cat(all_distances, dim=0)
    ids = torch.cat(all_ids, dim=0)

    search_time_seconds = max(batch_timings["search_core"] + batch_timings["post_search_sync"], 1e-12)
    average_search_time_ms = (search_time_seconds / queries.shape[0]) * 1000
    qps = queries.shape[0] / search_time_seconds
    strict_recall_at_topk = compute_recall_at_k(ids, groundtruth, TOPK)
    if DEVICE.type == "cuda":
        peak_gpu_memory = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)

    result_path = SEARCH_RESULT_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "nprobes",
        "q_batch",
        "qps",
        "recall",
        "visited_vectors_per_query",
        "peak_gpu_memory",
    )
    has_content = result_path.exists() and result_path.stat().st_size > 0
    if has_content:
        with result_path.open("r", newline="", encoding="utf-8") as csv_file:
            existing_fieldnames = next(csv.reader(csv_file), [])
        if existing_fieldnames != list(fieldnames):
            result_path = result_path.with_name(
                f"{result_path.stem}_summary{result_path.suffix}"
            )
            has_content = result_path.exists() and result_path.stat().st_size > 0

    with result_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not has_content:
            writer.writeheader()
        writer.writerow(
            {
                "nprobes": NPROBE,
                "q_batch": q_batch,
                "qps": qps,
                "recall": strict_recall_at_topk,
                "visited_vectors_per_query": visit_stats["visited_vectors_per_query"],
                # This is peak allocated GPU memory in MiB.
                "peak_gpu_memory": peak_gpu_memory,
            }
        )

    print(f"Average search time (with sync): {average_search_time_ms:.4f} ms/query")
    print(f"QPS (with sync): {qps:.4f}")
    print(f"Strict Recall@{TOPK}: {strict_recall_at_topk:.4f}")
    print(f"CSV results appended to: {result_path}")
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
