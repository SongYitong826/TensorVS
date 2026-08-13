from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch

from tensorvs import (
    build_ivfsq_index,
    compute_recall_at_k,
    load_dataset,
    load_ivecs,
    load_ivfsq_index,
    save_ivfsq_index,
    search_ivfsq_index,
)

DIR_PATH = Path("/storage/vector_data/sift1m")
DATASET_NAME = DIR_PATH.name
DATASET_PATH = DIR_PATH / f"{DATASET_NAME}_base.fvecs"
QUERY_PATH = DIR_PATH / f"{DATASET_NAME}_query.fvecs"
GROUNDTRUTH_PATH = DIR_PATH / f"{DATASET_NAME}_groundtruth.ivecs"
OUTPUT_DIR = DIR_PATH / "GPUIndex"

NLIST = 1024
NITER = 25
USE_TRITON = True
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# IVFSQ currently supports residual scalar quantization with 4-bit or 8-bit
# codes.  int4 stores two scalar codes in one byte.
SQ_BITS = 8  # 4 / 8
TOPK = 10
NPROBE = 50
SELECTION_MODE = "triton_topk"
# triton_topk keeps the quantized code cache resident and decodes inside the
# list-grouped Triton scan. tensor_topk remains the portable fallback.

# Set this to True when inspecting the per-stage search timing breakdown.
# Profiled searches synchronize CUDA and should not be used for QPS numbers.
ENABLE_DETAILED_TIMING = False

# Set REFINE_FACTOR > 1 to retrieve more SQ candidates and rerank them with
# the original vectors. The raw vectors are not stored in the IVFSQ index.
REFINE_FACTOR = 1
NQ = 1024
# The list-grouped Triton path has better occupancy at 128 queries/batch on
# RTX 4090. Use 64 for a direct comparison with the cuVS example.
Q_BATCH = 64
WARMUP_ITERATIONS = 3
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
    """Append build statistics to statistics_ivfsq.csv."""
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    statistics_path = output_root_path / "statistics_ivfsq.csv"
    fieldnames = [
        "dataset_path",
        "output_dir",
        "nlist",
        "sq_bits",
        "sq_storage",
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


def load_saved_metadata(index_path: Path) -> dict[str, object]:
    if index_path.is_file():
        return torch.load(index_path, map_location="cpu")["metadata"]
    return json.loads((index_path / "metadata.json").read_text(encoding="utf-8"))


def main() -> None:
    index_path = Path(OUTPUT_DIR) / (
        f"TenVS_IVFSQResidual_nlist{NLIST}_bits{SQ_BITS}.bin"
    )

    if index_path.exists():
        print(f"Found existing index at: {index_path}")
    else:
        print(f"Index not found at: {index_path}")
        print("Building IVF-SQ index first...")
        base_vectors = load_dataset(DATASET_PATH)
        print(
            f"Successfully prepared dataset from {DATASET_PATH} "
            f"with shape {tuple(base_vectors.shape)} and dtype {base_vectors.dtype}"
        )
        print(
            f"Starting clustering with nlist={NLIST}, niter={NITER}, "
            f"sq_bits={SQ_BITS}, device={DEVICE}..."
        )
        build_start_time = time.perf_counter()
        build_result = build_ivfsq_index(
            vectors=base_vectors,
            nlist=NLIST,
            niter=NITER,
            use_triton=USE_TRITON,
            device=DEVICE,
            bits=SQ_BITS,
            seed=SEED,
            n_init=N_INIT,
        )
        build_time_seconds = round(time.perf_counter() - build_start_time, 4)
        print("Clustering and scalar quantization finished.")
        print("Saving IVF-SQ index files...")
        index_path = save_ivfsq_index(
            build_result=build_result,
            output_path=index_path,
            dataset_path=DATASET_PATH,
            build_time_seconds=build_time_seconds,
        )
        metadata = load_saved_metadata(index_path)
        statistics_path = append_statistics_row(OUTPUT_DIR, metadata)
        print(f"Statistics written to: {statistics_path}")

    index = load_ivfsq_index(index_path, load_lists=True)
    print(f"Loaded IVF-SQ index from: {index.index_dir}")
    print(f"SQ format: {index.metadata['sq_bits']}-bit ({index.metadata['sq_storage']})")
    print(f"Moving IVF-SQ index to resident device: {DEVICE}")
    index.reside_on_device(DEVICE)

    # Keep the raw vectors outside the index. They are needed only when exact
    # refinement is enabled and for reporting the dataset/query dimensions.
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
        f"topk={TOPK}, nprobe={NPROBE}, q_batch={q_batch}, "
        f"selection_mode={SELECTION_MODE}..."
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
        batch_size = end - start

        transfer_start_time = time.perf_counter()
        query_batch = queries[start:end].to(DEVICE)
        # Keep a stable search shape for the compiled Tensor distance path.
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
                search_ivfsq_index(
                    index=index,
                    queries=query_batch,
                    topk=TOPK,
                    nprobe=NPROBE,
                    device=DEVICE,
                    selection_mode=SELECTION_MODE,
                    refine_vectors=base_vectors if REFINE_FACTOR > 1 else None,
                    refine_factor=REFINE_FACTOR,
                    profile=warmup_profile,
                )
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            batch_timings["warmup"] += time.perf_counter() - warmup_start_time
            warmup_done = True
            if warmup_profile is not None:
                warmup_step_timings = warmup_profile

        batch_profile: dict[str, float] | None = {} if ENABLE_DETAILED_TIMING else None
        batch_search_start_time = time.perf_counter()
        batch_distances, batch_ids = search_ivfsq_index(
            index=index,
            queries=query_batch,
            topk=TOPK,
            nprobe=NPROBE,
            device=DEVICE,
            selection_mode=SELECTION_MODE,
            refine_vectors=base_vectors if REFINE_FACTOR > 1 else None,
            refine_factor=REFINE_FACTOR,
            profile=batch_profile,
        )
        batch_timings["search_core"] += time.perf_counter() - batch_search_start_time

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
    del distances  # IDs are sufficient for the recall metric below.

    search_time_seconds = max(
        batch_timings["search_core"] + batch_timings["post_search_sync"],
        1e-12,
    )
    average_search_time_ms = (search_time_seconds / max(queries.shape[0], 1)) * 1000
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
        print_timing_table(
            "Steady-state search kernel timing breakdown:",
            search_step_timings,
            queries.shape[0],
        )


if __name__ == "__main__":
    main()
