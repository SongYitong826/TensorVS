from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch

from tensorvs import (
    build_ivfflat_index,
    compute_recall_at_k,
    load_dataset,
    load_ivecs,
    load_ivfflat_index,
    save_ivfflat_index,
    search_ivfflat_index,
)

DIR_PATH = Path("/storage/vector_data/glove1.2m")
DATASET_NAME = DIR_PATH.name
DATASET_PATH = DIR_PATH / f"{DATASET_NAME}_base.fvecs"
QUERY_PATH = DIR_PATH / f"{DATASET_NAME}_query.fvecs"
GROUNDTRUTH_PATH = DIR_PATH / f"{DATASET_NAME}_groundtruth.ivecs"
OUTPUT_DIR = DIR_PATH / "GPUIndex"
NLIST = 1024
NITER = 25
USE_TRITON = True
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
TOPK = 10
NPROBE = 50
NQ = 1000
Q_BATCH = 128
SELECTION_MODE = "triton_topk"
# 可选："triton_topk" / "tensor_topk" / "tensor_padding_topk" / "tensor_list_topk"
# triton_topk 的batch越高越好？？为什么？
# Enable only when inspecting per-stage timing. It adds CUDA timing events and
# a per-search synchronization, so leave it False for QPS measurements.
ENABLE_DETAILED_TIMING = False
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
    """Append build statistics to statistics.csv, creating it if needed."""
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    statistics_path = output_root_path / "statistics.csv"
    fieldnames = [
        "dataset_path",
        "output_dir",
        "nlist",
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
    index_dir = Path(OUTPUT_DIR) / f"TenVS_IVFFlat_nlist{NLIST}.bin"

    if index_dir.exists():
        print(f"Found existing index at: {index_dir}")
    else:
        print(f"Index not found at: {index_dir}")
        print("Building IVF-Flat index first...")
        base_vectors = load_dataset(DATASET_PATH)
        print(
            f"Successfully prepared dataset from {DATASET_PATH} "
            f"with shape {tuple(base_vectors.shape)} and dtype {base_vectors.dtype}"
        )
        print(f"Starting clustering with nlist={NLIST}, niter={NITER}, device={DEVICE}...")
        build_start_time = time.perf_counter()
        build_result = build_ivfflat_index(
            vectors=base_vectors,
            nlist=NLIST,
            niter=NITER,
            use_triton=USE_TRITON,
            device=DEVICE,
            seed=SEED,
            n_init=N_INIT,
        )
        build_time_seconds = round(time.perf_counter() - build_start_time, 4)
        print("Clustering finished.")
        index_dir = Path(OUTPUT_DIR) / f"TenVS_IVFFlat_nlist{NLIST}.bin"
        print("Saving IVF-Flat index files...")
        index_dir = save_ivfflat_index(
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

    index = load_ivfflat_index(index_dir, load_lists=True)
    print(f"Loaded IVF-Flat index from: {index.index_dir}")
    print(f"Moving IVF-Flat index to resident device: {DEVICE}")
    index.reside_on_device(DEVICE)

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
        batch_size = end - start
        transfer_start_time = time.perf_counter()
        query_batch = queries[start:end].to(DEVICE)
        if batch_size < q_batch:
            # Keep the compiled search graph on the configured batch shape.
            # Duplicate the final real query only for execution, then discard
            # its extra results before aggregating outputs and recall.
            padding = query_batch[-1:].expand(q_batch - batch_size, -1)
            query_batch = torch.cat((query_batch, padding), dim=0)
        batch_timings["query_to_device"] += time.perf_counter() - transfer_start_time

        if DEVICE.type == "cuda":
            sync_start_time = time.perf_counter()
            torch.cuda.synchronize(DEVICE)
            batch_timings["pre_search_sync"] += time.perf_counter() - sync_start_time
        if not warmup_done:
            warmup_profile: dict[str, float] | None = {} if ENABLE_DETAILED_TIMING else None
            warmup_start_time = time.perf_counter()
            search_ivfflat_index(
                index=index,
                queries=query_batch,
                topk=TOPK,
                nprobe=NPROBE,
                selection_mode=SELECTION_MODE,
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
        batch_distances, batch_ids = search_ivfflat_index(
            index=index,
            queries=query_batch,
            topk=TOPK,
            nprobe=NPROBE,
            selection_mode=SELECTION_MODE,
            profile=batch_profile,
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
