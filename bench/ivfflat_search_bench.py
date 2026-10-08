"""Batch IVF-Flat experiments through the Triton TopK path.

Edit the configuration block below, then run from the repository root with:

    conda run -n tensorvs python bench/ivfflat_search_bench.py
"""

from __future__ import annotations

import csv
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
from tensorvs._ivfflat_triton import can_run_topk as can_run_triton_topk

try:
    from ._benchmark_utils import (
        probe_visit_statistics,
        print_result,
        run_one_nprobe,
        safe_filename,
    )
except ImportError:  # Direct execution: ``python bench/ivfflat_search_bench.py``.
    from _benchmark_utils import (
        probe_visit_statistics,
        print_result,
        run_one_nprobe,
        safe_filename,
    )


# ============================ Manual configuration =========================
DIR_PATH = Path("/storage/vector_data/deep1m")  # Path to the dataset directory
DATASET_NAME = DIR_PATH.name
DATASET_PATH = DIR_PATH / f"{DATASET_NAME}_base.fvecs"
QUERY_PATH = DIR_PATH / f"{DATASET_NAME}_query.fvecs"
GROUNDTRUTH_PATH = DIR_PATH / f"{DATASET_NAME}_groundtruth.ivecs"
RESULT_DIR = Path(__file__).resolve().parents[1] / "result"

NLIST = 1024
NITER = 25
# Faiss-style KMeans training cap: each stage samples at most k * this many
# vectors. The full dataset is still assigned to the trained centroids.
MAX_POINTS_PER_CENTROID = 256
SEED = 1234
N_INIT = 1
USE_TRITON = True
DEVICE = torch.device("cuda:6")

TOPKS = (20, 40, 60, 80, 100)
# NPROBES = (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150) # sift1m
NPROBES = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100, 120, 140, 160, 200, 250) # deep1m
# NPROBES = (50, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400) # glove1.2m
# NPROBES = (10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250) # tiny5m
# NPROBES = (10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300) # gist
# NPROBES = (10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 250, 300) # openai1536
Q_BATCH = 128

NQ = 1024
WARMUP_ITERATIONS = 2
SELECTION_MODE = "triton_topk"
INDEX_PATH = (
    DIR_PATH
    / "GPUIndex"
    / f"TenVS_IVFFlat_nlist{NLIST}.bin"
)
# ============================================================================


def append_index_statistics(
    index_path: Path,
    index_type: str,
    dataset_name: str,
    build_time_seconds: float,
) -> None:
    """Append build time and on-disk index size to the GPU index statistics."""
    if index_path.is_file():
        index_size_bytes = index_path.stat().st_size
    else:
        index_size_bytes = sum(
            path.stat().st_size for path in index_path.rglob("*") if path.is_file()
        )

    statistics_path = index_path.parent / "statistic.csv"
    fieldnames = (
        "Index Type",
        "Dataset Name",
        "Method Name",
        "Build Time (s)",
        "Index Size (MB)",
    )
    has_content = statistics_path.exists() and statistics_path.stat().st_size > 0
    with statistics_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not has_content:
            writer.writeheader()
        writer.writerow(
            {
                "Index Type": index_type,
                "Dataset Name": dataset_name,
                "Method Name": "TensorVS",
                "Build Time (s)": round(build_time_seconds, 4),
                "Index Size (MB)": round(index_size_bytes / (1024 * 1024), 4),
            }
        )


def load_or_build_index() -> object:
    if INDEX_PATH.exists():
        print(f"Found existing IVF-Flat index: {INDEX_PATH}")
        return load_ivfflat_index(INDEX_PATH, load_lists=True)

    print(f"Index not found; building IVF-Flat index at {INDEX_PATH}")
    base_vectors = load_dataset(DATASET_PATH)
    build_start = time.perf_counter()
    build_result = build_ivfflat_index(
        vectors=base_vectors,
        nlist=NLIST,
        niter=NITER,
        use_triton=USE_TRITON,
        device=DEVICE,
        seed=SEED,
        n_init=N_INIT,
        max_points_per_centroid=MAX_POINTS_PER_CENTROID,
    )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    build_seconds = time.perf_counter() - build_start
    saved_path = save_ivfflat_index(
        build_result=build_result,
        output_dir=INDEX_PATH,
        dataset_path=DATASET_PATH,
        build_time_seconds=build_seconds,
    )
    append_index_statistics(
        index_path=saved_path,
        index_type="IVF Flat",
        dataset_name=DATASET_NAME,
        build_time_seconds=build_seconds,
    )
    print(f"Built IVF-Flat index in {build_seconds:.4f}s: {saved_path}")
    return load_ivfflat_index(saved_path, load_lists=True)


def main() -> None:
    if DEVICE.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("IVF-Flat benchmark requires a CUDA device for Triton TopK")
    if not TOPKS or any(int(topk) <= 0 for topk in TOPKS):
        raise ValueError(f"TOPKS must contain positive values, got {TOPKS}")
    if len(set(TOPKS)) != len(TOPKS):
        raise ValueError(f"TOPKS contains duplicates: {TOPKS}")
    if not NPROBES:
        raise ValueError("NPROBES must contain at least one value")
    if len(set(NPROBES)) != len(NPROBES):
        raise ValueError(f"NPROBES contains duplicates: {NPROBES}")
    if any(int(value) <= 0 or int(value) > NLIST for value in NPROBES):
        raise ValueError(f"Every nprobe must be in [1, {NLIST}], got {NPROBES}")

    index = load_or_build_index()
    actual_nlist = int(index.centroids.shape[0])
    if actual_nlist != NLIST:
        raise ValueError(
            f"Configured NLIST={NLIST} does not match loaded index nlist={actual_nlist}: "
            f"{INDEX_PATH}"
        )
    index.reside_on_device(DEVICE)
    queries = load_dataset(QUERY_PATH)
    groundtruth = load_ivecs(GROUNDTRUTH_PATH)
    if queries.shape[0] != groundtruth.shape[0]:
        raise ValueError(
            f"Query count {queries.shape[0]} does not match groundtruth count "
            f"{groundtruth.shape[0]}"
        )
    nq = min(int(NQ), int(queries.shape[0]))
    queries = queries[:nq]
    groundtruth = groundtruth[:nq]
    q_batch = max(1, int(Q_BATCH))

    dim = int(index.centroids.shape[1])
    rows = []
    print(
        f"Running IVF-Flat Triton TopK on dataset={DATASET_NAME}, nq={nq}, "
        f"topks={list(TOPKS)}, q_batch={q_batch}, nprobes={list(NPROBES)}"
    )
    for topk in TOPKS:
        print(f"Starting K={topk}")
        for requested_nprobe in NPROBES:
            nprobe = min(int(requested_nprobe), int(index.centroids.shape[0]))
            if not can_run_triton_topk(
                device=DEVICE,
                topk=topk,
                num_queries=q_batch,
                nprobe=nprobe,
                dim=dim,
            ):
                raise RuntimeError(
                    "Triton IVF-Flat TopK does not support "
                    f"topk={topk}, q_batch={q_batch}, nprobe={nprobe}, dim={dim}"
                )

            visit_stats = probe_visit_statistics(
                index=index,
                queries=queries,
                nprobe=nprobe,
                device=DEVICE,
                data_bytes_per_vector=dim * 4,
            )
            row = run_one_nprobe(
                queries=queries,
                groundtruth=groundtruth,
                nprobe=nprobe,
                topk=topk,
                q_batch=q_batch,
                warmup_iterations=WARMUP_ITERATIONS,
                device=DEVICE,
                search_one=lambda query_batch, current_nprobe, current_topk=topk: search_ivfflat_index(
                    index=index,
                    queries=query_batch,
                    topk=current_topk,
                    nprobe=current_nprobe,
                    selection_mode=SELECTION_MODE,
                ),
                visit_stats=visit_stats,
                recall_fn=compute_recall_at_k,
            )
            row["requested_nprobe"] = int(requested_nprobe)
            row["topk"] = int(topk)
            print_result(row)
            rows.append(row)

    # Keep all K values for this dataset/index configuration in one file.
    file_stem = f"IVFFLAT_{safe_filename(DATASET_NAME)}_nlist{NLIST}"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR / f"{file_stem}.csv"
    fieldnames = (
        "nprobes",
        "q_batch",
        "K",
        "qps",
        "recall",
        "visited_vectors_per_query",
        "peak_gpu_memory",
    )
    has_content = csv_path.exists() and csv_path.stat().st_size > 0
    if has_content:
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            existing_fieldnames = next(csv.reader(csv_file), [])
        if existing_fieldnames != list(fieldnames):
            raise ValueError(
                f"Unexpected CSV header in {csv_path}: {existing_fieldnames}; "
                f"expected {list(fieldnames)}"
            )

    output_rows = [
        {
            "nprobes": row["requested_nprobe"],
            "q_batch": q_batch,
            "K": row["topk"],
            "qps": row["qps"],
            "recall": row["recall"],
            "visited_vectors_per_query": row["visited_vectors_per_query"],
            # This is peak CUDA-driver device memory in MiB.
            "peak_gpu_memory": row["peak_gpu_memory_driver_mb"],
        }
        for row in rows
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not has_content:
            writer.writeheader()
        writer.writerows(output_rows)
    print(f"CSV results appended to: {csv_path}")


if __name__ == "__main__":
    main()
