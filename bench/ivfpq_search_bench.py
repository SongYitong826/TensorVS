"""Batch residual IVF-PQ experiments through the Triton TopK path.

Edit the configuration block below, then run from the repository root with:

    conda run -n tensorvs python bench/ivfpq_search_bench.py
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch

from tensorvs import (
    build_ivfpq_index,
    compute_recall_at_k,
    load_ivfpq_index,
    save_ivfpq_index,
    search_ivfpq_index,
)
from tensorvs._ivfpq_triton import can_run_topk as can_run_triton_topk

try:
    from ._benchmark_utils import (
        process_gpu_memory_bytes,
        load_benchmark_vectors,
        load_benchmark_groundtruth,
        probe_visit_statistics,
        print_result,
        run_one_nprobe,
    )
except ImportError:  # Direct execution: ``python bench/ivfpq_search_bench.py``.
    from _benchmark_utils import (
        process_gpu_memory_bytes,
        load_benchmark_vectors,
        load_benchmark_groundtruth,
        probe_visit_statistics,
        print_result,
        run_one_nprobe,
    )


# ============================ Manual configuration =========================
DIR_PATH = Path(
    os.environ.get("TENSORVS_BENCH_DATASET", "/storage/vector_data/tiny5m")
)
DATASET_NAME = DIR_PATH.name
DATASET_PATH = DIR_PATH / f"{DATASET_NAME}_base.fvecs"
QUERY_PATH = DIR_PATH / f"{DATASET_NAME}_query.fvecs"
GROUNDTRUTH_PATH = DIR_PATH / f"{DATASET_NAME}_groundtruth.ivecs"
RESULT_DIR = Path(__file__).resolve().parents[1] / "result"

DATASET_PROFILES = {
    "sift1m": {
        "nlist": 1024,
        "pq_dim": 16,
        "nprobes": (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150),
    },
    "sift10m": {
        "nlist": 1024,
        "pq_dim": 16,
        "nprobes": (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150),
    },
    "sift100m": {
        "nlist": 1024,
        "pq_dim": 16,
        "nprobes": (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150),
    },
    "glove1.2m": {
        "nlist": 1024,
        "pq_dim": 25,
        "nprobes": (50, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400),
    },
    "deep1m": {
        "nlist": 1024,
        "pq_dim": 32,
        "nprobes": (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100, 120, 140, 160, 200, 250),
    },
    "tiny5m": {
        "nlist": 2048,
        "pq_dim": 64,
        "nprobes": (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500),
    },
    "hunyuan": {
        "nlist": 1024,
        "pq_dim": 384,
        "nprobes": (350, 380, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000),
    },
    "gist": {
        "nlist": 1024,
        "pq_dim": 120,
        "nprobes": (10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300),
    },
    "openai1536": {
        "nlist": 1024,
        "pq_dim": 192,
        "nprobes": (10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 250, 300),
    },
}

if DATASET_NAME not in DATASET_PROFILES:
    raise ValueError(
        f"No IVF-PQ profile for dataset {DATASET_NAME!r}; "
        f"available profiles: {tuple(DATASET_PROFILES)}"
    )
DATASET_PROFILE = DATASET_PROFILES[DATASET_NAME]

# Keep IVF/coarse and PQ training controls independent, following the
# corresponding Faiss clustering knobs.  The defaults are intentionally the
# same as Faiss' common IVF/PQ defaults; they can now be tuned independently.
IVF_NITER = 25
IVF_N_INIT = 1
COARSE_TRAINSET_FRACTION = 0.5
# Keep the IVF/PQ training budgets independent. This experiment uses
# cuVS-style Balanced KMeans for both the coarse IVF quantizer and PQ.
PQ_NITER = 50
PQ_N_INIT = 2
PQ_MAX_POINTS_PER_CENTROID = 512
KMEANS_INIT_METHOD = "kmeans++"
COARSE_KMEANS_METHOD = os.environ.get("TENSORVS_COARSE_KMEANS_METHOD", "balanced")
PQ_KMEANS_METHOD = os.environ.get("TENSORVS_PQ_KMEANS_METHOD", "balanced")
SEED = 1234
USE_TRITON = True
DEVICE = torch.device(os.environ.get("TENSORVS_BENCH_DEVICE", "cuda:6"))

NLIST = int(DATASET_PROFILE["nlist"])
NBITS = 8

TOPK = int(os.environ.get("TENSORVS_BENCH_TOPK", "10"))
M = int(DATASET_PROFILE["pq_dim"])
NPROBES = tuple(int(value) for value in DATASET_PROFILE["nprobes"])
Q_BATCH = int(os.environ.get("TENSORVS_BENCH_Q_BATCH", "16"))
UPDATE_PEAK_MEMORY_ONLY = (
    os.environ.get("TENSORVS_BENCH_UPDATE_PEAK_MEMORY_ONLY", "0") == "1"
)

NQ = 1024
WARMUP_ITERATIONS = 2
SELECTION_MODE = "triton_topk"
USE_PACKED_16B_CODES = True
USE_PREALLOCATED_WORKSPACE = (
    os.environ.get("TENSORVS_BENCH_PREALLOCATE_WORKSPACE", "1") != "0"
)
USE_CUDA_GRAPH = os.environ.get("TENSORVS_BENCH_CUDA_GRAPH", "1") != "0"
ENABLE_TRITON_AUTOTUNE = True
# Tune representative probe counts and reuse the nearest representative's
# launch configuration for the other nprobe values in this benchmark.
AUTOTUNE_NPROBE_ANCHORS = (5, 50, 150, 300, 500, 1000)
LUT_DTYPE = "fp16"
REFINE_FACTOR = 1
INDEX_PATH = (
    DIR_PATH
    / "GPUIndex"
    / f"TenVS_IVFPQResidual_nlist{NLIST}_m{M}_nbits{NBITS}"
    # Keep the legacy TensorVS index name. The suffix is intentionally not
    # used to identify the KMeans implementation or the effective training
    # budget.
    "_train256.bin"
)

RESULT_TRAIN_LABELS = {
    "sift1m": 256,
    "sift10m": 256,
    "sift100m": 256,
    "glove1.2m": 512,
    "deep1m": 256,
    "tiny5m": 256,
    "hunyuan": 256,
    "gist": 256,
    "openai1536": 256,
}
LEGACY_RESULT_FILENAMES = {
    dataset_name: (
        f"IVFPQ_{dataset_name}_nlist{profile['nlist']}_"
        f"M{profile['pq_dim']}_nbits{NBITS}_"
        f"train{RESULT_TRAIN_LABELS[dataset_name]}.csv"
    )
    for dataset_name, profile in DATASET_PROFILES.items()
}
if os.environ.get("TENSORVS_BENCH_INDEX_PATH"):
    INDEX_PATH = Path(os.environ["TENSORVS_BENCH_INDEX_PATH"])
if os.environ.get("TENSORVS_BENCH_RESULT_PATH"):
    result_override = Path(os.environ["TENSORVS_BENCH_RESULT_PATH"])
    RESULT_DIR = result_override.parent
    LEGACY_RESULT_FILENAMES[DATASET_NAME] = result_override.name
# ============================================================================


def append_index_statistics(
    index_path: Path,
    index_type: str,
    dataset_name: str,
    build_time_seconds: float | None,
) -> None:
    """Write the canonical CUDA Build CSV/JSON pair."""
    try:
        from ._cuda_build_results import save_tensorvs_build
    except ImportError:
        from _cuda_build_results import save_tensorvs_build
    name = LEGACY_RESULT_FILENAMES.get(dataset_name)
    save_tensorvs_build(index_path, dataset_name, "ivfpq", build_seconds=build_time_seconds,
                        query_path=RESULT_DIR / name if name else None)


def _select_autotune_representatives(nprobes: tuple[int, ...]) -> tuple[int, ...]:
    """Choose available nprobe values nearest to the configured anchor points."""
    available = tuple(sorted({int(value) for value in nprobes}))
    if not available:
        return ()
    return tuple(
        sorted(
            {
                min(
                    available,
                    key=lambda candidate, anchor=anchor: (
                        abs(candidate - anchor),
                        candidate,
                    ),
                )
                for anchor in AUTOTUNE_NPROBE_ANCHORS
            }
        )
    )


def _nearest_autotune_representative(
    nprobe: int,
    representatives: tuple[int, ...],
) -> int:
    """Return the representative whose probe count is nearest to ``nprobe``."""
    if not representatives:
        return int(nprobe)
    return min(
        representatives,
        key=lambda candidate: (abs(int(candidate) - int(nprobe)), int(candidate)),
    )


def load_or_build_index() -> object:
    if INDEX_PATH.exists():
        print(f"Found existing IVF-PQ index: {INDEX_PATH}")
        append_index_statistics(INDEX_PATH, "IVF PQ", DATASET_NAME, None)
        return load_ivfpq_index(INDEX_PATH, load_lists=True)

    print(f"Index not found; building IVF-PQ index at {INDEX_PATH}")
    base_vectors = load_benchmark_vectors(DATASET_PATH)
    build_start = time.perf_counter()
    build_result = build_ivfpq_index(
        vectors=base_vectors,
        nlist=NLIST,
        # ``niter``/``n_init`` are retained as compatibility fallbacks; the
        # explicit coarse/PQ arguments below are the actual settings.
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
        coarse_trainset_fraction=COARSE_TRAINSET_FRACTION,
        pq_max_points_per_centroid=PQ_MAX_POINTS_PER_CENTROID,
        coarse_init_method=KMEANS_INIT_METHOD,
        coarse_kmeans_method=COARSE_KMEANS_METHOD,
        pq_init_method=KMEANS_INIT_METHOD,
        pq_kmeans_method=PQ_KMEANS_METHOD,
    )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    build_seconds = time.perf_counter() - build_start
    saved_path = save_ivfpq_index(
        build_result=build_result,
        output_dir=INDEX_PATH,
        dataset_path=DATASET_PATH,
        build_time_seconds=build_seconds,
    )
    append_index_statistics(
        index_path=saved_path,
        index_type="IVF PQ",
        dataset_name=DATASET_NAME,
        build_time_seconds=build_seconds,
    )
    print(f"Built IVF-PQ index in {build_seconds:.4f}s: {saved_path}")
    return load_ivfpq_index(saved_path, load_lists=True)


def main() -> None:
    if DEVICE.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("IVF-PQ benchmark requires a CUDA device for Triton TopK")
    if not NPROBES:
        raise ValueError("NPROBES must contain at least one value")
    if len(set(NPROBES)) != len(NPROBES):
        raise ValueError(f"NPROBES contains duplicates: {NPROBES}")
    if any(int(value) <= 0 or int(value) > NLIST for value in NPROBES):
        raise ValueError(f"Every nprobe must be in [1, {NLIST}], got {NPROBES}")

    index = load_or_build_index()
    actual_nlist = int(index.centroids.shape[0])
    actual_m = int(index.metadata["m"])
    actual_nbits = int(index.metadata["nbits"])
    if (actual_nlist, actual_m, actual_nbits) != (NLIST, M, NBITS):
        raise ValueError(
            "Configured IVF-PQ parameters do not match the loaded index: "
            f"configured=(nlist={NLIST}, M={M}, nbits={NBITS}), "
            f"loaded=(nlist={actual_nlist}, M={actual_m}, nbits={actual_nbits}), "
            f"path={INDEX_PATH}"
        )
    actual_pq_kmeans_method = str(index.metadata.get("pq_kmeans_method", "lloyd"))
    if actual_pq_kmeans_method != PQ_KMEANS_METHOD:
        raise ValueError(
            "Configured PQ KMeans method does not match the loaded index: "
            f"configured={PQ_KMEANS_METHOD!r}, loaded={actual_pq_kmeans_method!r}, "
            f"path={INDEX_PATH}"
        )
    actual_coarse_kmeans_method = str(
        index.metadata.get("coarse_kmeans_method", "lloyd")
    )
    if actual_coarse_kmeans_method != COARSE_KMEANS_METHOD:
        raise ValueError(
            "Configured coarse KMeans method does not match the loaded index: "
            f"configured={COARSE_KMEANS_METHOD!r}, "
            f"loaded={actual_coarse_kmeans_method!r}, path={INDEX_PATH}"
        )
    index.reside_on_device(DEVICE)
    torch.cuda.synchronize(DEVICE)
    memory_baseline_process_mb = process_gpu_memory_bytes(DEVICE) / (1 << 20)
    queries = load_benchmark_vectors(QUERY_PATH)
    groundtruth = load_benchmark_groundtruth(GROUNDTRUTH_PATH)
    if queries.shape[0] != groundtruth.shape[0]:
        raise ValueError(
            f"Query count {queries.shape[0]} does not match groundtruth count "
            f"{groundtruth.shape[0]}"
        )
    nq = min(int(NQ), int(queries.shape[0]))
    queries = queries[:nq]
    groundtruth = groundtruth[:nq]
    q_batch = max(1, int(Q_BATCH))
    requested_nprobes = tuple(int(value) for value in NPROBES)
    autotune_representatives = _select_autotune_representatives(requested_nprobes)
    # Run representatives first so each reused configuration is tuned at the
    # representative value itself, even if NPROBES was configured out of order.
    execution_nprobes = tuple(
        dict.fromkeys((*autotune_representatives, *requested_nprobes))
    )
    autotuned_representatives: set[int] = set()
    if int(TOPK) <= 0:
        raise ValueError(f"TOPK must be positive, got {TOPK}")
    refine_vectors = load_benchmark_vectors(DATASET_PATH) if REFINE_FACTOR > 1 else None

    ksub = int(index.metadata["ksub"])
    search_topk = TOPK * REFINE_FACTOR if refine_vectors is not None else TOPK
    search_topk = min(search_topk, int(index.metadata.get("num_vectors", search_topk)))
    if index.persisted_list_codes_interleaved_by_subspace is None:
        raise ValueError("Loaded IVF-PQ index is missing its Triton interleaved code layout")
    if index.persisted_list_ids_interleaved is None:
        raise ValueError("Loaded IVF-PQ index is missing its Triton interleaved IDs")
    code_dtype = index.persisted_list_codes_interleaved_by_subspace.dtype
    num_code_vectors = int(index.persisted_list_ids_interleaved.numel())
    code_bytes_per_vector = int(index.metadata["m"]) * index.persisted_list_codes_interleaved_by_subspace.element_size()
    rows = []
    print(
        f"Running IVF-PQ Triton TopK on dataset={DATASET_NAME}, nq={nq}, "
        f"nlist={NLIST}, M={M}, nbits={NBITS}, topk={TOPK}, q_batch={q_batch}, "
        f"ivf=(niter={IVF_NITER}, n_init={IVF_N_INIT}, "
        f"trainset_fraction={COARSE_TRAINSET_FRACTION}, "
        f"method={COARSE_KMEANS_METHOD}), "
        f"pq=(niter={PQ_NITER}, n_init={PQ_N_INIT}, "
        f"max_points={PQ_MAX_POINTS_PER_CENTROID}, method={PQ_KMEANS_METHOD}), "
        f"nprobes={list(requested_nprobes)}, "
        f"preallocated_workspace={USE_PREALLOCATED_WORKSPACE}, "
        f"cuda_graph={USE_CUDA_GRAPH}, "
        "memory_metric=peak_process_gpu_memory_mib, "
        f"autotune_representatives={list(autotune_representatives)}"
    )
    for requested_nprobe in execution_nprobes:
        nprobe = min(int(requested_nprobe), int(index.centroids.shape[0]))
        if not can_run_triton_topk(
            device=DEVICE,
            topk=search_topk,
            m=int(index.metadata["m"]),
            ksub=ksub,
            nprobe=nprobe,
            num_code_vectors=num_code_vectors,
            code_dtype=code_dtype,
            use_packed_16b_codes=USE_PACKED_16B_CODES,
        ):
            raise RuntimeError(
                "Triton IVF-PQ TopK does not support "
                f"topk={search_topk}, M={M}, nbits={NBITS}, nprobe={nprobe}"
            )

        autotune_cache_nprobe = _nearest_autotune_representative(
            nprobe,
            autotune_representatives,
        )
        autotune_this_run = (
            ENABLE_TRITON_AUTOTUNE
            and autotune_cache_nprobe not in autotuned_representatives
        )
        visit_stats = probe_visit_statistics(
            index=index,
            queries=queries,
            nprobe=nprobe,
            device=DEVICE,
            data_bytes_per_vector=code_bytes_per_vector,
        )
        # Do not let a previous nprobe's cached workspace inflate this run's
        # query-time memory above the common resident-index baseline.
        index.search_workspace = None
        index.search_workspace_key = None
        index.triton_partial_keys = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize(DEVICE)
        row = run_one_nprobe(
            queries=queries,
            groundtruth=groundtruth,
            nprobe=nprobe,
            topk=TOPK,
            q_batch=q_batch,
            warmup_iterations=WARMUP_ITERATIONS,
            device=DEVICE,
            search_one=lambda query_batch, current_nprobe: search_ivfpq_index(
                index=index,
                queries=query_batch,
                topk=TOPK,
                nprobe=current_nprobe,
                selection_mode=SELECTION_MODE,
                refine_vectors=refine_vectors,
                refine_factor=REFINE_FACTOR,
                use_packed_16b_codes=USE_PACKED_16B_CODES,
                triton_autotune=autotune_this_run,
                triton_autotune_cache_nprobe=(
                    autotune_cache_nprobe if ENABLE_TRITON_AUTOTUNE else None
                ),
                lut_dtype=LUT_DTYPE,
                use_preallocated_workspace=USE_PREALLOCATED_WORKSPACE,
                use_cuda_graph=USE_CUDA_GRAPH,
            ),
            visit_stats=visit_stats,
            recall_fn=compute_recall_at_k,
            memory_baseline_process_mb=memory_baseline_process_mb,
        )
        if ENABLE_TRITON_AUTOTUNE:
            autotuned_representatives.add(autotune_cache_nprobe)
        row["requested_nprobe"] = int(requested_nprobe)
        print_result(row)
        rows.append(row)

    # Keep CSV output in the user-configured NPROBES order even though the
    # representative runs are intentionally executed first.
    row_order = {nprobe: position for position, nprobe in enumerate(requested_nprobes)}
    rows.sort(key=lambda row: row_order[int(row["requested_nprobe"])])

    # Keep all K values for this dataset/index configuration in one file.
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result_filename = LEGACY_RESULT_FILENAMES[DATASET_NAME]
    except KeyError as error:
        raise ValueError(
            f"No legacy IVF-PQ result filename for dataset {DATASET_NAME!r}"
        ) from error
    csv_path = RESULT_DIR / result_filename
    fieldnames = (
        "nprobes",
        "q_batch",
        "K",
        "qps",
        "recall",
        "visited_vectors_per_query",
        "peak_gpu_memory",
    )
    has_existing_results = csv_path.exists() and csv_path.stat().st_size > 0
    needs_leading_newline = False
    if has_existing_results:
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            existing_fieldnames = next(csv.reader(csv_file), [])
        if existing_fieldnames != list(fieldnames):
            raise ValueError(
                f"Unexpected CSV header in {csv_path}: {existing_fieldnames}; "
                f"expected {list(fieldnames)}"
            )
        with csv_path.open("rb") as csv_file:
            csv_file.seek(-1, 2)
            needs_leading_newline = csv_file.read(1) not in (b"\n", b"\r")

    output_rows = [
        {
            "nprobes": row["requested_nprobe"],
            "q_batch": q_batch,
            "K": TOPK,
            "qps": row["qps"],
            "recall": row["recall"],
            "visited_vectors_per_query": row["visited_vectors_per_query"],
            # Peak logical query allocation above the resident index (MiB).
            "peak_gpu_memory": row["peak_gpu_memory"],
        }
        for row in rows
    ]
    replace_existing = os.environ.get("TENSORVS_BENCH_REPLACE_EXISTING", "0") == "1"
    if UPDATE_PEAK_MEMORY_ONLY:
        if not has_existing_results:
            raise FileNotFoundError(
                f"Peak-memory update mode requires an existing result file: {csv_path}"
            )
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            existing_rows = list(csv.DictReader(csv_file))
        peak_by_nprobe = {
            int(row["nprobes"]): row["peak_gpu_memory"] for row in output_rows
        }
        updated = 0
        for existing_row in existing_rows:
            if (
                int(existing_row["q_batch"]) == q_batch
                and int(existing_row["K"]) == TOPK
                and int(existing_row["nprobes"]) in peak_by_nprobe
            ):
                existing_row["peak_gpu_memory"] = peak_by_nprobe[
                    int(existing_row["nprobes"])
                ]
                updated += 1
        if updated == 0:
            raise ValueError(
                f"No existing rows matched q_batch={q_batch}, K={TOPK} in {csv_path}"
            )
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
        print(
            f"Peak memory updated for {updated} existing rows at "
            f"q_batch={q_batch}, K={TOPK}: {csv_path}"
        )
    elif replace_existing:
        if not has_existing_results:
            raise FileNotFoundError(
                f"Replacement mode requires an existing result file: {csv_path}"
            )
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            existing_rows = list(csv.DictReader(csv_file))
        retained_rows = [
            row
            for row in existing_rows
            if not (
                int(row["q_batch"]) == q_batch
                and int(row["K"]) == TOPK
            )
        ]
        nprobe_order = {
            int(value): position for position, value in enumerate(requested_nprobes)
        }
        replacement_rows = retained_rows + output_rows
        replacement_rows.sort(
            key=lambda row: (
                int(row["q_batch"]),
                int(row["K"]),
                nprobe_order.get(int(row["nprobes"]), int(row["nprobes"])),
            )
        )
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(replacement_rows)
        print(f"CSV results replaced for q_batch={q_batch}, K={TOPK}: {csv_path}")
    else:
        with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            if needs_leading_newline:
                csv_file.write("\n")
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not has_existing_results:
                writer.writeheader()
            writer.writerows(output_rows)
        print(f"CSV results appended: {csv_path}")


if __name__ == "__main__":
    main()
