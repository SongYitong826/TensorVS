"""Profile and plot IVF-Flat/IVF-PQ query-time breakdowns.

The default dataset is GloVe. The figure contains two horizontal stacked-bar
plots: panel A is IVF-Flat and panel B is IVF-PQ. Each panel compares
TensorVS, CUVS, and FAISS.

Detailed mode uses benchmark-only non-fused TensorVS scan kernels and the
maintained multi-pass CUVS/FAISS paths where available. CUVS and FAISS are
profiled through optional source instrumentation enabled only in benchmark
workers via ``IVF_STAGE_PROFILE_PATH``. IVF-Flat reports list scheduling
separately and keeps vector reads plus distance arithmetic as one fused stage;
the libraries' production search paths remain unchanged.

Examples::

    conda run -n tensorvs python plot/query_time_breakdown.py
    conda run -n tensorvs python plot/query_time_breakdown.py --dataset gist
    conda run -n tensorvs python plot/query_time_breakdown.py --nq 128

Outputs are written to ``plot/Figures`` and ``plot/results`` by default.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DEVICE = "cuda:1"
DEFAULT_TOPK = 10
# 120 is present in the existing GloVe CUVS and FAISS result CSVs for both
# IVF-Flat and IVF-PQ. Override with --nprobe when fresh benchmark results are
# available for another probe count.
DEFAULT_NPROBE = 120
# Match the existing CUVS/FAISS IVF-Flat benchmark default. This is also
# passed to all three backends so a comparison does not mix batch sizes.
DEFAULT_QUERY_BATCH = 128
DEFAULT_WARMUP_ITERATIONS = 2
DEFAULT_TIMING_REPETITIONS = 5
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_FIGURES_DIR = SCRIPT_DIR / "Figures"
FAISS_PROFILE_PYTHON = Path(
    "/home/songyitong/faiss/build-profile/faiss/python/pybuild/lib"
)
CUVS_PROFILE_LIB = Path("/home/songyitong/cuvs/cpp/build")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    query_candidates: tuple[Path, ...]
    ivfflat_filename: str
    ivfpq_filename: str
    cuvs_ivfflat_filename: str
    cuvs_ivfpq_filename: str
    faiss_ivfflat_filename: str
    faiss_ivfpq_filename: str
    cuvs_pq_result_stem: str
    faiss_pq_result_stem: str

    @property
    def tensorvs_ivfflat_path(self) -> Path:
        return self.root / "GPUIndex" / self.ivfflat_filename

    @property
    def tensorvs_ivfpq_path(self) -> Path:
        return self.root / "GPUIndex" / self.ivfpq_filename

    @property
    def cuvs_ivfflat_path(self) -> Path:
        return self.root / "GPUIndex" / self.cuvs_ivfflat_filename

    @property
    def cuvs_ivfpq_path(self) -> Path:
        return self.root / "GPUIndex" / self.cuvs_ivfpq_filename

    @property
    def faiss_ivfflat_path(self) -> Path:
        return self.root / "GPUIndex" / self.faiss_ivfflat_filename

    @property
    def faiss_ivfpq_path(self) -> Path:
        return self.root / "GPUIndex" / self.faiss_ivfpq_filename


DATASETS = (
    DatasetConfig(
        name="glove1.2m",
        root=Path("/storage/vector_data/glove1.2m"),
        query_candidates=(Path("glove1.2m_query.fvecs"),),
        ivfflat_filename="TenVS_IVFFlat_nlist1024.bin",
        ivfpq_filename="TenVS_IVFPQResidual_nlist1024_m25_nbits8_train256.bin",
        cuvs_ivfflat_filename="cuVS_IVFFlat_nlist1024.bin",
        cuvs_ivfpq_filename="cuVS_IVFPQ_nlist1024_pqdim25_pqbits8.bin",
        faiss_ivfflat_filename="Faiss_IVFFlat_nlist1024.index",
        faiss_ivfpq_filename="Faiss_IVFPQ_nlist1024_M25_nbits8.index",
        cuvs_pq_result_stem="IVFPQ_glove1.2m_nlist1024_pqdim25_pqbits8",
        faiss_pq_result_stem="IVFPQ_glove1.2m_nlist1024_M25_nbits8",
    ),
    DatasetConfig(
        name="gist",
        root=Path("/storage/vector_data/gist"),
        query_candidates=(Path("gist_query.fvecs"), Path("gist/gist_query.fvecs")),
        ivfflat_filename="TenVS_IVFFlat_nlist1024.bin",
        ivfpq_filename="TenVS_IVFPQResidual_nlist1024_m120_nbits8_train256.bin",
        cuvs_ivfflat_filename="cuVS_IVFFlat_nlist1024.bin",
        cuvs_ivfpq_filename="cuVS_IVFPQ_nlist1024_pqdim120_pqbits8.bin",
        faiss_ivfflat_filename="Faiss_IVFFlat_nlist1024.index",
        faiss_ivfpq_filename="Faiss_IVFPQ_nlist1024_M96_nbits8.index",
        cuvs_pq_result_stem="IVFPQ_gist_nlist1024_pqdim120_pqbits8",
        faiss_pq_result_stem="IVFPQ_gist_nlist1024_M96_nbits8",
    ),
)


METHODS = ("TensorVS", "CUVS", "FAISS")
PLOT_METHODS = ("FAISS", "CUVS", "TensorVS")
METHOD_LABELS = {
    "FAISS": "Faiss GPU",
    "CUVS": "cuVS",
    "TensorVS": "TensorVS",
}
FLAT_STAGES = (
    "coarse_ivf_selection",
    "list_scheduling_preparation",
    "vector_read_distance",
    # Legacy detailed stage names remain accepted when reading older
    # profiling CSV files, but new figures merge them into the fused stage.
    "vector_gather_read",
    "dot_product_computation",
    "l2_epilogue_distance_writeback",
    "vector_read_distance_topk_fused",
    "topk_selection",
    "topk_final_merge",
)
PQ_STAGES = (
    "coarse_ivf_selection",
    "list_scheduling_preparation",
    "query_transform",
    "lut_computation",
    "lut_build_lookup_accumulation",
    "lut_lookup_accumulation",
    "lut_lookup_topk_fused",
    "topk_selection",
    "topk_final_merge",
)
STAGE_LABELS = {
    "coarse_ivf_selection": "Coarse IVF Selection",
    "list_scheduling_preparation": "Data Preparation",
    "vector_gather_read": "Vector Gather / Read",
    "dot_product_computation": "Dot-product Computation",
    "l2_epilogue_distance_writeback": "Distance Finalization + Write-back",
    "vector_read_distance": "Vector Read + Distance Computation",
    "vector_read_distance_topk_fused": "Vector read + distance + Top-K (fused)",
    "topk_selection": "Top-K selection",
    "topk_final_merge": "Final Top-K merge",
    "query_transform": "Query transform",
    "lut_computation": "LUT Construction",
    "lut_build_lookup_accumulation": "LUT computation + lookup/accumulation (fused)",
    "lut_lookup_accumulation": "LUT lookup + accumulation",
    "lut_lookup_topk_fused": "LUT lookup/accumulation + Top-K (fused)",
    "unsplit_library_search": "Unsplit library search",
}
STAGE_COLORS = {
    "coarse_ivf_selection": "#4C78A8",
    "list_scheduling_preparation": "#59A14F",
    "vector_gather_read": "#59A14F",
    "dot_product_computation": "#EDC948",
    "l2_epilogue_distance_writeback": "#9C755F",
    "vector_read_distance": "#EDC948",
    "vector_read_distance_topk_fused": "#59A14F",
    "topk_selection": "#E15759",
    "topk_final_merge": "#E15759",
    "query_transform": "#F28E2B",
    "lut_computation": "#B279A2",
    "lut_build_lookup_accumulation": "#76B7B2",
    "lut_lookup_accumulation": "#76B7B2",
    "lut_lookup_topk_fused": "#76B7B2",
    "unsplit_library_search": "#A6A6A6",
}


def first_existing(root: Path, candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    attempted = ", ".join(str(root / candidate) for candidate in candidates)
    raise FileNotFoundError(f"None of the candidate files exists: {attempted}")


def validate_device(device: torch.device) -> None:
    import torch

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Query-time profiling requires a CUDA device")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device}, but only {torch.cuda.device_count()} CUDA devices exist"
        )


def tensorvs_flow_stages(profile: dict[str, float], method: str) -> dict[str, float]:
    """Map TensorVS profile events to the algorithm-flow names.

    TensorVS IVF-Flat uses one fused Triton kernel for raw-vector loading,
    distance calculation, and local Top-K. Detailed mode reports list/task
    scheduling separately, but keeps vector reads and distance arithmetic as
    one fused stage because they overlap in the scan kernel.
    IVF-PQ exposes LUT preparation, a fused scan/local-Top-K path, and the
    final merge separately.
    """

    def total(*names: str) -> float:
        return sum(float(profile.get(name, 0.0)) for name in names)

    # Keep this definition aligned with the source-level CUVS/FAISS events:
    # tensor/workspace preparation is API overhead, not coarse IVF selection.
    coarse = total("centroid_distance", "centroid_topk")
    if method == "IVFFlat":
        if "list_scheduling_preparation" in profile:
            return {
                "coarse_ivf_selection": coarse,
                "list_scheduling_preparation": total(
                    "list_scheduling_preparation"
                ),
                "vector_read_distance": total("vector_read_distance"),
                "l2_epilogue_distance_writeback": total(
                    "l2_epilogue_distance_writeback"
                ),
                "topk_selection": total("topk_selection"),
            }
        if "vector_gather_read" in profile:
            return {
                "coarse_ivf_selection": coarse,
                "list_scheduling_preparation": 0.0,
                "vector_read_distance": total(
                    "vector_gather_read", "dot_product_computation"
                ),
                "l2_epilogue_distance_writeback": total(
                    "l2_epilogue_distance_writeback"
                ),
                "topk_selection": total("topk_selection"),
            }
        if "vector_read_distance" in profile:
            return {
                "coarse_ivf_selection": coarse,
                "list_scheduling_preparation": 0.0,
                "vector_read_distance": total("vector_read_distance"),
                "topk_selection": total("topk_selection"),
            }
        return {
            "coarse_ivf_selection": coarse,
            "vector_read_distance_topk_fused": total(
                "triton_prepare_layout", "triton_topk_select"
            ),
            "topk_final_merge": 0.0,
        }
    if "lut_lookup_accumulation" in profile:
        return {
            "coarse_ivf_selection": coarse,
            "list_scheduling_preparation": total(
                "prepare_tensors", "init_outputs", "triton_prepare_layout",
                "triton_packed_prepare", "list_scheduling_preparation",
            ),
            "query_transform": 0.0,
            "lut_computation": total("LUT_prepare"),
            "lut_lookup_accumulation": total("lut_lookup_accumulation"),
            "topk_selection": total("topk_selection"),
        }
    return {
        "coarse_ivf_selection": coarse,
        "list_scheduling_preparation": total(
            "prepare_tensors", "init_outputs", "triton_prepare_layout",
            "triton_packed_prepare", "list_scheduling_preparation",
        ),
        "query_transform": 0.0,
        "lut_computation": total("LUT_prepare"),
        "lut_lookup_topk_fused": total(
            "triton_scan"
        ),
        "topk_final_merge": total("triton_final_merge"),
    }


def profile_tensorvs(
    *,
    method: str,
    index_path: Path,
    queries: torch.Tensor,
    device: torch.device,
    topk: int,
    nprobe: int,
    query_batch: int,
    warmup_iterations: int,
    timing_repetitions: int,
    nq: int | None,
    detailed_profile: bool,
) -> dict[str, Any]:
    import torch

    from tensorvs import load_ivfflat_index, load_ivfpq_index

    if nq is not None:
        queries = queries[: min(int(nq), int(queries.shape[0]))]
    if queries.shape[0] == 0:
        raise ValueError("At least one query is required")
    index = (
        load_ivfflat_index(index_path, load_lists=True)
        if method == "IVFFlat"
        else load_ivfpq_index(index_path, load_lists=True)
    )
    index.reside_on_device(device)

    q_batch = max(1, int(query_batch))
    first_batch = queries[: min(q_batch, int(queries.shape[0]))]
    if first_batch.shape[0] < q_batch:
        first_batch = torch.cat(
            (first_batch, first_batch[-1:].expand(q_batch - first_batch.shape[0], -1))
        )

    unfused_runner = None
    if detailed_profile:
        from _tensorvs_unfused_profile import TensorVSUnfusedRunner

        unfused_runner = TensorVSUnfusedRunner(
            index=index,
            algorithm=method,
            device=device,
            query_batch=q_batch,
            nprobe=nprobe,
            topk=topk,
            lut_dtype="fp16",
        )

    def run_search(batch: torch.Tensor, profile: dict[str, float] | None) -> None:
        if unfused_runner is not None:
            unfused_runner(batch, profile)
        elif method == "IVFFlat":
            from tensorvs import search_ivfflat_index

            search_ivfflat_index(
                index, batch, topk=topk, nprobe=nprobe, device=device,
                selection_mode="triton_topk", profile=profile,
            )
        else:
            from tensorvs import search_ivfpq_index

            search_ivfpq_index(
                index, batch, topk=topk, nprobe=nprobe, device=device,
                selection_mode="triton_topk", profile=profile,
                use_packed_16b_codes=True, lut_dtype="fp16",
                use_preallocated_workspace=False, use_cuda_graph=False,
            )

    print(f"[TensorVS/{method}] warming up {index_path}")
    with torch.inference_mode():
        for _ in range(max(1, int(warmup_iterations))):
            run_search(first_batch, None)
    torch.cuda.synchronize(device)
    del first_batch

    stage_names = FLAT_STAGES if method == "IVFFlat" else PQ_STAGES
    totals = {stage: 0.0 for stage in stage_names}
    executed_queries = 0
    pass_ms_per_query: list[float] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max(1, int(timing_repetitions))):
            pass_started = time.perf_counter()
            pass_queries = 0
            for start in range(0, int(queries.shape[0]), q_batch):
                end = min(start + q_batch, int(queries.shape[0]))
                batch = queries[start:end]
                actual_size = int(batch.shape[0])
                if actual_size < q_batch:
                    batch = torch.cat(
                        (batch, batch[-1:].expand(q_batch - actual_size, -1))
                    )
                profile: dict[str, float] = {}
                run_search(batch, profile)
                torch.cuda.synchronize(device)
                for stage, seconds in tensorvs_flow_stages(profile, method).items():
                    totals[stage] += seconds
                # Count real queries, while the measured event also includes
                # the same last-batch padding used by the external benchmarks.
                executed_queries += actual_size
                pass_queries += actual_size
                del batch
            pass_ms_per_query.append(
                (time.perf_counter() - pass_started) * 1000.0 /
                max(pass_queries, 1)
            )

    divisor = max(executed_queries, 1)
    stages_ms = {stage: value * 1000.0 / divisor for stage, value in totals.items()}
    total_ms = sum(stages_ms.values())
    normal_end_to_end_ms = None
    if not detailed_profile:
        normal_seconds = 0.0
        normal_executed = 0
        normal_pass_ms_per_query: list[float] = []
        with torch.inference_mode():
            for _ in range(max(1, int(timing_repetitions))):
                pass_seconds = 0.0
                pass_queries = 0
                for start in range(0, int(queries.shape[0]), q_batch):
                    end = min(start + q_batch, int(queries.shape[0]))
                    batch = queries[start:end]
                    actual_size = int(batch.shape[0])
                    if actual_size < q_batch:
                        batch = torch.cat(
                            (batch, batch[-1:].expand(q_batch - actual_size, -1))
                        )
                    normal_started = time.perf_counter()
                    run_search(batch, None)
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - normal_started
                    normal_seconds += elapsed
                    pass_seconds += elapsed
                    normal_executed += actual_size
                    pass_queries += actual_size
                    del batch
                normal_pass_ms_per_query.append(
                    pass_seconds * 1000.0 / max(pass_queries, 1)
                )
        normal_end_to_end_ms = (
            normal_seconds * 1000.0 / max(normal_executed, 1)
        )
    else:
        normal_pass_ms_per_query = []
    print(
        f"[TensorVS/{method}] {executed_queries} executed queries, "
        f"profile wall time={time.perf_counter() - started:.2f}s, "
        f"flow total={total_ms:.6f} ms/query"
    )
    del index
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "method": "TensorVS",
        "index_path": str(index_path),
        "nprobe": int(nprobe),
        "topk": int(topk),
        "query_count": int(queries.shape[0]),
        "executed_queries": int(executed_queries),
        "timing_repetitions": max(1, int(timing_repetitions)),
        "profile_pass_ms_per_query": pass_ms_per_query,
        "profile_pass_stddev_ms_per_query": (
            statistics.stdev(pass_ms_per_query)
            if len(pass_ms_per_query) > 1 else 0.0
        ),
        "normal_pass_ms_per_query": normal_pass_ms_per_query,
        "normal_pass_stddev_ms_per_query": (
            statistics.stdev(normal_pass_ms_per_query)
            if len(normal_pass_ms_per_query) > 1 else None
        ),
        "query_batch": int(query_batch),
        "stages_ms_per_query": stages_ms,
        "total_ms_per_query": total_ms,
        "normal_end_to_end_ms_per_query": normal_end_to_end_ms,
        "timing_source": "TensorVS CUDA-event profile",
        "stage_notes": (
            "Benchmark-only non-fused candidate materialization followed by "
            "an independent Top-K. Profiling-only candidate-buffer clearing is "
            "excluded. List scheduling is reported separately, while vector "
            "reads and distance arithmetic remain one fused scan stage. The "
            "normal TensorVS search API is unchanged."
            if detailed_profile
            else "Normal TensorVS fused search path."
        ),
    }


def _load_fvecs_for_worker(path: Path, max_rows: int | None) -> Any:
    """Read fvecs without importing TensorVS in the CUVS/FAISS environments."""
    import numpy as np

    rows: list[np.ndarray] = []
    with path.open("rb") as handle:
        while max_rows is None or len(rows) < max_rows:
            dim_bytes = handle.read(4)
            if not dim_bytes:
                break
            dim = int(np.frombuffer(dim_bytes, dtype=np.int32)[0])
            vector = np.fromfile(handle, dtype=np.float32, count=dim)
            if vector.size != dim:
                raise ValueError(f"Truncated fvecs record in {path}")
            rows.append(vector)
    if not rows:
        raise ValueError(f"No vectors found in {path}")
    return np.asarray(rows, dtype=np.float32)


def _padded_numpy_batch(queries: Any, start: int, q_batch: int) -> tuple[Any, int]:
    import numpy as np

    batch = np.ascontiguousarray(queries[start : start + q_batch], dtype=np.float32)
    actual = int(batch.shape[0])
    if actual < q_batch:
        batch = np.concatenate((batch, np.repeat(batch[-1:], q_batch - actual, axis=0)))
    return batch, actual


def _read_stage_profile(
    path: Path, backend: str, algorithm: str, executed_queries: int
) -> dict[str, float]:
    allowed = set(FLAT_STAGES if algorithm == "IVFFlat" else PQ_STAGES)
    totals = {stage: 0.0 for stage in allowed}
    matching_rows = 0
    if not path.exists():
        raise RuntimeError(
            f"{backend}/{algorithm} produced no CUDA-event profile at {path}. "
            "Rebuild the instrumented library or verify that its rebuilt "
            "shared library is the one loaded by Python."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 4:
                continue
            row_backend, row_algorithm, stage, milliseconds = row
            if row_backend.lower() != backend.lower() or row_algorithm != algorithm:
                continue
            if stage not in allowed:
                continue
            totals[stage] += float(milliseconds)
            matching_rows += 1
    if matching_rows == 0:
        raise RuntimeError(
            f"{path} contains no recognized {backend}/{algorithm} stage rows"
        )
    divisor = max(executed_queries, 1)
    return {stage: value / divisor for stage, value in totals.items()}


def run_external_worker(args: argparse.Namespace) -> None:
    """Run one external backend in its own installed Conda environment.

    This worker deliberately times the same synchronized query loop for CUVS
    and FAISS. The parent process remains in the tensorvs environment, while
    the worker imports the backend that is actually being measured.
    """
    import numpy as np

    backend = args.backend_worker
    gpu_id = int(args.gpu_id)
    if args.detailed_profile:
        os.environ["IVF_STAGE_PROFILE_DETAIL"] = "1"
    queries = _load_fvecs_for_worker(Path(args.query_path), args.nq)
    q_batch = max(1, int(args.query_batch))
    if backend == "cuvs":
        import cupy as cp
        from cuvs.neighbors import ivf_flat, ivf_pq

        cp.cuda.Device(gpu_id).use()
        index = (
            ivf_flat.load(args.index_path)
            if args.algorithm == "IVFFlat"
            else ivf_pq.load(args.index_path)
        )
        if args.algorithm == "IVFFlat":
            search_params = ivf_flat.SearchParams(n_probes=int(args.nprobe))

            def search(batch: Any) -> Any:
                return ivf_flat.search(search_params, index, batch, int(args.topk))
        else:
            search_params = ivf_pq.SearchParams(
                n_probes=int(args.nprobe),
                lut_dtype=np.float16,
                internal_distance_dtype=np.float32,
                coarse_search_dtype=np.float32,
            )

            def search(batch: Any) -> Any:
                return ivf_pq.search(search_params, index, batch, int(args.topk))

        def sync() -> None:
            cp.cuda.Device(gpu_id).synchronize()

    elif backend == "faiss":
        sys.path.insert(0, "/home/songyitong/faiss/exp")
        import faiss
        from _benchmark_utils import maybe_to_gpu

        index_cpu = faiss.read_index(args.index_path)
        # Detailed and normal IVF-Flat runs intentionally use the same layout
        # selected by Faiss's GPU cloner (the IVFFlat config defaults to the
        # interleaved layout). Both scan layouts are instrumented; production
        # has no clock/counter work unless the benchmark detail flag is set.
        index, resources = maybe_to_gpu(
            index_cpu,
            True,
            gpu_id,
            ivfpq_interleaved_layout=bool(args.faiss_interleaved_layout),
            lut_dtype="fp16",
        )
        index.nprobe = int(args.nprobe)

        def search(batch: Any) -> Any:
            return index.search(batch, int(args.topk))

        from _benchmark_utils import cuda_synchronize_device

        def sync() -> None:
            cuda_synchronize_device(gpu_id)
    else:  # pragma: no cover - argparse limits this value
        raise ValueError(f"Unsupported backend worker: {backend}")

    first_batch, _ = _padded_numpy_batch(queries, 0, q_batch)
    for _ in range(max(1, int(args.warmup_iterations))):
        search(cp.asarray(first_batch) if backend == "cuvs" else first_batch)
        sync()
    del first_batch

    profile_path = Path(args.worker_output).with_suffix(".stages.csv")
    profile_path.unlink(missing_ok=True)
    os.environ["IVF_STAGE_PROFILE_PATH"] = str(profile_path)
    total_seconds = 0.0
    executed = 0
    profile_pass_ms_per_query: list[float] = []
    try:
        for _ in range(max(1, int(args.timing_repetitions))):
            pass_seconds = 0.0
            pass_queries = 0
            for start in range(0, int(queries.shape[0]), q_batch):
                batch, actual = _padded_numpy_batch(queries, start, q_batch)
                if backend == "cuvs":
                    batch = cp.asarray(batch)
                started = time.perf_counter()
                output = search(batch)
                sync()
                elapsed = time.perf_counter() - started
                total_seconds += elapsed
                pass_seconds += elapsed
                executed += actual
                pass_queries += actual
                del batch, output
            profile_pass_ms_per_query.append(
                pass_seconds * 1000.0 / max(pass_queries, 1)
            )
    finally:
        os.environ.pop("IVF_STAGE_PROFILE_PATH", None)
        os.environ.pop("IVF_STAGE_PROFILE_DETAIL", None)

    normal_end_to_end_ms = None
    normal_pass_ms_per_query: list[float] = []
    if not args.detailed_profile:
        normal_seconds = 0.0
        normal_executed = 0
        for _ in range(max(1, int(args.timing_repetitions))):
            pass_seconds = 0.0
            pass_queries = 0
            for start in range(0, int(queries.shape[0]), q_batch):
                batch, actual = _padded_numpy_batch(queries, start, q_batch)
                if backend == "cuvs":
                    batch = cp.asarray(batch)
                normal_started = time.perf_counter()
                output = search(batch)
                sync()
                elapsed = time.perf_counter() - normal_started
                normal_seconds += elapsed
                pass_seconds += elapsed
                normal_executed += actual
                pass_queries += actual
                del batch, output
            normal_pass_ms_per_query.append(
                pass_seconds * 1000.0 / max(pass_queries, 1)
            )
        normal_end_to_end_ms = (
            normal_seconds * 1000.0 / max(normal_executed, 1)
        )

    stages_ms = _read_stage_profile(profile_path, backend, args.algorithm, executed)
    profiled_total_ms = sum(stages_ms.values())

    result = {
        "method": "CUVS" if backend == "cuvs" else "FAISS",
        "algorithm": args.algorithm,
        "index_path": args.index_path,
        "nprobe": int(args.nprobe),
        "topk": int(args.topk),
        "query_count": int(queries.shape[0]),
        "executed_queries": executed,
        "timing_repetitions": max(1, int(args.timing_repetitions)),
        "profile_pass_ms_per_query": profile_pass_ms_per_query,
        "profile_pass_stddev_ms_per_query": (
            statistics.stdev(profile_pass_ms_per_query)
            if len(profile_pass_ms_per_query) > 1 else 0.0
        ),
        "normal_pass_ms_per_query": normal_pass_ms_per_query,
        "normal_pass_stddev_ms_per_query": (
            statistics.stdev(normal_pass_ms_per_query)
            if len(normal_pass_ms_per_query) > 1 else None
        ),
        "query_batch": q_batch,
        "stages_ms_per_query": stages_ms,
        "total_ms_per_query": profiled_total_ms,
        "end_to_end_ms_per_query": total_seconds * 1000.0 / max(executed, 1),
        "normal_end_to_end_ms_per_query": normal_end_to_end_ms,
        "timing_source": f"{backend} source-level CUDA-event profile",
        "stage_notes": (
            "IVF-Flat list scheduling is measured at its library launch "
            "boundary. Raw vector-read and distance-arithmetic counters are "
            "merged into one fused Vector Read + Distance Computation stage; "
            "L2 epilogue/write-back and Top-K remain separate. Both libraries "
            "use one Python-side per-query normalization, and their production "
            "paths remain unchanged when detailed profiling is disabled."
            if args.algorithm == "IVFFlat" and args.detailed_profile
            else "CUVS IVF-PQ LUT computation and LUT lookup/accumulation are "
            "attributed from clock64 block-cycle counters and normalized to "
            "the enclosing CUDA-event duration; other stages follow actual "
            "library launch boundaries."
            if backend == "cuvs" and args.algorithm == "IVFPQ"
            and args.detailed_profile
            else "CUDA events follow actual library launch boundaries; fused "
            "kernels are reported as fused stages."
        ),
    }
    Path(args.worker_output).write_text(json.dumps(result, indent=2), encoding="utf-8")


def profile_external(
    *, method: str, algorithm: str, dataset: DatasetConfig, query_path: Path,
    nprobe: int, topk: int, query_batch: int, warmup_iterations: int,
    timing_repetitions: int, nq: int | None, gpu_id: int,
    detailed_profile: bool,
) -> dict[str, Any]:
    backend = "cuvs" if method == "CUVS" else "faiss"
    index_path = (
        dataset.cuvs_ivfflat_path if method == "CUVS" and algorithm == "IVFFlat"
        else dataset.cuvs_ivfpq_path if method == "CUVS"
        else dataset.faiss_ivfflat_path if algorithm == "IVFFlat"
        else dataset.faiss_ivfpq_path
    )
    if not index_path.exists():
        raise FileNotFoundError(f"{method} index not found: {index_path}")
    env_name = "cuvs" if backend == "cuvs" else "faiss"
    with tempfile.TemporaryDirectory(prefix="query_breakdown_") as temp_dir:
        output_path = Path(temp_dir) / "result.json"
        command = [
            "conda", "run", "-n", env_name, "python", str(Path(__file__).resolve()),
            "--backend-worker", backend, "--algorithm", algorithm,
            "--index-path", str(index_path), "--query-path", str(query_path),
            "--worker-output", str(output_path), "--nprobe", str(nprobe),
            "--topk", str(topk), "--query-batch", str(query_batch),
            "--warmup-iterations", str(warmup_iterations), "--gpu-id", str(gpu_id),
            "--timing-repetitions", str(timing_repetitions),
        ]
        if method == "FAISS" and algorithm == "IVFPQ" and dataset.name == "glove1.2m":
            command.append("--faiss-interleaved-layout")
        if detailed_profile:
            command.append("--detailed-profile")
        if nq is not None:
            command.extend(("--nq", str(nq)))
        print(f"[{method}/{algorithm}] running with nprobe={nprobe}, qbatch={query_batch}")
        worker_env = os.environ.copy()
        if method == "CUVS":
            cuvs_library_dirs = (CUVS_PROFILE_LIB / "c", CUVS_PROFILE_LIB)
            required_libraries = (
                cuvs_library_dirs[0] / "libcuvs_c.so",
                cuvs_library_dirs[1] / "libcuvs.so",
            )
            missing = [path for path in required_libraries if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Rebuilt CUVS libraries not found: {missing}")
            previous = worker_env.get("LD_LIBRARY_PATH")
            rebuilt_paths = os.pathsep.join(str(path) for path in cuvs_library_dirs)
            worker_env["LD_LIBRARY_PATH"] = (
                f"{rebuilt_paths}{os.pathsep}{previous}"
                if previous else rebuilt_paths
            )
            # The CUVS Conda Python runtime contains transitive DT_RPATH
            # entries that take precedence over LD_LIBRARY_PATH. Preloading
            # the two ABI-compatible rebuilt libraries guarantees that this
            # worker, and only this worker, uses the instrumented build.
            previous_preload = worker_env.get("LD_PRELOAD")
            rebuilt_libraries = os.pathsep.join(
                str(path) for path in reversed(required_libraries)
            )
            worker_env["LD_PRELOAD"] = (
                f"{rebuilt_libraries}{os.pathsep}{previous_preload}"
                if previous_preload else rebuilt_libraries
            )
        else:
            if not FAISS_PROFILE_PYTHON.exists():
                raise FileNotFoundError(
                    f"Rebuilt FAISS Python package not found: {FAISS_PROFILE_PYTHON}"
                )
            previous = worker_env.get("PYTHONPATH")
            worker_env["PYTHONPATH"] = (
                f"{FAISS_PROFILE_PYTHON}{os.pathsep}{previous}"
                if previous else str(FAISS_PROFILE_PYTHON)
            )
        subprocess.run(command, check=True, env=worker_env)
        return json.loads(output_path.read_text(encoding="utf-8"))


def merge_external_stage_times(
    result: dict[str, Any], stage_times_path: Path | None, algorithm: str,
) -> None:
    if stage_times_path is None:
        return
    data = json.loads(stage_times_path.read_text(encoding="utf-8"))
    method_data = data.get(result["method"], {})
    algorithm_data = method_data.get(algorithm, method_data.get(algorithm.lower(), {}))
    stages = algorithm_data.get("stages_ms_per_query", algorithm_data)
    allowed = set(FLAT_STAGES if algorithm == "IVFFlat" else PQ_STAGES)
    parsed = {key: float(value) for key, value in stages.items() if key in allowed}
    if set(parsed) != allowed:
        missing = ", ".join(sorted(allowed - set(parsed)))
        raise ValueError(
            f"{stage_times_path}: missing {result['method']}/{algorithm} stages: {missing}"
        )
    if any(value < 0.0 for value in parsed.values()):
        raise ValueError(f"Negative stage time in {stage_times_path}")
    result["stages_ms_per_query"] = parsed
    result["total_ms_per_query"] = sum(parsed.values())
    result["timing_source"] = f"{result['method']} library-side stage timing JSON"
    result["stage_times_json"] = str(stage_times_path)


def attribute_detailed_profile(
    detailed: dict[str, Any], fused: dict[str, Any], algorithm: str,
) -> dict[str, Any]:
    """Scale detailed stage proportions to the normal fused-path total.

    De-fusing can add candidate-buffer traffic that is absent from the normal
    implementation. The detailed run therefore supplies attribution weights,
    while the fused run supplies the real bar length. Stages already exposed
    by the normal path retain their directly measured fused-path duration.
    """

    if algorithm == "IVFFlat":
        canonical_stages = (
            "coarse_ivf_selection", "list_scheduling_preparation",
            "vector_read_distance", "l2_epilogue_distance_writeback",
            "topk_selection",
        )
        aliases = {
            "vector_gather_read": "vector_read_distance",
            "dot_product_computation": "vector_read_distance",
            "vector_read_distance_topk_fused": "vector_read_distance",
            "topk_final_merge": "topk_selection",
        }
    else:
        canonical_stages = (
            "coarse_ivf_selection", "list_scheduling_preparation",
            "query_transform", "lut_computation", "lut_lookup_accumulation",
            "topk_selection",
        )
        aliases = {
            "lut_build_lookup_accumulation": "lut_lookup_accumulation",
            "lut_lookup_topk_fused": "lut_lookup_accumulation",
            "topk_final_merge": "topk_selection",
        }

    def canonicalize(profile: dict[str, Any]) -> dict[str, float]:
        values = {stage: 0.0 for stage in canonical_stages}
        for stage, value in profile["stages_ms_per_query"].items():
            target = aliases.get(stage, stage)
            if target in values:
                values[target] += float(value)
        return values

    detailed_values = canonicalize(detailed)
    fused_values = canonicalize(fused)
    # External-library event rows intentionally cover only the instrumented
    # GPU stages. Their synchronized API duration is the fair quantity to
    # compare against TensorVS's normal search path, particularly for the
    # short IVF-PQ query. TensorVS has no separate end-to-end field here.
    target_total = float(
        fused.get("normal_end_to_end_ms_per_query")
        or fused.get("end_to_end_ms_per_query")
        or fused["total_ms_per_query"]
    )
    method = str(fused["method"])

    # These stages have honest launch boundaries in the normal fused path and
    # do not need attribution from the slower de-fused run.
    fixed_stages = {"coarse_ivf_selection"}
    if algorithm == "IVFPQ" and method == "CUVS":
        fixed_stages.add("query_transform")
    if algorithm == "IVFPQ" and method == "TensorVS":
        fixed_stages.update(("list_scheduling_preparation", "lut_computation"))

    attributed = {stage: 0.0 for stage in canonical_stages}
    for stage in fixed_stages:
        attributed[stage] = fused_values.get(stage, 0.0)
    fixed_raw_total = sum(attributed.values())
    if fixed_raw_total > target_total and fixed_raw_total > 0.0:
        fixed_scale = target_total / fixed_raw_total
        for stage in fixed_stages:
            attributed[stage] *= fixed_scale
        fixed_total = target_total
    else:
        fixed_total = fixed_raw_total
    remaining = max(0.0, target_total - fixed_total)
    weight_total = sum(
        detailed_values[stage] for stage in canonical_stages
        if stage not in fixed_stages
    )
    if weight_total > 0.0:
        scale = remaining / weight_total
        for stage in canonical_stages:
            if stage not in fixed_stages:
                attributed[stage] = detailed_values[stage] * scale
    else:
        # A profiling kernel may produce no usable cycle counters when the
        # batch is large enough for the counter reduction to be empty. Keep
        # the independent stages intact and conservatively assign the
        # unaccounted scan time to the distance phase rather than silently
        # dropping it from the total.
        scale = 0.0
        if "vector_read_distance" not in fixed_stages:
            attributed["vector_read_distance"] = remaining

    result = dict(fused)
    result["stages_ms_per_query"] = attributed
    result["total_ms_per_query"] = sum(attributed.values())
    result["normal_fused_total_ms_per_query"] = target_total
    result["normal_fused_profiled_stages_ms_per_query"] = float(
        fused["total_ms_per_query"]
    )
    result["normal_fused_profiled_stage_values_ms_per_query"] = dict(
        fused["stages_ms_per_query"]
    )
    result["instrumented_fused_end_to_end_ms_per_query"] = fused.get(
        "end_to_end_ms_per_query"
    )
    result["raw_detailed_total_ms_per_query"] = float(
        detailed["total_ms_per_query"]
    )
    result["raw_detailed_stages_ms_per_query"] = dict(
        detailed["stages_ms_per_query"]
    )
    result["raw_detailed_end_to_end_ms_per_query"] = detailed.get(
        "end_to_end_ms_per_query"
    )
    result["detailed_stage_notes"] = detailed.get("stage_notes")
    result["attribution_scale"] = scale
    result["timing_source"] = (
        "uninstrumented normal fused-path API total with normalized "
        "detailed-path attribution"
    )
    result["stage_notes"] = (
        "Bar length is the normal fused implementation. De-fused/multi-pass "
        "measurements provide stage proportions; directly exposed fused-path "
        "stages retain their measured duration. IVF-Flat keeps vector reads "
        "and distance arithmetic in one fused segment because their GPU "
        "execution overlaps. Bars use a "
        "separately measured, "
        "synchronized, instrumentation-disabled normal API duration so profiler "
        "overhead is excluded while launch/dispatch overhead is retained."
    )
    return result


def plot_breakdown(result: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ms_to_us = 1000.0
    algorithms = (("IVFFlat", FLAT_STAGES), ("IVFPQ", PQ_STAGES))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.65), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.77, wspace=0.16)

    legend_stages: set[str] = set()
    for axis, (algorithm, stages_order) in zip(axes, algorithms):
        panel_max_total = max(
            float(result["methods"][method][algorithm].get("total_ms_per_query", 0.0))
            * ms_to_us
            for method in PLOT_METHODS
        )
        max_total = 0.0
        for y_position, method in enumerate(PLOT_METHODS):
            method_result = result["methods"][method][algorithm]
            stage_values = method_result.get("stages_ms_per_query", {})
            total = (
                float(method_result.get("total_ms_per_query", 0.0)) * ms_to_us
            )
            max_total = max(max_total, total)
            left = 0.0
            has_real_stage_values = bool(stage_values)
            for stage in stages_order:
                value = float(stage_values.get(stage, 0.0)) * ms_to_us
                if value <= 0.0:
                    continue
                legend_stages.add(stage)
                axis.barh(
                    y_position, value, left=left, height=0.54,
                    color=STAGE_COLORS[stage], edgecolor="white", linewidth=0.65,
                )
                # Avoid printing a value wider than a narrow segment. The
                # color and legend still identify every such segment, while
                # the exact value remains available in the JSON output.
                if value >= max(total * 0.10, panel_max_total * 0.055, 0.001):
                    axis.text(left + value / 2.0, y_position, f"{value:.1f}",
                              ha="center", va="center", fontsize=7.2, color="white")
                left += value
            if not has_real_stage_values and total > 0.0:
                legend_stages.add("unsplit_library_search")
                axis.barh(
                    y_position, total, left=0.0, height=0.54,
                    color=STAGE_COLORS["unsplit_library_search"],
                    edgecolor="white", linewidth=0.65, hatch="//",
                )
                axis.text(total / 2.0, y_position, f"{total:.1f}",
                          ha="center", va="center", fontsize=7.2, color="black")

        axis.set_xlabel("Average query time (μs)", fontsize=9)
        axis.set_xlim(0.0, max_total * 1.16 if max_total > 0.0 else 1.0)
        axis.set_yticks(
            np.arange(len(PLOT_METHODS)),
            [METHOD_LABELS[method] for method in PLOT_METHODS],
        )
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].invert_yaxis()
    # Stable de-duplication is required because coarse selection and final
    # Top-K merge can appear in both algorithm panels.
    all_stages = tuple(
        dict.fromkeys((*FLAT_STAGES, *PQ_STAGES, "unsplit_library_search"))
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=STAGE_COLORS[stage],
                      hatch="//" if stage == "unsplit_library_search" else None)
        for stage in all_stages if stage in legend_stages
    ]
    labels = [STAGE_LABELS[stage] for stage in all_stages if stage in legend_stages]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.91),
               ncol=4, fontsize=7.1, frameon=False)
    axes[0].text(0.5, -0.28, "(a) IVF-Flat query-time breakdown",
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=9)
    axes[1].text(0.5, -0.28, "(b) IVF-PQ query-time breakdown",
                 transform=axes[1].transAxes, ha="center", va="top", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[item.name for item in DATASETS],
                        default="glove1.2m")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--nprobe", type=int, default=DEFAULT_NPROBE)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--query-batch", type=int, default=DEFAULT_QUERY_BATCH)
    parser.add_argument("--nq", type=int, default=None)
    parser.add_argument("--warmup-iterations", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument(
        "--timing-repetitions", type=int, default=DEFAULT_TIMING_REPETITIONS,
        help="Repeat each timed pass and average the measurements (default: 5)",
    )
    parser.add_argument(
        "--profile-mode", choices=("detailed", "fused"), default="detailed",
        help="Use official non-fused/multi-pass paths where available",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--external-stage-times", type=Path, default=None,
        help="JSON containing exact CUVS/FAISS stages; see module docstring",
    )
    parser.add_argument("--plot-only", action="store_true")
    # Internal worker options. They are intentionally hidden from the normal
    # command-line workflow; the parent launches one worker per external
    # backend in that backend's Conda environment.
    parser.add_argument("--backend-worker", choices=("cuvs", "faiss"), default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", choices=("IVFFlat", "IVFPQ"), default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--index-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--query-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--faiss-interleaved-layout", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--detailed-profile", action="store_true",
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend_worker is not None:
        required = (args.algorithm, args.index_path, args.query_path, args.worker_output, args.gpu_id)
        if any(value is None for value in required):
            raise ValueError("backend worker is missing a required option")
        run_external_worker(args)
        return

    import torch

    config = next(item for item in DATASETS if item.name == args.dataset)
    result_path = args.results_dir / f"query_time_breakdown_{args.dataset}.json"
    figure_path = args.figures_dir / f"{args.dataset}_query_time_breakdown.pdf"

    if args.plot_only:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        plot_breakdown(result, figure_path)
        print(f"Figure written to {figure_path}")
        return

    if (
        args.nprobe <= 0 or args.topk <= 0 or args.query_batch <= 0
        or args.timing_repetitions <= 0
    ):
        raise ValueError(
            "nprobe, topk, query-batch, and timing-repetitions must be positive"
        )
    device = torch.device(args.device)
    validate_device(device)
    gpu_id = 0 if device.index is None else int(device.index)
    query_path = first_existing(config.root, config.query_candidates)
    from tensorvs import load_dataset

    queries = load_dataset(query_path)
    if args.nq is not None:
        queries = queries[: min(int(args.nq), int(queries.shape[0]))]
    print(f"Dataset={config.name}, queries={tuple(queries.shape)}, nprobe={args.nprobe}, topk={args.topk}")

    result: dict[str, Any] = {
        "dataset": config.name,
        "query_path": str(query_path),
        "nprobe": int(args.nprobe),
        "topk": int(args.topk),
        "query_batch": int(args.query_batch),
        "timing_repetitions": int(args.timing_repetitions),
        "profile_mode": args.profile_mode,
        "methods": {method: {} for method in METHODS},
    }
    tensorvs_paths = {
        "IVFFlat": config.tensorvs_ivfflat_path,
        "IVFPQ": config.tensorvs_ivfpq_path,
    }
    for algorithm, index_path in tensorvs_paths.items():
        if not index_path.exists():
            raise FileNotFoundError(f"TensorVS index not found: {index_path}")
        tensorvs_profile = profile_tensorvs(
            method=algorithm, index_path=index_path, queries=queries, device=device,
            topk=args.topk, nprobe=args.nprobe, query_batch=args.query_batch,
            warmup_iterations=args.warmup_iterations,
            timing_repetitions=args.timing_repetitions, nq=args.nq,
            detailed_profile=args.profile_mode == "detailed",
        )
        if args.profile_mode == "detailed":
            fused_profile = profile_tensorvs(
                method=algorithm, index_path=index_path, queries=queries, device=device,
                topk=args.topk, nprobe=args.nprobe, query_batch=args.query_batch,
                warmup_iterations=args.warmup_iterations,
                timing_repetitions=args.timing_repetitions, nq=args.nq,
                detailed_profile=False,
            )
            tensorvs_profile = attribute_detailed_profile(
                tensorvs_profile, fused_profile, algorithm
            )
        result["methods"]["TensorVS"][algorithm] = tensorvs_profile

    for method in ("CUVS", "FAISS"):
        for algorithm in ("IVFFlat", "IVFPQ"):
            external_profile = profile_external(
                method=method, algorithm=algorithm, dataset=config,
                query_path=query_path, nprobe=args.nprobe, topk=args.topk,
                query_batch=args.query_batch, warmup_iterations=args.warmup_iterations,
                timing_repetitions=args.timing_repetitions,
                nq=args.nq, gpu_id=gpu_id,
                detailed_profile=args.profile_mode == "detailed",
            )
            if args.profile_mode == "detailed":
                fused_profile = profile_external(
                    method=method, algorithm=algorithm, dataset=config,
                    query_path=query_path, nprobe=args.nprobe, topk=args.topk,
                    query_batch=args.query_batch,
                    warmup_iterations=args.warmup_iterations,
                    timing_repetitions=args.timing_repetitions,
                    nq=args.nq, gpu_id=gpu_id, detailed_profile=False,
                )
                external_profile = attribute_detailed_profile(
                    external_profile, fused_profile, algorithm
                )
            result["methods"][method][algorithm] = external_profile
            merge_external_stage_times(
                result["methods"][method][algorithm], args.external_stage_times, algorithm
            )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_breakdown(result, figure_path)
    print(f"Timing data written to {result_path}")
    print(f"Figure written to {figure_path}")


if __name__ == "__main__":
    main()
