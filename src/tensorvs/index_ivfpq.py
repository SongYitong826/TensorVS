from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ._kmeans import fit_best_kmeans
from ._ivf_interleaved import (
    INTERLEAVED_GROUP_SIZE,
    build_interleaved_pq_lists,
    build_interleaved_pq_scalar_cache,
)
from ._ivfpq_triton import can_run as can_run_ivfpq_triton_topk
from ._ivfpq_triton import triton_topk as triton_ivfpq_topk
from .distance import squared_l2_distance
from .refine import refine_search_results


_LUT_STORAGE_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}

_LUT_STRATEGIES = {
    "decomposed_vector_correction",
}


def _normalize_lut_dtype(lut_dtype: str) -> str:
    """Validate the shared Tensor/Triton IVF-PQ LUT precision switch."""
    normalized = str(lut_dtype).lower()
    if normalized not in _LUT_STORAGE_DTYPES:
        raise ValueError(
            f"Unsupported LUT dtype {lut_dtype!r}; expected one of {tuple(_LUT_STORAGE_DTYPES)}"
        )
    return normalized


def _normalize_lut_strategy(lut_strategy: str) -> str:
    """Validate the residual-LUT construction/scanning strategy."""
    normalized = str(lut_strategy).lower()
    if normalized not in _LUT_STRATEGIES:
        raise ValueError(
            f"Unsupported LUT strategy {lut_strategy!r}; expected one of {tuple(sorted(_LUT_STRATEGIES))}"
        )
    return normalized


def _pq_code_dtype(nbits: int) -> torch.dtype:
    if nbits <= 8:
        return torch.uint8
    if nbits <= 16:
        return torch.int16
    return torch.int32


def _tensor_topk_pq_distance_impl(
    tensor_query_luts: torch.Tensor,
    list_codes: torch.Tensor,
    list_ids_flat: torch.Tensor,
    probe_list_prefix: torch.Tensor,
    probe_list_count: torch.Tensor,
    candidate_offsets: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the nearest PQ candidates per query without sorting all pairs."""
    safe_candidate_positions, probe_idx, valid_mask = _tensor_topk_pq_candidate_map_impl(
        list_codes,
        probe_list_prefix,
        probe_list_count,
        candidate_offsets,
    )
    candidate_codes = _tensor_topk_pq_code_gather_impl(list_codes, safe_candidate_positions)
    pair_distances = _tensor_topk_pq_lut_reduce_impl(
        tensor_query_luts,
        candidate_codes,
        probe_idx,
        valid_mask,
    )
    topk_distances, topk_positions = _tensor_topk_pq_select_impl(pair_distances, topk)
    topk_ids = _tensor_topk_pq_id_lookup_impl(
        list_ids_flat,
        safe_candidate_positions,
        valid_mask,
        topk_positions,
    )
    return topk_distances, topk_ids


def _tensor_topk_pq_candidate_map_impl(
    list_codes: torch.Tensor,
    probe_list_prefix: torch.Tensor,
    probe_list_count: torch.Tensor,
    candidate_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map padded query-local candidate offsets into flat PQ-code positions."""
    num_queries, nprobe = probe_list_count.shape
    query_candidate_count = probe_list_count.sum(dim=1)
    candidate_offsets_2d = candidate_offsets.unsqueeze(0).repeat(num_queries, 1)
    valid_mask = candidate_offsets_2d < query_candidate_count.unsqueeze(1)
    probe_end = torch.cumsum(probe_list_count, dim=1)
    probe_idx = torch.searchsorted(probe_end, candidate_offsets_2d, right=True)
    probe_idx = probe_idx.clamp_max(nprobe - 1)
    probe_start = probe_end - probe_list_count
    candidate_in_list = candidate_offsets_2d - torch.gather(probe_start, 1, probe_idx)
    candidate_positions = torch.gather(probe_list_prefix, 1, probe_idx) + candidate_in_list
    safe_candidate_positions = candidate_positions.clamp(0, list_codes.shape[0] - 1)
    return safe_candidate_positions, probe_idx, valid_mask


def _tensor_topk_pq_code_gather_impl(
    list_codes: torch.Tensor,
    safe_candidate_positions: torch.Tensor,
) -> torch.Tensor:
    """Gather every selected candidate's PQ codes as advanced-index inputs."""
    return list_codes[safe_candidate_positions].to(torch.int64)


def _tensor_topk_pq_lut_reduce_impl(
    tensor_query_luts: torch.Tensor,
    candidate_codes: torch.Tensor,
    probe_idx: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Look up and sum ADC values for every query-candidate pair."""
    num_queries = probe_idx.shape[0]
    nprobe = tensor_query_luts.shape[0] // num_queries
    max_candidates = probe_idx.shape[1]
    m = int(tensor_query_luts.shape[1])
    ksub = int(tensor_query_luts.shape[2])

    # Tensor search stores every (query, probe) LUT row contiguously as
    # [m, ksub]. Map (lut row, subspace, code) directly into its contiguous
    # linear offset so lookup uses one torch.take instead of 3-D advanced
    # indexing.
    lut_rows = (
        torch.arange(num_queries, device=tensor_query_luts.device, dtype=torch.int64).unsqueeze(1) * nprobe
        + probe_idx
    ).reshape(-1)
    subspace_idx = torch.arange(m, device=tensor_query_luts.device, dtype=torch.int64)
    lut_offsets = (
        (lut_rows.unsqueeze(1) * m + subspace_idx.unsqueeze(0)) * ksub
        + candidate_codes.reshape(-1, m)
    )
    # FP16 LUTs save lookup bandwidth, but retain FP32 accumulation so the
    # only intentional precision change is the stored LUT value itself.
    pair_distances = torch.take(tensor_query_luts.reshape(-1), lut_offsets).sum(
        dim=1,
        dtype=torch.float32,
    ).reshape(num_queries, max_candidates)
    return pair_distances.masked_fill(~valid_mask, float("inf"))


def _tensor_topk_pq_select_impl(
    pair_distances: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform only the per-query distance selection."""
    return torch.topk(pair_distances, k=topk, dim=1, largest=False)


def _tensor_topk_pq_id_lookup_impl(
    list_ids_flat: torch.Tensor,
    safe_candidate_positions: torch.Tensor,
    valid_mask: torch.Tensor,
    topk_positions: torch.Tensor,
) -> torch.Tensor:
    """Translate top-k candidate columns into vector IDs and invalid markers."""
    topk_vector_positions = torch.gather(safe_candidate_positions, 1, topk_positions)
    topk_ids = list_ids_flat[topk_vector_positions]
    topk_valid = torch.gather(valid_mask, 1, topk_positions)
    return topk_ids.masked_fill(~topk_valid, -1)


@lru_cache(maxsize=1)
def _get_tensor_topk_pq_distance_fn():
    try:
        # Keep this whole candidate-map/ADC/top-k path in one Dynamo graph.
        # Dynamic candidate widths avoid recompiling on each query batch.
        return torch.compile(_tensor_topk_pq_distance_impl, fullgraph=True, dynamic=True)
    except Exception:
        return _tensor_topk_pq_distance_impl


@lru_cache(maxsize=1)
def _get_tensor_topk_pq_candidate_map_fn():
    try:
        return torch.compile(_tensor_topk_pq_candidate_map_impl, fullgraph=True, dynamic=True)
    except Exception:
        return _tensor_topk_pq_candidate_map_impl


@lru_cache(maxsize=1)
def _get_tensor_topk_pq_code_gather_fn():
    try:
        return torch.compile(_tensor_topk_pq_code_gather_impl, fullgraph=True, dynamic=True)
    except Exception:
        return _tensor_topk_pq_code_gather_impl


@lru_cache(maxsize=1)
def _get_tensor_topk_pq_lut_reduce_fn():
    try:
        return torch.compile(_tensor_topk_pq_lut_reduce_impl, fullgraph=True, dynamic=True)
    except Exception:
        return _tensor_topk_pq_lut_reduce_impl


@lru_cache(maxsize=1)
def _get_tensor_topk_pq_select_fn():
    try:
        return torch.compile(_tensor_topk_pq_select_impl, fullgraph=True, dynamic=True)
    except Exception:
        return _tensor_topk_pq_select_impl


@lru_cache(maxsize=1)
def _get_tensor_topk_pq_id_lookup_fn():
    try:
        return torch.compile(_tensor_topk_pq_id_lookup_impl, fullgraph=True, dynamic=True)
    except Exception:
        return _tensor_topk_pq_id_lookup_impl


def _get_interleaved_pq_lists(
    index: IVFPQIndex,
    list_codes: torch.Tensor,
    list_ids_flat: torch.Tensor,
    list_prefix: torch.Tensor,
    list_count: torch.Tensor,
    search_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retrieve the persisted interleaved lists or build a legacy fallback."""
    has_persisted_layout = (
        index.persisted_list_codes_interleaved_by_subspace is not None
        and index.persisted_list_ids_interleaved is not None
        and index.persisted_list_group_prefix is not None
    )

    def persisted_layout() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert index.persisted_list_codes_interleaved_by_subspace is not None
        assert index.persisted_list_ids_interleaved is not None
        assert index.persisted_list_group_prefix is not None
        return (
            index.persisted_list_codes_interleaved_by_subspace.to(
                device=search_device,
                copy=False,
            ),
            index.persisted_list_ids_interleaved.to(
                dtype=torch.int64,
                device=search_device,
                copy=False,
            ),
            index.persisted_list_group_prefix.to(
                dtype=torch.int64,
                device=search_device,
                copy=False,
            ),
        )

    if index.resident_device is not None and index.resident_device == search_device:
        if (
            index.list_codes_interleaved_by_subspace is None
            or index.list_ids_interleaved is None
            or index.list_group_prefix is None
        ):
            if has_persisted_layout:
                (
                    index.list_codes_interleaved_by_subspace,
                    index.list_ids_interleaved,
                    index.list_group_prefix,
                ) = persisted_layout()
            else:
                layout = build_interleaved_pq_lists(list_codes, list_ids_flat, list_prefix, list_count)
                index.list_codes_interleaved_by_subspace = layout.codes_by_subspace
                index.list_ids_interleaved = layout.ids
                index.list_group_prefix = layout.group_prefix
        return index.list_codes_interleaved_by_subspace, index.list_ids_interleaved, index.list_group_prefix

    if has_persisted_layout:
        return persisted_layout()

    layout = build_interleaved_pq_lists(list_codes, list_ids_flat, list_prefix, list_count)
    return layout.codes_by_subspace, layout.ids, layout.group_prefix


def _get_interleaved_pq_static_corrections(
    index: IVFPQIndex,
    list_codes: torch.Tensor,
    list_prefix: torch.Tensor,
    list_count: torch.Tensor,
    search_device: torch.device,
) -> torch.Tensor:
    """Return the physical-order static centroid/PQ scalar cache for Triton."""
    if index.resident_device is not None and index.resident_device == search_device:
        if index.list_pq_static_corrections_interleaved is None:
            if index.persisted_list_pq_static_corrections_interleaved is not None:
                index.list_pq_static_corrections_interleaved = (
                    index.persisted_list_pq_static_corrections_interleaved.to(
                        dtype=torch.float32,
                        device=search_device,
                        copy=False,
                    )
                )
            else:
                index.list_pq_static_corrections_interleaved = (
                    _build_interleaved_pq_static_corrections(
                        list_codes,
                        list_prefix,
                        list_count,
                        index.centroids,
                        index.pq_codebooks,
                    )
                )
        return index.list_pq_static_corrections_interleaved

    if index.persisted_list_pq_static_corrections_interleaved is not None:
        return index.persisted_list_pq_static_corrections_interleaved.to(
            dtype=torch.float32,
            device=search_device,
            copy=False,
        )
    return _build_interleaved_pq_static_corrections(
        list_codes,
        list_prefix,
        list_count,
        index.centroids,
        index.pq_codebooks,
    )


def _build_query_luts_bmm(
    residual_q: torch.Tensor,
    pq_codebooks: torch.Tensor,
    pq_codebook_norms: torch.Tensor,
    *,
    num_queries: int,
    nprobe: int,
    m: int,
    subdim: int,
    use_tf32: bool = False,
) -> torch.Tensor:
    """Build residual-PQ lookup tables via batched GEMM instead of broadcast subtraction.

    ``use_tf32`` keeps FP32 LUT storage and FP32 ADC accumulation, while
    allowing CUDA Tensor Cores to accelerate the FP32 BMM inputs. The CUDA
    matmul setting is restored immediately after the launch so this search
    choice does not leak into unrelated caller operations.
    """
    q_sub_by_m = (
        residual_q.view(num_queries, nprobe, m, subdim)
        .permute(2, 0, 1, 3)
        .reshape(m, num_queries * nprobe, subdim)
        .contiguous()
    )
    query_norms = (q_sub_by_m * q_sub_by_m).sum(dim=-1, keepdim=True)
    query_luts = query_norms + pq_codebook_norms.unsqueeze(1)
    if not use_tf32 or not residual_q.is_cuda:
        return torch.baddbmm(
            query_luts,
            q_sub_by_m,
            pq_codebooks.transpose(1, 2),
            beta=1.0,
            alpha=-2.0,
        ).clamp_min_(0.0)

    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.baddbmm(
            query_luts,
            q_sub_by_m,
            pq_codebooks.transpose(1, 2),
            beta=1.0,
            alpha=-2.0,
        ).clamp_min_(0.0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32


def _build_interleaved_pq_static_corrections(
    list_codes: torch.Tensor,
    list_prefix: torch.Tensor,
    list_count: torch.Tensor,
    centroids: torch.Tensor,
    pq_codebooks: torch.Tensor,
) -> torch.Tensor:
    """Precompute one centroid/PQ correction scalar for every physical row.

    For a vector ``i`` in list ``l`` with codes ``z[i, m]``, this stores
    ``gamma[i] = sum_m 2 * <c[l, m], p[m, z[i, m]]>`` at the same padded
    interleaved position as its PQ codes.  Query-time ADC can therefore add
    one scalar per candidate instead of gathering ``R[l, m, code]`` for every
    subspace.
    """
    if list_codes.ndim != 2:
        raise ValueError("List PQ codes must have shape [num_vectors, m]")
    if list_prefix.ndim != 1 or list_count.ndim != 1 or list_prefix.shape != list_count.shape:
        raise ValueError("List prefix/count tensors must have matching [nlist] shapes")
    if centroids.ndim != 2 or pq_codebooks.ndim != 3:
        raise ValueError("Expected centroids [nlist, dim] and PQ codebooks [m, ksub, subdim]")
    num_vectors, m = list_codes.shape
    _, _, subdim = pq_codebooks.shape
    if (
        centroids.shape[0] != list_count.numel()
        or pq_codebooks.shape[0] != m
        or centroids.shape[1] != m * subdim
    ):
        raise ValueError("Centroids/codebooks do not match the IVF lists or PQ subspaces")
    if int(list_count.sum().item()) != num_vectors:
        raise ValueError("List counts do not cover all PQ code rows")

    values = torch.zeros(
        (num_vectors,),
        dtype=torch.float32,
        device=list_codes.device,
    )
    if num_vectors > 0:
        list_ids = torch.repeat_interleave(
            torch.arange(list_count.numel(), dtype=torch.int64, device=list_codes.device),
            list_count.to(device=list_codes.device, dtype=torch.int64),
        )
        code_values = list_codes.to(torch.int64)
        centroid_subvectors = centroids.to(
            device=list_codes.device,
            dtype=torch.float32,
        ).contiguous().view(-1, m, subdim)
        codebooks = pq_codebooks.to(device=list_codes.device, dtype=torch.float32).contiguous()
        for subspace_idx in range(m):
            codes_m = code_values[:, subspace_idx]
            values += 2.0 * (
                centroid_subvectors[list_ids, subspace_idx]
                * codebooks[subspace_idx, codes_m]
            ).sum(dim=1)

    return build_interleaved_pq_scalar_cache(
        values,
        list_prefix.to(device=list_codes.device, dtype=torch.int64),
        list_count.to(device=list_codes.device, dtype=torch.int64),
    )


def _build_global_query_luts_bmm(
    queries: torch.Tensor,
    pq_codebooks: torch.Tensor,
    pq_codebook_norms: torch.Tensor,
    *,
    m: int,
    subdim: int,
    use_tf32: bool,
) -> torch.Tensor:
    """Build the query-only ``||q_m - p_mk||^2`` table as ``[m, q, k]``."""
    return _build_query_luts_bmm(
        queries[:, None, :],
        pq_codebooks,
        pq_codebook_norms,
        num_queries=queries.shape[0],
        nprobe=1,
        m=m,
        subdim=subdim,
        use_tf32=use_tf32,
    )


@dataclass
class IVFPQIndex:
    """In-memory representation of IVF-PQ index."""

    index_dir: Path
    metadata: dict[str, Any]
    vectors: torch.Tensor | None
    assignments: torch.Tensor
    centroids: torch.Tensor
    pq_codebooks: torch.Tensor
    pq_codebook_norms: torch.Tensor
    ids: torch.Tensor
    list_codes: torch.Tensor | None
    list_ids_flat: torch.Tensor | None
    list_prefix: torch.Tensor | None
    list_count: torch.Tensor | None
    # Device-resident padded 32-vector code cache for the CUDA direct scanner.
    # Each subspace stream is contiguous across a warp group.
    list_codes_interleaved_by_subspace: torch.Tensor | None = None
    list_ids_interleaved: torch.Tensor | None = None
    list_group_prefix: torch.Tensor | None = None
    # One FP32 scalar per padded physical vector: sum_m R[list, m, code_m].
    # It lets the query-major Triton path avoid a per-subspace R-table gather.
    list_pq_static_corrections_interleaved: torch.Tensor | None = None
    # CPU copy loaded from a dual-layout index. ``reside_on_device`` keeps a
    # matching interleaved copy beside the row-major Tensor layout.
    persisted_list_codes_interleaved_by_subspace: torch.Tensor | None = None
    persisted_list_ids_interleaved: torch.Tensor | None = None
    persisted_list_group_prefix: torch.Tensor | None = None
    persisted_list_pq_static_corrections_interleaved: torch.Tensor | None = None
    # Reused by stable-batch Triton searches to avoid allocator traffic.
    triton_partial_keys: torch.Tensor | None = None
    resident_device: torch.device | None = None

    def reside_on_device(self, device: torch.device | str) -> IVFPQIndex:
        """Move search-critical tensors onto a target device and keep them resident."""
        if self.list_codes is None or self.list_ids_flat is None or self.list_prefix is None or self.list_count is None:
            raise ValueError("Index PQ lists are not loaded. Call load_ivfpq_index(..., load_lists=True).")

        target_device = torch.device(device)
        self.centroids = self.centroids.to(dtype=torch.float32, device=target_device, copy=False)
        self.pq_codebooks = self.pq_codebooks.to(dtype=torch.float32, device=target_device, copy=False)
        self.pq_codebook_norms = self.pq_codebook_norms.to(
            dtype=torch.float32,
            device=target_device,
            copy=False,
        )
        self.list_codes = self.list_codes.to(device=target_device, copy=False)
        self.list_ids_flat = self.list_ids_flat.to(dtype=torch.int64, device=target_device, copy=False)
        self.list_prefix = self.list_prefix.to(dtype=torch.int64, device=target_device, copy=False)
        self.list_count = self.list_count.to(dtype=torch.int64, device=target_device, copy=False)
        if (
            self.persisted_list_codes_interleaved_by_subspace is not None
            and self.persisted_list_ids_interleaved is not None
            and self.persisted_list_group_prefix is not None
        ):
            self.list_codes_interleaved_by_subspace = (
                self.persisted_list_codes_interleaved_by_subspace.to(
                    device=target_device,
                    copy=False,
                )
            )
            self.list_ids_interleaved = self.persisted_list_ids_interleaved.to(
                dtype=torch.int64,
                device=target_device,
                copy=False,
            )
            self.list_group_prefix = self.persisted_list_group_prefix.to(
                dtype=torch.int64,
                device=target_device,
                copy=False,
            )
            if self.persisted_list_pq_static_corrections_interleaved is not None:
                self.list_pq_static_corrections_interleaved = (
                    self.persisted_list_pq_static_corrections_interleaved.to(
                        dtype=torch.float32,
                        device=target_device,
                        copy=False,
                    )
                )
        else:
            # Legacy indexes do not have the persisted interleaved files, so
            # materialize an equivalent device-resident cache from row-major
            # lists during residency setup.
            layout = build_interleaved_pq_lists(
                self.list_codes,
                self.list_ids_flat,
                self.list_prefix,
                self.list_count,
            )
            self.list_codes_interleaved_by_subspace = layout.codes_by_subspace
            self.list_ids_interleaved = layout.ids
            self.list_group_prefix = layout.group_prefix
        if self.list_pq_static_corrections_interleaved is None:
            self.list_pq_static_corrections_interleaved = _build_interleaved_pq_static_corrections(
                self.list_codes,
                self.list_prefix,
                self.list_count,
                self.centroids,
                self.pq_codebooks,
            )
        self.triton_partial_keys = None
        self.resident_device = target_device
        return self

    def release_device_residency(self) -> IVFPQIndex:
        """Move resident search tensors back to CPU."""
        if self.list_codes is not None:
            self.centroids = self.centroids.to(dtype=torch.float32, device="cpu", copy=False)
            self.pq_codebooks = self.pq_codebooks.to(dtype=torch.float32, device="cpu", copy=False)
            self.pq_codebook_norms = self.pq_codebook_norms.to(dtype=torch.float32, device="cpu", copy=False)
            self.list_codes = self.list_codes.to(device="cpu", copy=False)
            self.list_ids_flat = self.list_ids_flat.to(dtype=torch.int64, device="cpu", copy=False)
            self.list_prefix = self.list_prefix.to(dtype=torch.int64, device="cpu", copy=False)
            self.list_count = self.list_count.to(dtype=torch.int64, device="cpu", copy=False)
        self.list_codes_interleaved_by_subspace = None
        self.list_ids_interleaved = None
        self.list_group_prefix = None
        self.list_pq_static_corrections_interleaved = None
        self.triton_partial_keys = None
        self.resident_device = None
        return self


@dataclass
class IVFPQBuildResult:
    """In-memory result of IVF-PQ training before serialization."""

    vectors: torch.Tensor
    labels: torch.Tensor
    centroids: torch.Tensor
    pq_codebooks: torch.Tensor
    pq_codes: torch.Tensor
    device: torch.device
    m: int
    nbits: int
    seed: int = 0
    n_init: int = 1
    coarse_inertia: float | None = None
    coarse_seed: int | None = None
    pq_inertias: list[float] = field(default_factory=list)
    pq_seeds: list[int] = field(default_factory=list)


def _encode_vectors_with_pq(
    vectors: torch.Tensor,
    pq_codebooks: torch.Tensor,
    m: int,
    *,
    code_dtype: torch.dtype,
) -> torch.Tensor:
    num_vectors, dim = vectors.shape
    subdim = dim // m
    vectors_view = vectors.view(num_vectors, m, subdim)
    pq_codes = torch.empty((num_vectors, m), dtype=code_dtype, device=vectors.device)

    for subspace_idx in range(m):
        sub_vectors = vectors_view[:, subspace_idx, :]
        sub_codebook = pq_codebooks[subspace_idx]
        distances = squared_l2_distance(sub_vectors, sub_codebook)
        pq_codes[:, subspace_idx] = torch.argmin(distances, dim=1).to(code_dtype)

    return pq_codes


def save_ivfpq_index(
    build_result: IVFPQBuildResult,
    output_dir: str | Path,
    dataset_path: str | Path | None = None,
    build_time_seconds: float | None = None,
) -> Path:
    """Persist row-major Tensor and interleaved Triton IVF-PQ list layouts."""
    output_path = Path(output_dir)

    vectors_cpu = build_result.vectors.cpu()
    labels_cpu = build_result.labels.cpu().to(torch.int64)
    centroids_cpu = build_result.centroids.cpu().to(torch.float32)
    pq_codebooks_cpu = build_result.pq_codebooks.cpu().to(torch.float32)
    pq_codebook_norms_cpu = (pq_codebooks_cpu * pq_codebooks_cpu).sum(dim=-1)
    pq_codes_cpu = build_result.pq_codes.cpu()
    ids = torch.arange(vectors_cpu.shape[0], dtype=torch.int64)

    if labels_cpu.numel() > 0:
        sort_perm = torch.argsort(labels_cpu, stable=True)
        labels_sorted = labels_cpu[sort_perm]
        list_codes = pq_codes_cpu[sort_perm]
        list_ids_flat = ids[sort_perm]
        list_count = torch.bincount(labels_sorted, minlength=centroids_cpu.shape[0]).to(torch.int64)
        list_prefix = torch.cumsum(list_count, dim=0) - list_count
    else:
        list_codes = torch.empty((0, build_result.m), dtype=pq_codes_cpu.dtype)
        list_ids_flat = torch.empty((0,), dtype=torch.int64)
        list_count = torch.zeros((centroids_cpu.shape[0],), dtype=torch.int64)
        list_prefix = torch.zeros((centroids_cpu.shape[0],), dtype=torch.int64)

    interleaved_layout = build_interleaved_pq_lists(
        list_codes,
        list_ids_flat,
        list_prefix,
        list_count,
    )
    interleaved_static_corrections = _build_interleaved_pq_static_corrections(
        list_codes,
        list_prefix,
        list_count,
        centroids_cpu,
        pq_codebooks_cpu,
    )

    cluster_sizes = {
        str(cluster_id): int(count)
        for cluster_id, count in enumerate(list_count.tolist())
    }

    is_single_file = output_path.suffix != ""
    metadata_output = str(output_path)
    metadata: dict[str, Any] = {
        "dataset_path": str(Path(dataset_path)) if dataset_path is not None else "in_memory_tensor",
        "output_dir": metadata_output,
        "num_vectors": int(vectors_cpu.shape[0]),
        "dim": int(vectors_cpu.shape[1]),
        "nlist": int(centroids_cpu.shape[0]),
        "m": int(build_result.m),
        "nbits": int(build_result.nbits),
        "ksub": int(1 << build_result.nbits),
        "subdim": int(vectors_cpu.shape[1] // build_result.m),
        "device": str(build_result.device),
        "seed": int(build_result.seed),
        "n_init": int(build_result.n_init),
        "coarse_inertia": build_result.coarse_inertia,
        "coarse_seed": build_result.coarse_seed,
        "pq_inertias": build_result.pq_inertias,
        "pq_seeds": build_result.pq_seeds,
        "pq_mode": "residual",
        "has_pq_static_vector_corrections": True,
        "stores_full_vectors": False,
        "cluster_sizes": cluster_sizes,
        "storage_format": "single_file" if is_single_file else "directory_flat",
        "interleaved_layout": {
            "available": True,
            "group_size": INTERLEAVED_GROUP_SIZE,
            "code_layout": "subspace_major",
        },
    }
    if build_time_seconds is not None:
        metadata["build_time_seconds"] = build_time_seconds

    if is_single_file:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata,
            "assignments": labels_cpu,
            "centroids": centroids_cpu,
            "pq_codebooks": pq_codebooks_cpu,
            "pq_codebook_norms": pq_codebook_norms_cpu,
            "pq_codes": pq_codes_cpu,
            "ids": ids,
            "list_codes": list_codes,
            "list_ids_flat": list_ids_flat,
            "list_prefix": list_prefix,
            "list_count": list_count,
            "list_codes_interleaved_by_subspace": interleaved_layout.codes_by_subspace,
            "list_ids_interleaved": interleaved_layout.ids,
            "list_group_prefix": interleaved_layout.group_prefix,
            "list_pq_static_corrections_interleaved": interleaved_static_corrections,
        }
        torch.save(payload, output_path)
        index_size_bytes = output_path.stat().st_size
        metadata["index_size_bytes"] = index_size_bytes
        metadata["index_size_mb"] = round(index_size_bytes / (1024 * 1024), 4)
        payload["metadata"] = metadata
        torch.save(payload, output_path)
        return output_path

    out_dir = output_path
    out_dir.mkdir(parents=True, exist_ok=True)
    # A previous index version persisted R[nlist,m,ksub]. The sole remaining
    # scheme stores only per-vector gamma, so reclaim this known obsolete file
    # when overwriting an existing directory index.
    obsolete_correction_path = out_dir / "pq_centroid_corrections.pt"
    if obsolete_correction_path.is_file():
        obsolete_correction_path.unlink()
    torch.save(labels_cpu, out_dir / "assignments.pt")
    torch.save(centroids_cpu, out_dir / "centroids.pt")
    torch.save(pq_codebooks_cpu, out_dir / "pq_codebooks.pt")
    torch.save(pq_codebook_norms_cpu, out_dir / "pq_codebook_norms.pt")
    torch.save(pq_codes_cpu, out_dir / "pq_codes.pt")
    torch.save(ids, out_dir / "ids.pt")
    torch.save(list_codes, out_dir / "list_codes.pt")
    torch.save(list_ids_flat, out_dir / "list_ids_flat.pt")
    torch.save(list_prefix, out_dir / "list_prefix.pt")
    torch.save(list_count, out_dir / "list_count.pt")
    torch.save(
        interleaved_layout.codes_by_subspace,
        out_dir / "list_codes_interleaved_by_subspace.pt",
    )
    torch.save(interleaved_layout.ids, out_dir / "list_ids_interleaved.pt")
    torch.save(interleaved_layout.group_prefix, out_dir / "list_group_prefix.pt")
    torch.save(
        interleaved_static_corrections,
        out_dir / "list_pq_static_corrections_interleaved.pt",
    )

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    index_size_bytes = sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())
    metadata["index_size_bytes"] = index_size_bytes
    metadata["index_size_mb"] = round(index_size_bytes / (1024 * 1024), 4)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def load_ivfpq_index(index_dir: str | Path, load_lists: bool = True) -> IVFPQIndex:
    """Load a saved IVF-PQ index directory or single-file archive."""
    index_path = Path(index_dir)
    if not index_path.exists():
        raise FileNotFoundError(f"Index directory not found: {index_path}")

    if index_path.is_file():
        payload = torch.load(index_path, map_location="cpu")
        metadata = payload["metadata"]
        assignments = payload["assignments"]
        centroids = payload["centroids"].to(torch.float32)
        pq_codebooks = payload["pq_codebooks"].to(torch.float32)
        pq_codebook_norms = payload.get("pq_codebook_norms")
        if pq_codebook_norms is None:
            pq_codebook_norms = (pq_codebooks * pq_codebooks).sum(dim=-1)
        else:
            pq_codebook_norms = pq_codebook_norms.to(torch.float32)
        ids = payload["ids"]
        if load_lists:
            list_codes = payload["list_codes"]
            list_ids_flat = payload["list_ids_flat"].to(torch.int64)
            list_prefix = payload["list_prefix"].to(torch.int64)
            list_count = payload["list_count"].to(torch.int64)
            persisted_list_codes_interleaved_by_subspace = payload.get(
                "list_codes_interleaved_by_subspace"
            )
            persisted_list_ids_interleaved = payload.get("list_ids_interleaved")
            persisted_list_group_prefix = payload.get("list_group_prefix")
            persisted_list_pq_static_corrections_interleaved = payload.get(
                "list_pq_static_corrections_interleaved"
            )
            if persisted_list_pq_static_corrections_interleaved is None:
                persisted_list_pq_static_corrections_interleaved = (
                    _build_interleaved_pq_static_corrections(
                        list_codes,
                        list_prefix,
                        list_count,
                        centroids,
                        pq_codebooks,
                    )
                )
            else:
                persisted_list_pq_static_corrections_interleaved = (
                    persisted_list_pq_static_corrections_interleaved.to(torch.float32)
                )
            if (
                persisted_list_codes_interleaved_by_subspace is not None
                and persisted_list_ids_interleaved is not None
                and persisted_list_group_prefix is not None
            ):
                persisted_list_ids_interleaved = persisted_list_ids_interleaved.to(torch.int64)
                persisted_list_group_prefix = persisted_list_group_prefix.to(torch.int64)
            else:
                persisted_list_codes_interleaved_by_subspace = None
                persisted_list_ids_interleaved = None
                persisted_list_group_prefix = None
        else:
            list_codes = None
            list_ids_flat = None
            list_prefix = None
            list_count = None
            persisted_list_codes_interleaved_by_subspace = None
            persisted_list_ids_interleaved = None
            persisted_list_group_prefix = None
            persisted_list_pq_static_corrections_interleaved = None

        return IVFPQIndex(
            index_dir=index_path,
            metadata=metadata,
            vectors=None,
            assignments=assignments,
            centroids=centroids,
            pq_codebooks=pq_codebooks,
            pq_codebook_norms=pq_codebook_norms,
            ids=ids,
            list_codes=list_codes,
            list_ids_flat=list_ids_flat,
            list_prefix=list_prefix,
            list_count=list_count,
            persisted_list_codes_interleaved_by_subspace=persisted_list_codes_interleaved_by_subspace,
            persisted_list_ids_interleaved=persisted_list_ids_interleaved,
            persisted_list_group_prefix=persisted_list_group_prefix,
            persisted_list_pq_static_corrections_interleaved=(
                persisted_list_pq_static_corrections_interleaved
            ),
        )

    metadata_path = index_path / "metadata.json"
    assignments_path = index_path / "assignments.pt"
    centroids_path = index_path / "centroids.pt"
    pq_codebooks_path = index_path / "pq_codebooks.pt"
    pq_codebook_norms_path = index_path / "pq_codebook_norms.pt"
    ids_path = index_path / "ids.pt"
    list_codes_path = index_path / "list_codes.pt"
    list_ids_flat_path = index_path / "list_ids_flat.pt"
    list_prefix_path = index_path / "list_prefix.pt"
    list_count_path = index_path / "list_count.pt"
    list_codes_interleaved_by_subspace_path = index_path / "list_codes_interleaved_by_subspace.pt"
    list_ids_interleaved_path = index_path / "list_ids_interleaved.pt"
    list_group_prefix_path = index_path / "list_group_prefix.pt"
    list_pq_static_corrections_interleaved_path = (
        index_path / "list_pq_static_corrections_interleaved.pt"
    )

    required_files = [
        metadata_path,
        assignments_path,
        centroids_path,
        pq_codebooks_path,
        ids_path,
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing IVF-PQ index files: {missing_files}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assignments = torch.load(assignments_path, map_location="cpu")
    centroids = torch.load(centroids_path, map_location="cpu").to(torch.float32)
    pq_codebooks = torch.load(pq_codebooks_path, map_location="cpu").to(torch.float32)
    if pq_codebook_norms_path.exists():
        pq_codebook_norms = torch.load(pq_codebook_norms_path, map_location="cpu").to(torch.float32)
    else:
        # Indexes created before codebook norms were persisted remain usable.
        pq_codebook_norms = (pq_codebooks * pq_codebooks).sum(dim=-1)
    ids = torch.load(ids_path, map_location="cpu")

    if load_lists:
        list_files = [
            list_codes_path,
            list_ids_flat_path,
            list_prefix_path,
            list_count_path,
        ]
        missing_list_files = [str(path) for path in list_files if not path.exists()]
        if missing_list_files:
            raise FileNotFoundError(f"Missing IVF-PQ list files: {missing_list_files}")
        list_codes = torch.load(list_codes_path, map_location="cpu")
        list_ids_flat = torch.load(list_ids_flat_path, map_location="cpu").to(torch.int64)
        list_prefix = torch.load(list_prefix_path, map_location="cpu").to(torch.int64)
        list_count = torch.load(list_count_path, map_location="cpu").to(torch.int64)
        interleaved_layout_paths = [
            list_codes_interleaved_by_subspace_path,
            list_ids_interleaved_path,
            list_group_prefix_path,
        ]
        if all(path.exists() for path in interleaved_layout_paths):
            persisted_list_codes_interleaved_by_subspace = torch.load(
                list_codes_interleaved_by_subspace_path,
                map_location="cpu",
            )
            persisted_list_ids_interleaved = torch.load(
                list_ids_interleaved_path,
                map_location="cpu",
            ).to(torch.int64)
            persisted_list_group_prefix = torch.load(
                list_group_prefix_path,
                map_location="cpu",
            ).to(torch.int64)
        else:
            persisted_list_codes_interleaved_by_subspace = None
            persisted_list_ids_interleaved = None
            persisted_list_group_prefix = None
        if list_pq_static_corrections_interleaved_path.exists():
            persisted_list_pq_static_corrections_interleaved = torch.load(
                list_pq_static_corrections_interleaved_path,
                map_location="cpu",
            ).to(torch.float32)
        else:
            persisted_list_pq_static_corrections_interleaved = (
                _build_interleaved_pq_static_corrections(
                    list_codes,
                    list_prefix,
                    list_count,
                    centroids,
                    pq_codebooks,
                )
            )
    else:
        list_codes = None
        list_ids_flat = None
        list_prefix = None
        list_count = None
        persisted_list_codes_interleaved_by_subspace = None
        persisted_list_ids_interleaved = None
        persisted_list_group_prefix = None
        persisted_list_pq_static_corrections_interleaved = None

    return IVFPQIndex(
        index_dir=index_path,
        metadata=metadata,
        vectors=None,
        assignments=assignments,
        centroids=centroids,
        pq_codebooks=pq_codebooks,
        pq_codebook_norms=pq_codebook_norms,
        ids=ids,
        list_codes=list_codes,
        list_ids_flat=list_ids_flat,
        list_prefix=list_prefix,
        list_count=list_count,
        persisted_list_codes_interleaved_by_subspace=persisted_list_codes_interleaved_by_subspace,
        persisted_list_ids_interleaved=persisted_list_ids_interleaved,
        persisted_list_group_prefix=persisted_list_group_prefix,
        persisted_list_pq_static_corrections_interleaved=(
            persisted_list_pq_static_corrections_interleaved
        ),
    )


def search_ivfpq_index(
    index: IVFPQIndex,
    queries: torch.Tensor,
    topk: int = 10,
    nprobe: int = 10,
    device: torch.device | str | None = None,
    selection_mode: str = "tensor_topk",
    profile: dict[str, float] | None = None,
    refine_vectors: torch.Tensor | None = None,
    refine_factor: int = 1,
    detailed_tensor_topk_profile: bool = False,
    lut_dtype: str = "fp32",
    lut_strategy: str = "decomposed_vector_correction",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search IVF-PQ, optionally reranking an expanded candidate set exactly.

    ``tensor_topk`` is the portable Tensor-native path. ``triton_topk`` uses
    a CUDA list-tiled, 32-way-interleaved ADC/top-k kernel when supported and
    otherwise falls back to ``tensor_topk``.

    Set ``detailed_tensor_topk_profile=True`` together with ``profile`` to
    time the Tensor path's candidate map, code gather, LUT reduction, top-k,
    ID lookup, and output writeback separately. This diagnostic path splits a
    normally fused graph, so use its individual timings for attribution rather
    than comparing its aggregate time to normal search throughput.

    The only Triton LUT strategy is ``"decomposed_vector_correction"``.
    It builds query-global ``G`` in FP32 with TF32 Tensor Cores, then stores
    it as FP32 or FP16 according to ``lut_dtype``. The direct scan always
    converts LUT values to FP32 before ADC accumulation and adds the persisted
    FP32 scalar correction for each physical vector. It performs no list
    bucketing and has no ``R[nlist,m,ksub]`` table.

    """

    search_device: torch.device | None = None
    cuda_timing_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def start_timing() -> float | torch.cuda.Event:
        if profile is None:
            return 0.0
        if search_device is not None and search_device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(search_device))
            return start_event
        return time.perf_counter()

    def record_timing(name: str, start_time: float | torch.cuda.Event) -> None:
        if profile is None:
            return
        if search_device is not None and search_device.type == "cuda":
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream(search_device))
            cuda_timing_events.append((name, start_time, end_event))
            return
        profile[name] = profile.get(name, 0.0) + (time.perf_counter() - start_time)

    def flush_timing() -> None:
        if profile is None or not cuda_timing_events:
            return
        assert search_device is not None
        torch.cuda.synchronize(search_device)
        for name, start_event, end_event in cuda_timing_events:
            profile[name] = profile.get(name, 0.0) + (start_event.elapsed_time(end_event) / 1000.0)

    if queries.ndim != 2:
        raise ValueError(f"Query tensor must have shape (num_queries, dim), got {tuple(queries.shape)}")
    if queries.shape[1] != index.centroids.shape[1]:
        raise ValueError(
            f"Query dimension {queries.shape[1]} does not match index dimension {index.centroids.shape[1]}"
        )
    pq_mode = str(index.metadata.get("pq_mode", "global"))
    if pq_mode != "residual":
        raise ValueError(
            "This IVF-PQ index uses the legacy global-PQ layout. Rebuild the index with the current "
            "residual-PQ implementation before searching."
        )
    if index.list_codes is None or index.list_ids_flat is None or index.list_prefix is None or index.list_count is None:
        raise ValueError("Index PQ lists are not loaded. Call load_ivfpq_index(..., load_lists=True).")
    if selection_mode not in {"triton_topk", "tensor_topk"}:
        raise ValueError(f"Unsupported IVFPQ selection_mode: {selection_mode}")
    lut_dtype = _normalize_lut_dtype(lut_dtype)
    lut_strategy = _normalize_lut_strategy(lut_strategy)

    if device is None:
        if queries.is_cuda:
            search_device = queries.device
        elif torch.cuda.is_available():
            search_device = torch.device("cuda:0")
        else:
            search_device = torch.device("cpu")
    else:
        search_device = torch.device(device)

    step_start = start_timing()
    q = queries.to(dtype=torch.float32, device=search_device, copy=False).contiguous()
    if index.resident_device is not None and index.resident_device == search_device:
        centroids = index.centroids
        pq_codebooks = index.pq_codebooks
        pq_codebook_norms = index.pq_codebook_norms
        list_codes = index.list_codes
        list_ids_flat = index.list_ids_flat
        list_prefix = index.list_prefix
        list_count = index.list_count
    else:
        centroids = index.centroids.to(dtype=torch.float32, device=search_device, copy=False)
        pq_codebooks = index.pq_codebooks.to(dtype=torch.float32, device=search_device, copy=False)
        pq_codebook_norms = index.pq_codebook_norms.to(
            dtype=torch.float32,
            device=search_device,
            copy=False,
        )
        list_codes = index.list_codes.to(device=search_device, copy=False)
        list_ids_flat = index.list_ids_flat.to(dtype=torch.int64, device=search_device, copy=False)
        list_prefix = index.list_prefix.to(dtype=torch.int64, device=search_device, copy=False)
        list_count = index.list_count.to(dtype=torch.int64, device=search_device, copy=False)
    record_timing("prepare_tensors", step_start)

    topk = max(1, int(topk))
    refine_factor = int(refine_factor)
    if refine_factor < 1:
        raise ValueError(f"refine_factor must be >= 1, got {refine_factor}")
    if refine_vectors is None and refine_factor != 1:
        raise ValueError("refine_factor requires refine_vectors with the raw dataset")
    search_topk = topk * refine_factor if refine_vectors is not None else topk
    search_topk = min(search_topk, int(index.metadata.get("num_vectors", search_topk)))
    if search_topk < topk:
        search_topk = topk
    nprobe = max(1, min(int(nprobe), centroids.shape[0]))
    num_queries = q.shape[0]

    step_start = start_timing()
    result_distances = torch.full(
        (num_queries, search_topk),
        float("inf"),
        dtype=torch.float32,
        device=search_device,
    )
    result_ids = torch.full(
        (num_queries, search_topk),
        -1,
        dtype=torch.int64,
        device=search_device,
    )
    record_timing("init_outputs", step_start)

    # Step 1: compute the full query-centroid squared L2 distance matrix, then
    # keep only the nearest `nprobe` coarse centroids for each query.
    step_start = start_timing()
    dist_qc = squared_l2_distance(q, centroids)
    record_timing("centroid_distance", step_start)
    step_start = start_timing()
    probe_distances, probe_cids = torch.topk(dist_qc, k=nprobe, dim=1, largest=False)
    record_timing("centroid_topk", step_start)

    m = int(index.metadata["m"])
    subdim = int(index.metadata["subdim"])
    triton_topk_supported = selection_mode == "triton_topk" and can_run_ivfpq_triton_topk(
        device=search_device,
        topk=search_topk,
        num_queries=num_queries,
        m=m,
        ksub=int(index.metadata["ksub"]),
        nprobe=nprobe,
        num_code_vectors=list_ids_flat.numel(),
        code_dtype=list_codes.dtype,
    )
    query_luts: torch.Tensor | None = None
    global_query_luts: torch.Tensor | None = None
    probe_distance_bias: torch.Tensor | None = None
    step_start = start_timing()
    if triton_topk_supported:
        # Build G in FP32 (with TF32 Tensor Core BMM on CUDA), then optionally
        # compact only its storage. The scan always accumulates in FP32.
        global_query_luts = _build_global_query_luts_bmm(
            q,
            pq_codebooks,
            pq_codebook_norms,
            m=m,
            subdim=subdim,
            use_tf32=True,
        )
        if lut_dtype == "fp16":
            global_query_luts = global_query_luts.to(torch.float16)
        probe_distance_bias = probe_distances - (q * q).sum(dim=1, keepdim=True)
    else:
        # Portable Tensor fallback retains its own query/probe-major LUT.
        residual_q = (q[:, None, :] - centroids[probe_cids]).contiguous()
        query_luts = _build_query_luts_bmm(
            residual_q,
            pq_codebooks,
            pq_codebook_norms,
            num_queries=num_queries,
            nprobe=nprobe,
            m=m,
            subdim=subdim,
            use_tf32=lut_dtype == "fp32",
        )
        if lut_dtype != "fp32":
            query_luts = query_luts.to(_LUT_STORAGE_DTYPES[lut_dtype])
    record_timing("LUT_prepare", step_start)

    active_selection_mode = selection_mode
    if triton_topk_supported:
        step_start = start_timing()
        interleaved_codes, interleaved_ids, list_group_prefix = _get_interleaved_pq_lists(
            index,
            list_codes,
            list_ids_flat,
            list_prefix,
            list_count,
            search_device,
        )
        interleaved_static_corrections = _get_interleaved_pq_static_corrections(
            index,
            list_codes,
            list_prefix,
            list_count,
            search_device,
        )
        record_timing("triton_prepare_layout", step_start)

        step_start = start_timing()
        probe_group_prefix = list_group_prefix[probe_cids]
        probe_list_count = list_count[probe_cids]
        record_timing("lookup_probe_lists", step_start)

        direct_scan_start = start_timing()
        final_merge_start: float | torch.cuda.Event | None = None

        def record_direct_scan_complete() -> None:
            nonlocal final_merge_start
            record_timing("triton_direct_scan", direct_scan_start)
            final_merge_start = start_timing()

        assert global_query_luts is not None
        assert probe_distance_bias is not None
        partial_keys = triton_ivfpq_topk(
            global_query_luts=global_query_luts,
            static_vector_corrections=interleaved_static_corrections,
            probe_distance_bias=probe_distance_bias,
            list_codes_interleaved=interleaved_codes,
            list_ids_interleaved=interleaved_ids,
            probe_group_prefix=probe_group_prefix,
            probe_list_count=probe_list_count,
            topk=search_topk,
            output_distances=result_distances,
            output_ids=result_ids,
            partial_keys=index.triton_partial_keys,
            on_direct_scan_complete=record_direct_scan_complete,
        )
        if index.resident_device is not None and index.resident_device == search_device:
            index.triton_partial_keys = partial_keys
        assert final_merge_start is not None
        record_timing("triton_final_merge", final_merge_start)
        active_selection_mode = "done"
    elif selection_mode == "triton_topk":
        # Preserve the requested search semantics on CPU, for non-uint8
        # codes, and for kernel shapes outside the fixed Triton limits.
        active_selection_mode = "tensor_topk"

    if active_selection_mode == "tensor_topk":
        # The portable path keeps candidates query-major and executes one
        # batched torch.topk.  Its host reduction is intentionally confined to
        # this fallback path; the Triton path has no per-batch .item() sync.
        step_start = start_timing()
        # Triton consumes [m, query * probe, ksub], while Tensor ADC benefits
        # from keeping each query/probe's complete [m, ksub] table contiguous.
        assert query_luts is not None
        tensor_query_luts = query_luts.permute(1, 0, 2).contiguous()
        del query_luts
        record_timing("tensor_LUT_layout", step_start)

        step_start = start_timing()
        probe_list_prefix = list_prefix[probe_cids]
        probe_list_count = list_count[probe_cids]
        record_timing("lookup_probe_lists", step_start)

        step_start = start_timing()
        max_candidates = int(probe_list_count.sum(dim=1).max().item())
        record_timing("max_candidates", step_start)
        if max_candidates > 0:
            selected_topk = min(search_topk, max_candidates)
            step_start = start_timing()
            candidate_offsets = torch.arange(max_candidates, device=search_device, dtype=torch.int64)
            record_timing("expand_query_candidates", step_start)

            step_start = start_timing()
            if detailed_tensor_topk_profile and profile is not None:
                stage_start = start_timing()
                candidate_map = _get_tensor_topk_pq_candidate_map_fn()
                safe_candidate_positions, probe_idx, valid_mask = candidate_map(
                    list_codes,
                    probe_list_prefix,
                    probe_list_count,
                    candidate_offsets,
                )
                record_timing("tensor_topk_candidate_map", stage_start)

                stage_start = start_timing()
                code_gather = _get_tensor_topk_pq_code_gather_fn()
                candidate_codes = code_gather(list_codes, safe_candidate_positions)
                record_timing("tensor_topk_code_gather", stage_start)

                stage_start = start_timing()
                lut_reduce = _get_tensor_topk_pq_lut_reduce_fn()
                pair_distances = lut_reduce(tensor_query_luts, candidate_codes, probe_idx, valid_mask)
                record_timing("tensor_topk_lut_reduce", stage_start)

                stage_start = start_timing()
                tensor_select = _get_tensor_topk_pq_select_fn()
                topk_distances, topk_positions = tensor_select(pair_distances, selected_topk)
                record_timing("tensor_topk_select_only", stage_start)

                stage_start = start_timing()
                id_lookup = _get_tensor_topk_pq_id_lookup_fn()
                topk_ids = id_lookup(
                    list_ids_flat,
                    safe_candidate_positions,
                    valid_mask,
                    topk_positions,
                )
                record_timing("tensor_topk_id_lookup", stage_start)
            else:
                tensor_topk = _get_tensor_topk_pq_distance_fn()
                topk_distances, topk_ids = tensor_topk(
                    tensor_query_luts,
                    list_codes,
                    list_ids_flat,
                    probe_list_prefix,
                    probe_list_count,
                    candidate_offsets,
                    selected_topk,
                )

            stage_start = start_timing()
            result_distances[:, :selected_topk] = topk_distances
            result_ids[:, :selected_topk] = topk_ids
            record_timing("tensor_topk_writeback", stage_start)
            record_timing("tensor_topk_select", step_start)

    if refine_vectors is not None:
        step_start = start_timing()
        refined = refine_search_results(
            q,
            result_ids,
            refine_vectors,
            topk,
            device=search_device,
        )
        record_timing("exact_refine", step_start)
        flush_timing()
        return refined
    flush_timing()
    return result_distances, result_ids


def build_ivfpq_index(
    vectors: torch.Tensor,
    nlist: int,
    niter: int,
    use_triton: bool,
    device: torch.device | str,
    m: int,
    nbits: int,
    seed: int = 0,
    n_init: int = 2,
) -> IVFPQBuildResult:
    """Cluster a dataset with IVF and train residual PQ codebooks."""
    if vectors.ndim != 2:
        raise ValueError(f"Dataset tensor must have shape (num_vectors, dim), got {tuple(vectors.shape)}")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if nbits <= 0:
        raise ValueError(f"nbits must be positive, got {nbits}")

    vectors = vectors.to(dtype=torch.float32, copy=False).contiguous()
    num_vectors, dim = vectors.shape
    if dim % m != 0:
        raise ValueError(f"Vector dimension {dim} must be divisible by m={m}")

    ksub = 1 << int(nbits)
    if num_vectors < ksub:
        raise ValueError(f"Dataset size {num_vectors} must be >= 2**nbits ({ksub}) for PQ training")

    target_device = torch.device(device)
    coarse = fit_best_kmeans(
        vectors,
        d=dim,
        k=nlist,
        niter=niter,
        use_triton=use_triton,
        device=target_device,
        seed=seed,
        n_init=n_init,
        stage=0,
    )
    centroids = coarse.centroids
    labels = coarse.labels

    vectors_device = vectors.to(dtype=torch.float32, device=target_device, copy=False)
    labels_device = labels.to(dtype=torch.int64, device=target_device, copy=False)
    residual_vectors = (vectors_device - centroids[labels_device]).contiguous()
    subdim = dim // m

    pq_code_dtype = _pq_code_dtype(nbits)
    pq_codebooks = torch.empty((m, ksub, subdim), dtype=torch.float32, device=target_device)
    pq_codes = torch.empty((num_vectors, m), dtype=pq_code_dtype, device=target_device)
    pq_inertias: list[float] = []
    pq_seeds: list[int] = []
    residual_view = residual_vectors.view(num_vectors, m, subdim)
    for subspace_idx in range(m):
        trained_subspace = fit_best_kmeans(
            residual_view[:, subspace_idx, :],
            d=subdim,
            k=ksub,
            niter=niter,
            use_triton=use_triton,
            device=target_device,
            seed=seed,
            n_init=n_init,
            stage=subspace_idx + 1,
        )
        pq_codebooks[subspace_idx] = trained_subspace.centroids
        pq_codes[:, subspace_idx] = trained_subspace.labels.to(pq_code_dtype)
        pq_inertias.append(trained_subspace.inertia)
        pq_seeds.append(trained_subspace.seed)

    return IVFPQBuildResult(
        vectors=vectors,
        labels=labels,
        centroids=centroids,
        pq_codebooks=pq_codebooks,
        pq_codes=pq_codes,
        device=target_device,
        m=m,
        nbits=nbits,
        seed=int(seed),
        n_init=int(n_init),
        coarse_inertia=coarse.inertia,
        coarse_seed=coarse.seed,
        pq_inertias=pq_inertias,
        pq_seeds=pq_seeds,
    )
