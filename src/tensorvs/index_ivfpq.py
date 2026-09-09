from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from flash_kmeans import (
    FAISS_DEFAULT_MAX_POINTS_PER_CENTROID,
    FAISS_DEFAULT_MIN_POINTS_PER_CENTROID,
    FAISS_DEFAULT_NITER,
    FAISS_DEFAULT_NREDO,
    FAISS_DEFAULT_SEED,
    fit_best_kmeans,
    select_training_indices,
)
from ._ivf_interleaved import (
    INTERLEAVED_GROUP_SIZE,
    build_interleaved_pq_lists,
    build_interleaved_pq_scalar_cache,
)
from ._ivfpq_triton import (
    can_run_topk as can_run_ivfpq_triton_topk,
    plan_probe_chunks,
)
from ._ivfpq_triton import triton_topk as triton_ivfpq_topk
from .distance import squared_l2_distance
from .refine import refine_search_results


_LUT_STORAGE_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}

def _normalize_lut_dtype(lut_dtype: str) -> str:
    """Validate the shared Tensor/Triton IVF-PQ LUT precision switch."""
    normalized = str(lut_dtype).lower()
    if normalized not in _LUT_STORAGE_DTYPES:
        raise ValueError(
            f"Unsupported LUT dtype {lut_dtype!r}; expected one of {tuple(_LUT_STORAGE_DTYPES)}"
        )
    return normalized


def _pq_code_dtype(nbits: int) -> torch.dtype:
    if nbits <= 8:
        return torch.uint8
    if nbits <= 16:
        return torch.int16
    return torch.int32


def _pq_working_dim(dim: int, m: int) -> int:
    """Return the cuVS-style PQ working dimension for ``m`` subspaces."""
    if dim <= 0 or m <= 0:
        raise ValueError(f"dim and m must be positive, got dim={dim}, m={m}")
    return ((dim + m - 1) // m) * m


def _make_pq_rotation_matrix(
    dim: int,
    working_dim: int,
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor | None:
    """Make the tall orthogonal PQ transform used when ``dim % m != 0``.

    cuVS maps a non-divisible input dimension into the next dimension that is
    divisible by ``pq_dim`` with a random orthogonal transform.  A tall matrix
    with orthonormal columns preserves all L2 distances while making the
    padded working space suitable for equal-width PQ subspaces.  A divisible
    shape needs no transform and returns ``None`` to preserve the existing
    fast path and on-disk format.
    """
    if working_dim < dim:
        raise ValueError(
            f"PQ working dimension must be >= input dimension, got {working_dim} < {dim}"
        )
    if working_dim == dim:
        return None

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        (working_dim, dim),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    q, r = torch.linalg.qr(matrix, mode="reduced")
    # Canonicalize QR column signs so a fixed seed produces a stable transform.
    signs = torch.where(torch.diagonal(r) < 0.0, -1.0, 1.0).to(q.dtype)
    return (q * signs.unsqueeze(0)).contiguous()


def _apply_pq_rotation(
    vectors: torch.Tensor,
    rotation_matrix: torch.Tensor | None,
) -> torch.Tensor:
    """Map original-space vectors into the PQ working space when required."""
    if rotation_matrix is None:
        return vectors
    if vectors.shape[1] != rotation_matrix.shape[1]:
        raise ValueError(
            "PQ rotation/input dimensions do not match: "
            f"{vectors.shape[1]} != {rotation_matrix.shape[1]}"
        )
    return (vectors @ rotation_matrix.transpose(0, 1)).contiguous()


def _restore_pq_transform(
    metadata: dict[str, Any],
    centroids: torch.Tensor,
    pq_codebooks: torch.Tensor,
    rotation_matrix: torch.Tensor | None,
    pq_centroids: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Validate and restore optional PQ-space transform tensors from an index."""
    working_dim = int(pq_codebooks.shape[0] * pq_codebooks.shape[2])
    recorded_working_dim = int(metadata.get("pq_working_dim", working_dim))
    if recorded_working_dim != working_dim:
        raise ValueError(
            "IVF-PQ metadata/codebook working dimensions disagree: "
            f"{recorded_working_dim} != {working_dim}"
        )
    if rotation_matrix is None:
        if working_dim != centroids.shape[1]:
            raise ValueError(
                "This IVF-PQ index has a padded PQ working space but is missing "
                "pq_rotation_matrix. Rebuild the index."
            )
        return None, None

    rotation_matrix = rotation_matrix.to(torch.float32)
    expected_rotation_shape = (working_dim, centroids.shape[1])
    if tuple(rotation_matrix.shape) != expected_rotation_shape:
        raise ValueError(
            "Invalid IVF-PQ rotation shape: "
            f"expected {expected_rotation_shape}, got {tuple(rotation_matrix.shape)}"
        )
    if pq_centroids is None:
        pq_centroids = centroids @ rotation_matrix.transpose(0, 1)
    else:
        pq_centroids = pq_centroids.to(torch.float32)
    expected_centroid_shape = (centroids.shape[0], working_dim)
    if tuple(pq_centroids.shape) != expected_centroid_shape:
        raise ValueError(
            "Invalid IVF-PQ transformed-centroid shape: "
            f"expected {expected_centroid_shape}, got {tuple(pq_centroids.shape)}"
        )
    return rotation_matrix, pq_centroids.contiguous()


def _get_interleaved_pq_lists(
    index: IVFPQIndex,
    search_device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Retrieve the sole persisted list layout used by Triton TopK."""
    if (
        index.persisted_list_codes_interleaved_by_subspace is None
        or index.persisted_list_ids_interleaved is None
        or index.persisted_list_group_prefix is None
    ):
        raise ValueError(
            "Triton IVF-PQ interleaved lists are not loaded. Rebuild the index."
        )

    if index.resident_device is not None and index.resident_device == search_device:
        if (
            index.list_ids_interleaved is None
            or index.list_group_prefix is None
            or (
                index.list_codes_interleaved_by_subspace is None
                and index.list_codes_interleaved_packed_16b is None
            )
        ):
            raise ValueError("Resident Triton IVF-PQ list layout is incomplete")
        return (
            index.list_codes_interleaved_by_subspace,
            index.list_ids_interleaved,
            index.list_group_prefix,
        )

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


def _get_interleaved_pq_static_corrections(
    index: IVFPQIndex,
    search_device: torch.device,
) -> torch.Tensor:
    """Return the physical-order scalar cache persisted for Triton TopK."""
    if index.resident_device is not None and index.resident_device == search_device:
        if index.list_pq_static_corrections_interleaved is None:
            raise ValueError("Resident Triton IVF-PQ correction cache is missing")
        return index.list_pq_static_corrections_interleaved
    if index.persisted_list_pq_static_corrections_interleaved is None:
        raise ValueError(
            "Triton IVF-PQ correction cache is not loaded. Rebuild the index."
        )
    return index.persisted_list_pq_static_corrections_interleaved.to(
        dtype=torch.float32,
        device=search_device,
        copy=False,
    )


def _pack_interleaved_pq_code_chunks(
    interleaved_codes: torch.Tensor,
) -> torch.Tensor | None:
    """Convert subspace-major codes to the packed 16-byte scanner layout."""
    m = int(interleaved_codes.shape[0])
    if m <= 16 or interleaved_codes.dtype != torch.uint8:
        return None

    num_groups = interleaved_codes.shape[1] // INTERLEAVED_GROUP_SIZE
    num_chunks = (m + 15) // 16
    padded_m = num_chunks * 16
    codes = interleaved_codes.reshape(
        m,
        num_groups,
        INTERLEAVED_GROUP_SIZE,
    ).permute(1, 2, 0)
    if padded_m != m:
        padded_codes = torch.zeros(
            (num_groups, INTERLEAVED_GROUP_SIZE, padded_m),
            dtype=interleaved_codes.dtype,
            device=interleaved_codes.device,
        )
        padded_codes[..., :m] = codes
        codes = padded_codes
    return (
        codes.reshape(num_groups, INTERLEAVED_GROUP_SIZE, num_chunks, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(torch.uint64)
    )


def _get_interleaved_pq_code_chunks(
    index: IVFPQIndex,
    interleaved_codes: torch.Tensor | None,
    search_device: torch.device,
) -> torch.Tensor | None:
    """Return the cuVS-style 16-byte PQ layout used by the packed scanner."""
    if index.resident_device == search_device:
        cache = index.list_codes_interleaved_packed_16b
        if cache is not None and cache.device == search_device:
            return cache
    if interleaved_codes is None:
        raise ValueError("Packed IVF-PQ search is missing both resident code layouts")
    cache = _pack_interleaved_pq_code_chunks(interleaved_codes)
    if index.resident_device == search_device:
        index.list_codes_interleaved_packed_16b = cache
    return cache


def _build_pq_static_corrections(
    list_codes: torch.Tensor,
    list_prefix: torch.Tensor,
    list_count: torch.Tensor,
    centroids: torch.Tensor,
    pq_codebooks: torch.Tensor,
) -> torch.Tensor:
    """Precompute one centroid/PQ correction scalar for every row-major code.

    For a vector ``i`` in list ``l`` with codes ``z[i, m]``, this stores
    ``gamma[i] = sum_m 2 * <c[l, m], p[m, z[i, m]]>``. Query-time ADC can add
    one scalar per candidate to a query-global LUT lookup instead of gathering
    a list/subspace/code correction for every subspace.
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

    return values


def _build_interleaved_pq_static_corrections(
    list_codes: torch.Tensor,
    list_prefix: torch.Tensor,
    list_count: torch.Tensor,
    centroids: torch.Tensor,
    pq_codebooks: torch.Tensor,
) -> torch.Tensor:
    """Place full row-major static corrections in the Triton physical layout."""
    values = _build_pq_static_corrections(
        list_codes,
        list_prefix,
        list_count,
        centroids,
        pq_codebooks,
    )
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
    """Build ``||p_mk||^2 - 2 q_m^T p_mk`` as a global LUT.

    The returned Global LUT uses ``[m, q, k]`` storage for the Triton TopK
    scanner.

    The omitted ``||q||^2`` is intentionally carried by neither this table
    nor the probe bias.  For a candidate in list ``l`` the direct scanner
    evaluates ``probe_distance(q, l) + gamma[i] + sum_m G[q, m, code_m]``;
    this is algebraically identical to the residual-PQ distance because the
    former decomposition's query-norm terms cancel exactly.  Keeping them out
    avoids two query-norm reductions and their temporary global-memory tensors.
    Individual entries of this table may be negative, so unlike a squared-L2
    LUT it must not be clamped before ADC accumulation.
    """
    num_queries = queries.shape[0]
    q_sub_by_m = (
        queries.view(num_queries, m, subdim)
        .permute(1, 0, 2)
        .contiguous()
    )
    codebooks = pq_codebooks.transpose(1, 2)
    norm_base = pq_codebook_norms.unsqueeze(1)

    # baddbmm broadcasts the static codeword-norm table across the Query
    # dimension, without materializing a separate query-norm tensor.
    if not use_tf32 or not queries.is_cuda:
        values = torch.baddbmm(
            norm_base,
            q_sub_by_m,
            codebooks,
            beta=1.0,
            alpha=-2.0,
        )
    else:
        previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            values = torch.baddbmm(
                norm_base,
                q_sub_by_m,
                codebooks,
                beta=1.0,
                alpha=-2.0,
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32

    return values


def _coarse_squared_l2_distance_addmm(
    queries: torch.Tensor,
    centroids: torch.Tensor,
    centroid_norms: torch.Tensor,
) -> torch.Tensor:
    """Compute ``||q-c||^2`` with a cached centroid-norm table.

    ``centroid_norms`` is static index data.  Feeding it as the broadcast C
    operand of ``addmm`` avoids recomputing a `[nlist, dim]` reduction for
    every query batch.  Coarse top-k selection is deliberately strict FP32:
    small TF32 dot-product perturbations can change the nprobe boundary and
    therefore the candidate lists searched.
    """
    query_norms = (queries * queries).sum(dim=1, keepdim=True)
    centroid_norm_base = centroid_norms.unsqueeze(0)
    if not queries.is_cuda:
        distances = torch.addmm(
            centroid_norm_base,
            queries,
            centroids.transpose(0, 1),
            beta=1.0,
            alpha=-2.0,
        )
    else:
        previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            distances = torch.addmm(
                centroid_norm_base,
                queries,
                centroids.transpose(0, 1),
                beta=1.0,
                alpha=-2.0,
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
    return distances.add_(query_norms).clamp_min_(0.0)


@dataclass
class _IVFPQSearchWorkspace:
    """Reusable fixed-shape buffers and an optional captured CUDA graph."""

    static_queries: torch.Tensor
    result_distances: torch.Tensor
    result_ids: torch.Tensor
    triton_partial_keys: torch.Tensor | None = None
    graph: Any = None
    graph_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
    capture_stream: Any = None


@dataclass
class IVFPQIndex:
    """In-memory representation of IVF-PQ index."""

    index_dir: Path
    metadata: dict[str, Any]
    vectors: torch.Tensor | None
    assignments: torch.Tensor
    centroids: torch.Tensor
    centroid_norms: torch.Tensor
    pq_codebooks: torch.Tensor
    pq_codebook_norms: torch.Tensor
    ids: torch.Tensor
    list_count: torch.Tensor | None
    # cuVS-style tall orthogonal transform and coarse centers in its PQ
    # working space. They are populated only when ``dim % m != 0``.
    pq_rotation_matrix: torch.Tensor | None = None
    pq_centroids: torch.Tensor | None = None
    # Unpacked device cache for M<=16 and explicit compatibility fallback.
    # Wide uint8 PQ indices use only the packed 16-byte cache on CUDA.
    list_codes_interleaved_by_subspace: torch.Tensor | None = None
    # [group, 16-byte-chunk, lane, two uint64 words] packed PQ cache.
    list_codes_interleaved_packed_16b: torch.Tensor | None = None
    list_ids_interleaved: torch.Tensor | None = None
    list_group_prefix: torch.Tensor | None = None
    # One FP32 scalar per padded physical vector: sum_m R[list, m, code_m].
    list_pq_static_corrections_interleaved: torch.Tensor | None = None
    # CPU copies of the sole persisted Triton layout.
    persisted_list_codes_interleaved_by_subspace: torch.Tensor | None = None
    persisted_list_ids_interleaved: torch.Tensor | None = None
    persisted_list_group_prefix: torch.Tensor | None = None
    persisted_list_pq_static_corrections_interleaved: torch.Tensor | None = None
    # Reused by the one-CTA-per-(query, probe-group) Triton route.
    triton_partial_keys: torch.Tensor | None = None
    # Computed once, then used to address the persistent Triton autotune cache
    # without synchronizing a GPU-resident ``list_count`` on every search.
    triton_list_count_stats: tuple[int, int, int] | None = None
    resident_device: torch.device | None = None
    # A single fixed-shape workspace is enough for the benchmark's
    # warmup-then-measure access pattern and bounds CUDA Graph pool growth as
    # nprobe changes.
    search_workspace_key: tuple[object, ...] | None = field(default=None, repr=False)
    search_workspace: _IVFPQSearchWorkspace | None = field(default=None, repr=False)

    def reside_on_device(self, device: torch.device | str) -> IVFPQIndex:
        """Move search-critical tensors onto a target device and keep them resident."""
        if self.list_count is None:
            raise ValueError("Index PQ lists are not loaded. Call load_ivfpq_index(..., load_lists=True).")
        if (
            self.persisted_list_codes_interleaved_by_subspace is None
            or self.persisted_list_ids_interleaved is None
            or self.persisted_list_group_prefix is None
            or self.persisted_list_pq_static_corrections_interleaved is None
        ):
            raise ValueError("Triton IVF-PQ layout is incomplete. Rebuild the index.")

        target_device = torch.device(device)
        if (
            self.resident_device == target_device
            and self.list_ids_interleaved is not None
            and self.list_group_prefix is not None
            and self.list_pq_static_corrections_interleaved is not None
            and (
                self.list_codes_interleaved_by_subspace is not None
                or self.list_codes_interleaved_packed_16b is not None
            )
        ):
            return self
        self.centroids = self.centroids.to(dtype=torch.float32, device=target_device, copy=False)
        self.centroid_norms = self.centroid_norms.to(
            dtype=torch.float32,
            device=target_device,
            copy=False,
        )
        self.pq_codebooks = self.pq_codebooks.to(dtype=torch.float32, device=target_device, copy=False)
        self.pq_codebook_norms = self.pq_codebook_norms.to(
            dtype=torch.float32,
            device=target_device,
            copy=False,
        )
        if self.pq_rotation_matrix is not None:
            self.pq_rotation_matrix = self.pq_rotation_matrix.to(
                dtype=torch.float32,
                device=target_device,
                copy=False,
            )
        if self.pq_centroids is not None:
            self.pq_centroids = self.pq_centroids.to(
                dtype=torch.float32,
                device=target_device,
                copy=False,
            )
        self.list_count = self.list_count.to(dtype=torch.int64, device=target_device, copy=False)
        persisted_codes = self.persisted_list_codes_interleaved_by_subspace
        # The M>16 Triton hot path consumes only the 16-byte packed layout.
        # Pack on the CPU before upload so the GPU never owns both complete
        # code layouts at once. Keep the original layout for M<=16 and for
        # non-CUDA residency, where the unpacked scanner is still used.
        if (
            target_device.type == "cuda"
            and int(persisted_codes.shape[0]) > 16
            and persisted_codes.dtype == torch.uint8
        ):
            packed_codes = _pack_interleaved_pq_code_chunks(persisted_codes)
            assert packed_codes is not None
            self.list_codes_interleaved_packed_16b = packed_codes.to(
                device=target_device,
                copy=False,
            )
            self.list_codes_interleaved_by_subspace = None
        else:
            self.list_codes_interleaved_by_subspace = persisted_codes.to(
                device=target_device,
                copy=False,
            )
            self.list_codes_interleaved_packed_16b = None
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
        self.list_pq_static_corrections_interleaved = (
            self.persisted_list_pq_static_corrections_interleaved.to(
                dtype=torch.float32,
                device=target_device,
                copy=False,
            )
        )
        self.triton_partial_keys = None
        self.search_workspace_key = None
        self.search_workspace = None
        self.resident_device = target_device
        return self

    def release_device_residency(self) -> IVFPQIndex:
        """Move resident search tensors back to CPU."""
        self.centroids = self.centroids.to(dtype=torch.float32, device="cpu", copy=False)
        self.centroid_norms = self.centroid_norms.to(dtype=torch.float32, device="cpu", copy=False)
        self.pq_codebooks = self.pq_codebooks.to(dtype=torch.float32, device="cpu", copy=False)
        self.pq_codebook_norms = self.pq_codebook_norms.to(dtype=torch.float32, device="cpu", copy=False)
        if self.pq_rotation_matrix is not None:
            self.pq_rotation_matrix = self.pq_rotation_matrix.to(
                dtype=torch.float32,
                device="cpu",
                copy=False,
            )
        if self.pq_centroids is not None:
            self.pq_centroids = self.pq_centroids.to(
                dtype=torch.float32,
                device="cpu",
                copy=False,
            )
        if self.list_count is not None:
            self.list_count = self.list_count.to(dtype=torch.int64, device="cpu", copy=False)
        self.list_codes_interleaved_by_subspace = None
        self.list_codes_interleaved_packed_16b = None
        self.list_ids_interleaved = None
        self.list_group_prefix = None
        self.list_pq_static_corrections_interleaved = None
        self.triton_partial_keys = None
        self.search_workspace_key = None
        self.search_workspace = None
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
    pq_rotation_matrix: torch.Tensor | None = None
    pq_centroids: torch.Tensor | None = None
    seed: int = 0
    n_init: int = 1
    coarse_inertia: float | None = None
    coarse_seed: int | None = None
    pq_inertias: list[float] = field(default_factory=list)
    pq_seeds: list[int] = field(default_factory=list)
    # Keep IVF/coarse and PQ training controls independent.  ``n_init`` above
    # remains as a legacy compatibility field for callers that used the old
    # shared setting.
    coarse_niter: int = FAISS_DEFAULT_NITER
    pq_niter: int = FAISS_DEFAULT_NITER
    coarse_n_init: int = FAISS_DEFAULT_NREDO
    pq_n_init: int = FAISS_DEFAULT_NREDO
    coarse_max_points_per_centroid: int | None = FAISS_DEFAULT_MAX_POINTS_PER_CENTROID
    coarse_trainset_fraction: float | None = None
    pq_max_points_per_centroid: int | None = FAISS_DEFAULT_MAX_POINTS_PER_CENTROID
    coarse_init_method: str = "kmeans++"
    coarse_kmeans_method: str = "balanced"
    pq_init_method: str = "kmeans++"
    pq_kmeans_method: str = "balanced"
    # Diagnostic fields retained in metadata to make the PQ sampling policy
    # explicit. The default benchmark keeps PQ sampling independent.
    pq_training_rows: int | None = None
    pq_reuses_coarse_trainset: bool = False


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
    """Persist the interleaved IVF-PQ layout consumed by Triton TopK."""
    output_path = Path(output_dir)

    vectors_cpu = build_result.vectors.cpu()
    labels_cpu = build_result.labels.cpu().to(torch.int64)
    centroids_cpu = build_result.centroids.cpu().to(torch.float32)
    centroid_norms_cpu = (centroids_cpu * centroids_cpu).sum(dim=-1)
    pq_codebooks_cpu = build_result.pq_codebooks.cpu().to(torch.float32)
    pq_codebook_norms_cpu = (pq_codebooks_cpu * pq_codebooks_cpu).sum(dim=-1)
    pq_working_dim = int(pq_codebooks_cpu.shape[0] * pq_codebooks_cpu.shape[2])
    pq_rotation_matrix_cpu = (
        build_result.pq_rotation_matrix.cpu().to(torch.float32)
        if build_result.pq_rotation_matrix is not None
        else None
    )
    if pq_rotation_matrix_cpu is None:
        if pq_working_dim != centroids_cpu.shape[1]:
            raise ValueError(
                "A non-divisible IVF-PQ build result must include its PQ rotation matrix"
            )
        pq_centroids_cpu = centroids_cpu
    elif build_result.pq_centroids is None:
        pq_centroids_cpu = (centroids_cpu @ pq_rotation_matrix_cpu.transpose(0, 1)).contiguous()
    else:
        pq_centroids_cpu = build_result.pq_centroids.cpu().to(torch.float32)
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
        pq_centroids_cpu,
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
        "subdim": int(pq_codebooks_cpu.shape[2]),
        "pq_working_dim": pq_working_dim,
        "pq_rotation": "random_orthogonal" if pq_rotation_matrix_cpu is not None else "identity",
        "device": str(build_result.device),
        "seed": int(build_result.seed),
        # ``n_init`` is kept for compatibility with older index readers.
        "n_init": int(build_result.n_init),
        "coarse_niter": int(build_result.coarse_niter),
        "pq_niter": int(build_result.pq_niter),
        "coarse_n_init": int(build_result.coarse_n_init),
        "pq_n_init": int(build_result.pq_n_init),
        "coarse_max_points_per_centroid": build_result.coarse_max_points_per_centroid,
        "coarse_trainset_fraction": build_result.coarse_trainset_fraction,
        "pq_max_points_per_centroid": build_result.pq_max_points_per_centroid,
        "coarse_init_method": str(build_result.coarse_init_method),
        "coarse_kmeans_method": str(build_result.coarse_kmeans_method),
        "pq_init_method": str(build_result.pq_init_method),
        "pq_kmeans_method": str(build_result.pq_kmeans_method),
        "pq_training_rows": build_result.pq_training_rows,
        "pq_reuses_coarse_trainset": bool(build_result.pq_reuses_coarse_trainset),
        "coarse_inertia": build_result.coarse_inertia,
        "coarse_seed": build_result.coarse_seed,
        "pq_inertias": build_result.pq_inertias,
        "pq_seeds": build_result.pq_seeds,
        "pq_mode": "residual",
        "has_pq_static_vector_corrections": True,
        "stores_full_vectors": False,
        "cluster_sizes": cluster_sizes,
        "storage_format": "single_file" if is_single_file else "directory_interleaved",
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
            "centroid_norms": centroid_norms_cpu,
            "pq_codebooks": pq_codebooks_cpu,
            "pq_codebook_norms": pq_codebook_norms_cpu,
            "pq_codes": pq_codes_cpu,
            "ids": ids,
            "list_count": list_count,
            "list_codes_interleaved_by_subspace": interleaved_layout.codes_by_subspace,
            "list_ids_interleaved": interleaved_layout.ids,
            "list_group_prefix": interleaved_layout.group_prefix,
            "list_pq_static_corrections_interleaved": interleaved_static_corrections,
        }
        if pq_rotation_matrix_cpu is not None:
            payload["pq_rotation_matrix"] = pq_rotation_matrix_cpu
            payload["pq_centroids"] = pq_centroids_cpu
        torch.save(payload, output_path)
        index_size_bytes = output_path.stat().st_size
        metadata["index_size_bytes"] = index_size_bytes
        metadata["index_size_mb"] = round(index_size_bytes / (1024 * 1024), 4)
        payload["metadata"] = metadata
        torch.save(payload, output_path)
        return output_path

    out_dir = output_path
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove files from the former row-major/Tensor layout when overwriting an
    # existing directory index. Triton TopK now has one physical list layout.
    for obsolete_path in (
        out_dir / "list_codes.pt",
        out_dir / "list_ids_flat.pt",
        out_dir / "list_prefix.pt",
        out_dir / "list_pq_static_corrections.pt",
        out_dir / "pq_centroid_corrections.pt",
    ):
        if obsolete_path.is_file():
            obsolete_path.unlink()
    torch.save(labels_cpu, out_dir / "assignments.pt")
    torch.save(centroids_cpu, out_dir / "centroids.pt")
    torch.save(centroid_norms_cpu, out_dir / "centroid_norms.pt")
    torch.save(pq_codebooks_cpu, out_dir / "pq_codebooks.pt")
    torch.save(pq_codebook_norms_cpu, out_dir / "pq_codebook_norms.pt")
    if pq_rotation_matrix_cpu is not None:
        torch.save(pq_rotation_matrix_cpu, out_dir / "pq_rotation_matrix.pt")
        torch.save(pq_centroids_cpu, out_dir / "pq_centroids.pt")
    else:
        for stale_path in (out_dir / "pq_rotation_matrix.pt", out_dir / "pq_centroids.pt"):
            if stale_path.is_file():
                stale_path.unlink()
    torch.save(pq_codes_cpu, out_dir / "pq_codes.pt")
    torch.save(ids, out_dir / "ids.pt")
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
        centroid_norms = payload.get("centroid_norms")
        if centroid_norms is None:
            centroid_norms = (centroids * centroids).sum(dim=-1)
        else:
            centroid_norms = centroid_norms.to(torch.float32)
        pq_codebooks = payload["pq_codebooks"].to(torch.float32)
        pq_codebook_norms = payload.get("pq_codebook_norms")
        if pq_codebook_norms is None:
            pq_codebook_norms = (pq_codebooks * pq_codebooks).sum(dim=-1)
        else:
            pq_codebook_norms = pq_codebook_norms.to(torch.float32)
        pq_rotation_matrix, pq_centroids = _restore_pq_transform(
            metadata,
            centroids,
            pq_codebooks,
            payload.get("pq_rotation_matrix"),
            payload.get("pq_centroids"),
        )
        ids = payload["ids"]
        if load_lists:
            required_payload_keys = (
                "list_count",
                "list_codes_interleaved_by_subspace",
                "list_ids_interleaved",
                "list_group_prefix",
                "list_pq_static_corrections_interleaved",
            )
            missing_payload_keys = [key for key in required_payload_keys if key not in payload]
            if missing_payload_keys:
                raise FileNotFoundError(
                    "Triton IVF-PQ interleaved layout is missing from this archive; "
                    f"rebuild the index. Missing keys: {missing_payload_keys}"
                )
            list_count = payload["list_count"].to(torch.int64)
            persisted_list_codes_interleaved_by_subspace = payload[
                "list_codes_interleaved_by_subspace"
            ]
            persisted_list_ids_interleaved = payload["list_ids_interleaved"].to(torch.int64)
            persisted_list_group_prefix = payload["list_group_prefix"].to(torch.int64)
            persisted_list_pq_static_corrections_interleaved = payload[
                "list_pq_static_corrections_interleaved"
            ].to(torch.float32)
        else:
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
            centroid_norms=centroid_norms,
            pq_codebooks=pq_codebooks,
            pq_codebook_norms=pq_codebook_norms,
            ids=ids,
            list_count=list_count,
            pq_rotation_matrix=pq_rotation_matrix,
            pq_centroids=pq_centroids,
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
    centroid_norms_path = index_path / "centroid_norms.pt"
    pq_codebooks_path = index_path / "pq_codebooks.pt"
    pq_codebook_norms_path = index_path / "pq_codebook_norms.pt"
    pq_rotation_matrix_path = index_path / "pq_rotation_matrix.pt"
    pq_centroids_path = index_path / "pq_centroids.pt"
    ids_path = index_path / "ids.pt"
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
    if centroid_norms_path.exists():
        centroid_norms = torch.load(centroid_norms_path, map_location="cpu").to(torch.float32)
    else:
        # Indexes created before centroid norms were persisted remain usable.
        centroid_norms = (centroids * centroids).sum(dim=-1)
    pq_codebooks = torch.load(pq_codebooks_path, map_location="cpu").to(torch.float32)
    if pq_codebook_norms_path.exists():
        pq_codebook_norms = torch.load(pq_codebook_norms_path, map_location="cpu").to(torch.float32)
    else:
        # Indexes created before codebook norms were persisted remain usable.
        pq_codebook_norms = (pq_codebooks * pq_codebooks).sum(dim=-1)
    pq_rotation_matrix = (
        torch.load(pq_rotation_matrix_path, map_location="cpu").to(torch.float32)
        if pq_rotation_matrix_path.exists()
        else None
    )
    pq_centroids = (
        torch.load(pq_centroids_path, map_location="cpu").to(torch.float32)
        if pq_centroids_path.exists()
        else None
    )
    pq_rotation_matrix, pq_centroids = _restore_pq_transform(
        metadata,
        centroids,
        pq_codebooks,
        pq_rotation_matrix,
        pq_centroids,
    )
    ids = torch.load(ids_path, map_location="cpu")

    if load_lists:
        list_files = [
            list_count_path,
            list_codes_interleaved_by_subspace_path,
            list_ids_interleaved_path,
            list_group_prefix_path,
            list_pq_static_corrections_interleaved_path,
        ]
        missing_list_files = [str(path) for path in list_files if not path.exists()]
        if missing_list_files:
            raise FileNotFoundError(
                "Triton IVF-PQ interleaved layout is incomplete; rebuild the index. "
                f"Missing files: {missing_list_files}"
            )
        list_count = torch.load(list_count_path, map_location="cpu").to(torch.int64)
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
        persisted_list_pq_static_corrections_interleaved = torch.load(
            list_pq_static_corrections_interleaved_path,
            map_location="cpu",
        ).to(torch.float32)
    else:
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
        centroid_norms=centroid_norms,
        pq_codebooks=pq_codebooks,
        pq_codebook_norms=pq_codebook_norms,
        ids=ids,
        list_count=list_count,
        pq_rotation_matrix=pq_rotation_matrix,
        pq_centroids=pq_centroids,
        persisted_list_codes_interleaved_by_subspace=persisted_list_codes_interleaved_by_subspace,
        persisted_list_ids_interleaved=persisted_list_ids_interleaved,
        persisted_list_group_prefix=persisted_list_group_prefix,
        persisted_list_pq_static_corrections_interleaved=(
            persisted_list_pq_static_corrections_interleaved
        ),
    )


def _search_ivfpq_index_eager(
    index: IVFPQIndex,
    queries: torch.Tensor,
    topk: int = 10,
    nprobe: int = 10,
    device: torch.device | str | None = None,
    selection_mode: str = "triton_topk",
    profile: dict[str, float] | None = None,
    use_packed_16b_codes: bool = True,
    triton_autotune: bool = False,
    triton_autotune_cache_nprobe: int | None = None,
    refine_vectors: torch.Tensor | None = None,
    refine_factor: int = 1,
    lut_dtype: str = "fp32",
    _workspace: _IVFPQSearchWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search IVF-PQ, optionally reranking an expanded candidate set exactly.

    ``triton_topk`` assigns one CTA to each ``(query, probed-list group)``
    edge, keeps an exact group-local top-k in that CTA, then runs only the
    query-level final merge. Unsupported CUDA shapes raise an error instead
    of silently selecting another search implementation.

    ``use_packed_16b_codes`` selects cuVS-style 16-byte code storage.  M=17-32
    and M=33-1024 both use a rolled chunk-loop ADC
    whose compiled body and register lifetime do not grow linearly with M.
    Set it to ``False`` to retain the subspace-major path where supported.

    Set ``triton_autotune=True`` to measure a small local sweep of CTA tile,
    warp, and probe-list grouping choices on the first matching workload. The
    winner is persisted under ``~/.cache/tensorvs`` and reused automatically
    by later searches, including when ``triton_autotune`` is False. Set
    ``TENSORVS_TRITON_AUTOTUNE_CACHE`` to choose a different path or to an
    empty string to disable persistence.

    ``triton_autotune_cache_nprobe`` optionally replaces the actual ``nprobe``
    in the autotune cache key. This lets a benchmark tune representative
    nprobe values and reuse their launch configuration for nearby values; it
    does not change the actual number of probed lists.

    The Triton path uses decomposed query-global ``G`` in FP32 with TF32
    Tensor Cores, stored as FP32 or FP16 according to ``lut_dtype``. ADC adds
    the persisted FP32 scalar correction and selected coarse-list distance;
    it does not materialize a per-probe ``R[nlist,m,ksub]`` table or true
    residual LUT.

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
    if index.list_count is None:
        raise ValueError("Index PQ lists are not loaded. Call load_ivfpq_index(..., load_lists=True).")
    if selection_mode != "triton_topk":
        raise ValueError(
            "Unsupported IVFPQ selection_mode: "
            f"{selection_mode}; only 'triton_topk' is supported"
        )
    lut_dtype = _normalize_lut_dtype(lut_dtype)

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
        centroid_norms = index.centroid_norms
        pq_codebooks = index.pq_codebooks
        pq_codebook_norms = index.pq_codebook_norms
        pq_rotation_matrix = index.pq_rotation_matrix
        list_count = index.list_count
    else:
        centroids = index.centroids.to(dtype=torch.float32, device=search_device, copy=False)
        centroid_norms = index.centroid_norms.to(
            dtype=torch.float32,
            device=search_device,
            copy=False,
        )
        pq_codebooks = index.pq_codebooks.to(dtype=torch.float32, device=search_device, copy=False)
        pq_codebook_norms = index.pq_codebook_norms.to(
            dtype=torch.float32,
            device=search_device,
            copy=False,
        )
        pq_rotation_matrix = (
            index.pq_rotation_matrix.to(
                dtype=torch.float32,
                device=search_device,
                copy=False,
            )
            if index.pq_rotation_matrix is not None
            else None
        )
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
    if _workspace is None:
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
    else:
        expected_shape = (num_queries, search_topk)
        if (
            _workspace.result_distances.shape != expected_shape
            or _workspace.result_distances.device != search_device
            or _workspace.result_ids.shape != expected_shape
            or _workspace.result_ids.device != search_device
        ):
            raise ValueError("IVFPQ workspace does not match the current search shape")
        result_distances = _workspace.result_distances
        result_ids = _workspace.result_ids
        result_distances.fill_(float("inf"))
        result_ids.fill_(-1)
    record_timing("init_outputs", step_start)

    # Step 1: compute the full query-centroid squared L2 distance matrix, then
    # keep only the nearest `nprobe` coarse centroids for each query.
    step_start = start_timing()
    dist_qc = _coarse_squared_l2_distance_addmm(
        q,
        centroids,
        centroid_norms,
    )
    record_timing("centroid_distance", step_start)
    step_start = start_timing()
    probe_distances, probe_cids = torch.topk(dist_qc, k=nprobe, dim=1, largest=False)
    record_timing("centroid_topk", step_start)

    m = int(index.metadata["m"])
    subdim = int(index.metadata["subdim"])
    step_start = start_timing()
    interleaved_codes, interleaved_ids, list_group_prefix = _get_interleaved_pq_lists(
        index,
        search_device,
    )
    if interleaved_codes is None and not (use_packed_16b_codes and m > 16):
        # Packed-only residency is the default for wide uint8 PQ codes. Keep
        # the public unpacked fallback available by uploading the persisted
        # layout only for callers that explicitly disable packed scanning.
        assert index.persisted_list_codes_interleaved_by_subspace is not None
        interleaved_codes = index.persisted_list_codes_interleaved_by_subspace.to(
            device=search_device,
            copy=False,
        )
    interleaved_static_corrections = _get_interleaved_pq_static_corrections(
        index,
        search_device,
    )
    record_timing("triton_prepare_layout", step_start)

    triton_topk_supported = True
    global_query_luts: torch.Tensor | None = None
    probe_distance_bias: torch.Tensor | None = None
    step_start = start_timing()
    # Build the single Global LUT layout consumed by Triton TopK.
    pq_queries = _apply_pq_rotation(q, pq_rotation_matrix)
    if pq_queries.shape[1] != m * subdim:
        raise ValueError(
            "PQ query working dimension does not match the codebooks: "
            f"{pq_queries.shape[1]} != {m * subdim}"
        )
    global_query_luts = _build_global_query_luts_bmm(
        pq_queries,
        pq_codebooks,
        pq_codebook_norms,
        m=m,
        subdim=subdim,
        use_tf32=True,
    )
    if lut_dtype == "fp16":
        global_query_luts = global_query_luts.to(torch.float16)
    probe_distance_bias = probe_distances
    record_timing("LUT_prepare", step_start)

    if triton_topk_supported:
        packed_code_chunks: torch.Tensor | None = None
        if use_packed_16b_codes and m > 16:
            step_start = start_timing()
            packed_code_chunks = _get_interleaved_pq_code_chunks(
                index,
                interleaved_codes,
                search_device,
            )
            record_timing("triton_packed_prepare", step_start)

        if index.triton_list_count_stats is None:
            count_cpu = list_count.detach().to(device="cpu", dtype=torch.float32)
            index.triton_list_count_stats = (
                int(count_cpu.mean().item()),
                int(torch.quantile(count_cpu, 0.95).item()),
                int(count_cpu.max().item()),
            )

        scan_chunk_groups, scan_max_chunks = plan_probe_chunks(
            nprobe=nprobe,
            max_list_count=index.triton_list_count_stats[2],
        )
        scan_nprobe = nprobe * scan_max_chunks
        triton_topk_capable = can_run_ivfpq_triton_topk(
            device=search_device,
            topk=search_topk,
            m=m,
            ksub=int(index.metadata["ksub"]),
            nprobe=scan_nprobe,
            num_code_vectors=interleaved_ids.numel(),
            code_dtype=(
                interleaved_codes.dtype
                if interleaved_codes is not None
                else torch.uint8
            ),
            use_packed_16b_codes=packed_code_chunks is not None,
        )
        if not triton_topk_capable:
            raise RuntimeError(
                "IVF-PQ search requires the Triton TopK path and a supported CUDA shape"
            )
        if triton_autotune_cache_nprobe is None:
            autotune_cache_nprobe = nprobe
        else:
            autotune_cache_nprobe = max(
                1,
                min(int(triton_autotune_cache_nprobe), int(centroids.shape[0])),
            )
        gpu_name = torch.cuda.get_device_name(search_device)
        compute_capability = torch.cuda.get_device_capability(search_device)
        autotune_key: tuple[object, ...] = (
            "query_major",
            gpu_name,
            compute_capability,
            torch.__version__,
            torch.version.cuda,
            lut_dtype,
            m,
            int(index.metadata["ksub"]),
            autotune_cache_nprobe * scan_max_chunks,
            search_topk,
            num_queries,
            bool(packed_code_chunks is not None),
            scan_chunk_groups,
            scan_max_chunks,
            *index.triton_list_count_stats,
        )

        list_scan_start = start_timing()
        final_merge_start: float | torch.cuda.Event | None = None

        def record_triton_scan_complete() -> None:
            nonlocal final_merge_start
            record_timing("triton_scan", list_scan_start)
            final_merge_start = start_timing()

        assert global_query_luts is not None
        assert probe_distance_bias is not None
        partial_keys = triton_ivfpq_topk(
            global_query_luts=global_query_luts,
            static_vector_corrections=interleaved_static_corrections,
            probe_distance_bias=probe_distance_bias,
            list_codes_interleaved=interleaved_codes,
            list_ids_interleaved=interleaved_ids,
            probe_group_prefix=None,
            probe_list_count=None,
            probe_cids=probe_cids,
            list_group_prefix=list_group_prefix,
            list_count=list_count,
            scan_chunk_groups=scan_chunk_groups,
            scan_max_chunks=scan_max_chunks,
            topk=search_topk,
            output_distances=result_distances,
            output_ids=result_ids,
            list_partial_keys=(
                _workspace.triton_partial_keys
                if _workspace is not None
                else index.triton_partial_keys
            ),
            list_codes_packed_16b=packed_code_chunks,
            autotune=triton_autotune,
            autotune_key=autotune_key,
            on_list_scan_complete=record_triton_scan_complete,
        )
        if _workspace is not None:
            _workspace.triton_partial_keys = partial_keys
        elif index.resident_device is not None and index.resident_device == search_device:
            index.triton_partial_keys = partial_keys
        assert final_merge_start is not None
        record_timing("triton_final_merge", final_merge_start)

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


def search_ivfpq_index(
    index: IVFPQIndex,
    queries: torch.Tensor,
    topk: int = 10,
    nprobe: int = 10,
    device: torch.device | str | None = None,
    selection_mode: str = "triton_topk",
    profile: dict[str, float] | None = None,
    use_packed_16b_codes: bool = True,
    triton_autotune: bool = False,
    triton_autotune_cache_nprobe: int | None = None,
    refine_vectors: torch.Tensor | None = None,
    refine_factor: int = 1,
    lut_dtype: str = "fp32",
    use_preallocated_workspace: bool = False,
    use_cuda_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search IVF-PQ with optional fixed-shape workspace and CUDA Graph replay.

    CUDA Graph mode implies workspace reuse and requires a resident CUDA
    index, fixed query shape, and no exact-refine or per-stage profiling. The
    returned tensors are workspace-backed and are overwritten by the next
    workspace/graph search on this index. Call ``clone()`` if they must outlive
    the next search.

    ``use_preallocated_workspace`` provides the same output and Triton scratch
    reuse without graph capture. It has the same output-lifetime rule.
    """
    graph_eligible = (
        use_cuda_graph
        and profile is None
        and refine_vectors is None
        # A wide packed-only resident index has no persistent unpacked code
        # pointer for CUDA Graph capture. Explicit unpacked fallback remains
        # eager so its temporary upload cannot outlive the search call.
        and (
            use_packed_16b_codes
            or int(index.metadata["m"]) <= 16
            or index.list_codes_interleaved_by_subspace is not None
        )
    )
    if not use_preallocated_workspace and not graph_eligible:
        return _search_ivfpq_index_eager(
            index=index,
            queries=queries,
            topk=topk,
            nprobe=nprobe,
            device=device,
            selection_mode=selection_mode,
            profile=profile,
            use_packed_16b_codes=use_packed_16b_codes,
            triton_autotune=triton_autotune,
            triton_autotune_cache_nprobe=triton_autotune_cache_nprobe,
            refine_vectors=refine_vectors,
            refine_factor=refine_factor,
            lut_dtype=lut_dtype,
        )

    if queries.ndim != 2:
        raise ValueError(
            f"Query tensor must have shape (num_queries, dim), got {tuple(queries.shape)}"
        )
    if queries.shape[1] != index.centroids.shape[1]:
        raise ValueError(
            f"Query dimension {queries.shape[1]} does not match index dimension "
            f"{index.centroids.shape[1]}"
        )
    if device is None:
        if queries.is_cuda:
            search_device = queries.device
        elif torch.cuda.is_available():
            search_device = torch.device("cuda:0")
        else:
            search_device = torch.device("cpu")
    else:
        search_device = torch.device(device)
    q = queries.to(dtype=torch.float32, device=search_device, copy=False).contiguous()

    normalized_topk = max(1, int(topk))
    normalized_refine_factor = int(refine_factor)
    if normalized_refine_factor < 1:
        raise ValueError(
            f"refine_factor must be >= 1, got {normalized_refine_factor}"
        )
    search_topk = (
        normalized_topk * normalized_refine_factor
        if refine_vectors is not None
        else normalized_topk
    )
    search_topk = min(
        search_topk,
        int(index.metadata.get("num_vectors", search_topk)),
    )
    search_topk = max(search_topk, normalized_topk)
    normalized_nprobe = max(1, min(int(nprobe), int(index.centroids.shape[0])))

    # Graph capture is only safe when all index pointers are persistent. For
    # CPU or a non-resident index, retain workspace reuse but execute eagerly.
    graph_eligible = (
        graph_eligible
        and search_device.type == "cuda"
        and index.resident_device == search_device
    )
    workspace_key: tuple[object, ...] = (
        str(search_device),
        tuple(q.shape),
        normalized_topk,
        search_topk,
        normalized_nprobe,
        selection_mode,
        bool(use_packed_16b_codes),
        str(lut_dtype).lower(),
        triton_autotune_cache_nprobe,
        bool(triton_autotune),
        bool(graph_eligible),
    )
    if index.search_workspace_key != workspace_key or index.search_workspace is None:
        index.search_workspace_key = workspace_key
        index.search_workspace = _IVFPQSearchWorkspace(
            static_queries=torch.empty_like(q),
            result_distances=torch.empty(
                (q.shape[0], search_topk),
                dtype=torch.float32,
                device=search_device,
            ),
            result_ids=torch.empty(
                (q.shape[0], search_topk),
                dtype=torch.int64,
                device=search_device,
            ),
        )
    workspace = index.search_workspace

    def run_eager(
        search_queries: torch.Tensor,
        *,
        autotune: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _search_ivfpq_index_eager(
            index=index,
            queries=search_queries,
            topk=normalized_topk,
            nprobe=normalized_nprobe,
            device=search_device,
            selection_mode=selection_mode,
            profile=profile,
            use_packed_16b_codes=use_packed_16b_codes,
            triton_autotune=autotune,
            triton_autotune_cache_nprobe=triton_autotune_cache_nprobe,
            refine_vectors=refine_vectors,
            refine_factor=normalized_refine_factor,
            lut_dtype=lut_dtype,
            _workspace=workspace,
        )

    if not graph_eligible:
        return run_eager(q, autotune=triton_autotune)

    workspace.static_queries.copy_(q)
    if workspace.graph is None:
        capture_stream = torch.cuda.Stream(device=search_device)
        workspace.capture_stream = capture_stream
        capture_stream.wait_stream(torch.cuda.current_stream(search_device))
        # Warm every library/Triton path and finish autotuning before capture;
        # synchronization and compilation are illegal inside a CUDA Graph.
        with torch.cuda.stream(capture_stream):
            run_eager(workspace.static_queries, autotune=triton_autotune)
            run_eager(workspace.static_queries, autotune=False)
        torch.cuda.current_stream(search_device).wait_stream(capture_stream)
        torch.cuda.synchronize(search_device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            graph_outputs = run_eager(workspace.static_queries, autotune=False)
        workspace.graph = graph
        workspace.graph_outputs = graph_outputs

    workspace.graph.replay()
    assert workspace.graph_outputs is not None
    return workspace.graph_outputs


def build_ivfpq_index(
    vectors: torch.Tensor,
    nlist: int,
    niter: int,
    use_triton: bool,
    device: torch.device | str,
    m: int,
    nbits: int,
    seed: int = FAISS_DEFAULT_SEED,
    n_init: int = FAISS_DEFAULT_NREDO,
    *,
    coarse_niter: int | None = None,
    pq_niter: int | None = None,
    coarse_n_init: int | None = None,
    pq_n_init: int | None = None,
    coarse_max_points_per_centroid: int | None = None,
    coarse_trainset_fraction: float | None = None,
    pq_max_points_per_centroid: int | None = None,
    coarse_min_points_per_centroid: int | None = None,
    pq_min_points_per_centroid: int | None = None,
    coarse_init_method: str = "kmeans++",
    coarse_kmeans_method: str = "balanced",
    pq_init_method: str = "kmeans++",
    pq_kmeans_method: str = "balanced",
    max_points_per_centroid: int | None = FAISS_DEFAULT_MAX_POINTS_PER_CENTROID,
    min_points_per_centroid: int = FAISS_DEFAULT_MIN_POINTS_PER_CENTROID,
) -> IVFPQBuildResult:
    """Cluster a dataset with IVF and train residual PQ codebooks.

    The coarse IVF quantizer and the PQ codebooks have independent KMeans
    settings. ``coarse_trainset_fraction`` is an alternative to the coarse
    per-centroid cap and takes precedence when provided. It only controls the
    coarse quantizer; PQ samples its larger training set independently for
    each subspace. The legacy ``niter``, ``n_init``,
    ``max_points_per_centroid`` and ``min_points_per_centroid`` arguments are
    used as fallbacks when their corresponding coarse/PQ-specific argument is
    omitted. ``coarse_kmeans_method`` and ``pq_kmeans_method`` select the
    KMeans implementation independently for IVF and PQ. ``balanced`` selects
    the cuVS-style balancing loop. Labels and PQ codes are still generated for
    the complete input dataset.
    """
    if vectors.ndim != 2:
        raise ValueError(f"Dataset tensor must have shape (num_vectors, dim), got {tuple(vectors.shape)}")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if nbits <= 0:
        raise ValueError(f"nbits must be positive, got {nbits}")

    resolved_coarse_niter = int(niter if coarse_niter is None else coarse_niter)
    resolved_pq_niter = int(niter if pq_niter is None else pq_niter)
    resolved_coarse_n_init = int(n_init if coarse_n_init is None else coarse_n_init)
    resolved_pq_n_init = int(n_init if pq_n_init is None else pq_n_init)
    resolved_coarse_max_points = (
        None
        if coarse_trainset_fraction is not None
        else max_points_per_centroid
        if coarse_max_points_per_centroid is None
        else coarse_max_points_per_centroid
    )
    resolved_pq_max_points = (
        max_points_per_centroid
        if pq_max_points_per_centroid is None
        else pq_max_points_per_centroid
    )
    resolved_coarse_min_points = (
        min_points_per_centroid
        if coarse_min_points_per_centroid is None
        else coarse_min_points_per_centroid
    )
    resolved_pq_min_points = (
        min_points_per_centroid
        if pq_min_points_per_centroid is None
        else pq_min_points_per_centroid
    )

    vectors = vectors.to(dtype=torch.float32, copy=False).contiguous()
    num_vectors, dim = vectors.shape
    working_dim = _pq_working_dim(dim, m)
    subdim = working_dim // m

    ksub = 1 << int(nbits)
    if num_vectors < ksub:
        raise ValueError(f"Dataset size {num_vectors} must be >= 2**nbits ({ksub}) for PQ training")

    target_device = torch.device(device)
    # Sample the coarse trainset once. PQ deliberately does not reuse this
    # sample: each PQ subspace gets its own independent training sample.
    shared_train_indices: torch.Tensor | None = None
    coarse_training_data: torch.Tensor | None = None
    if coarse_trainset_fraction is not None:
        shared_train_indices = select_training_indices(
            int(num_vectors),
            k=nlist,
            max_points_per_centroid=None,
            trainset_fraction=coarse_trainset_fraction,
            seed=seed,
            stage=0,
            device=vectors.device,
        )
        coarse_training_data = vectors.index_select(0, shared_train_indices)

    coarse = fit_best_kmeans(
        vectors,
        d=dim,
        k=nlist,
        niter=resolved_coarse_niter,
        use_triton=use_triton,
        device=target_device,
        seed=seed,
        n_init=resolved_coarse_n_init,
        stage=0,
        max_points_per_centroid=resolved_coarse_max_points,
        min_points_per_centroid=resolved_coarse_min_points,
        trainset_fraction=coarse_trainset_fraction,
        init_method=coarse_init_method,
        algorithm=coarse_kmeans_method,
        training_data=coarse_training_data,
        training_indices=shared_train_indices,
        # cuVS uses its hierarchical balanced builder for the IVF coarse
        # quantizer.  The flag is ignored for Lloyd and PQ uses the flat
        # balanced builder below.
        balanced_hierarchical=True,
    )
    centroids = coarse.centroids
    labels = coarse.labels

    labels_device = labels.to(dtype=torch.int64, device=target_device, copy=False)
    # cuVS applies a random orthogonal transform whenever ``dim`` is not a
    # multiple of ``pq_dim``.  Its tall matrix preserves original-space L2
    # distances while exposing equal-width subspaces in ``working_dim``.
    pq_rotation_matrix = _make_pq_rotation_matrix(
        dim,
        working_dim,
        device=target_device,
        seed=int(seed) + 104729,
    )
    pq_centroids = _apply_pq_rotation(centroids, pq_rotation_matrix)

    # A high-dimensional base set can be several GiB in FP32.  PQ training
    # only needs each residual subspace in turn, so keep the full residual
    # working set on the host when it would create a large device allocation;
    # ``fit_best_kmeans`` still moves its bounded training sample to the target
    # device and streams final labels for the complete dataset.
    pq_working_device = target_device
    if target_device.type == "cuda" and vectors.numel() * torch.float32.itemsize > (1 << 30):
        pq_working_device = torch.device("cpu")
    if pq_working_device.type == "cpu":
        residual_base = vectors.to(device="cpu", dtype=torch.float32, copy=False)
        residual_centroids = centroids.to(device="cpu", dtype=torch.float32, copy=True)
        residual_labels = labels_device.to(device="cpu", dtype=torch.int64, copy=True)
        residual_vectors = (residual_base - residual_centroids[residual_labels]).contiguous()
        pq_rotation = (
            pq_rotation_matrix.to(device="cpu", dtype=torch.float32, copy=True)
            if pq_rotation_matrix is not None
            else None
        )
    else:
        vectors_device = vectors.to(dtype=torch.float32, device=target_device, copy=False)
        residual_vectors = (vectors_device - centroids[labels_device]).contiguous()
        pq_rotation = pq_rotation_matrix
    residual_vectors = _apply_pq_rotation(residual_vectors, pq_rotation)

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
            niter=resolved_pq_niter,
            use_triton=use_triton,
            device=target_device,
            seed=seed,
            n_init=resolved_pq_n_init,
            stage=subspace_idx + 1,
            max_points_per_centroid=resolved_pq_max_points,
            min_points_per_centroid=resolved_pq_min_points,
            init_method=pq_init_method,
            algorithm=pq_kmeans_method,
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
        pq_rotation_matrix=pq_rotation_matrix,
        pq_centroids=pq_centroids if pq_rotation_matrix is not None else None,
        seed=int(seed),
        n_init=int(n_init),
        coarse_inertia=coarse.inertia,
        coarse_seed=coarse.seed,
        pq_inertias=pq_inertias,
        pq_seeds=pq_seeds,
        coarse_niter=resolved_coarse_niter,
        pq_niter=resolved_pq_niter,
        coarse_n_init=resolved_coarse_n_init,
        pq_n_init=resolved_pq_n_init,
        coarse_max_points_per_centroid=resolved_coarse_max_points,
        coarse_trainset_fraction=(
            None
            if coarse_trainset_fraction is None
            else float(coarse_trainset_fraction)
        ),
        pq_max_points_per_centroid=resolved_pq_max_points,
        coarse_init_method=str(coarse_init_method),
        coarse_kmeans_method=str(coarse_kmeans_method),
        pq_init_method=str(pq_init_method),
        pq_kmeans_method=str(pq_kmeans_method),
        pq_training_rows=None,
        pq_reuses_coarse_trainset=False,
    )
