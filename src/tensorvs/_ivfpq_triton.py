"""Triton query-major IVF-PQ scan with static per-vector corrections.

The index stores one FP32 scalar for every padded physical PQ vector:
``gamma[i] = sum_m 2 * <c[list(i), m], p[m, code(i, m)]>``.  Search builds a
query-global ``G[q,m,k] = ||q_m - p_mk||^2`` LUT and scans codes directly:
``distance(q, i) = bias(q, list(i)) + gamma[i] + sum_m G[q,m,code(i,m)]``.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CUDA caller fallback covers this
    triton = None
    tl = None


INTERLEAVED_GROUP_SIZE = 32
SCAN_BLOCK_CANDIDATES = 128
MAX_LOCAL_K = 128
MAX_FINAL_WIDTH = 2048
DIRECT_SCAN_PROGRAMS_PER_SM = 2
DIRECT_SCAN_FALLBACK_TARGET_PROGRAMS = 256
DIRECT_SCAN_MAX_GRID_X = 8
DIRECT_SCAN_MAX_NPROBE = 128
MAX_SUBSPACES = 64
MAX_KSUB = 256
MAX_INDEXED_VECTORS = (1 << 31) - 1
_INF_KEY = 0x7F800000FFFFFFFF
_SUPPORTED_GLOBAL_LUT_DTYPES = {torch.float32, torch.float16}


if triton is not None:

    @triton.jit
    def _direct_interleaved_pq_topk_kernel(
        global_query_luts_ptr,
        static_vector_corrections_ptr,
        probe_distance_bias_ptr,
        codes_by_subspace_ptr,
        probe_group_prefix_ptr,
        probe_list_count_ptr,
        partial_keys_ptr,
        num_queries,
        num_code_vectors,
        nprobe,
        GRID_X: tl.constexpr,
        M: tl.constexpr,
        KSUB: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        SCAN_BLOCK_C: tl.constexpr,
        LOCAL_K: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Scan query-owned probe-list partitions without list bucketing.

        ``G[q,m,code]`` is read through the global/L2 cache.  It deliberately
        is not retained as a program-wide block tensor: on the common
        M=16/K=256 shape that would consume 16 KiB of compiler-managed on-chip
        storage and reduce occupancy.
        """
        partition_idx = tl.program_id(0)
        query_idx = tl.program_id(1)
        candidate_offsets = tl.arange(0, SCAN_BLOCK_C)
        group_offsets = candidate_offsets // GROUP_SIZE
        lane_offsets = candidate_offsets % GROUP_SIZE
        best_keys = tl.full((LOCAL_K,), INF_KEY, tl.int64)

        for probe_idx in tl.range(0, nprobe, 1):
            probe_offset = query_idx * nprobe + probe_idx
            list_count = tl.load(probe_list_count_ptr + probe_offset).to(tl.int32)
            group_prefix = tl.load(probe_group_prefix_ptr + probe_offset).to(tl.int64)
            distance_bias = tl.load(probe_distance_bias_ptr + probe_offset).to(tl.float32)
            group_count = (list_count + GROUP_SIZE - 1) // GROUP_SIZE
            owns_probe = (probe_idx % GRID_X) == partition_idx
            owned_group_count = tl.where(owns_probe, group_count, 0)

            for local_group_start in tl.range(0, owned_group_count, 4):
                local_group_ids = local_group_start + group_offsets
                local_vector_offsets = local_group_ids * GROUP_SIZE + lane_offsets
                valid_candidates = local_vector_offsets < list_count
                physical_positions = (
                    (group_prefix + local_group_ids.to(tl.int64)) * GROUP_SIZE + lane_offsets
                )
                # gamma is a one-use streaming scalar, so keep its traffic out
                # of the per-SM L1 cache together with G's random LUT gathers.
                distances = tl.load(
                    static_vector_corrections_ptr + physical_positions,
                    mask=valid_candidates,
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)
                for subspace_idx in tl.static_range(0, M):
                    codes = tl.load(
                        codes_by_subspace_ptr + subspace_idx * num_code_vectors + physical_positions,
                        mask=valid_candidates,
                        other=0,
                    ).to(tl.int64)
                    lut_values = tl.load(
                        global_query_luts_ptr
                        + subspace_idx * num_queries * KSUB
                        + query_idx * KSUB
                        + codes,
                        mask=valid_candidates,
                        other=0.0,
                        cache_modifier=".cg",
                    )
                    # FP16 LUTs are compact storage only. ADC remains FP32.
                    distances += lut_values.to(tl.float32)

                distances = tl.maximum(distances + distance_bias, 0.0)
                distance_bits = distances.to(tl.uint32, bitcast=True).to(tl.int64)
                candidate_keys = (distance_bits << 32) | physical_positions
                candidate_keys = tl.where(valid_candidates, candidate_keys, INF_KEY)
                if LOCAL_K <= GROUP_SIZE:
                    tile_best_keys = ~tl.topk(~candidate_keys, LOCAL_K)
                    merged_keys = tl.ravel(tl.join(best_keys, tile_best_keys))
                    best_keys = ~tl.topk(~merged_keys, LOCAL_K)
                else:
                    merged_keys = tl.ravel(tl.join(best_keys, candidate_keys))
                    best_keys = ~tl.topk(~merged_keys, LOCAL_K)

        tl.store(
            partial_keys_ptr
            + (query_idx * GRID_X + partition_idx) * LOCAL_K
            + tl.arange(0, LOCAL_K),
            best_keys,
        )


    @triton.jit
    def _finalize_direct_pq_topk_kernel(
        partial_keys_ptr,
        list_ids_interleaved_ptr,
        output_distances_ptr,
        output_ids_ptr,
        topk,
        GRID_X: tl.constexpr,
        LOCAL_K: tl.constexpr,
        FINAL_WIDTH: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Merge query partition-local candidates and emit IDs/distances."""
        query_idx = tl.program_id(0)
        offsets = tl.arange(0, FINAL_WIDTH)
        keys = tl.load(
            partial_keys_ptr + query_idx * GRID_X * LOCAL_K + offsets,
            mask=offsets < GRID_X * LOCAL_K,
            other=INF_KEY,
        )
        final_keys = ~tl.topk(~keys, LOCAL_K)

        ranks = tl.arange(0, LOCAL_K)
        physical_positions = (final_keys & 0xFFFFFFFF).to(tl.int64)
        distance_bits = (final_keys >> 32).to(tl.uint32)
        distances = distance_bits.to(tl.float32, bitcast=True)
        valid_result = (ranks < topk) & (final_keys < INF_KEY)
        ids = tl.load(list_ids_interleaved_ptr + physical_positions, mask=valid_result, other=-1)
        distances = tl.where(valid_result, distances, float("inf"))

        tl.store(output_distances_ptr + query_idx * topk + ranks, distances, mask=ranks < topk)
        tl.store(output_ids_ptr + query_idx * topk + ranks, ids, mask=ranks < topk)


def is_available(device: torch.device) -> bool:
    """Return whether the vector-correction direct scan can run on ``device``."""
    return triton is not None and device.type == "cuda"


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


@lru_cache(maxsize=None)
def _target_programs_for_device(device_index: int) -> int:
    try:
        properties = torch.cuda.get_device_properties(device_index)
        return max(1, int(properties.multi_processor_count) * DIRECT_SCAN_PROGRAMS_PER_SM)
    except (AssertionError, RuntimeError):  # pragma: no cover - CUDA unavailable in CPU tests
        return DIRECT_SCAN_FALLBACK_TARGET_PROGRAMS


def _target_programs(device: torch.device) -> int:
    if device.type != "cuda":
        return DIRECT_SCAN_FALLBACK_TARGET_PROGRAMS
    device_index = torch.cuda.current_device() if device.index is None else device.index
    return _target_programs_for_device(int(device_index))


def _direct_scan_plan(
    *,
    topk: int,
    num_queries: int,
    nprobe: int,
    target_programs: int = DIRECT_SCAN_FALLBACK_TARGET_PROGRAMS,
) -> tuple[int, int, int] | None:
    if (
        topk <= 0
        or num_queries <= 0
        or nprobe <= 0
        or nprobe > DIRECT_SCAN_MAX_NPROBE
        or target_programs <= 0
    ):
        return None
    local_k = max(
        2,
        _next_power_of_two(
            topk if topk <= INTERLEAVED_GROUP_SIZE else max(SCAN_BLOCK_CANDIDATES, topk)
        ),
    )
    if local_k > MAX_LOCAL_K:
        return None
    grid_x = min(
        nprobe,
        DIRECT_SCAN_MAX_GRID_X,
        max(1, math.ceil(target_programs / num_queries)),
    )
    final_width = _next_power_of_two(grid_x * local_k)
    if final_width > MAX_FINAL_WIDTH:
        return None
    return local_k, grid_x, final_width


def can_run(
    *,
    device: torch.device,
    topk: int,
    num_queries: int,
    m: int,
    ksub: int,
    nprobe: int,
    num_code_vectors: int,
    code_dtype: torch.dtype,
) -> bool:
    """Check CUDA, PQ, and launch-shape requirements for the direct scan."""
    return (
        is_available(device)
        and code_dtype == torch.uint8
        and 0 < m <= MAX_SUBSPACES
        and 0 < ksub <= MAX_KSUB
        and 0 < num_code_vectors <= MAX_INDEXED_VECTORS
        and _direct_scan_plan(
            topk=topk,
            num_queries=num_queries,
            nprobe=nprobe,
            target_programs=_target_programs(device),
        )
        is not None
    )


def triton_topk(
    *,
    global_query_luts: torch.Tensor,
    static_vector_corrections: torch.Tensor,
    probe_distance_bias: torch.Tensor,
    list_codes_interleaved: torch.Tensor,
    list_ids_interleaved: torch.Tensor,
    probe_group_prefix: torch.Tensor,
    probe_list_count: torch.Tensor,
    topk: int,
    output_distances: torch.Tensor,
    output_ids: torch.Tensor,
    partial_keys: torch.Tensor | None = None,
    on_direct_scan_complete: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Run the only Triton IVFPQ strategy: static-vector-correction ADC.

    ``global_query_luts`` may be FP32 or FP16. FP16 is produced from an FP32
    LUT build and is converted back to FP32 for accumulation in the kernel.
    """
    if global_query_luts.ndim != 3:
        raise ValueError("Global query LUT must have shape [m, query, ksub]")
    m, num_queries, ksub = global_query_luts.shape
    if global_query_luts.dtype not in _SUPPORTED_GLOBAL_LUT_DTYPES:
        raise ValueError("Global query LUT must use FP32 or FP16 storage")
    if probe_group_prefix.ndim != 2 or probe_group_prefix.shape != probe_list_count.shape:
        raise ValueError("Probe group-prefix/count tensors must have matching [query, nprobe] shapes")
    if probe_distance_bias.shape != probe_group_prefix.shape:
        raise ValueError("Probe distance bias must match [query, nprobe] probe metadata")
    nprobe = probe_group_prefix.shape[1]
    if probe_group_prefix.shape[0] != num_queries:
        raise ValueError("Probe metadata query count does not match the global query LUT")
    if not can_run(
        device=global_query_luts.device,
        topk=topk,
        num_queries=num_queries,
        m=m,
        ksub=ksub,
        nprobe=nprobe,
        num_code_vectors=list_ids_interleaved.numel(),
        code_dtype=list_codes_interleaved.dtype,
    ):
        raise ValueError("IVFPQ Triton top-k is unavailable for these search parameters")
    if list_codes_interleaved.shape != (m, list_ids_interleaved.numel()):
        raise ValueError("Interleaved PQ code cache must have shape [m, padded_num_indexed_vectors]")
    if static_vector_corrections.shape != (list_ids_interleaved.numel(),):
        raise ValueError("Static vector corrections must match padded interleaved vector count")
    if static_vector_corrections.dtype != torch.float32:
        raise ValueError("Static vector corrections must use FP32 storage")
    if output_distances.shape[0] != num_queries or output_distances.shape[1] < topk:
        raise ValueError("Output distance tensor must have shape [num_queries, >= topk]")
    if output_ids.shape[0] != num_queries or output_ids.shape[1] < topk:
        raise ValueError("Output ID tensor must have shape [num_queries, >= topk]")

    tensors = (
        global_query_luts,
        static_vector_corrections,
        probe_distance_bias,
        list_codes_interleaved,
        list_ids_interleaved,
        probe_group_prefix,
        probe_list_count,
        output_distances,
        output_ids,
    )
    if any(tensor.device != global_query_luts.device for tensor in tensors):
        raise ValueError("All Triton IVF-PQ tensors must reside on the same device")

    global_query_luts = global_query_luts.contiguous()
    static_vector_corrections = static_vector_corrections.contiguous()
    probe_distance_bias = probe_distance_bias.contiguous()
    list_codes_interleaved = list_codes_interleaved.contiguous()
    list_ids_interleaved = list_ids_interleaved.contiguous()
    probe_group_prefix = probe_group_prefix.contiguous()
    probe_list_count = probe_list_count.contiguous()

    plan = _direct_scan_plan(
        topk=topk,
        num_queries=num_queries,
        nprobe=nprobe,
        target_programs=_target_programs(global_query_luts.device),
    )
    if plan is None:  # pragma: no cover - guarded by can_run above
        raise ValueError("Unsupported IVFPQ Triton direct top-k plan")
    local_k, grid_x, final_width = plan
    partial_shape = (num_queries, grid_x, local_k)
    if (
        partial_keys is None
        or partial_keys.shape != partial_shape
        or partial_keys.dtype != torch.int64
        or partial_keys.device != global_query_luts.device
    ):
        partial_keys = torch.empty(partial_shape, device=global_query_luts.device, dtype=torch.int64)

    _direct_interleaved_pq_topk_kernel[(grid_x, num_queries)](
        global_query_luts,
        static_vector_corrections,
        probe_distance_bias,
        list_codes_interleaved,
        probe_group_prefix,
        probe_list_count,
        partial_keys,
        num_queries,
        list_ids_interleaved.numel(),
        nprobe,
        GRID_X=grid_x,
        M=m,
        KSUB=ksub,
        GROUP_SIZE=INTERLEAVED_GROUP_SIZE,
        SCAN_BLOCK_C=SCAN_BLOCK_CANDIDATES,
        LOCAL_K=local_k,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )
    if on_direct_scan_complete is not None:
        on_direct_scan_complete()
    _finalize_direct_pq_topk_kernel[(num_queries,)](
        partial_keys,
        list_ids_interleaved,
        output_distances,
        output_ids,
        topk,
        GRID_X=grid_x,
        LOCAL_K=local_k,
        FINAL_WIDTH=final_width,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )
    return partial_keys
