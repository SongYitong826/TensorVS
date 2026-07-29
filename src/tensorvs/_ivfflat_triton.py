"""Triton list-grouped and direct-scan top-k implementations for IVF-Flat."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from ._ivf_interleaved import (
    INTERLEAVED_GROUP_SIZE,
    interleaved_vector_width,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - covered by the caller fallback
    triton = None
    tl = None


MAX_LOCAL_K = 128
MAX_FINAL_WIDTH = 2048
DIRECT_SCAN_BLOCK_CANDIDATES = 128
LIST_GROUPED_QUERY_TILE = 8
LIST_GROUPED_VECTOR_TILE = 32
LIST_GROUPED_DIM_TILE = 32
LIST_GROUPED_MAX_FINAL_WIDTH = 2048
# Target two scan programs per SM.  The direct kernel is register-heavy, so
# this is deliberately lower than the theoretical maximum occupancy.  On the
# RTX 4090 used by the benchmark (128 SMs), it yields 256 programs: GRID_X=2
# for a 128-query batch and GRID_X=4 for a 64-query batch.
DIRECT_SCAN_PROGRAMS_PER_SM = 2
DIRECT_SCAN_FALLBACK_TARGET_PROGRAMS = 256
DIRECT_SCAN_MAX_GRID_X = 8
DIRECT_SCAN_MAX_NPROBE = 128
_INF_KEY = 0x7F800000FFFFFFFF


if triton is not None:

    @triton.jit
    def _direct_interleaved_topk_kernel(
        queries_ptr,
        interleaved_vectors_ptr,
        probe_group_prefix_ptr,
        probe_list_count_ptr,
        partial_keys_ptr,
        nprobe,
        GRID_X: tl.constexpr,
        SCAN_BLOCK_CANDIDATES: tl.constexpr,
        D: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        VECTOR_WIDTH: tl.constexpr,
        LOCAL_K: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Scan selected IVF lists directly, without materializing candidate positions.

        A program uses four warps to scan four 32-row interleaved groups at a
        time. Each lane owns one vector and therefore reads one contiguous
        ``veclen`` vector for every dimension chunk. ``best_keys`` persists
        across all lists assigned to the program.
        """
        partition_idx = tl.program_id(0)
        query_idx = tl.program_id(1)
        candidate_offsets = tl.arange(0, SCAN_BLOCK_CANDIDATES)
        group_offsets = candidate_offsets // GROUP_SIZE
        lane_offsets = candidate_offsets % GROUP_SIZE
        component_offsets = tl.arange(0, VECTOR_WIDTH)
        best_keys = tl.full((LOCAL_K,), INF_KEY, tl.int64)

        # Keep both loops runtime-controlled.  In particular, the cuVS scalar
        # fallback for a non-divisible dimension has one chunk per component;
        # statically unrolling that many chunks makes the direct fallback's
        # first compilation impractically large.
        for probe_idx in tl.range(0, nprobe, 1):
            probe_offset = query_idx * nprobe + probe_idx
            list_count = tl.load(probe_list_count_ptr + probe_offset).to(tl.int32)
            group_prefix = tl.load(probe_group_prefix_ptr + probe_offset).to(tl.int64)
            group_count = (list_count + GROUP_SIZE - 1) // GROUP_SIZE
            owns_probe = (probe_idx % GRID_X) == partition_idx

            # ``tl.range`` keeps the list length dynamic: empty or short
            # lists do no extra work, while the local top-k remains in
            # registers across every group assigned to this program.
            for local_group_start in tl.range(0, group_count, 4):
                local_group_ids = local_group_start + group_offsets
                local_vector_offsets = local_group_ids * GROUP_SIZE + lane_offsets
                valid_candidates = owns_probe & (local_vector_offsets < list_count)
                physical_positions = (group_prefix + local_group_ids.to(tl.int64)) * GROUP_SIZE + lane_offsets

                distances = tl.zeros((SCAN_BLOCK_CANDIDATES,), tl.float32)
                for chunk_idx in tl.range(0, NUM_CHUNKS, 1):
                    query_offsets = chunk_idx * VECTOR_WIDTH + component_offsets
                    valid_components = query_offsets < D
                    # The query load is loop-invariant with respect to
                    # ``local_group_start``. ``.ca`` keeps it cacheable
                    # without the register-heavy explicit broadcast tried
                    # here previously.
                    query_values = tl.load(
                        queries_ptr + query_idx * D + query_offsets,
                        mask=valid_components,
                        other=0.0,
                        cache_modifier=".ca",
                    ).to(tl.float32)
                    vector_values = tl.load(
                        interleaved_vectors_ptr
                        + (
                            (
                                (group_prefix + local_group_ids.to(tl.int64)) * NUM_CHUNKS
                                + chunk_idx
                            )
                            * GROUP_SIZE
                            + lane_offsets
                        )[:, None]
                        * VECTOR_WIDTH
                        + component_offsets[None, :],
                        mask=valid_candidates[:, None] & valid_components[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    delta = vector_values - query_values[None, :]
                    distances += tl.sum(delta * delta, axis=1)

                distance_bits = distances.to(tl.uint32, bitcast=True).to(tl.int64)
                candidate_keys = (distance_bits << 32) | physical_positions
                candidate_keys = tl.where(valid_candidates, candidate_keys, INF_KEY)
                if LOCAL_K <= GROUP_SIZE:
                    # For normal ANN k (for example k=10 -> LOCAL_K=16),
                    # reduce this 128-candidate tile to LOCAL_K before
                    # merging it with the running LOCAL_K. This is a
                    # 128 -> 16 -> 16 hierarchy, replacing the former full
                    # 256-wide top-128 sort.
                    tile_best_keys = ~tl.topk(~candidate_keys, LOCAL_K)
                    merged_keys = tl.ravel(tl.join(best_keys, tile_best_keys))
                    best_keys = ~tl.topk(~merged_keys, LOCAL_K)
                else:
                    # Retain the original wider merge for uncommon k > 32,
                    # which cannot be represented by a single 32-lane group.
                    merged_keys = tl.ravel(tl.join(best_keys, candidate_keys))
                    best_keys = ~tl.topk(~merged_keys, LOCAL_K)

        output_offsets = tl.arange(0, LOCAL_K)
        tl.store(
            partial_keys_ptr + (query_idx * GRID_X + partition_idx) * LOCAL_K + output_offsets,
            best_keys,
        )


    @triton.jit
    def _finalize_direct_topk_kernel(
        partial_keys_ptr,
        list_ids_interleaved_ptr,
        output_distances_ptr,
        output_ids_ptr,
        GRID_X: tl.constexpr,
        topk,
        LOCAL_K: tl.constexpr,
        FINAL_WIDTH: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
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


    @triton.jit
    def _list_grouped_gemm_topk_kernel(
        queries_ptr,
        interleaved_vectors_ptr,
        list_group_prefix_ptr,
        list_count_ptr,
        edge_qid_sorted_ptr,
        edge_sorted_to_original_ptr,
        list_query_prefix_ptr,
        list_query_count_ptr,
        partial_keys_ptr,
        D: tl.constexpr,
        NUM_DIM_TILES: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        VECTOR_WIDTH: tl.constexpr,
        QUERY_TILE: tl.constexpr,
        VECTOR_TILE: tl.constexpr,
        DIM_TILE: tl.constexpr,
        LOCAL_K: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Compute one list's ``C @ V.T`` blocks and retain per-row local top-k.

        The grid is list-major.  ``edge_qid_sorted`` provides the rows of C
        (the queries that probe this list), while one interleaved 32-vector
        group forms a column tile of V.  Keeping a query tile together lets a
        vector tile be reused by up to ``QUERY_TILE`` rows of C.
        """
        list_idx = tl.program_id(0)
        query_count = tl.load(list_query_count_ptr + list_idx).to(tl.int32)
        query_prefix = tl.load(list_query_prefix_ptr + list_idx).to(tl.int64)
        vector_count = tl.load(list_count_ptr + list_idx).to(tl.int32)
        group_prefix = tl.load(list_group_prefix_ptr + list_idx).to(tl.int64)
        group_count = (vector_count + GROUP_SIZE - 1) // GROUP_SIZE

        query_tile_offsets = tl.arange(0, QUERY_TILE)
        vector_offsets = tl.arange(0, VECTOR_TILE)
        dim_offsets = tl.arange(0, DIM_TILE)
        rank_offsets = tl.arange(0, LOCAL_K)

        # ``tl.range`` keeps the number of rows per list dynamic.  Empty
        # lists and lists that received no probes therefore perform no work.
        for query_tile_start in tl.range(0, query_count, QUERY_TILE):
            query_positions = query_tile_start + query_tile_offsets
            valid_queries = query_positions < query_count
            sorted_edge_positions = query_prefix + query_positions.to(tl.int64)
            query_ids = tl.load(
                edge_qid_sorted_ptr + sorted_edge_positions,
                mask=valid_queries,
                other=0,
            ).to(tl.int64)
            original_edge_positions = tl.load(
                edge_sorted_to_original_ptr + sorted_edge_positions,
                mask=valid_queries,
                other=0,
            ).to(tl.int64)

            # Load the query rows once to calculate their squared norms.  The
            # same rows become C in every following C @ V.T tile.
            query_norms = tl.zeros((QUERY_TILE,), tl.float32)
            for dim_tile_idx in tl.static_range(0, NUM_DIM_TILES):
                current_dim_offsets = dim_tile_idx * DIM_TILE + dim_offsets
                valid_dimensions = current_dim_offsets < D
                query_values = tl.load(
                    queries_ptr + query_ids[:, None] * D + current_dim_offsets[None, :],
                    mask=valid_queries[:, None] & valid_dimensions[None, :],
                    other=0.0,
                ).to(tl.float32)
                query_norms += tl.sum(query_values * query_values, axis=1)

            best_keys = tl.full((QUERY_TILE, LOCAL_K), INF_KEY, tl.int64)
            # A physical interleaved group contains exactly VECTOR_TILE=32
            # vectors.  This lets tl.dot evaluate a [QUERY_TILE, DIM_TILE] @
            # [DIM_TILE, VECTOR_TILE] block for each list-vector tile.
            for local_group_idx in tl.range(0, group_count, 1):
                valid_vectors = local_group_idx * GROUP_SIZE + vector_offsets < vector_count
                dot_products = tl.zeros((QUERY_TILE, VECTOR_TILE), tl.float32)
                vector_norms = tl.zeros((VECTOR_TILE,), tl.float32)
                physical_positions = (group_prefix + local_group_idx.to(tl.int64)) * GROUP_SIZE + vector_offsets

                for dim_tile_idx in tl.static_range(0, NUM_DIM_TILES):
                    current_dim_offsets = dim_tile_idx * DIM_TILE + dim_offsets
                    valid_dimensions = current_dim_offsets < D
                    dim_chunk_idx = current_dim_offsets // VECTOR_WIDTH
                    component_offsets = current_dim_offsets % VECTOR_WIDTH
                    query_values = tl.load(
                        queries_ptr + query_ids[:, None] * D + current_dim_offsets[None, :],
                        mask=valid_queries[:, None] & valid_dimensions[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    # The interleaved layout is [group, dim_chunk, lane,
                    # veclen].  Loading it as [DIM_TILE, VECTOR_TILE] makes
                    # it the right-hand operand of the matrix product.
                    vector_values = tl.load(
                        interleaved_vectors_ptr
                        + (
                            (
                                (group_prefix + local_group_idx.to(tl.int64)) * NUM_CHUNKS
                                + dim_chunk_idx[:, None]
                            )
                            * GROUP_SIZE
                            + vector_offsets[None, :]
                        )
                        * VECTOR_WIDTH
                        + component_offsets[:, None],
                        mask=valid_dimensions[:, None] & valid_vectors[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    dot_products = tl.dot(
                        query_values,
                        vector_values,
                        dot_products,
                        input_precision="ieee",
                    )
                    vector_norms += tl.sum(vector_values * vector_values, axis=0)

                distances = (
                    query_norms[:, None] + vector_norms[None, :] - 2.0 * dot_products
                ).to(tl.float32)
                distances = tl.maximum(distances, 0.0)
                distance_bits = distances.to(tl.uint32, bitcast=True).to(tl.int64)
                candidate_keys = (distance_bits << 32) | physical_positions[None, :]
                valid_candidates = valid_queries[:, None] & valid_vectors[None, :]
                candidate_keys = tl.where(valid_candidates, candidate_keys, INF_KEY)
                tile_best_keys = ~tl.topk(~candidate_keys, LOCAL_K)
                merged_keys = tl.reshape(
                    tl.join(best_keys, tile_best_keys),
                    (QUERY_TILE, 2 * LOCAL_K),
                )
                best_keys = ~tl.topk(~merged_keys, LOCAL_K)

            tl.store(
                partial_keys_ptr
                + original_edge_positions[:, None] * LOCAL_K
                + rank_offsets[None, :],
                best_keys,
                mask=valid_queries[:, None],
            )


    @triton.jit
    def _finalize_list_grouped_topk_kernel(
        partial_keys_ptr,
        list_ids_interleaved_ptr,
        output_distances_ptr,
        output_ids_ptr,
        nprobe,
        topk,
        LOCAL_K: tl.constexpr,
        FINAL_WIDTH: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        query_idx = tl.program_id(0)
        offsets = tl.arange(0, FINAL_WIDTH)
        total_partial_keys = nprobe * LOCAL_K
        keys = tl.load(
            partial_keys_ptr + query_idx * total_partial_keys + offsets,
            mask=offsets < total_partial_keys,
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
    """Return whether the custom IVFFlat kernel can run on ``device``."""
    return triton is not None and device.type == "cuda"


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


@lru_cache(maxsize=None)
def _direct_scan_target_programs_for_device(device_index: int) -> int:
    """Return the target direct-scan program count for one CUDA device."""
    try:
        properties = torch.cuda.get_device_properties(device_index)
        return max(1, int(properties.multi_processor_count) * DIRECT_SCAN_PROGRAMS_PER_SM)
    except (AssertionError, RuntimeError):  # pragma: no cover - CUDA unavailable in CPU tests
        return DIRECT_SCAN_FALLBACK_TARGET_PROGRAMS


def _direct_scan_target_programs(device: torch.device) -> int:
    if device.type != "cuda":
        return DIRECT_SCAN_FALLBACK_TARGET_PROGRAMS
    device_index = torch.cuda.current_device() if device.index is None else device.index
    return _direct_scan_target_programs_for_device(int(device_index))


def _direct_scan_plan(
    *,
    topk: int,
    num_queries: int,
    nprobe: int,
    target_programs: int = DIRECT_SCAN_FALLBACK_TARGET_PROGRAMS,
) -> tuple[int, int, int] | None:
    """Plan the direct interleaved scan without inspecting GPU tensor values."""
    if (
        topk <= 0
        or num_queries <= 0
        or nprobe <= 0
        or nprobe > DIRECT_SCAN_MAX_NPROBE
        or target_programs <= 0
    ):
        return None
    # For k <= 32 a local partition needs only the requested top-k, rounded
    # to the power of two required by ``tl.topk``. Larger k retain the former
    # 128-wide merge, because one scan tile contains 128 candidates.
    local_k = max(
        2,
        _next_power_of_two(
            topk if topk <= INTERLEAVED_GROUP_SIZE else max(DIRECT_SCAN_BLOCK_CANDIDATES, topk)
        ),
    )
    if local_k > MAX_LOCAL_K:
        return None

    # Keep two programs per SM available for a normal query batch, while
    # allowing each program to retain its local top-k across many lists.
    grid_x = min(
        nprobe,
        DIRECT_SCAN_MAX_GRID_X,
        max(1, math.ceil(target_programs / num_queries)),
    )
    final_width = _next_power_of_two(grid_x * local_k)
    if final_width > MAX_FINAL_WIDTH:
        return None
    return local_k, grid_x, final_width


def _list_grouped_gemm_plan(
    *,
    topk: int,
    nprobe: int,
    dim: int,
) -> tuple[int, int] | None:
    """Plan the list-grouped ``C @ V.T`` path without reading CUDA values.

    One grouped program keeps a local top-k for every row in a small C tile.
    The final kernel then merges one such result from every probed list.  The
    vector tile is one 32-row interleaved group, so k > 32 uses the direct
    scan fallback instead of an invalid local reduction.
    """
    if (
        topk <= 0
        or nprobe <= 0
        or nprobe > DIRECT_SCAN_MAX_NPROBE
        or dim <= 0
    ):
        return None
    local_k = max(2, _next_power_of_two(topk))
    if local_k > LIST_GROUPED_VECTOR_TILE:
        return None
    final_width = _next_power_of_two(nprobe * local_k)
    if final_width > LIST_GROUPED_MAX_FINAL_WIDTH:
        return None
    return local_k, final_width


def can_run_topk(
    *,
    device: torch.device,
    topk: int,
    num_queries: int,
    nprobe: int,
    dim: int,
) -> bool:
    """Check the direct 32-lane interleaved scan kernel requirements."""
    return (
        is_available(device)
        and dim > 0
        and _direct_scan_plan(
            topk=topk,
            num_queries=num_queries,
            nprobe=nprobe,
            target_programs=_direct_scan_target_programs(device),
        )
        is not None
    )


def _run_direct_interleaved_topk(
    *,
    queries: torch.Tensor,
    list_vectors_interleaved: torch.Tensor,
    list_ids_interleaved: torch.Tensor,
    probe_group_prefix: torch.Tensor,
    probe_list_count: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the legacy query-major direct scan as a grouped-path fallback."""
    num_queries, nprobe = probe_list_count.shape
    dim = int(queries.shape[1])
    _, num_chunks, _, vector_width = list_vectors_interleaved.shape
    plan = _direct_scan_plan(
        topk=topk,
        num_queries=num_queries,
        nprobe=nprobe,
        target_programs=_direct_scan_target_programs(queries.device),
    )
    if plan is None:  # pragma: no cover - guarded by can_run_topk above
        raise ValueError("Unsupported IVFFlat Triton top-k plan")
    local_k, grid_x, final_width = plan
    partial_keys = torch.empty(
        (num_queries, grid_x, local_k),
        device=queries.device,
        dtype=torch.int64,
    )
    _direct_interleaved_topk_kernel[(grid_x, num_queries)](
        queries,
        list_vectors_interleaved,
        probe_group_prefix,
        probe_list_count,
        partial_keys,
        nprobe,
        GRID_X=grid_x,
        SCAN_BLOCK_CANDIDATES=DIRECT_SCAN_BLOCK_CANDIDATES,
        D=dim,
        NUM_CHUNKS=num_chunks,
        GROUP_SIZE=INTERLEAVED_GROUP_SIZE,
        VECTOR_WIDTH=vector_width,
        LOCAL_K=local_k,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )

    output_distances = torch.empty((num_queries, topk), device=queries.device, dtype=torch.float32)
    output_ids = torch.empty((num_queries, topk), device=queries.device, dtype=torch.int64)
    _finalize_direct_topk_kernel[(num_queries,)](
        partial_keys,
        list_ids_interleaved,
        output_distances,
        output_ids,
        GRID_X=grid_x,
        topk=topk,
        LOCAL_K=local_k,
        FINAL_WIDTH=final_width,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )
    return output_distances, output_ids


def _run_list_grouped_gemm_topk(
    *,
    queries: torch.Tensor,
    list_vectors_interleaved: torch.Tensor,
    list_ids_interleaved: torch.Tensor,
    list_group_prefix: torch.Tensor,
    list_count: torch.Tensor,
    probe_cids: torch.Tensor,
    topk: int,
    plan: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group probe edges by list, then evaluate list-local C @ V.T tiles."""
    num_queries, nprobe = probe_cids.shape
    dim = int(queries.shape[1])
    num_lists = int(list_count.numel())
    num_chunks = int(list_vectors_interleaved.shape[1])
    vector_width = int(list_vectors_interleaved.shape[3])
    local_k, final_width = plan

    # All preprocessing stays on the GPU and has fixed-size outputs.  ``perm``
    # maps list-major rows back to their query-major [query, probe] slot, so
    # the final reduction can be one contiguous program per query.
    edge_query_ids = torch.arange(
        num_queries,
        dtype=torch.int64,
        device=queries.device,
    ).repeat_interleave(nprobe)
    edge_cids = probe_cids.reshape(-1).to(torch.int64)
    _, edge_sorted_to_original = torch.sort(edge_cids)
    edge_query_ids_sorted = edge_query_ids[edge_sorted_to_original]
    list_query_count = torch.bincount(edge_cids, minlength=num_lists).to(torch.int64)
    list_query_prefix = torch.cumsum(list_query_count, dim=0) - list_query_count

    partial_keys = torch.full(
        (num_queries * nprobe, local_k),
        _INF_KEY,
        dtype=torch.int64,
        device=queries.device,
    )
    _list_grouped_gemm_topk_kernel[(num_lists,)](
        queries,
        list_vectors_interleaved,
        list_group_prefix,
        list_count,
        edge_query_ids_sorted,
        edge_sorted_to_original,
        list_query_prefix,
        list_query_count,
        partial_keys,
        D=dim,
        NUM_DIM_TILES=(dim + LIST_GROUPED_DIM_TILE - 1) // LIST_GROUPED_DIM_TILE,
        NUM_CHUNKS=num_chunks,
        GROUP_SIZE=INTERLEAVED_GROUP_SIZE,
        VECTOR_WIDTH=vector_width,
        QUERY_TILE=LIST_GROUPED_QUERY_TILE,
        VECTOR_TILE=LIST_GROUPED_VECTOR_TILE,
        DIM_TILE=LIST_GROUPED_DIM_TILE,
        LOCAL_K=local_k,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )

    output_distances = torch.empty((num_queries, topk), device=queries.device, dtype=torch.float32)
    output_ids = torch.empty((num_queries, topk), device=queries.device, dtype=torch.int64)
    _finalize_list_grouped_topk_kernel[(num_queries,)](
        partial_keys,
        list_ids_interleaved,
        output_distances,
        output_ids,
        nprobe,
        topk,
        LOCAL_K=local_k,
        FINAL_WIDTH=final_width,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )
    return output_distances, output_ids


def triton_topk(
    *,
    queries: torch.Tensor,
    list_vectors_interleaved: torch.Tensor,
    list_ids_interleaved: torch.Tensor,
    list_group_prefix: torch.Tensor,
    list_count: torch.Tensor,
    probe_cids: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact IVF-Flat top-k with GPU list-grouped C @ V.T blocks.

    Probe edges are grouped by list on the device.  Each list program forms
    query tiles C from the visiting queries, multiplies them with vector tiles
    from that list, and emits only each row's local top-k.  Unsupported
    dimensions or wide final merges retain the existing direct-scan path.
    """
    if probe_cids.ndim != 2:
        raise ValueError("Probe centroid IDs must have shape [queries, nprobe]")
    num_queries, nprobe = probe_cids.shape
    dim = int(queries.shape[1])
    if queries.shape[0] != num_queries:
        raise ValueError("Query count does not match probe metadata")
    if (
        list_group_prefix.ndim != 1
        or list_count.ndim != 1
        or list_group_prefix.shape != list_count.shape
    ):
        raise ValueError("List group prefixes and counts must share shape [nlist]")
    if not can_run_topk(
        device=queries.device,
        topk=topk,
        num_queries=num_queries,
        nprobe=nprobe,
        dim=dim,
    ):
        raise ValueError("IVFFlat Triton top-k is unavailable for these search parameters")
    if (
        list_vectors_interleaved.ndim != 4
        or list_vectors_interleaved.shape[2] != INTERLEAVED_GROUP_SIZE
    ):
        raise ValueError("Interleaved vectors must have shape [groups, chunks, 32, veclen]")
    num_groups, num_chunks, _, vector_width = list_vectors_interleaved.shape
    expected_vector_width = interleaved_vector_width(dim, list_vectors_interleaved.dtype)
    expected_num_chunks = dim // expected_vector_width
    if vector_width != expected_vector_width or num_chunks != expected_num_chunks:
        raise ValueError("Interleaved vector layout does not match the query dimension")
    if list_ids_interleaved.numel() != num_groups * INTERLEAVED_GROUP_SIZE:
        raise ValueError("Interleaved ID cache must contain one ID per padded vector lane")
    if not queries.is_contiguous():
        queries = queries.contiguous()
    if not list_vectors_interleaved.is_contiguous():
        list_vectors_interleaved = list_vectors_interleaved.contiguous()
    if not list_ids_interleaved.is_contiguous():
        list_ids_interleaved = list_ids_interleaved.contiguous()
    if not list_group_prefix.is_contiguous():
        list_group_prefix = list_group_prefix.contiguous()
    if not list_count.is_contiguous():
        list_count = list_count.contiguous()
    if not probe_cids.is_contiguous():
        probe_cids = probe_cids.contiguous()

    grouped_plan = _list_grouped_gemm_plan(topk=topk, nprobe=nprobe, dim=dim)
    if grouped_plan is not None:
        return _run_list_grouped_gemm_topk(
            queries=queries,
            list_vectors_interleaved=list_vectors_interleaved,
            list_ids_interleaved=list_ids_interleaved,
            list_group_prefix=list_group_prefix,
            list_count=list_count,
            probe_cids=probe_cids,
            topk=topk,
            plan=grouped_plan,
        )

    probe_group_prefix = list_group_prefix[probe_cids]
    probe_list_count = list_count[probe_cids]
    return _run_direct_interleaved_topk(
        queries=queries,
        list_vectors_interleaved=list_vectors_interleaved,
        list_ids_interleaved=list_ids_interleaved,
        probe_group_prefix=probe_group_prefix,
        probe_list_count=probe_list_count,
        topk=topk,
    )
