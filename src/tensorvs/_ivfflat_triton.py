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
# Top-k values above ``MAX_LOCAL_K`` use the scan-at-fixed-width path below.
# Keep this bound a power of two so the public path covers the requested
# practical range (including K < 10,000) while retaining a finite allocation
# guard for accidentally unbounded requests.
MAX_SUPPORTED_TOPK = 16_384
MAX_FINAL_WIDTH = 2048
DIRECT_SCAN_BLOCK_CANDIDATES = 128
LIST_GROUPED_QUERY_TILE = 8
LIST_GROUPED_VECTOR_TILE = 32
LIST_GROUPED_DIM_TILE = 32
# The regular grouped GEMM kernel keeps a statically bounded dimension tile
# loop. Higher-dimensional indexes use the chunked task kernel below, whose
# dimension loop is runtime-controlled so high-dimensional vectors do not
# cause a huge statically-unrolled kernel.
LIST_GROUPED_MAX_DIM_TILES = 8
LIST_GROUPED_MAX_FINAL_WIDTH = 2048
# Each active list is partitioned into bounded vector-group chunks.  Combined
# with its active query tiles, this yields exactly the number of Triton
# programs needed for that list's current search work, regardless of dataset.
LIST_GROUPED_TASK_VECTOR_GROUPS = 8
# Large-K scans retain all candidates from a bounded four-group tile.  The
# resulting 128-key rows are merged by the generic hierarchical reducer, so
# the requested K never becomes a register-resident Triton dimension.
LARGE_K_GROUPS_PER_CHUNK = MAX_LOCAL_K // LIST_GROUPED_VECTOR_TILE
LARGE_K_QUERY_BATCH = 32
LARGE_K_MERGE_WIDTH = 32_768
# Target two scan programs per SM.  The direct kernel is register-heavy, so
# this is deliberately lower than the theoretical maximum occupancy.  On the
# RTX 4090 used by the benchmark (128 SMs), it yields 256 programs: GRID_X=2
# for a 128-query batch and GRID_X=4 for a 64-query batch.
DIRECT_SCAN_PROGRAMS_PER_SM = 2
DIRECT_SCAN_FALLBACK_TARGET_PROGRAMS = 256
DIRECT_SCAN_MAX_GRID_X = 8
# The direct kernel partitions the probe lists across GRID_X programs, so its
# final merge width depends on GRID_X rather than on the full nprobe value.
# Keep the planner generous enough for the configured IVF nlist values.
DIRECT_SCAN_MAX_NPROBE = 4096
_INF_KEY = 0x7F800000FFFFFFFF


if triton is not None:

    @triton.jit
    def _merge_group_candidate_topk(
        best_keys,
        candidate_keys,
        QUERY_TILE: tl.constexpr,
        VECTOR_TILE: tl.constexpr,
        LOCAL_K: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Merge one vector tile into a per-query local top-k.

        A vector tile contains 32 candidates, while larger requested K values
        use LOCAL_K=64 or 128.  Pad the candidate tile to LOCAL_K before the
        merge so every tl.topk invocation still receives a power-of-two width.
        """
        if LOCAL_K <= VECTOR_TILE:
            tile_best_keys = ~tl.topk(~candidate_keys, LOCAL_K)
            merged_keys = tl.reshape(
                tl.join(best_keys, tile_best_keys),
                (QUERY_TILE, 2 * LOCAL_K),
            )
        else:
            padded_candidates = tl.cat(
                candidate_keys,
                tl.full((QUERY_TILE, VECTOR_TILE), INF_KEY, tl.int64),
                dim=1,
            )
            if LOCAL_K > 2 * VECTOR_TILE:
                padded_candidates = tl.cat(
                    padded_candidates,
                    tl.full((QUERY_TILE, 2 * VECTOR_TILE), INF_KEY, tl.int64),
                    dim=1,
                )
            merged_keys = tl.reshape(
                tl.join(best_keys, padded_candidates),
                (QUERY_TILE, 2 * LOCAL_K),
            )
        return ~tl.topk(~merged_keys, LOCAL_K)

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
        interleaved_vector_norms_ptr,
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
        USE_PRECOMPUTED_NORMS: tl.constexpr,
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
                physical_positions = (group_prefix + local_group_idx.to(tl.int64)) * GROUP_SIZE + vector_offsets
                if USE_PRECOMPUTED_NORMS:
                    vector_norms = tl.load(
                        interleaved_vector_norms_ptr + physical_positions,
                        mask=valid_vectors,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    vector_norms = tl.zeros((VECTOR_TILE,), tl.float32)

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
                    if not USE_PRECOMPUTED_NORMS:
                        vector_norms += tl.sum(vector_values * vector_values, axis=0)

                distances = (
                    query_norms[:, None] + vector_norms[None, :] - 2.0 * dot_products
                ).to(tl.float32)
                distances = tl.maximum(distances, 0.0)
                distance_bits = distances.to(tl.uint32, bitcast=True).to(tl.int64)
                candidate_keys = (distance_bits << 32) | physical_positions[None, :]
                valid_candidates = valid_queries[:, None] & valid_vectors[None, :]
                candidate_keys = tl.where(valid_candidates, candidate_keys, INF_KEY)
                best_keys = _merge_group_candidate_topk(
                    best_keys,
                    candidate_keys,
                    QUERY_TILE=QUERY_TILE,
                    VECTOR_TILE=VECTOR_TILE,
                    LOCAL_K=LOCAL_K,
                    INF_KEY=INF_KEY,
                )

            tl.store(
                partial_keys_ptr
                + original_edge_positions[:, None] * LOCAL_K
                + rank_offsets[None, :],
                best_keys,
                mask=valid_queries[:, None],
            )


    @triton.jit
    def _list_grouped_chunked_gemm_topk_kernel(
        queries_ptr,
        interleaved_vectors_ptr,
        list_group_prefix_ptr,
        list_count_ptr,
        edge_qid_sorted_ptr,
        edge_sorted_to_original_ptr,
        list_query_prefix_ptr,
        list_query_count_ptr,
        chunk_partial_keys_ptr,
        D: tl.constexpr,
        NUM_DIM_TILES: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        VECTOR_WIDTH: tl.constexpr,
        QUERY_TILE: tl.constexpr,
        VECTOR_TILE: tl.constexpr,
        DIM_TILE: tl.constexpr,
        GROUPS_PER_CHUNK: tl.constexpr,
        MAX_VECTOR_CHUNKS: tl.constexpr,
        LOCAL_K: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Evaluate one bounded vector-group run for one IVF list.

        The first program dimension is the IVF list and the second is a
        chunk of consecutive interleaved 32-vector groups.  Query tiles are
        retained so a vector tile remains shared by up to ``QUERY_TILE``
        queries.  Each query/list edge receives a local top-k for this run;
        a follow-up kernel merges its runs exactly.
        """
        list_idx = tl.program_id(0)
        vector_chunk_idx = tl.program_id(1)
        query_count = tl.load(list_query_count_ptr + list_idx).to(tl.int32)
        query_prefix = tl.load(list_query_prefix_ptr + list_idx).to(tl.int64)
        vector_count = tl.load(list_count_ptr + list_idx).to(tl.int32)
        group_prefix = tl.load(list_group_prefix_ptr + list_idx).to(tl.int64)
        group_count = (vector_count + GROUP_SIZE - 1) // GROUP_SIZE
        group_start = vector_chunk_idx * GROUPS_PER_CHUNK
        group_end = tl.minimum(group_start + GROUPS_PER_CHUNK, group_count)

        # Keep out-of-range list chunks entirely empty.  This is important
        # because the launch grid is rectangular while IVF lists are ragged.
        query_count = tl.where(group_start < group_count, query_count, 0)

        query_tile_offsets = tl.arange(0, QUERY_TILE)
        vector_offsets = tl.arange(0, VECTOR_TILE)
        dim_offsets = tl.arange(0, DIM_TILE)
        rank_offsets = tl.arange(0, LOCAL_K)

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
            for local_group_idx in tl.range(group_start, group_end, 1):
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
                best_keys = _merge_group_candidate_topk(
                    best_keys,
                    candidate_keys,
                    QUERY_TILE=QUERY_TILE,
                    VECTOR_TILE=VECTOR_TILE,
                    LOCAL_K=LOCAL_K,
                    INF_KEY=INF_KEY,
                )

            tl.store(
                chunk_partial_keys_ptr
                + (
                    original_edge_positions[:, None] * MAX_VECTOR_CHUNKS
                    + vector_chunk_idx
                )
                * LOCAL_K
                + rank_offsets[None, :],
                best_keys,
                mask=valid_queries[:, None],
            )


    @triton.jit
    def _u8_centered_norms_kernel(
        vectors_ptr, norms_ptr, N: tl.constexpr, D: tl.constexpr,
        VECTOR_WIDTH: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        dim = tl.arange(0, D)
        addr = (
            ((row[None, :].to(tl.int64) // GROUP * (D // VECTOR_WIDTH) + dim[:, None] // VECTOR_WIDTH)
             * GROUP + row[None, :] % GROUP) * VECTOR_WIDTH
            + dim[:, None] % VECTOR_WIDTH
        )
        values = tl.load(vectors_ptr + addr, row[None, :] < N, other=128).to(tl.int32) - 128
        tl.store(norms_ptr + row, tl.sum(values * values, axis=0), row < N)


    @triton.jit
    def _list_grouped_tasked_gemm_topk_kernel(
        queries_ptr,
        interleaved_vectors_ptr,
        interleaved_vector_norms_ptr,
        list_group_prefix_ptr,
        list_count_ptr,
        edge_qid_sorted_ptr,
        edge_sorted_to_original_ptr,
        list_query_prefix_ptr,
        list_query_count_ptr,
        task_list_ids_ptr,
        task_query_tile_ids_ptr,
        task_vector_chunk_ids_ptr,
        chunk_partial_keys_ptr,
        D: tl.constexpr,
        NUM_DIM_TILES: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        VECTOR_WIDTH: tl.constexpr,
        QUERY_TILE: tl.constexpr,
        VECTOR_TILE: tl.constexpr,
        DIM_TILE: tl.constexpr,
        GROUPS_PER_CHUNK: tl.constexpr,
        MAX_VECTOR_CHUNKS: tl.constexpr,
        USE_PRECOMPUTED_NORMS: tl.constexpr,
        INTEGER_SCAN: tl.constexpr,
        LOCAL_K: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Run one compact ``(list, query tile, vector-group chunk)`` task."""
        task_idx = tl.program_id(0)
        list_idx = tl.load(task_list_ids_ptr + task_idx).to(tl.int64)
        query_tile_idx = tl.load(task_query_tile_ids_ptr + task_idx).to(tl.int32)
        vector_chunk_idx = tl.load(task_vector_chunk_ids_ptr + task_idx).to(tl.int32)

        query_count = tl.load(list_query_count_ptr + list_idx).to(tl.int32)
        query_prefix = tl.load(list_query_prefix_ptr + list_idx).to(tl.int64)
        vector_count = tl.load(list_count_ptr + list_idx).to(tl.int32)
        group_prefix = tl.load(list_group_prefix_ptr + list_idx).to(tl.int64)
        group_count = (vector_count + GROUP_SIZE - 1) // GROUP_SIZE
        group_start = vector_chunk_idx * GROUPS_PER_CHUNK
        group_end = tl.minimum(group_start + GROUPS_PER_CHUNK, group_count)

        query_tile_offsets = tl.arange(0, QUERY_TILE)
        vector_offsets = tl.arange(0, VECTOR_TILE)
        dim_offsets = tl.arange(0, DIM_TILE)
        rank_offsets = tl.arange(0, LOCAL_K)
        query_positions = query_tile_idx * QUERY_TILE + query_tile_offsets
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

        ACC_DTYPE: tl.constexpr = tl.int32 if INTEGER_SCAN else tl.float32
        query_norms = tl.zeros((QUERY_TILE,), ACC_DTYPE)
        # This is deliberately a runtime loop.  A static loop over every
        # dimension tile makes compilation and register allocation
        # unnecessarily expensive.  Every iteration still evaluates a fixed
        # [QUERY_TILE, DIM_TILE] @ [DIM_TILE, VECTOR_TILE] GEMM tile.
        for dim_tile_idx in tl.range(0, NUM_DIM_TILES, 1):
            current_dim_offsets = dim_tile_idx * DIM_TILE + dim_offsets
            valid_dimensions = current_dim_offsets < D
            query_values = tl.load(
                queries_ptr + query_ids[:, None] * D + current_dim_offsets[None, :],
                mask=valid_queries[:, None] & valid_dimensions[None, :],
                other=0.0,
            ).to(ACC_DTYPE)
            if INTEGER_SCAN:
                query_values -= 128
            query_norms += tl.sum(query_values * query_values, axis=1)

        best_keys = tl.full((QUERY_TILE, LOCAL_K), INF_KEY, tl.int64)
        for local_group_idx in tl.range(group_start, group_end, 1):
            valid_vectors = local_group_idx * GROUP_SIZE + vector_offsets < vector_count
            dot_products = tl.zeros((QUERY_TILE, VECTOR_TILE), ACC_DTYPE)
            physical_positions = (group_prefix + local_group_idx.to(tl.int64)) * GROUP_SIZE + vector_offsets
            if USE_PRECOMPUTED_NORMS:
                vector_norms = tl.load(
                    interleaved_vector_norms_ptr + physical_positions,
                    mask=valid_vectors,
                    other=0.0,
                ).to(ACC_DTYPE)
            else:
                vector_norms = tl.zeros((VECTOR_TILE,), ACC_DTYPE)

            for dim_tile_idx in tl.range(0, NUM_DIM_TILES, 1):
                current_dim_offsets = dim_tile_idx * DIM_TILE + dim_offsets
                valid_dimensions = current_dim_offsets < D
                dim_chunk_idx = current_dim_offsets // VECTOR_WIDTH
                component_offsets = current_dim_offsets % VECTOR_WIDTH
                query_values = tl.load(
                    queries_ptr + query_ids[:, None] * D + current_dim_offsets[None, :],
                    mask=valid_queries[:, None] & valid_dimensions[None, :],
                    other=0.0,
                )
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
                )
                if INTEGER_SCAN:
                    # Flip the sign bit to represent each byte minus 128 as int8.
                    q = (query_values ^ 128).to(tl.uint8).to(tl.int8, bitcast=True)
                    x = (vector_values ^ 128).to(tl.uint8).to(tl.int8, bitcast=True)
                    dot_products = tl.dot(q, x, dot_products, out_dtype=tl.int32)
                else:
                    dot_products = tl.dot(
                        query_values.to(tl.float32),
                        vector_values.to(tl.float32),
                        dot_products,
                        input_precision="ieee",
                    )
                if not USE_PRECOMPUTED_NORMS:
                    vectors_f32 = vector_values.to(tl.float32)
                    vector_norms += tl.sum(vectors_f32 * vectors_f32, axis=0)

            if INTEGER_SCAN:
                # Both operands and norms use the same translation, preserving squared L2.
                distances = (query_norms[:, None] + vector_norms[None, :] - 2 * dot_products).to(tl.float32)
            else:
                distances = (query_norms[:, None] + vector_norms[None, :] - 2.0 * dot_products).to(tl.float32)
                distances = tl.maximum(distances, 0.0)
            distance_bits = distances.to(tl.uint32, bitcast=True).to(tl.int64)
            candidate_keys = (distance_bits << 32) | physical_positions[None, :]
            valid_candidates = valid_queries[:, None] & valid_vectors[None, :]
            candidate_keys = tl.where(valid_candidates, candidate_keys, INF_KEY)
            best_keys = _merge_group_candidate_topk(
                best_keys,
                candidate_keys,
                QUERY_TILE=QUERY_TILE,
                VECTOR_TILE=VECTOR_TILE,
                LOCAL_K=LOCAL_K,
                INF_KEY=INF_KEY,
            )

        tl.store(
            chunk_partial_keys_ptr
            + (
                original_edge_positions[:, None] * MAX_VECTOR_CHUNKS
                + vector_chunk_idx
            )
            * LOCAL_K
            + rank_offsets[None, :],
            best_keys,
            mask=valid_queries[:, None],
        )


    @triton.jit
    def _merge_list_chunk_topk_kernel(
        chunk_partial_keys_ptr,
        list_partial_keys_ptr,
        MERGE_WIDTH: tl.constexpr,
        MAX_VECTOR_CHUNKS: tl.constexpr,
        LOCAL_K: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Merge an IVF edge's chunk-local results into its exact local top-k."""
        edge_idx = tl.program_id(0)
        offsets = tl.arange(0, MERGE_WIDTH)
        chunk_key_count = MAX_VECTOR_CHUNKS * LOCAL_K
        keys = tl.load(
            chunk_partial_keys_ptr + edge_idx * chunk_key_count + offsets,
            mask=offsets < chunk_key_count,
            other=INF_KEY,
        )
        best_keys = ~tl.topk(~keys, LOCAL_K)
        ranks = tl.arange(0, LOCAL_K)
        tl.store(
            list_partial_keys_ptr + edge_idx * LOCAL_K + ranks,
            best_keys,
        )


    @triton.jit
    def _merge_list_chunk_groups_topk_kernel(
        chunk_partial_keys_ptr,
        list_partial_keys_ptr,
        MAX_VECTOR_CHUNKS: tl.constexpr,
        CHUNKS_PER_MERGE: tl.constexpr,
        NUM_MERGE_GROUPS: tl.constexpr,
        MERGE_WIDTH: tl.constexpr,
        LOCAL_K: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Merge bounded vector-chunk groups, retaining one row per group."""
        edge_idx = tl.program_id(0)
        merge_group_idx = tl.program_id(1)
        chunk_start = merge_group_idx * CHUNKS_PER_MERGE
        remaining_chunks = MAX_VECTOR_CHUNKS - chunk_start
        chunk_count = tl.minimum(CHUNKS_PER_MERGE, remaining_chunks)
        offsets = tl.arange(0, MERGE_WIDTH)
        keys = tl.load(
            chunk_partial_keys_ptr
            + (
                edge_idx * MAX_VECTOR_CHUNKS * LOCAL_K
                + chunk_start * LOCAL_K
                + offsets
            ),
            mask=offsets < chunk_count * LOCAL_K,
            other=INF_KEY,
        )
        best_keys = ~tl.topk(~keys, LOCAL_K)
        ranks = tl.arange(0, LOCAL_K)
        tl.store(
            list_partial_keys_ptr
            + (edge_idx * NUM_MERGE_GROUPS + merge_group_idx) * LOCAL_K
            + ranks,
            best_keys,
        )


    @triton.jit
    def _merge_probe_chunk_topk_kernel(
        list_partial_keys_ptr,
        query_partial_keys_ptr,
        nprobe,
        num_probe_chunks,
        PROBE_CHUNK_SIZE: tl.constexpr,
        LOCAL_K: tl.constexpr,
        MERGE_WIDTH: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Merge a bounded probe chunk for each query."""
        query_idx = tl.program_id(0)
        probe_chunk_idx = tl.program_id(1)
        offsets = tl.arange(0, MERGE_WIDTH)
        probe_offsets = probe_chunk_idx * PROBE_CHUNK_SIZE + offsets // LOCAL_K
        rank_offsets = offsets % LOCAL_K
        valid_probes = probe_offsets < nprobe
        keys = tl.load(
            list_partial_keys_ptr
            + (query_idx * nprobe + probe_offsets) * LOCAL_K
            + rank_offsets,
            mask=valid_probes,
            other=INF_KEY,
        )
        best_keys = ~tl.topk(~keys, LOCAL_K)
        output_ranks = tl.arange(0, LOCAL_K)
        tl.store(
            query_partial_keys_ptr
            + (query_idx * num_probe_chunks + probe_chunk_idx) * LOCAL_K
            + output_ranks,
            best_keys,
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
    vector tile is one 32-row interleaved group. Larger local K values are
    supported by padding each tile before the local reduction.
    """
    local_k = _list_grouped_local_topk_plan(topk=topk, nprobe=nprobe, dim=dim)
    if local_k is None:
        return None
    final_width = _next_power_of_two(nprobe * local_k)
    if final_width > LIST_GROUPED_MAX_FINAL_WIDTH:
        return None
    return local_k, final_width


def _list_grouped_local_topk_plan(
    *,
    topk: int,
    nprobe: int,
    dim: int,
) -> int | None:
    """Plan the per-list grouped scan without planning its final merge."""
    if (
        topk <= 0
        or nprobe <= 0
        or nprobe > DIRECT_SCAN_MAX_NPROBE
        or dim <= 0
        or math.ceil(dim / LIST_GROUPED_DIM_TILE) > LIST_GROUPED_MAX_DIM_TILES
    ):
        return None
    local_k = max(2, _next_power_of_two(topk))
    if local_k > MAX_LOCAL_K:
        return None
    return local_k


def _list_grouped_task_plan(
    *,
    topk: int,
    nprobe: int,
    dim: int,
    max_list_group_count: int,
) -> tuple[int, int, int] | None:
    """Return the exact adaptive list-task plan, or ``None`` when unsuitable."""
    # Unlike the one-program-per-list kernel, the task kernel processes a
    # bounded number of vector groups at a time.  It can therefore use a
    # runtime dimension loop and is also the high-dimensional path.
    local_k = _list_grouped_task_local_topk_plan(topk=topk, nprobe=nprobe, dim=dim)
    if local_k is None or max_list_group_count <= 0:
        return None

    max_vector_chunks = math.ceil(max_list_group_count / LIST_GROUPED_TASK_VECTOR_GROUPS)
    max_merge_chunks = max(1, LIST_GROUPED_MAX_FINAL_WIDTH // local_k)
    merge_width = _next_power_of_two(
        min(max_vector_chunks, max_merge_chunks) * local_k
    )
    return local_k, max_vector_chunks, merge_width


def _list_grouped_task_local_topk_plan(
    *,
    topk: int,
    nprobe: int,
    dim: int,
) -> int | None:
    """Plan the chunked grouped scan, including dimensions above 256."""
    if (
        topk <= 0
        or nprobe <= 0
        or nprobe > DIRECT_SCAN_MAX_NPROBE
        or dim <= 0
    ):
        return None
    local_k = max(2, _next_power_of_two(topk))
    if local_k > MAX_LOCAL_K:
        return None
    return local_k


def _list_grouped_large_k_task_plan(
    *,
    topk: int,
    nprobe: int,
    dim: int,
    max_list_group_count: int,
) -> tuple[int, int] | None:
    """Plan a fixed-width GEMM scan for K values wider than local TopK.

    Four 32-vector groups exactly fill the 128-key local row.  No candidate
    is discarded in that scan tile; later PyTorch TopK stages reduce those
    rows hierarchically to the requested K.
    """
    if (
        topk <= MAX_LOCAL_K
        or topk > MAX_SUPPORTED_TOPK
        or nprobe <= 0
        or nprobe > DIRECT_SCAN_MAX_NPROBE
        or dim <= 0
        or max_list_group_count <= 0
    ):
        return None
    max_vector_chunks = math.ceil(max_list_group_count / LARGE_K_GROUPS_PER_CHUNK)
    return MAX_LOCAL_K, max_vector_chunks


def can_run_topk(
    *,
    device: torch.device,
    topk: int,
    num_queries: int,
    nprobe: int,
    dim: int,
) -> bool:
    """Check whether a Triton IVF-Flat scan plan is available.

    K values above the register-local limit are handled by the fixed-width
    grouped GEMM path, so this check must not delegate to the direct-scan
    planner.
    """
    return (
        is_available(device)
        and dim > 0
        and num_queries > 0
        and 0 < nprobe <= DIRECT_SCAN_MAX_NPROBE
        and 0 < topk <= MAX_SUPPORTED_TOPK
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


def _merge_list_chunks_hierarchically(
    *,
    chunk_partial_keys: torch.Tensor,
    num_edges: int,
    max_vector_chunks: int,
    local_k: int,
) -> tuple[torch.Tensor, int]:
    """Merge vector chunks in bounded groups and return their row count."""
    total_width = max_vector_chunks * local_k
    if total_width <= LIST_GROUPED_MAX_FINAL_WIDTH:
        list_partial_keys = torch.empty(
            (num_edges, local_k),
            dtype=torch.int64,
            device=chunk_partial_keys.device,
        )
        _merge_list_chunk_topk_kernel[(num_edges,)](
            chunk_partial_keys,
            list_partial_keys,
            MERGE_WIDTH=_next_power_of_two(total_width),
            MAX_VECTOR_CHUNKS=max_vector_chunks,
            LOCAL_K=local_k,
            INF_KEY=_INF_KEY,
            num_warps=4,
        )
        return list_partial_keys, 1

    chunks_per_merge = max(1, LIST_GROUPED_MAX_FINAL_WIDTH // local_k)
    num_merge_groups = math.ceil(max_vector_chunks / chunks_per_merge)
    merge_width = _next_power_of_two(chunks_per_merge * local_k)
    list_partial_keys = torch.empty(
        (num_edges * num_merge_groups, local_k),
        dtype=torch.int64,
        device=chunk_partial_keys.device,
    )
    _merge_list_chunk_groups_topk_kernel[(num_edges, num_merge_groups)](
        chunk_partial_keys,
        list_partial_keys,
        MAX_VECTOR_CHUNKS=max_vector_chunks,
        CHUNKS_PER_MERGE=chunks_per_merge,
        NUM_MERGE_GROUPS=num_merge_groups,
        MERGE_WIDTH=merge_width,
        LOCAL_K=local_k,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )
    return list_partial_keys, num_merge_groups


def _merge_probe_results_hierarchically(
    *,
    partial_keys: torch.Tensor,
    num_queries: int,
    probe_count: int,
    local_k: int,
) -> tuple[torch.Tensor, int, int]:
    """Reduce probe rows until the final merge fits in one TopK tile."""
    final_partial_keys = partial_keys
    final_probe_count = int(probe_count)
    while _next_power_of_two(final_probe_count * local_k) > LIST_GROUPED_MAX_FINAL_WIDTH:
        probes_per_merge = max(1, LIST_GROUPED_MAX_FINAL_WIDTH // local_k)
        num_probe_chunks = math.ceil(final_probe_count / probes_per_merge)
        probe_merge_width = _next_power_of_two(probes_per_merge * local_k)
        query_partial_keys = torch.full(
            (num_queries * num_probe_chunks, local_k),
            _INF_KEY,
            dtype=torch.int64,
            device=final_partial_keys.device,
        )
        _merge_probe_chunk_topk_kernel[(num_queries, num_probe_chunks)](
            final_partial_keys,
            query_partial_keys,
            final_probe_count,
            num_probe_chunks,
            PROBE_CHUNK_SIZE=probes_per_merge,
            LOCAL_K=local_k,
            MERGE_WIDTH=probe_merge_width,
            INF_KEY=_INF_KEY,
            num_warps=4,
        )
        final_partial_keys = query_partial_keys
        final_probe_count = num_probe_chunks

    final_width = _next_power_of_two(final_probe_count * local_k)
    return final_partial_keys, final_probe_count, final_width


def _hierarchical_key_topk(
    keys: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select K ordered keys with bounded intermediate TopK widths.

    Each intermediate group retains ``min(topk, group_width)`` keys.  That is
    exact: if a group is narrower than K, every key is retained; otherwise no
    more than K keys from that group can reach the global top K.  This keeps
    large K out of Triton register tiles while still supporting arbitrary
    candidate counts on the GPU.
    """
    if keys.ndim == 0 or keys.shape[-1] <= 0:
        raise ValueError("Hierarchical TopK requires a non-empty last dimension")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    keys = keys.contiguous()
    width = int(keys.shape[-1])
    while width > LARGE_K_MERGE_WIDTH:
        group_count = math.ceil(width / LARGE_K_MERGE_WIDTH)
        padded_width = group_count * LARGE_K_MERGE_WIDTH
        if padded_width != width:
            padding = torch.full(
                (*keys.shape[:-1], padded_width - width),
                _INF_KEY,
                dtype=keys.dtype,
                device=keys.device,
            )
            keys = torch.cat((keys, padding), dim=-1)
        grouped = keys.reshape(
            (*keys.shape[:-1], group_count, LARGE_K_MERGE_WIDTH)
        )
        group_topk = min(topk, LARGE_K_MERGE_WIDTH)
        keys = torch.topk(
            grouped,
            k=group_topk,
            dim=-1,
            largest=False,
            sorted=True,
        ).values
        keys = keys.reshape(
            (*keys.shape[:-2], group_count * group_topk)
        )
        width = group_count * group_topk

    return torch.topk(
        keys,
        k=min(topk, width),
        dim=-1,
        largest=False,
        sorted=True,
    ).values


def _decode_key_topk(
    keys: torch.Tensor,
    *,
    topk: int,
    list_ids_interleaved: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode packed distance/physical-position keys into search outputs."""
    num_queries = int(keys.shape[0])
    output_distances = torch.full(
        (num_queries, topk),
        float("inf"),
        dtype=torch.float32,
        device=keys.device,
    )
    output_ids = torch.full(
        (num_queries, topk),
        -1,
        dtype=torch.int64,
        device=keys.device,
    )
    selected_k = min(topk, int(keys.shape[-1]))
    selected_keys = keys[:, :selected_k]
    physical_positions = (selected_keys & 0xFFFFFFFF).to(torch.int64)
    safe_positions = physical_positions.clamp_max(list_ids_interleaved.numel() - 1)
    distance_bits = (selected_keys >> 32).to(torch.uint32)
    distances = distance_bits.view(torch.float32)
    valid_result = selected_keys < _INF_KEY
    ids = list_ids_interleaved[safe_positions]
    ids = ids.masked_fill(~valid_result, -1)
    distances = distances.masked_fill(~valid_result, float("inf"))
    output_distances[:, :selected_k] = distances
    output_ids[:, :selected_k] = ids
    return output_distances, output_ids


def _run_list_grouped_large_k_gemm_topk(
    *,
    queries: torch.Tensor,
    list_vectors_interleaved: torch.Tensor,
    list_vector_norms_interleaved: torch.Tensor | None,
    list_ids_interleaved: torch.Tensor,
    list_group_prefix: torch.Tensor,
    list_count: torch.Tensor,
    probe_cids: torch.Tensor,
    topk: int,
    plan: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan all large-K candidates with fixed-width task GEMM tiles.

    The task kernel retains all 128 candidates in each four-group tile.  The
    two following GPU TopK reductions first collapse vector chunks per probe
    list and then collapse probe lists per query, both with bounded stages.
    """
    local_k, max_vector_chunks = plan
    chunk_partial_keys = _run_list_grouped_tasked_gemm_topk(
        queries=queries,
        list_vectors_interleaved=list_vectors_interleaved,
        list_vector_norms_interleaved=list_vector_norms_interleaved,
        list_ids_interleaved=list_ids_interleaved,
        list_group_prefix=list_group_prefix,
        list_count=list_count,
        probe_cids=probe_cids,
        topk=local_k,
        plan=(local_k, max_vector_chunks, 0),
        groups_per_chunk=LARGE_K_GROUPS_PER_CHUNK,
        return_chunk_partials=True,
    )
    assert isinstance(chunk_partial_keys, torch.Tensor)

    num_queries, nprobe = probe_cids.shape
    chunk_width = max_vector_chunks * local_k
    edge_keys = chunk_partial_keys.reshape(num_queries * nprobe, chunk_width)
    list_keys = _hierarchical_key_topk(edge_keys, topk).reshape(
        num_queries,
        nprobe,
        -1,
    )
    query_keys = list_keys.reshape(num_queries, -1)
    final_keys = _hierarchical_key_topk(query_keys, topk)
    return _decode_key_topk(
        final_keys,
        topk=topk,
        list_ids_interleaved=list_ids_interleaved,
    )


def _run_list_grouped_gemm_topk(
    *,
    queries: torch.Tensor,
    list_vectors_interleaved: torch.Tensor,
    list_vector_norms_interleaved: torch.Tensor | None,
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
    use_precomputed_norms = list_vector_norms_interleaved is not None
    norm_cache = list_vector_norms_interleaved
    if norm_cache is None:
        # This pointer is compile-time unused by the fallback specialization.
        norm_cache = list_vectors_interleaved

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
        norm_cache,
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
        USE_PRECOMPUTED_NORMS=use_precomputed_norms,
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


def _run_list_grouped_chunked_gemm_topk(
    *,
    queries: torch.Tensor,
    list_vectors_interleaved: torch.Tensor,
    list_ids_interleaved: torch.Tensor,
    list_group_prefix: torch.Tensor,
    list_count: torch.Tensor,
    probe_cids: torch.Tensor,
    topk: int,
    plan: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan long IVF lists in parallel vector-group chunks and merge exactly."""
    num_queries, nprobe = probe_cids.shape
    dim = int(queries.shape[1])
    num_lists = int(list_count.numel())
    num_chunks = int(list_vectors_interleaved.shape[1])
    vector_width = int(list_vectors_interleaved.shape[3])
    local_k, max_vector_chunks, _ = plan

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

    num_edges = num_queries * nprobe
    chunk_partial_keys = torch.full(
        (num_edges, max_vector_chunks, local_k),
        _INF_KEY,
        dtype=torch.int64,
        device=queries.device,
    )
    _list_grouped_chunked_gemm_topk_kernel[(num_lists, max_vector_chunks)](
        queries,
        list_vectors_interleaved,
        list_group_prefix,
        list_count,
        edge_query_ids_sorted,
        edge_sorted_to_original,
        list_query_prefix,
        list_query_count,
        chunk_partial_keys,
        D=dim,
        NUM_DIM_TILES=(dim + LIST_GROUPED_DIM_TILE - 1) // LIST_GROUPED_DIM_TILE,
        NUM_CHUNKS=num_chunks,
        GROUP_SIZE=INTERLEAVED_GROUP_SIZE,
        VECTOR_WIDTH=vector_width,
        QUERY_TILE=LIST_GROUPED_QUERY_TILE,
        VECTOR_TILE=LIST_GROUPED_VECTOR_TILE,
        DIM_TILE=LIST_GROUPED_DIM_TILE,
        GROUPS_PER_CHUNK=LIST_GROUPED_TASK_VECTOR_GROUPS,
        MAX_VECTOR_CHUNKS=max_vector_chunks,
        LOCAL_K=local_k,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )

    list_partial_keys, vector_chunk_merge_groups = _merge_list_chunks_hierarchically(
        chunk_partial_keys=chunk_partial_keys,
        num_edges=num_edges,
        max_vector_chunks=max_vector_chunks,
        local_k=local_k,
    )

    output_distances = torch.empty((num_queries, topk), device=queries.device, dtype=torch.float32)
    output_ids = torch.empty((num_queries, topk), device=queries.device, dtype=torch.int64)
    final_partial_keys, final_nprobe, final_width = _merge_probe_results_hierarchically(
        partial_keys=list_partial_keys,
        num_queries=num_queries,
        probe_count=nprobe * vector_chunk_merge_groups,
        local_k=local_k,
    )
    _finalize_list_grouped_topk_kernel[(num_queries,)](
        final_partial_keys,
        list_ids_interleaved,
        output_distances,
        output_ids,
        final_nprobe,
        topk,
        LOCAL_K=local_k,
        FINAL_WIDTH=final_width,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )
    return output_distances, output_ids


def _run_list_grouped_tasked_gemm_topk(
    *,
    queries: torch.Tensor,
    list_vectors_interleaved: torch.Tensor,
    list_vector_norms_interleaved: torch.Tensor | None,
    list_ids_interleaved: torch.Tensor,
    list_group_prefix: torch.Tensor,
    list_count: torch.Tensor,
    probe_cids: torch.Tensor,
    topk: int,
    plan: tuple[int, int, int],
    groups_per_chunk: int = LIST_GROUPED_TASK_VECTOR_GROUPS,
    return_chunk_partials: bool = False,
    integer_scan: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Use one Triton program per active list/query-tile/group-chunk task."""
    num_queries, nprobe = probe_cids.shape
    dim = int(queries.shape[1])
    num_lists = int(list_count.numel())
    num_chunks = int(list_vectors_interleaved.shape[1])
    vector_width = int(list_vectors_interleaved.shape[3])
    local_k, max_vector_chunks, _ = plan
    use_precomputed_norms = list_vector_norms_interleaved is not None
    norm_cache = list_vector_norms_interleaved
    if norm_cache is None:
        # The pointer is compile-time unused in this specialization.
        norm_cache = list_vectors_interleaved

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

    list_group_count = torch.div(
        list_count + (INTERLEAVED_GROUP_SIZE - 1),
        INTERLEAVED_GROUP_SIZE,
        rounding_mode="floor",
    )
    list_query_tile_count = torch.div(
        list_query_count + (LIST_GROUPED_QUERY_TILE - 1),
        LIST_GROUPED_QUERY_TILE,
        rounding_mode="floor",
    )
    list_vector_chunk_count = torch.div(
        list_group_count + (groups_per_chunk - 1),
        groups_per_chunk,
        rounding_mode="floor",
    )
    task_count_per_list = list_query_tile_count * list_vector_chunk_count
    task_prefix = torch.cumsum(task_count_per_list, dim=0) - task_count_per_list
    num_tasks = int(task_count_per_list.sum().item())

    if num_tasks == 0:  # pragma: no cover - non-empty probes normally have work
        if return_chunk_partials:
            return torch.full(
                (num_queries * nprobe, max_vector_chunks, local_k),
                _INF_KEY,
                dtype=torch.int64,
                device=queries.device,
            )
        output_distances = torch.full(
            (num_queries, topk),
            float("inf"),
            dtype=torch.float32,
            device=queries.device,
        )
        output_ids = torch.full(
            (num_queries, topk),
            -1,
            dtype=torch.int64,
            device=queries.device,
        )
        return output_distances, output_ids

    task_list_ids = torch.repeat_interleave(
        torch.arange(num_lists, device=queries.device, dtype=torch.int64),
        task_count_per_list,
        output_size=num_tasks,
    )
    task_positions = torch.arange(num_tasks, device=queries.device, dtype=torch.int64)
    task_local_positions = task_positions - task_prefix[task_list_ids]
    task_vector_chunk_counts = list_vector_chunk_count[task_list_ids]
    task_query_tile_ids = torch.div(
        task_local_positions,
        task_vector_chunk_counts,
        rounding_mode="floor",
    )
    task_vector_chunk_ids = torch.remainder(task_local_positions, task_vector_chunk_counts)

    num_edges = num_queries * nprobe
    chunk_partial_keys = torch.full(
        (num_edges, max_vector_chunks, local_k),
        _INF_KEY,
        dtype=torch.int64,
        device=queries.device,
    )
    _list_grouped_tasked_gemm_topk_kernel[(num_tasks,)](
        queries,
        list_vectors_interleaved,
        norm_cache,
        list_group_prefix,
        list_count,
        edge_query_ids_sorted,
        edge_sorted_to_original,
        list_query_prefix,
        list_query_count,
        task_list_ids,
        task_query_tile_ids,
        task_vector_chunk_ids,
        chunk_partial_keys,
        D=dim,
        NUM_DIM_TILES=(dim + LIST_GROUPED_DIM_TILE - 1) // LIST_GROUPED_DIM_TILE,
        NUM_CHUNKS=num_chunks,
        GROUP_SIZE=INTERLEAVED_GROUP_SIZE,
        VECTOR_WIDTH=vector_width,
        QUERY_TILE=LIST_GROUPED_QUERY_TILE,
        VECTOR_TILE=LIST_GROUPED_VECTOR_TILE,
        DIM_TILE=LIST_GROUPED_DIM_TILE,
        GROUPS_PER_CHUNK=groups_per_chunk,
        MAX_VECTOR_CHUNKS=max_vector_chunks,
        USE_PRECOMPUTED_NORMS=use_precomputed_norms,
        INTEGER_SCAN=integer_scan,
        LOCAL_K=local_k,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )

    if return_chunk_partials:
        return chunk_partial_keys

    list_partial_keys, vector_chunk_merge_groups = _merge_list_chunks_hierarchically(
        chunk_partial_keys=chunk_partial_keys,
        num_edges=num_edges,
        max_vector_chunks=max_vector_chunks,
        local_k=local_k,
    )

    output_distances = torch.empty((num_queries, topk), device=queries.device, dtype=torch.float32)
    output_ids = torch.empty((num_queries, topk), device=queries.device, dtype=torch.int64)
    final_partial_keys, final_nprobe, final_width = _merge_probe_results_hierarchically(
        partial_keys=list_partial_keys,
        num_queries=num_queries,
        probe_count=nprobe * vector_chunk_merge_groups,
        local_k=local_k,
    )
    _finalize_list_grouped_topk_kernel[(num_queries,)](
        final_partial_keys,
        list_ids_interleaved,
        output_distances,
        output_ids,
        final_nprobe,
        topk,
        LOCAL_K=local_k,
        FINAL_WIDTH=final_width,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )
    return output_distances, output_ids


def can_use_uint8_scan(queries: torch.Tensor, vectors: torch.Tensor, topk: int) -> bool:
    """Check the native CUDA integer scan's supported input and tile shape."""
    return (
        queries.dtype == vectors.dtype == torch.uint8
        and queries.shape[1] == 128
        and topk <= MAX_LOCAL_K
        and queries.device.type == "cuda"
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(queries.device)[0] >= 8
    )


def prepare_uint8_centered_norms(vectors: torch.Tensor) -> torch.Tensor:
    """Compute one int32 squared norm per interleaved row on the resident device."""
    vectors = vectors.contiguous()
    rows = vectors.shape[0] * INTERLEAVED_GROUP_SIZE
    norms = torch.empty(rows, dtype=torch.int32, device=vectors.device)
    if rows:
        _u8_centered_norms_kernel[(triton.cdiv(rows, 128),)](
            vectors,
            norms,
            N=rows,
            D=128,
            VECTOR_WIDTH=vectors.shape[3],
            GROUP=INTERLEAVED_GROUP_SIZE,
            BLOCK=128,
            num_warps=4,
        )
    return norms


def triton_topk(
    *,
    queries: torch.Tensor,
    list_vectors_interleaved: torch.Tensor,
    list_ids_interleaved: torch.Tensor,
    list_group_prefix: torch.Tensor,
    list_count: torch.Tensor,
    probe_cids: torch.Tensor,
    topk: int,
    list_vector_norms_interleaved: torch.Tensor | None = None,
    max_list_group_count: int | None = None,
    uint8_centered_norms: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact IVF-Flat top-k with GPU list-grouped C @ V.T blocks.

    Probe edges are grouped by list on the device.  Each list program forms
    query tiles C from the visiting queries, multiplies them with vector tiles
    from that list, and emits only each row's local top-k.  Wide final merges
    are reduced hierarchically in bounded chunks; unsupported shapes retain
    the existing direct-scan path.
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

    # Large-K reductions materialize fixed-width scan rows.  Bound the query
    # slice so the temporary key buffers do not scale without limit with an
    # unusually large Q_BATCH; the normal K<=128 path remains one launch.
    if topk > MAX_LOCAL_K and num_queries > LARGE_K_QUERY_BATCH:
        distance_parts = []
        id_parts = []
        for query_start in range(0, num_queries, LARGE_K_QUERY_BATCH):
            query_end = min(query_start + LARGE_K_QUERY_BATCH, num_queries)
            part_distances, part_ids = triton_topk(
                queries=queries[query_start:query_end],
                list_vectors_interleaved=list_vectors_interleaved,
                list_ids_interleaved=list_ids_interleaved,
                list_group_prefix=list_group_prefix,
                list_count=list_count,
                probe_cids=probe_cids[query_start:query_end],
                topk=topk,
                list_vector_norms_interleaved=list_vector_norms_interleaved,
                max_list_group_count=max_list_group_count,
            )
            distance_parts.append(part_distances)
            id_parts.append(part_ids)
        return torch.cat(distance_parts, dim=0), torch.cat(id_parts, dim=0)

    if max_list_group_count is None:
        max_list_group_count = int(
            torch.div(
                list_count.max() + (INTERLEAVED_GROUP_SIZE - 1),
                INTERLEAVED_GROUP_SIZE,
                rounding_mode="floor",
            ).item()
        )
    if max_list_group_count <= 0:
        return (
            torch.full(
                (num_queries, topk),
                float("inf"),
                dtype=torch.float32,
                device=queries.device,
            ),
            torch.full(
                (num_queries, topk),
                -1,
                dtype=torch.int64,
                device=queries.device,
            ),
        )

    large_k_plan = _list_grouped_large_k_task_plan(
        topk=topk,
        nprobe=nprobe,
        dim=dim,
        max_list_group_count=max_list_group_count,
    )
    if large_k_plan is not None:
        return _run_list_grouped_large_k_gemm_topk(
            queries=queries,
            list_vectors_interleaved=list_vectors_interleaved,
            list_vector_norms_interleaved=list_vector_norms_interleaved,
            list_ids_interleaved=list_ids_interleaved,
            list_group_prefix=list_group_prefix,
            list_count=list_count,
            probe_cids=probe_cids,
            topk=topk,
            plan=large_k_plan,
        )

    grouped_plan = _list_grouped_gemm_plan(topk=topk, nprobe=nprobe, dim=dim)
    task_plan = _list_grouped_task_plan(
        topk=topk,
        nprobe=nprobe,
        dim=dim,
        max_list_group_count=max_list_group_count,
    )
    if task_plan is not None:
        return _run_list_grouped_tasked_gemm_topk(
            queries=queries,
            list_vectors_interleaved=list_vectors_interleaved,
            list_vector_norms_interleaved=(
                list_vector_norms_interleaved if uint8_centered_norms is None else uint8_centered_norms
            ),
            list_ids_interleaved=list_ids_interleaved,
            list_group_prefix=list_group_prefix,
            list_count=list_count,
            probe_cids=probe_cids,
            topk=topk,
            plan=task_plan,
            integer_scan=uint8_centered_norms is not None,
        )
    if grouped_plan is not None:
        return _run_list_grouped_gemm_topk(
            queries=queries,
            list_vectors_interleaved=list_vectors_interleaved,
            list_vector_norms_interleaved=list_vector_norms_interleaved,
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
