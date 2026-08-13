"""Triton list-grouped scan for residual IVF scalar quantization."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - caller falls back to Tensor search
    triton = None
    tl = None


GROUP_SIZE = 32
QUERY_TILE = 8
VECTOR_TILE = 32
DIM_TILE = 128
GROUPS_PER_CHUNK = 8
MAX_LOCAL_K = 32
MAX_FINAL_WIDTH = 2048
MAX_NPROBE = 128
DIRECT_MAX_GRID_X = 8
DIRECT_PROGRAMS_PER_SM = 2
_INF_KEY = 0x7F800000FFFFFFFF


if triton is not None:

    @triton.jit
    def _sq_query_major_topk_kernel(
        queries_ptr,
        residuals_by_dim_ptr,
        vector_norms_ptr,
        sq_base_ptr,
        probe_cids_ptr,
        list_group_prefix_ptr,
        list_count_ptr,
        partial_keys_ptr,
        nprobe,
        PADDED_VECTORS,
        GROUP_SIZE: tl.constexpr,
        DIM_TILE: tl.constexpr,
        GRID_X: tl.constexpr,
        D: tl.constexpr,
        LOCAL_K: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Scan one query partition, matching cuVS's query-major launch.

        A program owns one query and every ``GRID_X``-th probe.  It keeps the
        complete local top-k in registers, so no query/list edge expansion or
        per-task partial buffer is needed.  The query/list invariant terms are
        loaded once per probe and reused for all 32-row code groups.
        """
        partition_idx = tl.program_id(0)
        query_idx = tl.program_id(1)
        dim_offsets = tl.arange(0, DIM_TILE)
        vector_offsets = tl.arange(0, GROUP_SIZE)
        output_offsets = tl.arange(0, LOCAL_K)
        best_keys = tl.full((LOCAL_K,), INF_KEY, tl.int64)

        query_values = tl.load(
            queries_ptr + query_idx * D + dim_offsets,
            mask=dim_offsets < D,
            other=0.0,
        ).to(tl.float32)

        for probe_idx in tl.range(partition_idx, nprobe, GRID_X):
            probe_offset = query_idx * nprobe + probe_idx
            list_idx = tl.load(probe_cids_ptr + probe_offset).to(tl.int64)
            vector_count = tl.load(list_count_ptr + list_idx).to(tl.int32)
            group_prefix = tl.load(list_group_prefix_ptr + list_idx).to(tl.int64)
            group_count = (vector_count + (GROUP_SIZE - 1)) // GROUP_SIZE

            base_values = tl.load(
                sq_base_ptr + list_idx * D + dim_offsets,
                mask=dim_offsets < D,
                other=0.0,
            ).to(tl.float32)
            query_delta = query_values - base_values
            query_term = query_delta
            query_norm = tl.sum(query_delta * query_delta, axis=0)

            for group_idx in tl.range(0, group_count, 1):
                physical_positions = (
                    (group_prefix + group_idx.to(tl.int64)) * GROUP_SIZE
                    + vector_offsets
                )
                valid_vectors = group_idx * GROUP_SIZE + vector_offsets < vector_count
                vector_norms = tl.load(
                    vector_norms_ptr + physical_positions,
                    mask=valid_vectors,
                    other=0.0,
                ).to(tl.float32)
                code_values = tl.load(
                    residuals_by_dim_ptr
                    + dim_offsets[:, None].to(tl.int64) * PADDED_VECTORS
                    + physical_positions[None, :],
                    mask=(dim_offsets < D)[:, None] & valid_vectors[None, :],
                    other=0,
                ).to(tl.float32)
                dot_products = tl.ravel(
                    tl.dot(
                        query_term[None, :],
                        code_values,
                        input_precision="ieee",
                    )
                )
                distances = query_norm + vector_norms - 2.0 * dot_products
                distances = tl.maximum(distances, 0.0)
                distance_bits = distances.to(tl.uint32, bitcast=True).to(tl.int64)
                candidate_keys = (distance_bits << 32) | physical_positions
                candidate_keys = tl.where(valid_vectors, candidate_keys, INF_KEY)
                tile_best_keys = ~tl.topk(~candidate_keys, LOCAL_K)
                merged_keys = tl.ravel(tl.join(best_keys, tile_best_keys))
                best_keys = ~tl.topk(~merged_keys, LOCAL_K)

        tl.store(
            partial_keys_ptr
            + (query_idx * GRID_X + partition_idx) * LOCAL_K
            + output_offsets,
            best_keys,
        )

    @triton.jit
    def _sq_tasked_topk_kernel(
        queries_ptr,
        codes_by_dim_ptr,
        vector_norms_ptr,
        sq_base_ptr,
        sq_scale_ptr,
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
        GROUP_SIZE: tl.constexpr,
        D: tl.constexpr,
        NUM_DIM_TILES: tl.constexpr,
        PADDED_VECTORS,
        QUERY_TILE: tl.constexpr,
        VECTOR_TILE: tl.constexpr,
        DIM_TILE: tl.constexpr,
        GROUPS_PER_CHUNK: tl.constexpr,
        MAX_VECTOR_CHUNKS: tl.constexpr,
        LOCAL_K: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Run one ``(list, query tile, code-group chunk)`` task.

        The resident code cache uses a ``[dimension,
        padded_physical_vector]`` uint8 layout.  Codes are converted to FP16
        only for the tensor-core dot product, while the list scale is applied
        once per query/dimension tile.
        """
        task_idx = tl.program_id(0)
        list_idx = tl.load(task_list_ids_ptr + task_idx).to(tl.int64)
        query_tile_idx = tl.load(task_query_tile_ids_ptr + task_idx).to(tl.int32)
        vector_chunk_idx = tl.load(task_vector_chunk_ids_ptr + task_idx).to(tl.int32)

        query_count = tl.load(list_query_count_ptr + list_idx).to(tl.int32)
        query_prefix = tl.load(list_query_prefix_ptr + list_idx).to(tl.int64)
        vector_count = tl.load(list_count_ptr + list_idx).to(tl.int32)
        group_prefix = tl.load(list_group_prefix_ptr + list_idx).to(tl.int64)
        group_count = (vector_count + (GROUP_SIZE - 1)) // GROUP_SIZE
        group_start = vector_chunk_idx * GROUPS_PER_CHUNK
        group_end = tl.minimum(group_start + GROUPS_PER_CHUNK, group_count)

        query_offsets = tl.arange(0, QUERY_TILE)
        vector_offsets = tl.arange(0, VECTOR_TILE)
        dim_offsets = tl.arange(0, DIM_TILE)
        rank_offsets = tl.arange(0, LOCAL_K)
        query_positions = query_tile_idx * QUERY_TILE + query_offsets
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
            base_values = tl.load(
                sq_base_ptr + list_idx * D + current_dim_offsets,
                mask=valid_dimensions,
                other=0.0,
            ).to(tl.float32)
            query_delta = query_values - base_values[None, :]
            query_norms += tl.sum(query_delta * query_delta, axis=1)

        best_keys = tl.full((QUERY_TILE, LOCAL_K), INF_KEY, tl.int64)
        for local_group_idx in tl.range(group_start, group_end, 1):
            valid_vectors = local_group_idx * GROUP_SIZE + vector_offsets < vector_count
            physical_positions = (
                group_prefix + local_group_idx.to(tl.int64)
            ) * GROUP_SIZE + vector_offsets
            vector_norms = tl.load(
                vector_norms_ptr + physical_positions,
                mask=valid_vectors,
                other=0.0,
            ).to(tl.float32)
            dot_products = tl.zeros((QUERY_TILE, VECTOR_TILE), tl.float32)

            for dim_tile_idx in tl.static_range(0, NUM_DIM_TILES):
                current_dim_offsets = dim_tile_idx * DIM_TILE + dim_offsets
                valid_dimensions = current_dim_offsets < D
                query_values = tl.load(
                    queries_ptr + query_ids[:, None] * D + current_dim_offsets[None, :],
                    mask=valid_queries[:, None] & valid_dimensions[None, :],
                    other=0.0,
                ).to(tl.float32)
                base_values = tl.load(
                    sq_base_ptr + list_idx * D + current_dim_offsets,
                    mask=valid_dimensions,
                    other=0.0,
                ).to(tl.float32)
                query_delta = query_values - base_values[None, :]

                code_values = tl.load(
                    codes_by_dim_ptr
                    + current_dim_offsets[:, None].to(tl.int64) * PADDED_VECTORS
                    + physical_positions[None, :],
                    mask=valid_dimensions[:, None] & valid_vectors[None, :],
                    other=0.0,
                ).to(tl.float32)
                scale_values = tl.load(
                    sq_scale_ptr + list_idx * D + current_dim_offsets,
                    mask=valid_dimensions,
                    other=1.0,
                ).to(tl.float32)
                dot_products = tl.dot(
                    query_delta * scale_values[None, :],
                    code_values.to(tl.float32),
                    dot_products,
                    input_precision="ieee",
                )

            distances = (
                query_norms[:, None]
                + vector_norms[None, :]
                - 2.0 * dot_products
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
    def _merge_chunk_topk_kernel(
        chunk_partial_keys_ptr,
        list_partial_keys_ptr,
        MERGE_WIDTH: tl.constexpr,
        MAX_VECTOR_CHUNKS: tl.constexpr,
        LOCAL_K: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        edge_idx = tl.program_id(0)
        offsets = tl.arange(0, MERGE_WIDTH)
        keys = tl.load(
            chunk_partial_keys_ptr + edge_idx * MAX_VECTOR_CHUNKS * LOCAL_K + offsets,
            mask=offsets < MAX_VECTOR_CHUNKS * LOCAL_K,
            other=INF_KEY,
        )
        best_keys = ~tl.topk(~keys, LOCAL_K)
        tl.store(list_partial_keys_ptr + edge_idx * LOCAL_K + tl.arange(0, LOCAL_K), best_keys)


    @triton.jit
    def _finalize_topk_kernel(
        list_partial_keys_ptr,
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
        keys = tl.load(
            list_partial_keys_ptr + query_idx * nprobe * LOCAL_K + offsets,
            mask=offsets < nprobe * LOCAL_K,
            other=INF_KEY,
        )
        final_keys = ~tl.topk(~keys, LOCAL_K)
        ranks = tl.arange(0, LOCAL_K)
        physical_positions = (final_keys & 0xFFFFFFFF).to(tl.int64)
        distance_bits = (final_keys >> 32).to(tl.uint32)
        distances = distance_bits.to(tl.float32, bitcast=True)
        valid_result = (ranks < topk) & (final_keys < INF_KEY)
        ids = tl.load(
            list_ids_interleaved_ptr + physical_positions,
            mask=valid_result,
            other=-1,
        )
        distances = tl.where(valid_result, distances, float("inf"))
        tl.store(output_distances_ptr + query_idx * topk + ranks, distances, mask=ranks < topk)
        tl.store(output_ids_ptr + query_idx * topk + ranks, ids, mask=ranks < topk)

def is_available(device: torch.device) -> bool:
    return triton is not None and device.type == "cuda"


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _task_plan(*, topk: int, nprobe: int, max_list_group_count: int) -> tuple[int, int, int] | None:
    if topk <= 0 or nprobe <= 0 or nprobe > MAX_NPROBE or max_list_group_count <= 0:
        return None
    local_k = max(2, _next_power_of_two(topk))
    if local_k > MAX_LOCAL_K:
        return None
    max_vector_chunks = math.ceil(max_list_group_count / GROUPS_PER_CHUNK)
    merge_width = _next_power_of_two(max_vector_chunks * local_k)
    if merge_width > MAX_FINAL_WIDTH:
        return None
    return local_k, max_vector_chunks, merge_width


def can_run_topk(
    *,
    device: torch.device,
    topk: int,
    nprobe: int,
    max_list_group_count: int,
) -> bool:
    return is_available(device) and _task_plan(
        topk=topk,
        nprobe=nprobe,
        max_list_group_count=max_list_group_count,
    ) is not None


def _direct_scan_plan(*, topk: int, num_queries: int, nprobe: int, device: torch.device):
    if (
        topk <= 0
        or num_queries <= 0
        or nprobe <= 0
        or nprobe > MAX_NPROBE
        or not is_available(device)
    ):
        return None
    local_k = max(2, _next_power_of_two(topk))
    if local_k > 128:
        return None
    try:
        target_programs = int(torch.cuda.get_device_properties(device).multi_processor_count) * DIRECT_PROGRAMS_PER_SM
    except (AssertionError, RuntimeError):  # pragma: no cover - CUDA unavailable
        target_programs = 256
    grid_x = min(
        nprobe,
        DIRECT_MAX_GRID_X,
        max(1, math.ceil(target_programs / num_queries)),
    )
    final_width = _next_power_of_two(grid_x * local_k)
    if final_width > MAX_FINAL_WIDTH:
        return None
    return local_k, grid_x, final_width


def can_run_direct_topk(
    *,
    device: torch.device,
    topk: int,
    num_queries: int,
    nprobe: int,
    dim: int | None = None,
) -> bool:
    if dim is not None and dim > DIM_TILE:
        return False
    return _direct_scan_plan(
        topk=topk,
        num_queries=num_queries,
        nprobe=nprobe,
        device=device,
    ) is not None


can_run_query_major_topk = can_run_direct_topk


def triton_direct_topk(
    *,
    queries: torch.Tensor,
    residuals_by_dim: torch.Tensor,
    vector_norms_interleaved: torch.Tensor,
    sq_base: torch.Tensor,
    list_ids_interleaved: torch.Tensor,
    list_group_prefix: torch.Tensor,
    list_count: torch.Tensor,
    probe_cids: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Query-major SQ scan, matching cuVS's one-query block shape."""
    num_queries, nprobe = probe_cids.shape
    plan = _direct_scan_plan(
        topk=topk,
        num_queries=num_queries,
        nprobe=nprobe,
        device=queries.device,
    )
    if plan is None:
        raise ValueError("IVFSQ direct Triton top-k is unavailable for these search parameters")
    if int(queries.shape[1]) > DIM_TILE:
        raise ValueError(
            f"IVFSQ query-major Triton top-k supports dim <= {DIM_TILE}, "
            f"got {queries.shape[1]}"
        )
    local_k, grid_x, final_width = plan
    if not queries.is_contiguous():
        queries = queries.contiguous()
    if not residuals_by_dim.is_contiguous():
        residuals_by_dim = residuals_by_dim.contiguous()
    if not probe_cids.is_contiguous():
        probe_cids = probe_cids.contiguous()

    partial_keys = torch.full(
        (num_queries, grid_x, local_k),
        _INF_KEY,
        device=queries.device,
        dtype=torch.int64,
    )
    _sq_query_major_topk_kernel[(grid_x, num_queries)](
        queries,
        residuals_by_dim,
        vector_norms_interleaved,
        sq_base,
        probe_cids,
        list_group_prefix,
        list_count,
        partial_keys,
        nprobe,
        int(residuals_by_dim.shape[1]),
        GROUP_SIZE=GROUP_SIZE,
        DIM_TILE=DIM_TILE,
        GRID_X=grid_x,
        D=int(queries.shape[1]),
        LOCAL_K=local_k,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )
    output_distances = torch.empty(
        (num_queries, topk), device=queries.device, dtype=torch.float32
    )
    output_ids = torch.empty(
        (num_queries, topk), device=queries.device, dtype=torch.int64
    )
    _finalize_topk_kernel[(num_queries,)](
        partial_keys,
        list_ids_interleaved,
        output_distances,
        output_ids,
        grid_x,
        topk,
        LOCAL_K=local_k,
        FINAL_WIDTH=final_width,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )
    return output_distances, output_ids


triton_query_major_topk = triton_direct_topk


def triton_topk(
    *,
    queries: torch.Tensor,
    codes_by_dim: torch.Tensor,
    residuals_by_dim: torch.Tensor,
    vector_norms_interleaved: torch.Tensor,
    sq_base: torch.Tensor,
    sq_scale: torch.Tensor,
    list_ids_interleaved: torch.Tensor,
    list_group_prefix: torch.Tensor,
    list_count: torch.Tensor,
    probe_cids: torch.Tensor,
    topk: int,
    max_list_group_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan SQ codes in list/query tiles and keep local top-k in registers."""
    if not can_run_topk(
        device=queries.device,
        topk=topk,
        nprobe=int(probe_cids.shape[1]),
        max_list_group_count=max_list_group_count,
    ):
        raise ValueError("IVFSQ Triton top-k is unavailable for these search parameters")
    if not queries.is_contiguous():
        queries = queries.contiguous()
    if not codes_by_dim.is_contiguous():
        codes_by_dim = codes_by_dim.contiguous()
    if not probe_cids.is_contiguous():
        probe_cids = probe_cids.contiguous()

    num_queries, nprobe = probe_cids.shape
    dim = int(queries.shape[1])
    num_lists = int(list_count.numel())
    local_k, max_vector_chunks, merge_width = _task_plan(
        topk=topk,
        nprobe=nprobe,
        max_list_group_count=max_list_group_count,
    )
    assert local_k is not None

    edge_query_ids = torch.arange(
        num_queries,
        device=queries.device,
        dtype=torch.int64,
    ).repeat_interleave(nprobe)
    edge_cids = probe_cids.reshape(-1).to(torch.int64)
    _, edge_sorted_to_original = torch.sort(edge_cids)
    edge_query_ids_sorted = edge_query_ids[edge_sorted_to_original]
    list_query_count = torch.bincount(edge_cids, minlength=num_lists).to(torch.int64)
    list_query_prefix = torch.cumsum(list_query_count, dim=0) - list_query_count

    list_group_count = torch.div(
        list_count + (GROUP_SIZE - 1),
        GROUP_SIZE,
        rounding_mode="floor",
    )
    list_query_tile_count = torch.div(
        list_query_count + (QUERY_TILE - 1),
        QUERY_TILE,
        rounding_mode="floor",
    )
    list_vector_chunk_count = torch.div(
        list_group_count + (GROUPS_PER_CHUNK - 1),
        GROUPS_PER_CHUNK,
        rounding_mode="floor",
    )
    task_count_per_list = list_query_tile_count * list_vector_chunk_count
    task_prefix = torch.cumsum(task_count_per_list, dim=0) - task_count_per_list
    num_tasks = int(task_count_per_list.sum().item())
    num_edges = num_queries * nprobe

    output_distances = torch.full(
        (num_queries, topk),
        float("inf"),
        device=queries.device,
        dtype=torch.float32,
    )
    output_ids = torch.full(
        (num_queries, topk),
        -1,
        device=queries.device,
        dtype=torch.int64,
    )
    if num_tasks == 0:
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

    chunk_partial_keys = torch.full(
        (num_edges, max_vector_chunks, local_k),
        _INF_KEY,
        device=queries.device,
        dtype=torch.int64,
    )
    _sq_tasked_topk_kernel[(num_tasks,)](
        queries,
        codes_by_dim,
        vector_norms_interleaved,
        sq_base,
        sq_scale,
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
        GROUP_SIZE=GROUP_SIZE,
        D=dim,
        NUM_DIM_TILES=(dim + DIM_TILE - 1) // DIM_TILE,
        PADDED_VECTORS=int(codes_by_dim.shape[1]),
        QUERY_TILE=QUERY_TILE,
        VECTOR_TILE=VECTOR_TILE,
        DIM_TILE=DIM_TILE,
        GROUPS_PER_CHUNK=GROUPS_PER_CHUNK,
        MAX_VECTOR_CHUNKS=max_vector_chunks,
        LOCAL_K=local_k,
        INF_KEY=_INF_KEY,
        num_warps=8,
        num_stages=2,
    )

    list_partial_keys = torch.empty(
        (num_edges, local_k),
        device=queries.device,
        dtype=torch.int64,
    )
    _merge_chunk_topk_kernel[(num_edges,)](
        chunk_partial_keys,
        list_partial_keys,
        MERGE_WIDTH=merge_width,
        MAX_VECTOR_CHUNKS=max_vector_chunks,
        LOCAL_K=local_k,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )
    final_width = _next_power_of_two(nprobe * local_k)
    if final_width > MAX_FINAL_WIDTH:
        raise ValueError("IVFSQ final top-k merge is too wide")
    _finalize_topk_kernel[(num_queries,)](
        list_partial_keys,
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
