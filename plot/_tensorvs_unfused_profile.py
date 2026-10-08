"""Benchmark-only non-fused TensorVS IVF scan paths.

These kernels materialize candidate distances so Top-K can be timed as an
independent GPU stage. They are deliberately kept under ``plot/`` and are not
used by the TensorVS public search API.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


_GROUP_SIZE = 32


@triton.jit
def _globaltimer_ns():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;",
        "=l",
        [],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _materialize_f32(values):
    return tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=r,r",
        [values],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _ivfflat_grouped_distance_kernel(
    queries_ptr,
    query_norms_ptr,
    vectors_ptr,
    vector_norms_ptr,
    group_prefix_ptr,
    list_count_ptr,
    edge_qid_sorted_ptr,
    edge_sorted_to_original_ptr,
    list_query_prefix_ptr,
    list_query_count_ptr,
    task_list_ids_ptr,
    task_query_tile_ids_ptr,
    output_ptr,
    phase_cycles_ptr,
    DIM: tl.constexpr,
    MAX_LIST_COUNT: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    VECTOR_WIDTH: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    DIM_TILE: tl.constexpr,
    NUM_DIM_TILES: tl.constexpr,
):
    task = tl.program_id(0)
    list_id = tl.load(task_list_ids_ptr + task).to(tl.int64)
    query_tile_id = tl.load(task_query_tile_ids_ptr + task).to(tl.int32)
    query_count = tl.load(list_query_count_ptr + list_id).to(tl.int32)
    query_prefix = tl.load(list_query_prefix_ptr + list_id).to(tl.int64)
    vector_count = tl.load(list_count_ptr + list_id).to(tl.int32)
    group_prefix = tl.load(group_prefix_ptr + list_id).to(tl.int64)

    query_offsets = tl.arange(0, QUERY_TILE)
    lanes = tl.arange(0, 32)
    dim_offsets = tl.arange(0, DIM_TILE)
    query_positions = query_tile_id * QUERY_TILE + query_offsets
    valid_queries = query_positions < query_count
    sorted_edges = query_prefix + query_positions.to(tl.int64)
    query_ids = tl.load(
        edge_qid_sorted_ptr + sorted_edges, mask=valid_queries, other=0
    ).to(tl.int64)
    original_edges = tl.load(
        edge_sorted_to_original_ptr + sorted_edges, mask=valid_queries, other=0
    ).to(tl.int64)
    query_norms = tl.load(
        query_norms_ptr + query_ids, mask=valid_queries, other=0.0
    ).to(tl.float32)

    read_cycles = tl.zeros((), dtype=tl.int64)
    dot_cycles = tl.zeros((), dtype=tl.int64)
    epilogue_cycles = tl.zeros((), dtype=tl.int64)
    group_count = (vector_count + 31) // 32
    for group_in_list in tl.range(0, group_count, 1):
        vector_offsets = group_in_list * 32 + lanes
        valid_vectors = vector_offsets < vector_count
        physical = (group_prefix + group_in_list) * 32 + lanes
        dot_products = tl.zeros((QUERY_TILE, 32), dtype=tl.float32)
        for dim_tile in tl.range(0, NUM_DIM_TILES, 1):
            read_begin = _globaltimer_ns()
            dims = dim_tile * DIM_TILE + dim_offsets
            valid_dims = dims < DIM
            query_values = tl.load(
                queries_ptr + query_ids[:, None] * DIM + dims[None, :],
                mask=valid_queries[:, None] & valid_dims[None, :],
                other=0.0,
            ).to(tl.float32)
            chunks = dims // VECTOR_WIDTH
            components = dims % VECTOR_WIDTH
            vector_indices = (
                ((group_prefix + group_in_list) * NUM_CHUNKS + chunks[:, None])
                * 32
                + lanes[None, :]
            ) * VECTOR_WIDTH + components[:, None]
            vector_values = tl.load(
                vectors_ptr + vector_indices,
                mask=valid_dims[:, None] & valid_vectors[None, :],
                other=0.0,
            ).to(tl.float32)
            # Force the profiling-only read phase to make every loaded value
            # available before sampling the timer. Without this dependency,
            # load latency is charged to the following tl.dot instruction.
            query_values = _materialize_f32(query_values)
            vector_values = _materialize_f32(vector_values)
            read_end = _globaltimer_ns()
            dot_begin = read_end
            dot_products = tl.dot(
                query_values,
                vector_values,
                dot_products,
                input_precision="ieee",
            )
            dot_end = _globaltimer_ns()
            read_cycles += read_end - read_begin
            dot_cycles += dot_end - dot_begin
        epilogue_begin = _globaltimer_ns()
        vector_norms = tl.load(
            vector_norms_ptr + physical, mask=valid_vectors, other=0.0
        ).to(tl.float32)
        distances = tl.maximum(
            query_norms[:, None] + vector_norms[None, :] - 2.0 * dot_products,
            0.0,
        )
        output_indices = (
            original_edges[:, None] * MAX_LIST_COUNT + vector_offsets[None, :]
        )
        tl.store(
            output_ptr + output_indices,
            distances,
            mask=valid_queries[:, None] & valid_vectors[None, :],
        )
        epilogue_cycles += _globaltimer_ns() - epilogue_begin

    tl.atomic_add(phase_cycles_ptr, read_cycles)
    tl.atomic_add(phase_cycles_ptr + 1, dot_cycles)
    tl.atomic_add(phase_cycles_ptr + 2, epilogue_cycles)


@triton.jit
def _ivfpq_grouped_lookup_kernel(
    global_lut_ptr,
    static_corrections_ptr,
    probe_bias_ptr,
    codes_ptr,
    group_prefix_ptr,
    list_count_ptr,
    edge_qid_sorted_ptr,
    edge_sorted_to_original_ptr,
    list_query_prefix_ptr,
    list_query_count_ptr,
    task_list_ids_ptr,
    task_query_tile_ids_ptr,
    output_ptr,
    NUM_QUERIES: tl.constexpr,
    MAX_LIST_COUNT: tl.constexpr,
    PADDED_VECTORS: tl.constexpr,
    M: tl.constexpr,
    KSUB: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    task = tl.program_id(0)
    list_id = tl.load(task_list_ids_ptr + task).to(tl.int64)
    query_tile_id = tl.load(task_query_tile_ids_ptr + task).to(tl.int32)
    query_count = tl.load(list_query_count_ptr + list_id).to(tl.int32)
    query_prefix = tl.load(list_query_prefix_ptr + list_id).to(tl.int64)
    vector_count = tl.load(list_count_ptr + list_id).to(tl.int32)
    group_prefix = tl.load(group_prefix_ptr + list_id).to(tl.int64)

    query_offsets = tl.arange(0, QUERY_TILE)
    lanes = tl.arange(0, 32)
    query_positions = query_tile_id * QUERY_TILE + query_offsets
    valid_queries = query_positions < query_count
    sorted_edges = query_prefix + query_positions.to(tl.int64)
    query_ids = tl.load(
        edge_qid_sorted_ptr + sorted_edges, mask=valid_queries, other=0
    ).to(tl.int64)
    original_edges = tl.load(
        edge_sorted_to_original_ptr + sorted_edges, mask=valid_queries, other=0
    ).to(tl.int64)
    probe_bias = tl.load(
        probe_bias_ptr + original_edges, mask=valid_queries, other=0.0
    ).to(tl.float32)

    group_count = (vector_count + 31) // 32
    for group_in_list in tl.range(0, group_count, 1):
        vector_offsets = group_in_list * 32 + lanes
        valid_vectors = vector_offsets < vector_count
        physical = (group_prefix + group_in_list) * 32 + lanes
        distances = tl.zeros((QUERY_TILE, 32), dtype=tl.float32)
        for subspace_start in tl.static_range(0, M, BLOCK_M):
            subspaces = subspace_start + tl.arange(0, BLOCK_M)
            valid_subspaces = subspaces < M
            codes = tl.load(
                codes_ptr
                + subspaces[:, None] * PADDED_VECTORS
                + physical[None, :],
                mask=valid_subspaces[:, None] & valid_vectors[None, :],
                other=0,
            ).to(tl.int32)
            lut_offsets = (
                (subspaces[:, None, None] * NUM_QUERIES
                 + query_ids[None, :, None])
                * KSUB
                + codes[:, None, :]
            )
            lut_values = tl.load(
                global_lut_ptr + lut_offsets,
                mask=(
                    valid_subspaces[:, None, None]
                    & valid_queries[None, :, None]
                    & valid_vectors[None, None, :]
                ),
                other=0.0,
            ).to(tl.float32)
            distances += tl.sum(lut_values, axis=0)
        distances += tl.load(
            static_corrections_ptr + physical,
            mask=valid_vectors,
            other=0.0,
        )[None, :]
        distances += probe_bias[:, None]
        output_indices = (
            original_edges[:, None] * MAX_LIST_COUNT + vector_offsets[None, :]
        )
        tl.store(
            output_ptr + output_indices,
            distances,
            mask=valid_queries[:, None] & valid_vectors[None, :],
        )


class TensorVSUnfusedRunner:
    """Run exact candidate materialization followed by a separate Top-K."""

    def __init__(
        self,
        *,
        index: Any,
        algorithm: str,
        device: torch.device,
        query_batch: int,
        nprobe: int,
        topk: int,
        lut_dtype: str = "fp16",
    ) -> None:
        self.index = index
        self.algorithm = algorithm
        self.device = device
        self.query_batch = int(query_batch)
        self.nprobe = int(nprobe)
        self.topk = int(topk)
        self.lut_dtype = lut_dtype
        self.list_count = index.list_count
        if self.list_count is None:
            raise ValueError("TensorVS IVF lists are not loaded")
        self.max_list_count = int(self.list_count.max().item())
        self.candidate_distances = torch.empty(
            (self.query_batch, self.nprobe * self.max_list_count),
            device=device,
            dtype=torch.float32,
        )
        self.flat_phase_cycles = torch.zeros(
            3, device=device, dtype=torch.int64
        )

        if algorithm == "IVFFlat":
            from tensorvs.index_ivfflat import _get_interleaved_flat_lists

            (
                self.list_vectors,
                self.list_ids,
                self.list_group_prefix,
                self.list_vector_norms,
            ) = _get_interleaved_flat_lists(index, device)
            self.vector_width = int(self.list_vectors.shape[-1])
            self.num_chunks = int(self.list_vectors.shape[1])
        elif algorithm == "IVFPQ":
            from tensorvs.index_ivfpq import (
                _get_interleaved_pq_lists,
                _get_interleaved_pq_static_corrections,
            )

            codes, self.list_ids, self.list_group_prefix = _get_interleaved_pq_lists(
                index, device
            )
            if codes is None:
                persisted = index.persisted_list_codes_interleaved_by_subspace
                if persisted is None:
                    raise ValueError("Unpacked PQ codes are unavailable for profiling")
                codes = persisted.to(device=device, non_blocking=False).contiguous()
            self.list_codes = codes
            self.static_corrections = _get_interleaved_pq_static_corrections(
                index, device
            )
            self.m = int(index.metadata["m"])
            self.ksub = int(index.metadata["ksub"])
        else:
            raise ValueError(f"Unsupported TensorVS algorithm: {algorithm}")

    def __call__(
        self, queries: torch.Tensor, profile: dict[str, float] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from tensorvs.index_ivfflat import _coarse_squared_l2_distance_addmm

        events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

        def start() -> torch.cuda.Event:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event

        def finish(name: str, begin: torch.cuda.Event) -> None:
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(self.device))
            events.append((name, begin, end))

        q = queries.to(device=self.device, dtype=torch.float32, copy=False).contiguous()
        begin = start()
        distances_to_centroids = _coarse_squared_l2_distance_addmm(
            q, self.index.centroids, self.index.centroid_norms
        )
        finish("centroid_distance", begin)
        begin = start()
        probe_distances, probe_cids = torch.topk(
            distances_to_centroids, k=self.nprobe, dim=1, largest=False
        )
        finish("centroid_topk", begin)

        if self.algorithm == "IVFFlat":
            self.flat_phase_cycles.zero_()
            begin = start()
            num_lists = int(self.list_count.numel())
            edge_query_ids = torch.arange(
                self.query_batch, device=self.device, dtype=torch.int64
            ).repeat_interleave(self.nprobe)
            edge_cids = probe_cids.reshape(-1).to(torch.int64)
            _, edge_sorted_to_original = torch.sort(edge_cids)
            edge_query_ids_sorted = edge_query_ids[edge_sorted_to_original]
            list_query_count = torch.bincount(
                edge_cids, minlength=num_lists
            ).to(torch.int64)
            list_query_prefix = torch.cumsum(list_query_count, dim=0) - list_query_count
            query_tile = 16
            list_query_tile_count = torch.div(
                list_query_count + query_tile - 1,
                query_tile,
                rounding_mode="floor",
            )
            num_tasks = int(list_query_tile_count.sum().item())
            task_list_ids = torch.repeat_interleave(
                torch.arange(num_lists, device=self.device, dtype=torch.int64),
                list_query_tile_count,
                output_size=num_tasks,
            )
            task_prefix = torch.cumsum(list_query_tile_count, dim=0) - list_query_tile_count
            task_positions = torch.arange(
                num_tasks, device=self.device, dtype=torch.int64
            )
            task_query_tile_ids = task_positions - task_prefix[task_list_ids]
            finish("list_scheduling_preparation", begin)
            begin = start()
            query_norms = (q * q).sum(dim=1)
            finish("flat_distance_preparation", begin)
            # Clearing the full candidate matrix exists only in this
            # profiling-only de-fused path. Measure it so it is excluded from
            # List Scheduling / Preparation and from the displayed scan time.
            begin = start()
            self.candidate_distances.fill_(float("inf"))
            finish("profile_buffer_initialization", begin)
            begin = start()
            _ivfflat_grouped_distance_kernel[(num_tasks,)](
                q,
                query_norms,
                self.list_vectors,
                self.list_vector_norms,
                self.list_group_prefix,
                self.list_count,
                edge_query_ids_sorted,
                edge_sorted_to_original,
                list_query_prefix,
                list_query_count,
                task_list_ids,
                task_query_tile_ids,
                self.candidate_distances,
                self.flat_phase_cycles,
                DIM=int(q.shape[1]),
                MAX_LIST_COUNT=self.max_list_count,
                NUM_CHUNKS=self.num_chunks,
                VECTOR_WIDTH=self.vector_width,
                QUERY_TILE=query_tile,
                DIM_TILE=32,
                NUM_DIM_TILES=triton.cdiv(int(q.shape[1]), 32),
                num_warps=4,
            )
            finish("flat_scan_kernel", begin)
        else:
            from tensorvs.index_ivfpq import (
                _apply_pq_rotation,
                _build_global_query_luts_bmm,
            )

            begin = start()
            pq_queries = _apply_pq_rotation(q, self.index.pq_rotation_matrix)
            global_luts = _build_global_query_luts_bmm(
                pq_queries,
                self.index.pq_codebooks,
                self.index.pq_codebook_norms,
                m=self.m,
                subdim=int(self.index.metadata["subdim"]),
                use_tf32=True,
            )
            if self.lut_dtype == "fp16":
                global_luts = global_luts.to(torch.float16)
            finish("LUT_prepare", begin)
            begin = start()
            num_lists = int(self.list_count.numel())
            edge_query_ids = torch.arange(
                self.query_batch, device=self.device, dtype=torch.int64
            ).repeat_interleave(self.nprobe)
            edge_cids = probe_cids.reshape(-1).to(torch.int64)
            _, edge_sorted_to_original = torch.sort(edge_cids)
            edge_query_ids_sorted = edge_query_ids[edge_sorted_to_original]
            list_query_count = torch.bincount(
                edge_cids, minlength=num_lists
            ).to(torch.int64)
            list_query_prefix = torch.cumsum(list_query_count, dim=0) - list_query_count
            query_tile = 8
            list_query_tile_count = torch.div(
                list_query_count + query_tile - 1,
                query_tile,
                rounding_mode="floor",
            )
            num_tasks = int(list_query_tile_count.sum().item())
            task_list_ids = torch.repeat_interleave(
                torch.arange(num_lists, device=self.device, dtype=torch.int64),
                list_query_tile_count,
                output_size=num_tasks,
            )
            task_prefix = torch.cumsum(list_query_tile_count, dim=0) - list_query_tile_count
            task_positions = torch.arange(
                num_tasks, device=self.device, dtype=torch.int64
            )
            task_query_tile_ids = task_positions - task_prefix[task_list_ids]
            finish("list_scheduling_preparation", begin)
            # This full candidate-buffer clear exists only in the profiling
            # implementation, so keep it out of the production Preparation
            # attribution and the LUT lookup measurement.
            begin = start()
            self.candidate_distances.fill_(float("inf"))
            finish("profile_buffer_initialization", begin)
            begin = start()
            _ivfpq_grouped_lookup_kernel[(num_tasks,)](
                global_luts,
                self.static_corrections,
                probe_distances,
                self.list_codes,
                self.list_group_prefix,
                self.list_count,
                edge_query_ids_sorted,
                edge_sorted_to_original,
                list_query_prefix,
                list_query_count,
                task_list_ids,
                task_query_tile_ids,
                self.candidate_distances,
                NUM_QUERIES=self.query_batch,
                MAX_LIST_COUNT=self.max_list_count,
                PADDED_VECTORS=int(self.list_ids.numel()),
                M=self.m,
                KSUB=self.ksub,
                QUERY_TILE=query_tile,
                BLOCK_M=8,
                num_warps=4,
            )
            finish("lut_lookup_accumulation", begin)

        begin = start()
        topk_distances, candidate_positions = torch.topk(
            self.candidate_distances, k=self.topk, dim=1, largest=False
        )
        probe_slots = torch.div(
            candidate_positions, self.max_list_count, rounding_mode="floor"
        )
        offsets = candidate_positions % self.max_list_count
        selected_cids = torch.gather(probe_cids, 1, probe_slots)
        physical_positions = (
            self.list_group_prefix[selected_cids] * _GROUP_SIZE + offsets
        )
        topk_ids = self.list_ids[physical_positions]
        finish("topk_selection", begin)

        if profile is not None:
            torch.cuda.synchronize(self.device)
            measured: dict[str, float] = {}
            for name, event_start, event_end in events:
                measured[name] = event_start.elapsed_time(event_end) / 1000.0
            if self.algorithm == "IVFFlat":
                scan_seconds = measured.pop("flat_scan_kernel")
                distance_preparation_seconds = measured.pop(
                    "flat_distance_preparation"
                )
                measured.pop("profile_buffer_initialization")
                phase_cycles = self.flat_phase_cycles.cpu().tolist()
                phase_total = max(sum(phase_cycles), 1)
                measured["vector_read_distance"] = (
                    distance_preparation_seconds
                    + scan_seconds * (phase_cycles[0] + phase_cycles[1])
                    / phase_total
                )
                measured["l2_epilogue_distance_writeback"] = (
                    scan_seconds * phase_cycles[2] / phase_total
                )
            else:
                measured.pop("profile_buffer_initialization")
            for name, seconds in measured.items():
                profile[name] = profile.get(name, 0.0) + seconds
        return topk_distances, topk_ids
