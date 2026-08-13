"""Triton list-major IVF-PQ scanning.

This uses a list-major active-task schedule: one program is launched per
``(list, query-tile)`` pair. The task loops over candidate tiles while keeping
the query tile's local top-k in registers. The PQ
distance calculation uses the same global-LUT plus static-correction ADC
decomposition as the existing IVF-PQ Triton scanner.  The Global LUT remains
in its construction-friendly ``[M, Q, Ksub]`` layout.  For each active Query
tile the kernel loads a temporary ``[Ksub, QueryTile]`` Vector LUT and gathers
from it for all candidates in the tile.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from ._ivfpq_triton import (
    DIRECT_SCAN_MAX_NPROBE as _IVFPQ_MAX_NPROBE,
    INTERLEAVED_GROUP_SIZE as _IVFPQ_GROUP_SIZE,
    MAX_FINAL_WIDTH as _IVFPQ_MAX_FINAL_WIDTH,
    MAX_LOCAL_K as _IVFPQ_MAX_LOCAL_K,
    MAX_SUBSPACES as _IVFPQ_MAX_SUBSPACES,
    PACKED_CODE_CHUNK_BYTES as _IVFPQ_PACKED_CHUNK_BYTES,
    PACKED_CODE_WORD_BYTES as _IVFPQ_PACKED_WORD_BYTES,
    PACKED_CODE_WORDS as _IVFPQ_PACKED_WORDS,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CUDA caller fallback covers this
    triton = None
    tl = None


INTERLEAVED_GROUP_SIZE = _IVFPQ_GROUP_SIZE
# A candidate tile contains several 32-vector interleaved groups.  The old
# kernel rebuilt the Vector LUT once per group; the new kernel builds it once
# per candidate tile.  Autotuning tries smaller/larger values as well.
LIST_MAJOR_CANDIDATE_GROUPS = 4
LIST_MAJOR_CANDIDATE_GROUP_OPTIONS = (8, 4, 2, 1)
MAX_LOCAL_K = _IVFPQ_MAX_LOCAL_K
MAX_FINAL_WIDTH = _IVFPQ_MAX_FINAL_WIDTH
MAX_NPROBE = _IVFPQ_MAX_NPROBE
MAX_SUBSPACES = _IVFPQ_MAX_SUBSPACES
MAX_KSUB = 256
MAX_INDEXED_VECTORS = (1 << 31) - 1
PACKED_CODE_CHUNK_BYTES = _IVFPQ_PACKED_CHUNK_BYTES
PACKED_CODE_WORD_BYTES = _IVFPQ_PACKED_WORD_BYTES
PACKED_CODE_WORDS = _IVFPQ_PACKED_WORDS
_INF_KEY = 0x7F800000FFFFFFFF
_SUPPORTED_GLOBAL_LUT_DTYPES = {torch.float32, torch.float16}

_AUTOTUNE_WARMUP_RUNS = 1
_AUTOTUNE_TIMED_RUNS = 3
_AUTOTUNE_CACHE_FORMAT_VERSION = 4


@dataclass(frozen=True)
class _ListMajorLaunchConfig:
    """Compile-time tile settings for the active task kernel."""

    query_tile: int
    num_warps: int
    candidate_groups: int = LIST_MAJOR_CANDIDATE_GROUPS
    use_vector_lut: bool = True


_AUTOTUNE_CONFIG_CACHE: dict[tuple[object, ...], _ListMajorLaunchConfig] = {}
_PERSISTED_AUTOTUNE_CONFIGS: dict[str, _ListMajorLaunchConfig] | None = None


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _list_major_plan(*, topk: int, nprobe: int) -> tuple[int, int] | None:
    """Return ``(local_k, query-final-merge-width)`` for the list path."""
    if topk <= 0 or nprobe <= 0 or nprobe > MAX_NPROBE:
        return None
    local_k = max(2, _next_power_of_two(topk))
    if local_k > MAX_LOCAL_K:
        return None
    final_width = _next_power_of_two(nprobe * local_k)
    if final_width > MAX_FINAL_WIDTH:
        return None
    return local_k, final_width


def _estimated_query_tile(
    *,
    m: int,
    local_k: int,
    lut_dtype: torch.dtype,
    device: torch.device | None = None,
    ksub: int = INTERLEAVED_GROUP_SIZE,
    candidate_groups: int = 1,
    use_vector_lut: bool = True,
) -> int:
    """Choose a conservative query tile from the live tile footprint."""
    del m
    lut_bytes = torch.empty((), dtype=lut_dtype).element_size()
    candidate_tile = INTERLEAVED_GROUP_SIZE * candidate_groups
    bytes_per_query = candidate_tile * 4 + local_k * 8
    if use_vector_lut:
        bytes_per_query += ksub * lut_bytes
    budget = 3584
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        try:
            properties = torch.cuda.get_device_properties(device)
            registers_per_sm = int(properties.regs_per_multiprocessor)
            budget = max(2048, min(8192, (budget * registers_per_sm) // 65536))
        except (AttributeError, RuntimeError, TypeError):
            pass
    for query_tile in (16, 8, 4, 2, 1):
        if query_tile * bytes_per_query <= budget:
            return query_tile
    return 1


def _candidate_launch_configs(
    *,
    m: int,
    local_k: int,
    lut_dtype: torch.dtype,
    ksub: int,
    device: torch.device | None = None,
) -> tuple[_ListMajorLaunchConfig, ...]:
    configs: list[_ListMajorLaunchConfig] = []
    for candidate_groups in LIST_MAJOR_CANDIDATE_GROUP_OPTIONS:
        default_tile = _estimated_query_tile(
            m=m,
            local_k=local_k,
            lut_dtype=lut_dtype,
            device=device,
            ksub=ksub,
            candidate_groups=candidate_groups,
            use_vector_lut=False,
        )
        # Keep calibration comparable to ``triton_topk``. The previous sweep
        # tested every query tile for every candidate tile and ran the full
        # scan dozens of times before the first search.
        configs.append(
            _ListMajorLaunchConfig(
                query_tile=default_tile,
                num_warps=4,
                candidate_groups=candidate_groups,
                use_vector_lut=False,
            )
        )
        if default_tile <= 8:
            configs.append(
                _ListMajorLaunchConfig(
                    query_tile=default_tile,
                    num_warps=8,
                    candidate_groups=candidate_groups,
                    use_vector_lut=False,
                )
            )
    return tuple(configs)


def _autotune_cache_path() -> Path | None:
    configured = os.environ.get("TENSORVS_IVFPQ_LIST_MAJOR_AUTOTUNE_CACHE")
    if configured == "":
        return None
    if configured is not None:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "tensorvs" / "ivfpq_list_major_autotune.json"


def _serialized_autotune_key(cache_key: tuple[object, ...]) -> str:
    return json.dumps(cache_key, separators=(",", ":"), ensure_ascii=True)


def _load_persisted_autotune_configs() -> dict[str, _ListMajorLaunchConfig]:
    global _PERSISTED_AUTOTUNE_CONFIGS
    if _PERSISTED_AUTOTUNE_CONFIGS is not None:
        return _PERSISTED_AUTOTUNE_CONFIGS

    configs: dict[str, _ListMajorLaunchConfig] = {}
    path = _autotune_cache_path()
    if path is not None:
        try:
            payload = json.loads(path.read_text())
            if payload.get("version") == _AUTOTUNE_CACHE_FORMAT_VERSION:
                for key, values in payload.get("entries", {}).items():
                    config = _ListMajorLaunchConfig(*values)
                    if (
                        config.query_tile in (1, 2, 4, 8, 16)
                        and config.num_warps in (4, 8)
                        and config.candidate_groups in LIST_MAJOR_CANDIDATE_GROUP_OPTIONS
                        and isinstance(config.use_vector_lut, bool)
                    ):
                        configs[key] = config
        except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    _PERSISTED_AUTOTUNE_CONFIGS = configs
    return configs


def _persist_autotune_config(
    cache_key: tuple[object, ...],
    config: _ListMajorLaunchConfig,
) -> None:
    path = _autotune_cache_path()
    if path is None:
        return
    configs = _load_persisted_autotune_configs()
    configs[_serialized_autotune_key(cache_key)] = config
    payload = {
        "version": _AUTOTUNE_CACHE_FORMAT_VERSION,
        "entries": {
            key: [
                value.query_tile,
                value.num_warps,
                value.candidate_groups,
                value.use_vector_lut,
            ]
            for key, value in configs.items()
        },
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        os.replace(temporary_path, path)
    except OSError:
        pass


def _cached_autotune_config(
    cache_key: tuple[object, ...],
) -> _ListMajorLaunchConfig | None:
    cached = _AUTOTUNE_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cached = _load_persisted_autotune_configs().get(_serialized_autotune_key(cache_key))
    if cached is not None:
        _AUTOTUNE_CONFIG_CACHE[cache_key] = cached
    return cached


if triton is not None:

    @triton.jit
    def _list_major_pq_task_kernel(
        global_query_luts_ptr,
        static_vector_corrections_ptr,
        probe_distance_bias_flat_ptr,
        codes_by_subspace_ptr,
        list_group_prefix_ptr,
        list_count_ptr,
        edge_qid_sorted_ptr,
        edge_sorted_to_original_ptr,
        list_query_prefix_ptr,
        list_query_count_ptr,
        task_list_ids_ptr,
        task_query_tile_ids_ptr,
        list_partial_keys_ptr,
        num_queries,
        num_code_vectors,
        M: tl.constexpr,
        KSUB: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        QUERY_TILE: tl.constexpr,
        GROUPS_PER_CHUNK: tl.constexpr,
        LOCAL_K: tl.constexpr,
        USE_VECTOR_LUT: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Run one ``(list, query tile)`` task with PQ-ADC local top-k."""
        task_idx = tl.program_id(0)
        list_idx = tl.load(task_list_ids_ptr + task_idx).to(tl.int64)
        query_tile_idx = tl.load(task_query_tile_ids_ptr + task_idx).to(tl.int32)

        query_count = tl.load(list_query_count_ptr + list_idx).to(tl.int32)
        query_prefix = tl.load(list_query_prefix_ptr + list_idx).to(tl.int64)
        vector_count = tl.load(list_count_ptr + list_idx).to(tl.int32)
        group_prefix = tl.load(list_group_prefix_ptr + list_idx).to(tl.int64)
        group_count = (vector_count + GROUP_SIZE - 1) // GROUP_SIZE

        query_offsets = tl.arange(0, QUERY_TILE)
        vector_offsets = tl.arange(0, GROUP_SIZE * GROUPS_PER_CHUNK)
        lut_k_offsets = tl.arange(0, KSUB)
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
        edge_bias = tl.load(
            probe_distance_bias_flat_ptr + original_edge_positions,
            mask=valid_queries,
            other=0.0,
        ).to(tl.float32)
        best_keys = tl.full((QUERY_TILE, LOCAL_K), INF_KEY, tl.int64)

        for group_start in tl.range(0, group_count, GROUPS_PER_CHUNK):
            valid_vectors = group_start * GROUP_SIZE + vector_offsets < vector_count
            physical_positions = (
                (group_prefix + group_start) * GROUP_SIZE
                + vector_offsets
            )
            correction = tl.load(
                static_vector_corrections_ptr + physical_positions,
                mask=valid_vectors,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            # Give the loop-carried accumulator its full [QueryTile, C]
            # shape. Triton does not infer the broadcasted shape from the
            # later LUT gather when the initial value is [1, C].
            distances = tl.zeros(
                (QUERY_TILE, GROUP_SIZE * GROUPS_PER_CHUNK),
                tl.float32,
            )
            distances += correction[None, :]

            # Build one Ksub x QueryTile Vector LUT for this candidate tile.
            # This changes only the lookup layout; code loading, correction,
            # accumulation and local top-k follow the IVFPQ Triton scanner.
            # Match the IVFPQ query-major scanner: M is a compile-time
            # specialization, so keep this loop static and let Triton hoist
            # invariant LUT address arithmetic.
            for subspace_idx in tl.static_range(0, M):
                codes = tl.load(
                    codes_by_subspace_ptr
                    + subspace_idx * num_code_vectors
                    + physical_positions,
                    mask=valid_vectors,
                    other=0,
                    cache_modifier=".ca",
                ).to(tl.int64)
                if USE_VECTOR_LUT:
                    query_lut_qk = tl.load(
                        global_query_luts_ptr
                        + subspace_idx * num_queries * KSUB
                        + query_ids[:, None] * KSUB
                        + lut_k_offsets[None, :],
                        mask=valid_queries[:, None],
                        other=0.0,
                        cache_modifier=".ca",
                    )
                    vector_lut = tl.trans(query_lut_qk)
                    code_query_indices = codes[:, None] + tl.zeros(
                        (1, QUERY_TILE), dtype=tl.int64
                    )
                    lut_values = tl.trans(
                        tl.gather(vector_lut, code_query_indices, axis=0)
                    ).to(tl.float32)
                else:
                    lut_values = tl.load(
                        global_query_luts_ptr
                        + subspace_idx * num_queries * KSUB
                        + query_ids[:, None] * KSUB
                        + codes[None, :],
                        mask=valid_queries[:, None] & valid_vectors[None, :],
                        other=0.0,
                        cache_modifier=".ca",
                    ).to(tl.float32)
                distances += lut_values

            distances = tl.maximum(
                distances + edge_bias[:, None],
                0.0,
            )
            distance_bits = distances.to(tl.uint32, bitcast=True).to(tl.int64)
            candidate_keys = (distance_bits << 32) | physical_positions[None, :]
            valid_candidates = valid_queries[:, None] & valid_vectors[None, :]
            candidate_keys = tl.where(valid_candidates, candidate_keys, INF_KEY)
            if LOCAL_K <= GROUP_SIZE * GROUPS_PER_CHUNK:
                tile_best_keys = ~tl.topk(~candidate_keys, LOCAL_K)
                merged_keys = tl.reshape(
                    tl.join(best_keys, tile_best_keys),
                    (QUERY_TILE, 2 * LOCAL_K),
                )
            else:
                merged_keys = tl.reshape(
                    tl.join(best_keys, candidate_keys),
                    (QUERY_TILE, LOCAL_K + GROUP_SIZE * GROUPS_PER_CHUNK),
                )
            best_keys = ~tl.topk(~merged_keys, LOCAL_K)

        tl.store(
            list_partial_keys_ptr
            + original_edge_positions[:, None] * LOCAL_K
            + rank_offsets[None, :],
            best_keys,
            mask=valid_queries[:, None],
        )

    @triton.jit
    def _list_major_pq_packed_task_kernel(
        global_query_luts_ptr,
        static_vector_corrections_ptr,
        probe_distance_bias_flat_ptr,
        packed_codes_ptr,
        list_group_prefix_ptr,
        list_count_ptr,
        edge_qid_sorted_ptr,
        edge_sorted_to_original_ptr,
        list_query_prefix_ptr,
        list_query_count_ptr,
        task_list_ids_ptr,
        task_query_tile_ids_ptr,
        list_partial_keys_ptr,
        num_queries,
        num_code_vectors,
        M: tl.constexpr,
        KSUB: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        QUERY_TILE: tl.constexpr,
        GROUPS_PER_CHUNK: tl.constexpr,
        LOCAL_K: tl.constexpr,
        PACKED_CHUNKS: tl.constexpr,
        CODE_CHUNK_BYTES: tl.constexpr,
        CODE_WORD_BYTES: tl.constexpr,
        CODE_WORDS: tl.constexpr,
        USE_VECTOR_LUT: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """List-major PQ ADC using the same rolled packed-code loop as topk."""
        task_idx = tl.program_id(0)
        list_idx = tl.load(task_list_ids_ptr + task_idx).to(tl.int64)
        query_tile_idx = tl.load(task_query_tile_ids_ptr + task_idx).to(tl.int32)

        query_count = tl.load(list_query_count_ptr + list_idx).to(tl.int32)
        query_prefix = tl.load(list_query_prefix_ptr + list_idx).to(tl.int64)
        vector_count = tl.load(list_count_ptr + list_idx).to(tl.int32)
        group_prefix = tl.load(list_group_prefix_ptr + list_idx).to(tl.int64)
        group_count = (vector_count + GROUP_SIZE - 1) // GROUP_SIZE

        query_offsets = tl.arange(0, QUERY_TILE)
        candidate_offsets = tl.arange(0, GROUP_SIZE * GROUPS_PER_CHUNK)
        group_offsets = candidate_offsets // GROUP_SIZE
        lane_offsets = candidate_offsets % GROUP_SIZE
        lut_k_offsets = tl.arange(0, KSUB)
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
        edge_bias = tl.load(
            probe_distance_bias_flat_ptr + original_edge_positions,
            mask=valid_queries,
            other=0.0,
        ).to(tl.float32)
        best_keys = tl.full((QUERY_TILE, LOCAL_K), INF_KEY, tl.int64)

        for group_start in tl.range(0, group_count, GROUPS_PER_CHUNK):
            valid_vectors = group_start * GROUP_SIZE + candidate_offsets < vector_count
            physical_groups = group_prefix + group_start + group_offsets.to(tl.int64)
            physical_positions = physical_groups * GROUP_SIZE + lane_offsets
            packed_chunk_zero = (
                (physical_groups * PACKED_CHUNKS * GROUP_SIZE + lane_offsets)
                * CODE_WORDS
            )
            correction = tl.load(
                static_vector_corrections_ptr + physical_positions,
                mask=valid_vectors,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            distances = tl.zeros(
                (QUERY_TILE, GROUP_SIZE * GROUPS_PER_CHUNK),
                tl.float32,
            )
            distances += correction[None, :]

            # This is the packed analogue of triton_topk's rolled chunk loop.
            # Each packed word is read once for the whole Query tile; only the
            # LUT lookup is repeated across the tile's queries.
            for chunk_idx in tl.range(0, PACKED_CHUNKS, 1, loop_unroll_factor=1):
                packed_chunk = packed_chunk_zero + chunk_idx * GROUP_SIZE * CODE_WORDS
                for word_idx in tl.static_range(0, CODE_WORDS):
                    packed_word = tl.load(
                        packed_codes_ptr + packed_chunk + word_idx,
                        mask=valid_vectors,
                        other=0,
                        cache_modifier=".ca",
                    ).to(tl.uint64)
                    for byte_idx in tl.static_range(0, CODE_WORD_BYTES):
                        subspace_idx = (
                            chunk_idx * CODE_CHUNK_BYTES
                            + word_idx * CODE_WORD_BYTES
                            + byte_idx
                        )
                        subspace_valid = subspace_idx < M
                        codes = ((packed_word >> (byte_idx * 8)) & 0xFF).to(tl.int64)
                        if USE_VECTOR_LUT:
                            query_lut_qk = tl.load(
                                global_query_luts_ptr
                                + subspace_idx * num_queries * KSUB
                                + query_ids[:, None] * KSUB
                                + lut_k_offsets[None, :],
                                mask=valid_queries[:, None] & subspace_valid,
                                other=0.0,
                                cache_modifier=".ca",
                            )
                            vector_lut = tl.trans(query_lut_qk)
                            code_query_indices = codes[:, None] + tl.zeros(
                                (1, QUERY_TILE), dtype=tl.int64
                            )
                            lut_values = tl.trans(
                                tl.gather(vector_lut, code_query_indices, axis=0)
                            ).to(tl.float32)
                            distances += tl.where(subspace_valid, lut_values, 0.0)
                        else:
                            lut_values = tl.load(
                                global_query_luts_ptr
                                + subspace_idx * num_queries * KSUB
                                + query_ids[:, None] * KSUB
                                + codes[None, :],
                                mask=valid_queries[:, None]
                                & valid_vectors[None, :]
                                & subspace_valid,
                                other=0.0,
                                cache_modifier=".ca",
                            ).to(tl.float32)
                            distances += lut_values

            distances = tl.maximum(
                distances + edge_bias[:, None],
                0.0,
            )
            distance_bits = distances.to(tl.uint32, bitcast=True).to(tl.int64)
            candidate_keys = (distance_bits << 32) | physical_positions[None, :]
            valid_candidates = valid_queries[:, None] & valid_vectors[None, :]
            candidate_keys = tl.where(valid_candidates, candidate_keys, INF_KEY)
            if LOCAL_K <= GROUP_SIZE * GROUPS_PER_CHUNK:
                tile_best_keys = ~tl.topk(~candidate_keys, LOCAL_K)
                merged_keys = tl.reshape(
                    tl.join(best_keys, tile_best_keys),
                    (QUERY_TILE, 2 * LOCAL_K),
                )
            else:
                merged_keys = tl.reshape(
                    tl.join(best_keys, candidate_keys),
                    (QUERY_TILE, LOCAL_K + GROUP_SIZE * GROUPS_PER_CHUNK),
                )
            best_keys = ~tl.topk(~merged_keys, LOCAL_K)

        tl.store(
            list_partial_keys_ptr
            + original_edge_positions[:, None] * LOCAL_K
            + rank_offsets[None, :],
            best_keys,
            mask=valid_queries[:, None],
        )

    @triton.jit
    def _finalize_list_major_topk_kernel(
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
        """Merge one exact local top-k from each probed list per Query."""
        query_idx = tl.program_id(0)
        offsets = tl.arange(0, FINAL_WIDTH)
        keys = tl.load(
            partial_keys_ptr + query_idx * nprobe * LOCAL_K + offsets,
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
        tl.store(
            output_distances_ptr + query_idx * topk + ranks,
            distances,
            mask=ranks < topk,
        )
        tl.store(
            output_ids_ptr + query_idx * topk + ranks,
            ids,
            mask=ranks < topk,
        )


def is_available(device: torch.device) -> bool:
    return triton is not None and device.type == "cuda"


def _layout_list_major_global_lut(global_query_luts: torch.Tensor) -> torch.Tensor:
    """Validate and compact the Global LUT's canonical ``[M, Q, Ksub]`` layout."""
    if global_query_luts.ndim != 3:
        raise ValueError("Global query LUT must have shape [m, query, ksub]")
    return global_query_luts.contiguous()


def can_run_topk(
    *,
    device: torch.device,
    topk: int,
    m: int,
    ksub: int,
    nprobe: int,
    num_code_vectors: int,
    code_dtype: torch.dtype,
    use_packed_16b_codes: bool = False,
) -> bool:
    return (
        is_available(device)
        and code_dtype == torch.uint8
        and 0 < m <= (128 if use_packed_16b_codes else MAX_SUBSPACES)
        and 0 < ksub <= MAX_KSUB
        and 0 < num_code_vectors <= MAX_INDEXED_VECTORS
        and _list_major_plan(topk=topk, nprobe=nprobe) is not None
    )


def _build_list_major_edges(
    probe_cids: torch.Tensor,
    num_lists: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort probe edges by list for the IVFPQ active-task schedule."""
    num_queries, nprobe = probe_cids.shape
    edge_cids = probe_cids.reshape(-1).to(torch.int64)
    # The flattened probe edge order is query-major. Integer division creates
    # the matching query IDs without materializing repeat_interleave's extra
    # expanded index first.
    edge_query_ids = torch.arange(
        edge_cids.numel(),
        dtype=torch.int64,
        device=probe_cids.device,
    ).div(nprobe, rounding_mode="floor")
    _, edge_sorted_to_original = torch.sort(edge_cids)
    edge_query_ids_sorted = edge_query_ids[edge_sorted_to_original]
    list_query_count = torch.bincount(edge_cids, minlength=num_lists).to(torch.int64)
    list_query_prefix = torch.cumsum(list_query_count, dim=0) - list_query_count
    return (
        edge_query_ids_sorted.contiguous(),
        edge_sorted_to_original.contiguous(),
        list_query_prefix.contiguous(),
        list_query_count.contiguous(),
    )


def _build_list_major_task_plan(
    *,
    probe_cids: torch.Tensor,
    list_count: torch.Tensor,
    query_tile: int,
    candidate_groups: int = LIST_MAJOR_CANDIDATE_GROUPS,
) -> dict[str, torch.Tensor | int]:
    """Build compact ``(list, query-tile)`` tasks for the PQ scan."""
    if candidate_groups not in LIST_MAJOR_CANDIDATE_GROUP_OPTIONS:
        raise ValueError(f"Unsupported List-Major candidate-group tile: {candidate_groups}")
    num_lists = int(list_count.numel())
    edge_query_ids_sorted, edge_sorted_to_original, list_query_prefix, list_query_count = (
        _build_list_major_edges(probe_cids, num_lists)
    )
    list_query_tile_count = torch.div(
        list_query_count + (query_tile - 1),
        query_tile,
        rounding_mode="floor",
    )
    task_count_per_list = list_query_tile_count
    task_prefix = torch.cumsum(task_count_per_list, dim=0) - task_count_per_list
    num_tasks = int(task_count_per_list.sum().item())
    if num_tasks:
        task_list_ids = torch.repeat_interleave(
            torch.arange(num_lists, device=list_count.device, dtype=torch.int64),
            task_count_per_list,
            output_size=num_tasks,
        )
        task_positions = torch.arange(num_tasks, device=list_count.device, dtype=torch.int64)
        task_query_tile_ids = task_positions - task_prefix[task_list_ids]
    else:
        task_list_ids = torch.empty((0,), dtype=torch.int64, device=list_count.device)
        task_query_tile_ids = task_list_ids
    return {
        "edge_query_ids_sorted": edge_query_ids_sorted,
        "edge_sorted_to_original": edge_sorted_to_original,
        "list_query_prefix": list_query_prefix,
        "list_query_count": list_query_count,
        "task_list_ids": task_list_ids.contiguous(),
        "task_query_tile_ids": task_query_tile_ids.contiguous(),
        "max_vector_chunks": 1,
        "num_tasks": num_tasks,
    }


def _launch_scan(
    *,
    global_query_luts: torch.Tensor,
    static_vector_corrections: torch.Tensor,
    probe_distance_bias: torch.Tensor,
    list_codes_interleaved: torch.Tensor,
    list_codes_packed_16b: torch.Tensor | None,
    list_group_prefix: torch.Tensor,
    list_count: torch.Tensor,
    task_plan: dict[str, torch.Tensor | int],
    list_partial_keys: torch.Tensor,
    local_k: int,
    config: _ListMajorLaunchConfig,
) -> None:
    m, num_queries, ksub = global_query_luts.shape
    if list_codes_packed_16b is not None:
        _list_major_pq_packed_task_kernel[(int(task_plan["num_tasks"]),)](
            global_query_luts,
            static_vector_corrections,
            probe_distance_bias.reshape(-1),
            list_codes_packed_16b,
            list_group_prefix,
            list_count,
            task_plan["edge_query_ids_sorted"],
            task_plan["edge_sorted_to_original"],
            task_plan["list_query_prefix"],
            task_plan["list_query_count"],
            task_plan["task_list_ids"],
            task_plan["task_query_tile_ids"],
            list_partial_keys,
            num_queries,
            list_codes_interleaved.shape[1],
            M=m,
            KSUB=ksub,
            GROUP_SIZE=INTERLEAVED_GROUP_SIZE,
            QUERY_TILE=config.query_tile,
            GROUPS_PER_CHUNK=config.candidate_groups,
            LOCAL_K=local_k,
            PACKED_CHUNKS=(m + PACKED_CODE_CHUNK_BYTES - 1) // PACKED_CODE_CHUNK_BYTES,
            CODE_CHUNK_BYTES=PACKED_CODE_CHUNK_BYTES,
            CODE_WORD_BYTES=PACKED_CODE_WORD_BYTES,
            CODE_WORDS=PACKED_CODE_WORDS,
            USE_VECTOR_LUT=config.use_vector_lut,
            INF_KEY=_INF_KEY,
            num_warps=config.num_warps,
        )
        return
    _list_major_pq_task_kernel[(int(task_plan["num_tasks"]),)](
        global_query_luts,
        static_vector_corrections,
        probe_distance_bias.reshape(-1),
        list_codes_interleaved,
        list_group_prefix,
        list_count,
        task_plan["edge_query_ids_sorted"],
        task_plan["edge_sorted_to_original"],
        task_plan["list_query_prefix"],
        task_plan["list_query_count"],
        task_plan["task_list_ids"],
        task_plan["task_query_tile_ids"],
        list_partial_keys,
        num_queries,
        list_codes_interleaved.shape[1],
        M=m,
        KSUB=ksub,
        GROUP_SIZE=INTERLEAVED_GROUP_SIZE,
        QUERY_TILE=config.query_tile,
            GROUPS_PER_CHUNK=config.candidate_groups,
            LOCAL_K=local_k,
            USE_VECTOR_LUT=config.use_vector_lut,
            INF_KEY=_INF_KEY,
        num_warps=config.num_warps,
    )


def _autotune_launch_config(
    *,
    cache_key: tuple[object, ...],
    global_query_luts: torch.Tensor,
    static_vector_corrections: torch.Tensor,
    probe_distance_bias: torch.Tensor,
    list_codes_interleaved: torch.Tensor,
    list_codes_packed_16b: torch.Tensor | None,
    list_group_prefix: torch.Tensor,
    list_count: torch.Tensor,
    probe_cids: torch.Tensor,
    local_k: int,
) -> _ListMajorLaunchConfig:
    cached = _cached_autotune_config(cache_key)
    if cached is not None:
        return cached

    best_config: _ListMajorLaunchConfig | None = None
    best_ms = float("inf")
    for config in _candidate_launch_configs(
        m=int(global_query_luts.shape[0]),
        local_k=local_k,
        lut_dtype=global_query_luts.dtype,
        ksub=int(global_query_luts.shape[2]),
        device=global_query_luts.device,
    ):
        task_plan = _build_list_major_task_plan(
            probe_cids=probe_cids,
            list_count=list_count,
            query_tile=config.query_tile,
            candidate_groups=config.candidate_groups,
        )
        if int(task_plan["num_tasks"]) == 0:
            continue
        num_edges = int(probe_cids.numel())
        list_partial_keys = torch.full(
            (num_edges, local_k),
            _INF_KEY,
            dtype=torch.int64,
            device=global_query_luts.device,
        )
        for _ in range(_AUTOTUNE_WARMUP_RUNS):
            _launch_scan(
                global_query_luts=global_query_luts,
                static_vector_corrections=static_vector_corrections,
                probe_distance_bias=probe_distance_bias,
                list_codes_interleaved=list_codes_interleaved,
                list_codes_packed_16b=list_codes_packed_16b,
                list_group_prefix=list_group_prefix,
                list_count=list_count,
                task_plan=task_plan,
                list_partial_keys=list_partial_keys,
                local_k=local_k,
                config=config,
            )
        torch.cuda.synchronize(global_query_luts.device)
        elapsed_ms = 0.0
        for _ in range(_AUTOTUNE_TIMED_RUNS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(global_query_luts.device))
            _launch_scan(
                global_query_luts=global_query_luts,
                static_vector_corrections=static_vector_corrections,
                probe_distance_bias=probe_distance_bias,
                list_codes_interleaved=list_codes_interleaved,
                list_codes_packed_16b=list_codes_packed_16b,
                list_group_prefix=list_group_prefix,
                list_count=list_count,
                task_plan=task_plan,
                list_partial_keys=list_partial_keys,
                local_k=local_k,
                config=config,
            )
            end.record(torch.cuda.current_stream(global_query_luts.device))
            end.synchronize()
            elapsed_ms += start.elapsed_time(end)
        average_ms = elapsed_ms / _AUTOTUNE_TIMED_RUNS
        if average_ms < best_ms:
            best_ms = average_ms
            best_config = config

    if best_config is None:  # pragma: no cover - guarded by validation
        raise ValueError("No supported IVFPQ list-major launch configuration")
    _AUTOTUNE_CONFIG_CACHE[cache_key] = best_config
    _persist_autotune_config(cache_key, best_config)
    return best_config


def triton_topk(
    *,
    global_query_luts: torch.Tensor,
    static_vector_corrections: torch.Tensor,
    probe_distance_bias: torch.Tensor,
    list_codes_interleaved: torch.Tensor,
    list_ids_interleaved: torch.Tensor,
    list_group_prefix: torch.Tensor,
    list_count: torch.Tensor,
    probe_cids: torch.Tensor,
    topk: int,
    output_distances: torch.Tensor,
    output_ids: torch.Tensor,
    list_partial_keys: torch.Tensor | None = None,
    list_codes_packed_16b: torch.Tensor | None = None,
    autotune: bool = False,
    autotune_key: tuple[object, ...] | None = None,
    on_list_scan_complete: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Run active-task list-major IVF-PQ ADC and return local top-k keys."""
    if global_query_luts.ndim != 3:
        raise ValueError("List-major global query LUT must have shape [m, query, ksub]")
    m, num_queries, ksub = global_query_luts.shape
    if global_query_luts.dtype not in _SUPPORTED_GLOBAL_LUT_DTYPES:
        raise ValueError("Global query LUT must use FP32 or FP16 storage")
    if probe_cids.ndim != 2 or probe_cids.shape[0] != num_queries:
        raise ValueError("Probe centroid IDs must have shape [queries, nprobe]")
    nprobe = int(probe_cids.shape[1])
    plan = _list_major_plan(topk=topk, nprobe=nprobe)
    if not can_run_topk(
        device=global_query_luts.device,
        topk=topk,
        m=m,
        ksub=ksub,
        nprobe=nprobe,
        num_code_vectors=list_ids_interleaved.numel(),
        code_dtype=list_codes_interleaved.dtype,
        use_packed_16b_codes=list_codes_packed_16b is not None,
    ):
        raise ValueError("IVFPQ Triton list-major top-k is unavailable for these search parameters")
    assert plan is not None
    local_k, final_width = plan
    if list_codes_interleaved.shape != (m, list_ids_interleaved.numel()):
        raise ValueError("Interleaved PQ code cache must have shape [m, padded_num_indexed_vectors]")
    if static_vector_corrections.shape != (list_ids_interleaved.numel(),):
        raise ValueError("Static vector corrections must match padded interleaved vector count")
    if static_vector_corrections.dtype != torch.float32:
        raise ValueError("Static vector corrections must use FP32 storage")
    if list_codes_packed_16b is not None:
        if m <= 16:
            raise ValueError("Packed PQ code scanning requires M > 16")
        packed_shape = (
            list_ids_interleaved.numel() // INTERLEAVED_GROUP_SIZE,
            (m + PACKED_CODE_CHUNK_BYTES - 1) // PACKED_CODE_CHUNK_BYTES,
            INTERLEAVED_GROUP_SIZE,
            PACKED_CODE_WORDS,
        )
        if list_codes_packed_16b.shape != packed_shape:
            raise ValueError(
                "Packed PQ code cache must have shape "
                f"{packed_shape}, got {tuple(list_codes_packed_16b.shape)}"
            )
        if list_codes_packed_16b.dtype != torch.uint64:
            raise ValueError("Packed PQ code cache must use uint64 storage")
    if probe_distance_bias.shape != probe_cids.shape:
        raise ValueError("Probe distance bias must match probe centroid IDs")
    if output_distances.ndim != 2 or output_distances.shape[0] != num_queries or output_distances.shape[1] < topk:
        raise ValueError("Output distance tensor must have shape [queries, >= topk]")
    if output_ids.ndim != 2 or output_ids.shape[0] != num_queries or output_ids.shape[1] < topk:
        raise ValueError("Output ID tensor must have shape [queries, >= topk]")
    if list_group_prefix.shape != list_count.shape or list_group_prefix.ndim != 1:
        raise ValueError("List group prefixes and counts must share shape [nlist]")
    tensors = [
        global_query_luts,
        static_vector_corrections,
        probe_distance_bias,
        list_codes_interleaved,
        list_ids_interleaved,
        list_group_prefix,
        list_count,
        probe_cids,
        output_distances,
        output_ids,
    ]
    if list_codes_packed_16b is not None:
        tensors.append(list_codes_packed_16b)
    if any(tensor.device != global_query_luts.device for tensor in tensors):
        raise ValueError("All Triton IVF-PQ list-major tensors must reside on the same device")

    global_query_luts = global_query_luts.contiguous()
    static_vector_corrections = static_vector_corrections.contiguous()
    probe_distance_bias = probe_distance_bias.contiguous()
    list_codes_interleaved = list_codes_interleaved.contiguous()
    if list_codes_packed_16b is not None:
        list_codes_packed_16b = list_codes_packed_16b.contiguous()
    list_ids_interleaved = list_ids_interleaved.contiguous()
    list_group_prefix = list_group_prefix.contiguous()
    list_count = list_count.contiguous()
    probe_cids = probe_cids.contiguous()

    cache_key = autotune_key or (
        "active_task",
        str(global_query_luts.device),
        m,
        ksub,
        nprobe,
        topk,
        num_queries,
        str(global_query_luts.dtype),
        bool(list_codes_packed_16b is not None),
        int(list_count.numel()),
    )
    config = _cached_autotune_config(cache_key)
    if config is None and autotune:
        config = _autotune_launch_config(
            cache_key=cache_key,
            global_query_luts=global_query_luts,
            static_vector_corrections=static_vector_corrections,
            probe_distance_bias=probe_distance_bias,
            list_codes_interleaved=list_codes_interleaved,
            list_codes_packed_16b=list_codes_packed_16b,
            list_group_prefix=list_group_prefix,
            list_count=list_count,
            probe_cids=probe_cids,
            local_k=local_k,
        )
    if config is None:
        config = _ListMajorLaunchConfig(
            query_tile=_estimated_query_tile(
                m=m,
                local_k=local_k,
                lut_dtype=global_query_luts.dtype,
                device=global_query_luts.device,
                ksub=ksub,
                candidate_groups=LIST_MAJOR_CANDIDATE_GROUPS,
                use_vector_lut=False,
            ),
            num_warps=4,
            candidate_groups=LIST_MAJOR_CANDIDATE_GROUPS,
            use_vector_lut=False,
        )

    task_plan = _build_list_major_task_plan(
        probe_cids=probe_cids,
        list_count=list_count,
        query_tile=config.query_tile,
        candidate_groups=config.candidate_groups,
    )
    num_edges = int(probe_cids.numel())
    if int(task_plan["num_tasks"]) == 0:
        output_distances.fill_(float("inf"))
        output_ids.fill_(-1)
        return torch.empty((num_edges, local_k), dtype=torch.int64, device=global_query_luts.device)

    list_partial_shape = (num_edges, local_k)
    if (
        list_partial_keys is None
        or list_partial_keys.shape != list_partial_shape
        or list_partial_keys.dtype != torch.int64
        or list_partial_keys.device != global_query_luts.device
    ):
        list_partial_keys = torch.empty(
            list_partial_shape,
            dtype=torch.int64,
            device=global_query_luts.device,
        )
    # Empty IVF lists have probe edges but no scan work. Initialize all rows
    # so the final merge sees INF_KEY for those edges.
    list_partial_keys.fill_(_INF_KEY)
    _launch_scan(
        global_query_luts=global_query_luts,
        static_vector_corrections=static_vector_corrections,
        probe_distance_bias=probe_distance_bias,
        list_codes_interleaved=list_codes_interleaved,
        list_codes_packed_16b=list_codes_packed_16b,
        list_group_prefix=list_group_prefix,
        list_count=list_count,
        task_plan=task_plan,
        list_partial_keys=list_partial_keys,
        local_k=local_k,
        config=config,
    )
    if on_list_scan_complete is not None:
        on_list_scan_complete()

    _finalize_list_major_topk_kernel[(num_queries,)](
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
    return list_partial_keys


triton_list_major_topk = triton_topk
