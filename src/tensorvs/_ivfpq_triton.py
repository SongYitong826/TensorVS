"""Triton query-major IVF-PQ scan with static per-vector corrections.

The index stores one FP32 scalar for every padded physical PQ vector:
``gamma[i] = sum_m 2 * <c[list(i), m], p[m, code(i, m)]>``.  Search builds a
query-global ``G[q,m,k] = ||q_m - p_mk||^2`` LUT and scans codes directly:
``distance(q, i) = bias(q, list(i)) + gamma[i] + sum_m G[q,m,code(i,m)]``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CUDA caller fallback covers this
    triton = None
    tl = None


INTERLEAVED_GROUP_SIZE = 32
PACKED_CODE_CHUNK_BYTES = 16
PACKED_CODE_WORD_BYTES = 8
PACKED_CODE_WORDS = PACKED_CODE_CHUNK_BYTES // PACKED_CODE_WORD_BYTES
SCAN_BLOCK_CANDIDATES = 128
MAX_LOCAL_K = 128
MAX_FINAL_WIDTH = 2048
DIRECT_SCAN_MAX_NPROBE = 128
LIST_TASK_MAX_MERGE_WIDTH = 2048
MAX_SUBSPACES = 64
MAX_PACKED_CHUNKED_SUBSPACES = 128
MAX_KSUB = 256
MAX_INDEXED_VECTORS = (1 << 31) - 1
_INF_KEY = 0x7F800000FFFFFFFF
_SUPPORTED_GLOBAL_LUT_DTYPES = {torch.float32, torch.float16}
_AUTOTUNE_WARMUP_RUNS = 1
_AUTOTUNE_TIMED_RUNS = 3
_AUTOTUNE_CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class _MAdaptiveLaunchConfig:
    """Compile-time settings for one CTA scan specialization."""

    scan_block_candidates: int
    num_warps: int
    probe_lists_per_cta: int


_AUTOTUNE_CONFIG_CACHE: dict[tuple[object, ...], _MAdaptiveLaunchConfig] = {}
_PERSISTED_AUTOTUNE_CONFIGS: dict[str, _MAdaptiveLaunchConfig] | None = None


def _default_launch_config() -> _MAdaptiveLaunchConfig:
    """Return the portable no-cache fallback before a workload is tuned."""
    return _MAdaptiveLaunchConfig(
        scan_block_candidates=256,
        num_warps=4,
        probe_lists_per_cta=1,
    )


def _autotune_cache_path() -> Path | None:
    """Return the overridable per-user cache path, or None when disabled."""
    configured_path = os.environ.get("TENSORVS_TRITON_AUTOTUNE_CACHE")
    if configured_path == "":
        return None
    if configured_path is not None:
        return Path(configured_path).expanduser()
    return Path.home() / ".cache" / "tensorvs" / "ivfpq_triton_autotune.json"


def _serialized_autotune_key(cache_key: tuple[object, ...]) -> str:
    return json.dumps(cache_key, separators=(",", ":"), ensure_ascii=True)


def _load_persisted_autotune_configs() -> dict[str, _MAdaptiveLaunchConfig]:
    global _PERSISTED_AUTOTUNE_CONFIGS
    if _PERSISTED_AUTOTUNE_CONFIGS is not None:
        return _PERSISTED_AUTOTUNE_CONFIGS

    configs: dict[str, _MAdaptiveLaunchConfig] = {}
    cache_path = _autotune_cache_path()
    if cache_path is not None:
        try:
            payload = json.loads(cache_path.read_text())
            if payload.get("version") == _AUTOTUNE_CACHE_FORMAT_VERSION:
                for key, values in payload.get("entries", {}).items():
                    config = _MAdaptiveLaunchConfig(*values)
                    if (
                        config.scan_block_candidates in (128, 256, 512, 1024)
                        and config.num_warps in (4, 8)
                        and config.probe_lists_per_cta in (1, 2)
                    ):
                        configs[key] = config
        except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    _PERSISTED_AUTOTUNE_CONFIGS = configs
    return configs


def _persist_autotune_config(
    cache_key: tuple[object, ...],
    config: _MAdaptiveLaunchConfig,
) -> None:
    cache_path = _autotune_cache_path()
    if cache_path is None:
        return
    configs = _load_persisted_autotune_configs()
    configs[_serialized_autotune_key(cache_key)] = config
    payload = {
        "version": _AUTOTUNE_CACHE_FORMAT_VERSION,
        "entries": {
            key: [
                value.scan_block_candidates,
                value.num_warps,
                value.probe_lists_per_cta,
            ]
            for key, value in configs.items()
        },
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
        temporary_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        os.replace(temporary_path, cache_path)
    except OSError:
        pass


def _cached_autotune_config(
    cache_key: tuple[object, ...],
) -> _MAdaptiveLaunchConfig | None:
    cached = _AUTOTUNE_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cached = _load_persisted_autotune_configs().get(_serialized_autotune_key(cache_key))
    if cached is not None:
        _AUTOTUNE_CONFIG_CACHE[cache_key] = cached
    return cached


if triton is not None:
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


    @triton.jit
    def _list_cta_pq_topk_kernel(
        global_query_luts_ptr,
        static_vector_corrections_ptr,
        probe_distance_bias_ptr,
        codes_by_subspace_ptr,
        probe_group_prefix_ptr,
        probe_list_count_ptr,
        list_partial_keys_ptr,
        num_queries,
        num_code_vectors,
        nprobe,
        GRID_X: tl.constexpr,
        M: tl.constexpr,
        KSUB: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        SCAN_BLOCK_C: tl.constexpr,
        PROBE_LISTS_PER_CTA: tl.constexpr,
        LOCAL_K: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Scan one query-local group of complete probed lists in one CTA.

        ``PROBE_LISTS_PER_CTA`` is a small compile-time group (one or two
        lists).  The program keeps an exact top-k across every real list in
        that group, so the caller only needs the usual query-level merge.
        """
        query_idx = tl.program_id(0)
        cta_probe_group_idx = tl.program_id(1)
        probe_start = cta_probe_group_idx * PROBE_LISTS_PER_CTA

        candidate_offsets = tl.arange(0, SCAN_BLOCK_C)
        group_offsets = candidate_offsets // GROUP_SIZE
        lane_offsets = candidate_offsets % GROUP_SIZE
        best_keys = tl.full((LOCAL_K,), INF_KEY, tl.int64)

        for probe_offset in tl.static_range(0, PROBE_LISTS_PER_CTA):
            probe_idx = probe_start + probe_offset
            valid_probe = probe_idx < nprobe
            edge_idx = query_idx * nprobe + probe_idx
            vector_count = tl.load(
                probe_list_count_ptr + edge_idx,
                mask=valid_probe,
                other=0,
            ).to(tl.int32)
            group_prefix = tl.load(
                probe_group_prefix_ptr + edge_idx,
                mask=valid_probe,
                other=0,
            ).to(tl.int64)
            distance_bias = tl.load(
                probe_distance_bias_ptr + edge_idx,
                mask=valid_probe,
                other=0.0,
            ).to(tl.float32)
            group_count = (vector_count + GROUP_SIZE - 1) // GROUP_SIZE

            for local_group_start in tl.range(0, group_count, SCAN_BLOCK_C // GROUP_SIZE):
                local_group_ids = local_group_start + group_offsets
                local_vector_offsets = local_group_ids * GROUP_SIZE + lane_offsets
                valid_candidates = valid_probe & (local_vector_offsets < vector_count)
                physical_positions = (
                    (group_prefix + local_group_ids.to(tl.int64)) * GROUP_SIZE + lane_offsets
                )
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
                        cache_modifier=".ca",
                    )
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

        ranks = tl.arange(0, LOCAL_K)
        tl.store(
            list_partial_keys_ptr
            + (query_idx * GRID_X + cta_probe_group_idx) * LOCAL_K
            + ranks,
            best_keys,
        )


    @triton.jit
    def _list_cta_pq_packed_chunked_topk_kernel(
        global_query_luts_ptr,
        static_vector_corrections_ptr,
        probe_distance_bias_ptr,
        packed_codes_ptr,
        probe_group_prefix_ptr,
        probe_list_count_ptr,
        list_partial_keys_ptr,
        num_queries,
        nprobe,
        GRID_X: tl.constexpr,
        M: tl.constexpr,
        KSUB: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        SCAN_BLOCK_C: tl.constexpr,
        PROBE_LISTS_PER_CTA: tl.constexpr,
        LOCAL_K: tl.constexpr,
        PACKED_CHUNKS: tl.constexpr,
        CODE_CHUNK_BYTES: tl.constexpr,
        CODE_WORD_BYTES: tl.constexpr,
        CODE_WORDS: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Scan wide PQ codes one 16-byte chunk at a time.

        Only the distance vector and local top-k survive across chunk-loop
        iterations.  Keeping the chunk loop rolled avoids generating one
        straight-line LUT-gather sequence per subspace, whose code size and
        register live ranges become prohibitive at M=96.  The final partial
        chunk is zero padded in storage and masked here, so the same loop also
        supports M values that are not multiples of sixteen.
        """
        query_idx = tl.program_id(0)
        cta_probe_group_idx = tl.program_id(1)
        probe_start = cta_probe_group_idx * PROBE_LISTS_PER_CTA

        candidate_offsets = tl.arange(0, SCAN_BLOCK_C)
        group_offsets = candidate_offsets // GROUP_SIZE
        lane_offsets = candidate_offsets % GROUP_SIZE
        ranks = tl.arange(0, LOCAL_K)
        best_keys = tl.full((LOCAL_K,), INF_KEY, tl.int64)

        for probe_offset in tl.static_range(0, PROBE_LISTS_PER_CTA):
            probe_idx = probe_start + probe_offset
            valid_probe = probe_idx < nprobe
            edge_idx = query_idx * nprobe + probe_idx
            vector_count = tl.load(
                probe_list_count_ptr + edge_idx,
                mask=valid_probe,
                other=0,
            ).to(tl.int32)
            group_prefix = tl.load(
                probe_group_prefix_ptr + edge_idx,
                mask=valid_probe,
                other=0,
            ).to(tl.int64)
            distance_bias = tl.load(
                probe_distance_bias_ptr + edge_idx,
                mask=valid_probe,
                other=0.0,
            ).to(tl.float32)
            group_count = (vector_count + GROUP_SIZE - 1) // GROUP_SIZE

            for local_group_start in tl.range(0, group_count, SCAN_BLOCK_C // GROUP_SIZE):
                local_group_ids = local_group_start + group_offsets
                local_vector_offsets = local_group_ids * GROUP_SIZE + lane_offsets
                valid_candidates = valid_probe & (local_vector_offsets < vector_count)
                physical_groups = group_prefix + local_group_ids.to(tl.int64)
                physical_positions = physical_groups * GROUP_SIZE + lane_offsets
                packed_chunk_zero = (
                    (physical_groups * PACKED_CHUNKS * GROUP_SIZE + lane_offsets)
                    * CODE_WORDS
                )

                distances = tl.load(
                    static_vector_corrections_ptr + physical_positions,
                    mask=valid_candidates,
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)
                for chunk_idx in tl.range(
                    0,
                    PACKED_CHUNKS,
                    1,
                    loop_unroll_factor=1,
                    disable_licm=True,
                ):
                    packed_chunk = packed_chunk_zero + chunk_idx * GROUP_SIZE * CODE_WORDS
                    for word_idx in tl.static_range(0, CODE_WORDS):
                        packed_word = tl.load(
                            packed_codes_ptr + packed_chunk + word_idx,
                            mask=valid_candidates,
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
                            lut_values = tl.load(
                                global_query_luts_ptr
                                + subspace_idx * num_queries * KSUB
                                + query_idx * KSUB
                                + codes,
                                mask=valid_candidates & subspace_valid,
                                other=0.0,
                                cache_modifier=".ca",
                            )
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
            list_partial_keys_ptr
            + (query_idx * GRID_X + cta_probe_group_idx) * LOCAL_K
            + ranks,
            best_keys,
        )



def is_available(device: torch.device) -> bool:
    """Return whether the vector-correction direct scan can run on ``device``."""
    return triton is not None and device.type == "cuda"


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _list_cta_plan(
    *,
    topk: int,
    nprobe: int,
) -> tuple[int, int] | None:
    """Plan the one-CTA-per-probed-list route and its final query merge."""
    if topk <= 0 or nprobe <= 0 or nprobe > DIRECT_SCAN_MAX_NPROBE:
        return None
    local_k = max(2, _next_power_of_two(topk))
    if local_k > MAX_LOCAL_K:
        return None
    final_width = _next_power_of_two(nprobe * local_k)
    if final_width > LIST_TASK_MAX_MERGE_WIDTH:
        return None
    return local_k, final_width


def _list_cta_launch_plan(
    *,
    topk: int,
    nprobe: int,
    m: int,
) -> tuple[int, int, _MAdaptiveLaunchConfig, int] | None:
    """Plan the fallback CTA-list scanner and its final query merge.

    Returns ``(local_k, cta_probe_groups, launch_config, final_width)``.  A
    group emits one local top-k that is exact across its covered probe lists.
    """
    base_plan = _list_cta_plan(topk=topk, nprobe=nprobe)
    if base_plan is None:
        return None
    local_k, _ = base_plan
    launch_config = _default_launch_config()
    cta_probe_groups = math.ceil(nprobe / launch_config.probe_lists_per_cta)
    final_width = _next_power_of_two(cta_probe_groups * local_k)
    if final_width > LIST_TASK_MAX_MERGE_WIDTH:
        return None
    return local_k, cta_probe_groups, launch_config, final_width


def can_run_topk(
    *,
    device: torch.device,
    topk: int,
    m: int,
    ksub: int,
    nprobe: int,
    num_code_vectors: int,
    code_dtype: torch.dtype,
    use_packed_16b_codes: bool = True,
) -> bool:
    """Check static requirements for query/probe CTA scanning."""
    max_subspaces = (
        MAX_PACKED_CHUNKED_SUBSPACES if use_packed_16b_codes else MAX_SUBSPACES
    )
    return (
        is_available(device)
        and code_dtype == torch.uint8
        and 0 < m <= max_subspaces
        and 0 < ksub <= MAX_KSUB
        and 0 < num_code_vectors <= MAX_INDEXED_VECTORS
        and _list_cta_launch_plan(topk=topk, nprobe=nprobe, m=m) is not None
    )


def _candidate_launch_configs() -> tuple[_MAdaptiveLaunchConfig, ...]:
    """Return the small, device-independent Triton specialization sweep."""
    configs = [_default_launch_config()]
    raw_candidates = (
        (128, 4, 1),
        (256, 8, 1),
        (512, 4, 1),
        (512, 8, 1),
        (1024, 4, 1),
        (256, 4, 2),
    )
    configs.extend(_MAdaptiveLaunchConfig(*values) for values in raw_candidates)
    return tuple(dict.fromkeys(configs))


def _resolve_topk_launch_plan(
    *,
    m: int,
    topk: int,
    nprobe: int,
    launch_config: _MAdaptiveLaunchConfig | None = None,
) -> tuple[int, int, _MAdaptiveLaunchConfig, int] | None:
    base_plan = _list_cta_launch_plan(topk=topk, nprobe=nprobe, m=m)
    if base_plan is None:
        return None
    local_k, _, default_config, _ = base_plan
    config = launch_config or default_config
    cta_probe_groups = math.ceil(nprobe / config.probe_lists_per_cta)
    final_width = _next_power_of_two(cta_probe_groups * local_k)
    if final_width > LIST_TASK_MAX_MERGE_WIDTH:
        return None
    return local_k, cta_probe_groups, config, final_width


def _launch_topk_scan(
    *,
    global_query_luts: torch.Tensor,
    static_vector_corrections: torch.Tensor,
    probe_distance_bias: torch.Tensor,
    list_codes_interleaved: torch.Tensor,
    list_ids_interleaved: torch.Tensor,
    probe_group_prefix: torch.Tensor,
    probe_list_count: torch.Tensor,
    list_partial_keys: torch.Tensor,
    list_codes_packed_16b: torch.Tensor | None,
    m: int,
    ksub: int,
    num_queries: int,
    nprobe: int,
    local_k: int,
    cta_probe_groups: int,
    launch_config: _MAdaptiveLaunchConfig,
) -> None:
    """Launch only the CTA scan; shared by tuning and normal execution."""
    if list_codes_packed_16b is not None:
        _list_cta_pq_packed_chunked_topk_kernel[(num_queries, cta_probe_groups)](
            global_query_luts,
            static_vector_corrections,
            probe_distance_bias,
            list_codes_packed_16b,
            probe_group_prefix,
            probe_list_count,
            list_partial_keys,
            num_queries,
            nprobe,
            GRID_X=cta_probe_groups,
            M=m,
            KSUB=ksub,
            GROUP_SIZE=INTERLEAVED_GROUP_SIZE,
            SCAN_BLOCK_C=launch_config.scan_block_candidates,
            PROBE_LISTS_PER_CTA=launch_config.probe_lists_per_cta,
            LOCAL_K=local_k,
            PACKED_CHUNKS=math.ceil(m / PACKED_CODE_CHUNK_BYTES),
            CODE_CHUNK_BYTES=PACKED_CODE_CHUNK_BYTES,
            CODE_WORD_BYTES=PACKED_CODE_WORD_BYTES,
            CODE_WORDS=PACKED_CODE_WORDS,
            INF_KEY=_INF_KEY,
            num_warps=launch_config.num_warps,
        )
        return
    _list_cta_pq_topk_kernel[(num_queries, cta_probe_groups)](
        global_query_luts,
        static_vector_corrections,
        probe_distance_bias,
        list_codes_interleaved,
        probe_group_prefix,
        probe_list_count,
        list_partial_keys,
        num_queries,
        list_ids_interleaved.numel(),
        nprobe,
        GRID_X=cta_probe_groups,
        M=m,
        KSUB=ksub,
        GROUP_SIZE=INTERLEAVED_GROUP_SIZE,
        SCAN_BLOCK_C=launch_config.scan_block_candidates,
        PROBE_LISTS_PER_CTA=launch_config.probe_lists_per_cta,
        LOCAL_K=local_k,
        INF_KEY=_INF_KEY,
        num_warps=launch_config.num_warps,
    )


def _autotune_topk_launch_config(
    *,
    cache_key: tuple[object, ...],
    global_query_luts: torch.Tensor,
    static_vector_corrections: torch.Tensor,
    probe_distance_bias: torch.Tensor,
    list_codes_interleaved: torch.Tensor,
    list_ids_interleaved: torch.Tensor,
    probe_group_prefix: torch.Tensor,
    probe_list_count: torch.Tensor,
    list_codes_packed_16b: torch.Tensor | None,
    topk: int,
) -> _MAdaptiveLaunchConfig:
    """Measure a few CTA shapes once and cache the fastest scan-only plan."""
    cached = _cached_autotune_config(cache_key)
    if cached is not None:
        return cached

    m, num_queries, ksub = global_query_luts.shape
    nprobe = int(probe_group_prefix.shape[1])
    device = global_query_luts.device
    best_config: _MAdaptiveLaunchConfig | None = None
    best_ms = float("inf")
    for config in _candidate_launch_configs():
        plan = _resolve_topk_launch_plan(
            m=m,
            topk=topk,
            nprobe=nprobe,
            launch_config=config,
        )
        if plan is None:
            continue
        local_k, cta_probe_groups, _, _ = plan
        scratch = torch.empty(
            (num_queries, cta_probe_groups, local_k),
            dtype=torch.int64,
            device=device,
        )
        for _ in range(_AUTOTUNE_WARMUP_RUNS):
            _launch_topk_scan(
                global_query_luts=global_query_luts,
                static_vector_corrections=static_vector_corrections,
                probe_distance_bias=probe_distance_bias,
                list_codes_interleaved=list_codes_interleaved,
                list_ids_interleaved=list_ids_interleaved,
                probe_group_prefix=probe_group_prefix,
                probe_list_count=probe_list_count,
                list_partial_keys=scratch,
                list_codes_packed_16b=list_codes_packed_16b,
                m=m,
                ksub=ksub,
                num_queries=num_queries,
                nprobe=nprobe,
                local_k=local_k,
                cta_probe_groups=cta_probe_groups,
                launch_config=config,
            )
        torch.cuda.synchronize(device)
        elapsed_ms = 0.0
        for _ in range(_AUTOTUNE_TIMED_RUNS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(device))
            _launch_topk_scan(
                global_query_luts=global_query_luts,
                static_vector_corrections=static_vector_corrections,
                probe_distance_bias=probe_distance_bias,
                list_codes_interleaved=list_codes_interleaved,
                list_ids_interleaved=list_ids_interleaved,
                probe_group_prefix=probe_group_prefix,
                probe_list_count=probe_list_count,
                list_partial_keys=scratch,
                list_codes_packed_16b=list_codes_packed_16b,
                m=m,
                ksub=ksub,
                num_queries=num_queries,
                nprobe=nprobe,
                local_k=local_k,
                cta_probe_groups=cta_probe_groups,
                launch_config=config,
            )
            end.record(torch.cuda.current_stream(device))
            end.synchronize()
            elapsed_ms += start.elapsed_time(end)
        average_ms = elapsed_ms / _AUTOTUNE_TIMED_RUNS
        if average_ms < best_ms:
            best_ms = average_ms
            best_config = config

    if best_config is None:  # pragma: no cover - standard validation precedes tuning
        raise ValueError("No supported IVFPQ Triton autotune launch configuration")
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
    probe_group_prefix: torch.Tensor,
    probe_list_count: torch.Tensor,
    topk: int,
    output_distances: torch.Tensor,
    output_ids: torch.Tensor,
    list_partial_keys: torch.Tensor | None = None,
    list_codes_packed_16b: torch.Tensor | None = None,
    autotune: bool = False,
    autotune_key: tuple[object, ...] | None = None,
    on_list_scan_complete: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Run exact IVF-PQ ADC with query/probe CTA groups.

    A program scans one complete selected list by default and keeps its exact
    group-local top-k across all code tiles.  ``list_codes_packed_16b`` uses a
    rolled 16-byte chunk loop for M>16, avoiding code-size and register growth
    without lower-bound bookkeeping or a per-query shared kth threshold.
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
    nprobe = int(probe_group_prefix.shape[1])
    if probe_group_prefix.shape[0] != num_queries:
        raise ValueError("Probe metadata query count does not match the global query LUT")
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
        raise ValueError("IVFPQ Triton CTA-list top-k is unavailable for these search parameters")
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
    if list_codes_packed_16b is not None:
        if m <= 16:
            raise ValueError("Packed PQ code scanning requires M > 16")
        packed_shape = (
            list_ids_interleaved.numel() // INTERLEAVED_GROUP_SIZE,
            math.ceil(m / PACKED_CODE_CHUNK_BYTES),
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
        tensors += (list_codes_packed_16b,)
    if any(tensor.device != global_query_luts.device for tensor in tensors):
        raise ValueError("All Triton IVF-PQ tensors must reside on the same device")

    packed_codes = list_codes_packed_16b is not None
    plan = _resolve_topk_launch_plan(
        m=m,
        topk=topk,
        nprobe=nprobe,
    )
    if plan is None:  # pragma: no cover - guarded by can_run_topk
        raise ValueError("Unsupported IVFPQ Triton CTA-list top-k plan")
    local_k, cta_probe_groups, launch_config, final_width = plan

    list_partial_shape = (num_queries, cta_probe_groups, local_k)
    if (
        list_partial_keys is None
        or list_partial_keys.shape != list_partial_shape
        or list_partial_keys.dtype != torch.int64
        or list_partial_keys.device != global_query_luts.device
    ):
        list_partial_keys = torch.empty(
            list_partial_shape,
            device=global_query_luts.device,
            dtype=torch.int64,
        )

    global_query_luts = global_query_luts.contiguous()
    static_vector_corrections = static_vector_corrections.contiguous()
    probe_distance_bias = probe_distance_bias.contiguous()
    list_codes_interleaved = list_codes_interleaved.contiguous()
    list_ids_interleaved = list_ids_interleaved.contiguous()
    probe_group_prefix = probe_group_prefix.contiguous()
    probe_list_count = probe_list_count.contiguous()
    if list_codes_packed_16b is not None:
        list_codes_packed_16b = list_codes_packed_16b.contiguous()

    cache_key = autotune_key or (
        str(global_query_luts.device),
        m,
        ksub,
        nprobe,
        topk,
        num_queries,
        packed_codes,
    )
    cached_launch_config = _cached_autotune_config(cache_key)
    if cached_launch_config is not None:
        launch_config = cached_launch_config
    elif autotune:
        launch_config = _autotune_topk_launch_config(
            cache_key=cache_key,
            global_query_luts=global_query_luts,
            static_vector_corrections=static_vector_corrections,
            probe_distance_bias=probe_distance_bias,
            list_codes_interleaved=list_codes_interleaved,
            list_ids_interleaved=list_ids_interleaved,
            probe_group_prefix=probe_group_prefix,
            probe_list_count=probe_list_count,
            list_codes_packed_16b=list_codes_packed_16b,
            topk=topk,
        )
    if cached_launch_config is not None or autotune:
        tuned_plan = _resolve_topk_launch_plan(
            m=m,
            topk=topk,
            nprobe=nprobe,
            launch_config=launch_config,
        )
        assert tuned_plan is not None
        local_k, cta_probe_groups, launch_config, final_width = tuned_plan
        tuned_shape = (num_queries, cta_probe_groups, local_k)
        if list_partial_keys.shape != tuned_shape:
            list_partial_keys = torch.empty(tuned_shape, dtype=torch.int64, device=global_query_luts.device)

    _launch_topk_scan(
        global_query_luts=global_query_luts,
        static_vector_corrections=static_vector_corrections,
        probe_distance_bias=probe_distance_bias,
        list_codes_interleaved=list_codes_interleaved,
        list_ids_interleaved=list_ids_interleaved,
        probe_group_prefix=probe_group_prefix,
        probe_list_count=probe_list_count,
        list_partial_keys=list_partial_keys,
        list_codes_packed_16b=list_codes_packed_16b,
        m=m,
        ksub=ksub,
        num_queries=num_queries,
        nprobe=nprobe,
        local_k=local_k,
        cta_probe_groups=cta_probe_groups,
        launch_config=launch_config,
    )
    if on_list_scan_complete is not None:
        on_list_scan_complete()

    _finalize_direct_pq_topk_kernel[(num_queries,)](
        list_partial_keys,
        list_ids_interleaved,
        output_distances,
        output_ids,
        topk,
        GRID_X=cta_probe_groups,
        LOCAL_K=local_k,
        FINAL_WIDTH=final_width,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )
    return list_partial_keys
