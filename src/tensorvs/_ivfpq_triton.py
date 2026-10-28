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
# The original one-stage CTA-list merge was intentionally limited to this
# width.  Query-major IVF-PQ now reduces wider candidate sets in multiple
# stages, while keeping every individual ``tl.topk`` at MAX_FINAL_WIDTH.
MAX_QUERY_MAJOR_NPROBE = 4096
DIRECT_SCAN_MAX_NPROBE = 128
LIST_TASK_MAX_MERGE_WIDTH = 2048
MAX_SUBSPACES = 64
# One virtual scan task covers at most 128 interleaved 32-vector groups
# (4096 vectors).  Lists larger than this are represented by multiple probe
# tasks at search time, without changing the IVF assignment or candidates.
# The fixed rectangular shape avoids compiling a different merge kernel for
# different query batches with different numbers of selected non-empty chunks.
LIST_SCAN_CHUNK_GROUPS = 128
# The rolled 16-byte chunk scanner uses a runtime chunk loop, so keep the
# admission limit high enough for very wide PQ configurations. Actual Triton
# compiler/register/resource limits can still reject an individual workload.
MAX_PACKED_CHUNKED_SUBSPACES = 1024
MAX_KSUB = 256
MAX_INDEXED_VECTORS = (1 << 31) - 1
_INF_KEY = 0x7F800000FFFFFFFF
_SUPPORTED_GLOBAL_LUT_DTYPES = {torch.float32, torch.float16}
_AUTOTUNE_WARMUP_RUNS = 1
_AUTOTUNE_TIMED_RUNS = 3
# The packed scanner's tail handling and CTA grid mapping are part of the
# launch performance contract. Invalidate persisted choices when either
# specialization changes so cached choices are retuned for the current kernel.
_AUTOTUNE_CACHE_FORMAT_VERSION = 7


@dataclass(frozen=True)
class _MAdaptiveLaunchConfig:
    """Compile-time settings for one CTA scan specialization."""

    scan_block_candidates: int
    num_warps: int
    probe_lists_per_cta: int


_AUTOTUNE_CONFIG_CACHE: dict[tuple[object, ...], _MAdaptiveLaunchConfig] = {}
_PERSISTED_AUTOTUNE_CONFIGS: dict[str, _MAdaptiveLaunchConfig] | None = None


def plan_probe_chunks(
    *,
    nprobe: int,
    max_list_count: int,
    nprobe_limit: int = MAX_QUERY_MAJOR_NPROBE,
) -> tuple[int, int]:
    """Return ``(groups_per_chunk, max_chunks)`` without CUDA allocations."""
    if nprobe <= 0:
        raise ValueError("At least one probe list is required")
    if max_list_count < 0:
        raise ValueError(f"max_list_count must be non-negative, got {max_list_count}")
    if nprobe_limit <= 0:
        raise ValueError(f"nprobe_limit must be positive, got {nprobe_limit}")
    if nprobe > nprobe_limit:
        raise ValueError(
            f"Selected nprobe={nprobe} exceeds the scan limit {nprobe_limit}"
        )

    chunk_groups = LIST_SCAN_CHUNK_GROUPS
    max_chunks = max(
        1,
        math.ceil(max(int(max_list_count), 1) / (INTERLEAVED_GROUP_SIZE * chunk_groups)),
    )
    max_allowed_chunks = max(1, nprobe_limit // nprobe)
    if max_chunks > max_allowed_chunks:
        max_chunks = max_allowed_chunks
        chunk_groups = max(
            chunk_groups,
            math.ceil(
                max(int(max_list_count), 1)
                / (INTERLEAVED_GROUP_SIZE * max_chunks)
            ),
        )
    return int(chunk_groups), int(max_chunks)


def chunk_probe_metadata(
    probe_group_prefix: torch.Tensor,
    probe_list_count: torch.Tensor,
    probe_distance_bias: torch.Tensor,
    *,
    max_list_count: int,
    nprobe_limit: int = MAX_QUERY_MAJOR_NPROBE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Split selected oversized lists into virtual scan chunks.

    The returned tensors keep the original list-major order for each query.
    Oversized lists get multiple virtual slots, while shorter lists use
    zero-sized slots for the remaining positions. A virtual slot starts at a
    group boundary, so the existing interleaved scanner can scan it as an
    ordinary list without changing any candidate IDs or distances.

    ``max_list_count`` is taken from the resident index statistics rather than
    the query-specific selection.  This keeps the launch shape rectangular
    while avoiding a GPU synchronization for every search.
    """
    if probe_group_prefix.ndim != 2:
        raise ValueError("Probe group prefixes must have shape [query, nprobe]")
    if probe_list_count.shape != probe_group_prefix.shape:
        raise ValueError("Probe list counts must match probe group prefixes")
    if probe_distance_bias.shape != probe_group_prefix.shape:
        raise ValueError("Probe distance bias must match probe group prefixes")
    nprobe = int(probe_group_prefix.shape[1])
    chunk_groups, max_chunks = plan_probe_chunks(
        nprobe=nprobe,
        max_list_count=max_list_count,
        nprobe_limit=nprobe_limit,
    )

    device = probe_list_count.device
    chunk_indices = torch.arange(max_chunks, device=device, dtype=probe_list_count.dtype)
    group_starts = chunk_indices * int(chunk_groups)
    vector_starts = group_starts * INTERLEAVED_GROUP_SIZE

    chunk_prefix = probe_group_prefix.unsqueeze(-1) + group_starts
    remaining = probe_list_count.unsqueeze(-1) - vector_starts
    chunk_capacity = int(chunk_groups * INTERLEAVED_GROUP_SIZE)
    chunk_counts = remaining.clamp(min=0, max=chunk_capacity)
    chunk_bias = probe_distance_bias.unsqueeze(-1).expand(-1, -1, max_chunks)

    query_count = int(probe_group_prefix.shape[0])
    return (
        chunk_prefix.reshape(query_count, nprobe * max_chunks).contiguous(),
        chunk_counts.reshape(query_count, nprobe * max_chunks).contiguous(),
        chunk_bias.reshape(query_count, nprobe * max_chunks).contiguous(),
        int(chunk_groups),
        int(max_chunks),
    )


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
                        config.scan_block_candidates in (64, 128, 256, 512, 1024)
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
    def _merge_pq_partial_keys_kernel(
        input_keys_ptr,
        output_keys_ptr,
        input_count,
        output_groups,
        GRID_X: tl.constexpr,
        LOCAL_K: tl.constexpr,
        MERGE_WIDTH: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Reduce one contiguous candidate block to ``LOCAL_K`` keys.

        The scan kernel emits one local TopK row per probe-list group.  When
        those rows do not fit in one final ``tl.topk`` invocation, this
        kernel reduces several fixed-width chunks first.  Since every
        reduction retains at least ``LOCAL_K >= topk`` keys, the staged
        reduction is exact.
        """
        query_idx = tl.program_id(0)
        merge_group_idx = tl.program_id(1)
        offsets = tl.arange(0, MERGE_WIDTH)
        input_offset = merge_group_idx * MERGE_WIDTH + offsets
        keys = tl.load(
            input_keys_ptr + query_idx * input_count + input_offset,
            mask=input_offset < input_count,
            other=INF_KEY,
        )
        reduced_keys = ~tl.topk(~keys, LOCAL_K)
        ranks = tl.arange(0, LOCAL_K)
        tl.store(
            output_keys_ptr
            + (query_idx * output_groups + merge_group_idx) * LOCAL_K
            + ranks,
            reduced_keys,
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


    @triton.jit
    def _list_cta_pq_topk_kernel(
        global_query_luts_ptr,
        static_vector_corrections_ptr,
        probe_distance_bias_ptr,
        codes_by_subspace_ptr,
        probe_group_prefix_ptr,
        probe_list_count_ptr,
        probe_cids_ptr,
        list_group_prefix_ptr,
        global_list_count_ptr,
        list_partial_keys_ptr,
        num_queries,
        num_code_vectors,
        nprobe,
        scan_nprobe,
        GRID_X: tl.constexpr,
        M: tl.constexpr,
        KSUB: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        SCAN_BLOCK_C: tl.constexpr,
        PROBE_LISTS_PER_CTA: tl.constexpr,
        LOCAL_K: tl.constexpr,
        FUSED_METADATA: tl.constexpr,
        CHUNK_GROUPS: tl.constexpr,
        MAX_CHUNKS: tl.constexpr,
        INF_KEY: tl.constexpr,
    ):
        """Scan one query-local group of complete probed lists in one CTA.

        ``PROBE_LISTS_PER_CTA`` is a small compile-time group (one or two
        lists).  The program keeps an exact top-k across every real list in
        that group, so the caller only needs the usual query-level merge.
        """
        # Keep adjacent CTAs on the same query so the query-global LUT stays
        # hot in L2 while its probe groups are scanned.
        cta_probe_group_idx = tl.program_id(0)
        query_idx = tl.program_id(1)
        probe_start = cta_probe_group_idx * PROBE_LISTS_PER_CTA

        candidate_offsets = tl.arange(0, SCAN_BLOCK_C)
        group_offsets = candidate_offsets // GROUP_SIZE
        lane_offsets = candidate_offsets % GROUP_SIZE
        best_keys = tl.full((LOCAL_K,), INF_KEY, tl.int64)

        for probe_offset in tl.static_range(0, PROBE_LISTS_PER_CTA):
            probe_idx = probe_start + probe_offset
            valid_probe = probe_idx < scan_nprobe
            if FUSED_METADATA:
                source_probe_idx = probe_idx // MAX_CHUNKS
                chunk_idx = probe_idx % MAX_CHUNKS
                source_edge_idx = query_idx * nprobe + source_probe_idx
                centroid_id = tl.load(
                    probe_cids_ptr + source_edge_idx,
                    mask=valid_probe,
                    other=0,
                ).to(tl.int64)
                full_vector_count = tl.load(
                    global_list_count_ptr + centroid_id,
                    mask=valid_probe,
                    other=0,
                ).to(tl.int32)
                chunk_vector_start = chunk_idx * CHUNK_GROUPS * GROUP_SIZE
                vector_count = tl.minimum(
                    tl.maximum(full_vector_count - chunk_vector_start, 0),
                    CHUNK_GROUPS * GROUP_SIZE,
                )
                group_prefix = tl.load(
                    list_group_prefix_ptr + centroid_id,
                    mask=valid_probe,
                    other=0,
                ).to(tl.int64) + chunk_idx * CHUNK_GROUPS
                distance_bias = tl.load(
                    probe_distance_bias_ptr + source_edge_idx,
                    mask=valid_probe,
                    other=0.0,
                ).to(tl.float32)
            else:
                edge_idx = query_idx * scan_nprobe + probe_idx
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
                elif LOCAL_K <= SCAN_BLOCK_C:
                    tile_best_keys = ~tl.topk(~candidate_keys, LOCAL_K)
                    merged_keys = tl.ravel(tl.join(best_keys, tile_best_keys))
                    best_keys = ~tl.topk(~merged_keys, LOCAL_K)
                else:
                    padded_candidates = tl.cat(
                        candidate_keys,
                        tl.full((LOCAL_K - SCAN_BLOCK_C,), INF_KEY, tl.int64),
                        dim=0,
                    )
                    merged_keys = tl.ravel(tl.join(best_keys, padded_candidates))
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
        probe_cids_ptr,
        list_group_prefix_ptr,
        global_list_count_ptr,
        list_partial_keys_ptr,
        num_queries,
        nprobe,
        scan_nprobe,
        GRID_X: tl.constexpr,
        M: tl.constexpr,
        KSUB: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        SCAN_BLOCK_C: tl.constexpr,
        PROBE_LISTS_PER_CTA: tl.constexpr,
        LOCAL_K: tl.constexpr,
        FUSED_METADATA: tl.constexpr,
        CHUNK_GROUPS: tl.constexpr,
        MAX_CHUNKS: tl.constexpr,
        PACKED_CHUNKS: tl.constexpr,
        FULL_PACKED_CHUNKS: tl.constexpr,
        TAIL_CODE_WORDS: tl.constexpr,
        TAIL_CODE_BYTES: tl.constexpr,
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
        # Keep adjacent CTAs on the same query so the query-global LUT stays
        # hot in L2 while its probe groups are scanned.
        cta_probe_group_idx = tl.program_id(0)
        query_idx = tl.program_id(1)
        probe_start = cta_probe_group_idx * PROBE_LISTS_PER_CTA

        candidate_offsets = tl.arange(0, SCAN_BLOCK_C)
        group_offsets = candidate_offsets // GROUP_SIZE
        lane_offsets = candidate_offsets % GROUP_SIZE
        ranks = tl.arange(0, LOCAL_K)
        best_keys = tl.full((LOCAL_K,), INF_KEY, tl.int64)

        for probe_offset in tl.static_range(0, PROBE_LISTS_PER_CTA):
            probe_idx = probe_start + probe_offset
            valid_probe = probe_idx < scan_nprobe
            if FUSED_METADATA:
                source_probe_idx = probe_idx // MAX_CHUNKS
                chunk_idx = probe_idx % MAX_CHUNKS
                source_edge_idx = query_idx * nprobe + source_probe_idx
                centroid_id = tl.load(
                    probe_cids_ptr + source_edge_idx,
                    mask=valid_probe,
                    other=0,
                ).to(tl.int64)
                full_vector_count = tl.load(
                    global_list_count_ptr + centroid_id,
                    mask=valid_probe,
                    other=0,
                ).to(tl.int32)
                chunk_vector_start = chunk_idx * CHUNK_GROUPS * GROUP_SIZE
                vector_count = tl.minimum(
                    tl.maximum(full_vector_count - chunk_vector_start, 0),
                    CHUNK_GROUPS * GROUP_SIZE,
                )
                group_prefix = tl.load(
                    list_group_prefix_ptr + centroid_id,
                    mask=valid_probe,
                    other=0,
                ).to(tl.int64) + chunk_idx * CHUNK_GROUPS
                distance_bias = tl.load(
                    probe_distance_bias_ptr + source_edge_idx,
                    mask=valid_probe,
                    other=0.0,
                ).to(tl.float32)
            else:
                edge_idx = query_idx * scan_nprobe + probe_idx
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
                # Scan complete 16-byte chunks with the hot path shared by
                # both aligned and non-aligned M values.  The tail is handled
                # below with compile-time bounds, so an M=25 code performs
                # exactly 25 LUT gathers instead of masking seven synthetic
                # padding subspaces in the generic chunk loop.
                for chunk_idx in tl.range(
                    0,
                    FULL_PACKED_CHUNKS,
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
                            codes = ((packed_word >> (byte_idx * 8)) & 0xFF).to(tl.int64)
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

                if TAIL_CODE_BYTES > 0:
                    tail_chunk = packed_chunk_zero + FULL_PACKED_CHUNKS * GROUP_SIZE * CODE_WORDS
                    for word_idx in tl.static_range(0, TAIL_CODE_WORDS):
                        packed_word = tl.load(
                            packed_codes_ptr + tail_chunk + word_idx,
                            mask=valid_candidates,
                            other=0,
                            cache_modifier=".ca",
                        ).to(tl.uint64)
                        for byte_idx in tl.static_range(0, CODE_WORD_BYTES):
                            subspace_idx = (
                                FULL_PACKED_CHUNKS * CODE_CHUNK_BYTES
                                + word_idx * CODE_WORD_BYTES
                                + byte_idx
                            )
                            if subspace_idx < M:
                                codes = ((packed_word >> (byte_idx * 8)) & 0xFF).to(tl.int64)
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
                elif LOCAL_K <= SCAN_BLOCK_C:
                    tile_best_keys = ~tl.topk(~candidate_keys, LOCAL_K)
                    merged_keys = tl.ravel(tl.join(best_keys, tile_best_keys))
                    best_keys = ~tl.topk(~merged_keys, LOCAL_K)
                else:
                    padded_candidates = tl.cat(
                        candidate_keys,
                        tl.full((LOCAL_K - SCAN_BLOCK_C,), INF_KEY, tl.int64),
                        dim=0,
                    )
                    merged_keys = tl.ravel(tl.join(best_keys, padded_candidates))
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
    ``final_width`` is the unreduced width; wide values are handled by the
    staged merge planner below.
    """
    if topk <= 0 or nprobe <= 0 or nprobe > MAX_QUERY_MAJOR_NPROBE:
        return None
    local_k = max(2, _next_power_of_two(topk))
    if local_k > MAX_LOCAL_K:
        return None
    launch_config = _default_launch_config()
    cta_probe_groups = math.ceil(nprobe / launch_config.probe_lists_per_cta)
    final_width = _next_power_of_two(cta_probe_groups * local_k)
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
        (64, 4, 1),
        (64, 8, 1),
        (128, 4, 1),
        (128, 4, 2),
        (128, 8, 2),
        (256, 8, 1),
        (512, 4, 1),
        (512, 4, 2),
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
    return local_k, cta_probe_groups, config, final_width


def _hierarchical_merge_plan(
    *,
    cta_probe_groups: int,
    local_k: int,
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    """Return staged merge shapes and the final single-kernel merge width.

    Each pair is ``(input_groups, output_groups)`` for one intermediate
    reduction.  The returned final group count and width are guaranteed to
    fit in one ``MAX_FINAL_WIDTH`` merge.  The helper is intentionally pure
    Python so the planner can be tested without importing or launching CUDA.
    """
    input_groups = int(cta_probe_groups)
    if input_groups <= 0 or local_k <= 0:
        raise ValueError(
            f"cta_probe_groups and local_k must be positive, got {input_groups}, {local_k}"
        )

    stages: list[tuple[int, int]] = []
    while input_groups * local_k > MAX_FINAL_WIDTH:
        input_width = input_groups * local_k
        output_groups = math.ceil(input_width / MAX_FINAL_WIDTH)
        stages.append((input_groups, output_groups))
        input_groups = output_groups
    return tuple(stages), input_groups, _next_power_of_two(input_groups * local_k)


def _launch_topk_scan(
    *,
    global_query_luts: torch.Tensor,
    static_vector_corrections: torch.Tensor,
    probe_distance_bias: torch.Tensor,
    list_codes_interleaved: torch.Tensor | None,
    list_ids_interleaved: torch.Tensor,
    probe_group_prefix: torch.Tensor | None,
    probe_list_count: torch.Tensor | None,
    probe_cids: torch.Tensor | None,
    list_group_prefix: torch.Tensor | None,
    list_count: torch.Tensor | None,
    list_partial_keys: torch.Tensor,
    list_codes_packed_16b: torch.Tensor | None,
    m: int,
    ksub: int,
    num_queries: int,
    nprobe: int,
    scan_nprobe: int,
    chunk_groups: int,
    max_chunks: int,
    local_k: int,
    cta_probe_groups: int,
    launch_config: _MAdaptiveLaunchConfig,
) -> None:
    """Launch only the CTA scan; shared by tuning and normal execution."""
    fused_metadata = probe_cids is not None
    # Triton still requires a valid pointer for compile-time-dead arguments.
    dummy = probe_distance_bias
    legacy_prefix = probe_group_prefix if probe_group_prefix is not None else dummy
    legacy_count = probe_list_count if probe_list_count is not None else dummy
    fused_cids = probe_cids if probe_cids is not None else dummy
    global_prefix = list_group_prefix if list_group_prefix is not None else dummy
    global_count = list_count if list_count is not None else dummy
    if list_codes_packed_16b is not None:
        _list_cta_pq_packed_chunked_topk_kernel[(cta_probe_groups, num_queries)](
            global_query_luts,
            static_vector_corrections,
            probe_distance_bias,
            list_codes_packed_16b,
            legacy_prefix,
            legacy_count,
            fused_cids,
            global_prefix,
            global_count,
            list_partial_keys,
            num_queries,
            nprobe,
            scan_nprobe,
            GRID_X=cta_probe_groups,
            M=m,
            KSUB=ksub,
            GROUP_SIZE=INTERLEAVED_GROUP_SIZE,
            SCAN_BLOCK_C=launch_config.scan_block_candidates,
            PROBE_LISTS_PER_CTA=launch_config.probe_lists_per_cta,
            LOCAL_K=local_k,
            FUSED_METADATA=fused_metadata,
            CHUNK_GROUPS=chunk_groups,
            MAX_CHUNKS=max_chunks,
            PACKED_CHUNKS=math.ceil(m / PACKED_CODE_CHUNK_BYTES),
            FULL_PACKED_CHUNKS=m // PACKED_CODE_CHUNK_BYTES,
            TAIL_CODE_WORDS=math.ceil((m % PACKED_CODE_CHUNK_BYTES) / PACKED_CODE_WORD_BYTES),
            TAIL_CODE_BYTES=m % PACKED_CODE_CHUNK_BYTES,
            CODE_CHUNK_BYTES=PACKED_CODE_CHUNK_BYTES,
            CODE_WORD_BYTES=PACKED_CODE_WORD_BYTES,
            CODE_WORDS=PACKED_CODE_WORDS,
            INF_KEY=_INF_KEY,
            num_warps=launch_config.num_warps,
        )
        return
    if list_codes_interleaved is None:
        raise ValueError("Unpacked IVF-PQ scan requires interleaved PQ codes")
    _list_cta_pq_topk_kernel[(cta_probe_groups, num_queries)](
        global_query_luts,
        static_vector_corrections,
        probe_distance_bias,
        list_codes_interleaved,
        legacy_prefix,
        legacy_count,
        fused_cids,
        global_prefix,
        global_count,
        list_partial_keys,
        num_queries,
        list_ids_interleaved.numel(),
        nprobe,
        scan_nprobe,
        GRID_X=cta_probe_groups,
        M=m,
        KSUB=ksub,
        GROUP_SIZE=INTERLEAVED_GROUP_SIZE,
        SCAN_BLOCK_C=launch_config.scan_block_candidates,
        PROBE_LISTS_PER_CTA=launch_config.probe_lists_per_cta,
        LOCAL_K=local_k,
        FUSED_METADATA=fused_metadata,
        CHUNK_GROUPS=chunk_groups,
        MAX_CHUNKS=max_chunks,
        INF_KEY=_INF_KEY,
        num_warps=launch_config.num_warps,
    )


def _hierarchical_merge_partial_keys(
    *,
    partial_keys: torch.Tensor,
    local_k: int,
) -> tuple[torch.Tensor, int, int]:
    """Reduce scan partials until the final merge fits in one Triton CTA."""
    num_queries, cta_probe_groups, actual_local_k = partial_keys.shape
    if int(actual_local_k) != int(local_k):
        raise ValueError(
            "Partial-key local-k dimension does not match the launch plan: "
            f"{actual_local_k} != {local_k}"
        )
    stages, final_groups, final_width = _hierarchical_merge_plan(
        cta_probe_groups=cta_probe_groups,
        local_k=local_k,
    )
    current_keys = partial_keys
    for input_groups, output_groups in stages:
        output_keys = torch.empty(
            (num_queries, output_groups, local_k),
            dtype=torch.int64,
            device=partial_keys.device,
        )
        _merge_pq_partial_keys_kernel[(num_queries, output_groups)](
            current_keys,
            output_keys,
            input_groups * local_k,
            output_groups,
            GRID_X=output_groups,
            LOCAL_K=local_k,
            MERGE_WIDTH=MAX_FINAL_WIDTH,
            INF_KEY=_INF_KEY,
            num_warps=4,
        )
        current_keys = output_keys
    assert current_keys.shape[1] == final_groups
    return current_keys, final_groups, final_width


def _autotune_topk_launch_config(
    *,
    cache_key: tuple[object, ...],
    global_query_luts: torch.Tensor,
    static_vector_corrections: torch.Tensor,
    probe_distance_bias: torch.Tensor,
    list_codes_interleaved: torch.Tensor | None,
    list_ids_interleaved: torch.Tensor,
    probe_group_prefix: torch.Tensor | None,
    probe_list_count: torch.Tensor | None,
    probe_cids: torch.Tensor | None,
    list_group_prefix: torch.Tensor | None,
    list_count: torch.Tensor | None,
    list_codes_packed_16b: torch.Tensor | None,
    topk: int,
    nprobe: int,
    scan_nprobe: int,
    chunk_groups: int,
    max_chunks: int,
) -> _MAdaptiveLaunchConfig:
    """Measure a few CTA shapes once and cache the fastest scan-only plan."""
    cached = _cached_autotune_config(cache_key)
    if cached is not None:
        return cached

    m, num_queries, ksub = global_query_luts.shape
    device = global_query_luts.device
    candidates: list[tuple[_MAdaptiveLaunchConfig, int, int]] = []
    for config in _candidate_launch_configs():
        plan = _resolve_topk_launch_plan(
            m=m,
            topk=topk,
            nprobe=scan_nprobe,
            launch_config=config,
        )
        if plan is None:
            continue
        local_k, cta_probe_groups, _, _ = plan
        candidates.append((config, local_k, cta_probe_groups))
    if not candidates:  # pragma: no cover - standard validation precedes tuning
        raise ValueError("No supported IVFPQ Triton autotune launch configuration")

    # All candidates share one maximum scratch allocation. Repeatedly creating
    # differently sized tensors makes the CUDA caching allocator retain
    # multiple bins and inflates first-search peak memory without benefiting
    # the selected launch configuration.
    max_scratch_elements = max(
        num_queries * cta_probe_groups * local_k
        for _, local_k, cta_probe_groups in candidates
    )
    scratch_storage = torch.empty(
        (max_scratch_elements,),
        dtype=torch.int64,
        device=device,
    )

    best_config: _MAdaptiveLaunchConfig | None = None
    best_ms = float("inf")
    for config, local_k, cta_probe_groups in candidates:
        scratch_elements = num_queries * cta_probe_groups * local_k
        scratch = scratch_storage[:scratch_elements].view(
            num_queries, cta_probe_groups, local_k
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
                probe_cids=probe_cids,
                list_group_prefix=list_group_prefix,
                list_count=list_count,
                list_partial_keys=scratch,
                list_codes_packed_16b=list_codes_packed_16b,
                m=m,
                ksub=ksub,
                num_queries=num_queries,
                nprobe=nprobe,
                scan_nprobe=scan_nprobe,
                chunk_groups=chunk_groups,
                max_chunks=max_chunks,
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
                probe_cids=probe_cids,
                list_group_prefix=list_group_prefix,
                list_count=list_count,
                list_partial_keys=scratch,
                list_codes_packed_16b=list_codes_packed_16b,
                m=m,
                ksub=ksub,
                num_queries=num_queries,
                nprobe=nprobe,
                scan_nprobe=scan_nprobe,
                chunk_groups=chunk_groups,
                max_chunks=max_chunks,
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

    if best_config is None:  # pragma: no cover - candidates is non-empty
        raise ValueError("IVFPQ Triton autotuning did not select a configuration")
    _AUTOTUNE_CONFIG_CACHE[cache_key] = best_config
    _persist_autotune_config(cache_key, best_config)
    return best_config


def triton_topk(
    *,
    global_query_luts: torch.Tensor,
    static_vector_corrections: torch.Tensor,
    probe_distance_bias: torch.Tensor,
    list_codes_interleaved: torch.Tensor | None,
    list_ids_interleaved: torch.Tensor,
    probe_group_prefix: torch.Tensor | None,
    probe_list_count: torch.Tensor | None,
    topk: int,
    output_distances: torch.Tensor,
    output_ids: torch.Tensor,
    list_partial_keys: torch.Tensor | None = None,
    list_codes_packed_16b: torch.Tensor | None = None,
    probe_cids: torch.Tensor | None = None,
    list_group_prefix: torch.Tensor | None = None,
    list_count: torch.Tensor | None = None,
    scan_chunk_groups: int = LIST_SCAN_CHUNK_GROUPS,
    scan_max_chunks: int = 1,
    autotune: bool = False,
    autotune_key: tuple[object, ...] | None = None,
    on_list_scan_complete: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Run exact IVF-PQ ADC with query/probe-chunk CTA groups.

    Oversized selected lists are represented by multiple virtual probe chunks;
    each program keeps an exact group-local top-k for one chunk.  Wide probe
    sets are reduced by intermediate fixed-width TopK stages before the final
    query merge, so no single ``tl.topk`` invocation exceeds
    ``MAX_FINAL_WIDTH``.
    ``list_codes_packed_16b`` uses a rolled 16-byte chunk loop for M>16,
    avoiding code-size and register growth without lower-bound bookkeeping or
    a per-query shared kth threshold.
    """
    if global_query_luts.ndim != 3:
        raise ValueError("Global query LUT must have shape [m, query, ksub]")
    m, num_queries, ksub = global_query_luts.shape
    if global_query_luts.dtype not in _SUPPORTED_GLOBAL_LUT_DTYPES:
        raise ValueError("Global query LUT must use FP32 or FP16 storage")
    legacy_metadata = probe_group_prefix is not None or probe_list_count is not None
    fused_metadata = any(
        tensor is not None for tensor in (probe_cids, list_group_prefix, list_count)
    )
    if legacy_metadata and fused_metadata:
        raise ValueError("Provide either expanded probe metadata or fused list metadata, not both")
    if legacy_metadata:
        if probe_group_prefix is None or probe_list_count is None:
            raise ValueError("Both probe_group_prefix and probe_list_count are required")
        if probe_group_prefix.ndim != 2 or probe_group_prefix.shape != probe_list_count.shape:
            raise ValueError(
                "Probe group-prefix/count tensors must have matching [query, nprobe] shapes"
            )
        if probe_distance_bias.shape != probe_group_prefix.shape:
            raise ValueError("Probe distance bias must match [query, nprobe] probe metadata")
        nprobe = int(probe_group_prefix.shape[1])
        scan_nprobe = nprobe
        scan_chunk_groups = LIST_SCAN_CHUNK_GROUPS
        scan_max_chunks = 1
        metadata_query_count = int(probe_group_prefix.shape[0])
    else:
        if probe_cids is None or list_group_prefix is None or list_count is None:
            raise ValueError(
                "Fused metadata requires probe_cids, list_group_prefix, and list_count"
            )
        if probe_cids.ndim != 2:
            raise ValueError("Probe centroid IDs must have shape [query, nprobe]")
        if probe_distance_bias.shape != probe_cids.shape:
            raise ValueError("Probe distance bias must match probe centroid IDs")
        if list_group_prefix.ndim != 1 or list_count.ndim != 1:
            raise ValueError("Global list metadata must be one-dimensional")
        if list_group_prefix.shape != list_count.shape:
            raise ValueError("Global list prefix/count tensors must have matching shapes")
        if scan_chunk_groups <= 0 or scan_max_chunks <= 0:
            raise ValueError("Chunk groups and max chunks must be positive")
        nprobe = int(probe_cids.shape[1])
        scan_nprobe = nprobe * int(scan_max_chunks)
        metadata_query_count = int(probe_cids.shape[0])
    if metadata_query_count != num_queries:
        raise ValueError("Probe metadata query count does not match the global query LUT")
    if list_codes_interleaved is None and list_codes_packed_16b is None:
        raise ValueError("IVF-PQ scan requires an unpacked or packed code layout")
    code_dtype = (
        list_codes_interleaved.dtype
        if list_codes_interleaved is not None
        else torch.uint8
    )
    if not can_run_topk(
        device=global_query_luts.device,
        topk=topk,
        m=m,
        ksub=ksub,
        nprobe=scan_nprobe,
        num_code_vectors=list_ids_interleaved.numel(),
        code_dtype=code_dtype,
        use_packed_16b_codes=list_codes_packed_16b is not None,
    ):
        raise ValueError("IVFPQ Triton CTA-list top-k is unavailable for these search parameters")
    if (
        list_codes_interleaved is not None
        and list_codes_interleaved.shape != (m, list_ids_interleaved.numel())
    ):
        raise ValueError("Interleaved PQ code cache must have shape [m, padded_num_indexed_vectors]")
    if static_vector_corrections.shape != (list_ids_interleaved.numel(),):
        raise ValueError("Static vector corrections must match padded interleaved vector count")
    if static_vector_corrections.dtype != torch.float32:
        raise ValueError("Static vector corrections must use FP32 storage")
    if output_distances.shape[0] != num_queries or output_distances.shape[1] < topk:
        raise ValueError("Output distance tensor must have shape [num_queries, >= topk]")
    if output_ids.shape[0] != num_queries or output_ids.shape[1] < topk:
        raise ValueError("Output ID tensor must have shape [num_queries, >= topk]")

    tensors: tuple[torch.Tensor, ...] = (
        global_query_luts,
        static_vector_corrections,
        probe_distance_bias,
        list_ids_interleaved,
        output_distances,
        output_ids,
    )
    if list_codes_interleaved is not None:
        tensors += (list_codes_interleaved,)
    if legacy_metadata:
        assert probe_group_prefix is not None and probe_list_count is not None
        tensors += (probe_group_prefix, probe_list_count)
    else:
        assert probe_cids is not None and list_group_prefix is not None and list_count is not None
        tensors += (probe_cids, list_group_prefix, list_count)
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
        nprobe=scan_nprobe,
    )
    if plan is None:  # pragma: no cover - guarded by can_run_topk
        raise ValueError("Unsupported IVFPQ Triton CTA-list top-k plan")
    local_k, cta_probe_groups, launch_config, final_width = plan

    global_query_luts = global_query_luts.contiguous()
    static_vector_corrections = static_vector_corrections.contiguous()
    probe_distance_bias = probe_distance_bias.contiguous()
    if list_codes_interleaved is not None:
        list_codes_interleaved = list_codes_interleaved.contiguous()
    list_ids_interleaved = list_ids_interleaved.contiguous()
    if probe_group_prefix is not None:
        probe_group_prefix = probe_group_prefix.contiguous()
    if probe_list_count is not None:
        probe_list_count = probe_list_count.contiguous()
    if probe_cids is not None:
        probe_cids = probe_cids.contiguous()
    if list_group_prefix is not None:
        list_group_prefix = list_group_prefix.contiguous()
    if list_count is not None:
        list_count = list_count.contiguous()
    if list_codes_packed_16b is not None:
        list_codes_packed_16b = list_codes_packed_16b.contiguous()

    cache_key = autotune_key or (
        str(global_query_luts.device),
        m,
        ksub,
        scan_nprobe,
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
            probe_cids=probe_cids,
            list_group_prefix=list_group_prefix,
            list_count=list_count,
            list_codes_packed_16b=list_codes_packed_16b,
            topk=topk,
            nprobe=nprobe,
            scan_nprobe=scan_nprobe,
            chunk_groups=scan_chunk_groups,
            max_chunks=scan_max_chunks,
        )
    if cached_launch_config is not None or autotune:
        tuned_plan = _resolve_topk_launch_plan(
            m=m,
            topk=topk,
            nprobe=scan_nprobe,
            launch_config=launch_config,
        )
        assert tuned_plan is not None
        local_k, cta_probe_groups, launch_config, final_width = tuned_plan
    # Allocate only after the cached/autotuned launch shape is final.  The old
    # order allocated the default shape first and a second tuned shape later.
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

    _launch_topk_scan(
        global_query_luts=global_query_luts,
        static_vector_corrections=static_vector_corrections,
        probe_distance_bias=probe_distance_bias,
        list_codes_interleaved=list_codes_interleaved,
        list_ids_interleaved=list_ids_interleaved,
        probe_group_prefix=probe_group_prefix,
        probe_list_count=probe_list_count,
        probe_cids=probe_cids,
        list_group_prefix=list_group_prefix,
        list_count=list_count,
        list_partial_keys=list_partial_keys,
        list_codes_packed_16b=list_codes_packed_16b,
        m=m,
        ksub=ksub,
        num_queries=num_queries,
        nprobe=nprobe,
        scan_nprobe=scan_nprobe,
        chunk_groups=scan_chunk_groups,
        max_chunks=scan_max_chunks,
        local_k=local_k,
        cta_probe_groups=cta_probe_groups,
        launch_config=launch_config,
    )
    if on_list_scan_complete is not None:
        on_list_scan_complete()

    partial_keys, final_groups, final_width = _hierarchical_merge_partial_keys(
        partial_keys=list_partial_keys,
        local_k=local_k,
    )
    _finalize_direct_pq_topk_kernel[(num_queries,)](
        partial_keys,
        list_ids_interleaved,
        output_distances,
        output_ids,
        topk,
        GRID_X=final_groups,
        LOCAL_K=local_k,
        FINAL_WIDTH=final_width,
        INF_KEY=_INF_KEY,
        num_warps=4,
    )
    return list_partial_keys
