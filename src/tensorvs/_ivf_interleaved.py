"""cuVS-style 32-row interleaved caches for IVF list scans.

The persisted TensorVS indexes deliberately retain their simple row-major
layout.  These helpers build an optional resident-only cache where every IVF
list is padded to groups of 32 rows and stored in a layout suited to a warp
scanning the same dimension chunk across 32 candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


INTERLEAVED_GROUP_SIZE = 32
# Matches cuVS IVF-PQ's ``kIndexGroupVecLen``: one thread owns a 16-byte
# chunk of PQ codes, and a 32-thread group interleaves those chunks.
PQ_CODE_CHUNK_BYTES = 16
# cuVS vectorizes up to 16 bytes per load.  This is the maximum width for
# float32; the actual width is selected per index by
# ``interleaved_vector_width`` below.
INTERLEAVED_VECTOR_WIDTH = 4


@dataclass(frozen=True)
class InterleavedFlatLists:
    vectors: torch.Tensor
    ids: torch.Tensor
    group_prefix: torch.Tensor
    vector_width: int


@dataclass(frozen=True)
class InterleavedPQLists:
    codes_by_subspace: torch.Tensor
    ids: torch.Tensor
    group_prefix: torch.Tensor


def _build_physical_positions(
    list_prefix: torch.Tensor,
    list_count: torch.Tensor,
    num_vectors: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return each row's padded physical position and each list's group base."""
    group_count = torch.div(
        list_count + (INTERLEAVED_GROUP_SIZE - 1),
        INTERLEAVED_GROUP_SIZE,
        rounding_mode="floor",
    )
    group_prefix = torch.cumsum(group_count, dim=0) - group_count
    total_groups = int(group_count.sum().item())
    if num_vectors == 0:
        return (
            torch.empty((0,), device=list_count.device, dtype=torch.int64),
            group_prefix.contiguous(),
            total_groups,
        )

    original_positions = torch.arange(num_vectors, device=list_count.device, dtype=torch.int64)
    list_end = list_prefix + list_count
    list_idx = torch.searchsorted(list_end, original_positions, right=True).clamp_max(list_count.numel() - 1)
    within_list = original_positions - list_prefix[list_idx]
    physical_positions = group_prefix[list_idx] * INTERLEAVED_GROUP_SIZE + within_list
    return physical_positions.contiguous(), group_prefix.contiguous(), total_groups


def interleaved_vector_width(dim: int, dtype: torch.dtype) -> int:
    """Return cuVS's vector length for one dimension chunk.

    cuVS packs at most 16 bytes into a vectorized load.  It uses that maximum
    only when it divides the dimension exactly; otherwise it falls back to a
    scalar component so layouts never contain a padded vector tail.
    """
    if dim <= 0:
        raise ValueError(f"Interleaved IVFFlat requires a positive dimension, got {dim}")
    element_size = torch.empty((), dtype=dtype).element_size()
    max_vector_width = max(1, 16 // element_size)
    return max_vector_width if dim % max_vector_width == 0 else 1


def build_interleaved_flat_lists(
    list_vectors: torch.Tensor,
    list_ids_flat: torch.Tensor,
    list_prefix: torch.Tensor,
    list_count: torch.Tensor,
) -> InterleavedFlatLists:
    """Build cuVS-style `[group, dim_chunk, lane, veclen]` data and ID caches."""
    if list_vectors.ndim != 2:
        raise ValueError("IVFFlat vectors must have shape [num_vectors, dim]")
    if list_vectors.shape[0] != list_ids_flat.numel():
        raise ValueError("IVFFlat vector and ID counts must match")
    dim = int(list_vectors.shape[1])
    vector_width = interleaved_vector_width(dim, list_vectors.dtype)

    num_vectors = int(list_vectors.shape[0])
    physical_positions, group_prefix, total_groups = _build_physical_positions(
        list_prefix,
        list_count,
        num_vectors,
    )
    # Match cuVS: a 16-byte vector load is used only for a dimension that is
    # exactly divisible by its veclen.  Otherwise this is a scalar layout,
    # not a zero-padded float4 tail.
    num_chunks = dim // vector_width
    vectors = torch.zeros(
        (total_groups, num_chunks, INTERLEAVED_GROUP_SIZE, vector_width),
        dtype=list_vectors.dtype,
        device=list_vectors.device,
    )
    ids = torch.full(
        (total_groups * INTERLEAVED_GROUP_SIZE,),
        -1,
        dtype=torch.int64,
        device=list_ids_flat.device,
    )
    if num_vectors > 0:
        groups = torch.div(physical_positions, INTERLEAVED_GROUP_SIZE, rounding_mode="floor")
        lanes = physical_positions % INTERLEAVED_GROUP_SIZE
        vectors[groups, :, lanes, :] = list_vectors.reshape(
            num_vectors,
            num_chunks,
            vector_width,
        )
        ids[physical_positions] = list_ids_flat.to(dtype=torch.int64)
    return InterleavedFlatLists(
        vectors=vectors.contiguous(),
        ids=ids.contiguous(),
        group_prefix=group_prefix,
        vector_width=vector_width,
    )


def build_interleaved_pq_lists(
    list_codes: torch.Tensor,
    list_ids_flat: torch.Tensor,
    list_prefix: torch.Tensor,
    list_count: torch.Tensor,
) -> InterleavedPQLists:
    """Build subspace-major `[m, padded_group * 32]` PQ code storage."""
    if list_codes.ndim != 2:
        raise ValueError("PQ codes must have shape [num_vectors, m]")
    if list_codes.shape[0] != list_ids_flat.numel():
        raise ValueError("PQ code and ID counts must match")

    num_vectors, m = list_codes.shape
    physical_positions, group_prefix, total_groups = _build_physical_positions(
        list_prefix,
        list_count,
        int(num_vectors),
    )
    padded_vectors = total_groups * INTERLEAVED_GROUP_SIZE
    codes_by_subspace = torch.zeros(
        (m, padded_vectors),
        dtype=list_codes.dtype,
        device=list_codes.device,
    )
    ids = torch.full(
        (padded_vectors,),
        -1,
        dtype=torch.int64,
        device=list_ids_flat.device,
    )
    if num_vectors > 0:
        codes_by_subspace[:, physical_positions] = list_codes.transpose(0, 1)
        ids[physical_positions] = list_ids_flat.to(dtype=torch.int64)
    return InterleavedPQLists(
        codes_by_subspace=codes_by_subspace.contiguous(),
        ids=ids.contiguous(),
        group_prefix=group_prefix,
    )


def build_interleaved_pq_code_chunks(
    list_codes: torch.Tensor,
    list_prefix: torch.Tensor,
    list_count: torch.Tensor,
) -> torch.Tensor:
    """Build cuVS-style 16-byte PQ chunks in padded physical-list order.

    The underlying byte layout is ``[group, code_chunk, lane, 16]``; it is
    returned as ``[group, code_chunk, lane, 2]`` ``uint64`` words. Thus the
    32 lanes of one warp read adjacent 16-byte records for a code chunk, while
    a candidate's next chunk is exactly one group-stride away. The final chunk
    is zero padded; callers must still mask subspaces beyond the true ``m``.
    This cache is deliberately transient/device-resident because it is used
    only by the M>16 packed chunk scanner.
    """
    if list_codes.ndim != 2:
        raise ValueError("PQ codes must have shape [num_vectors, m]")
    if list_codes.dtype != torch.uint8:
        raise ValueError("Packed PQ code chunks require uint8 codes")

    num_vectors, m = list_codes.shape
    physical_positions, _, total_groups = _build_physical_positions(
        list_prefix,
        list_count,
        int(num_vectors),
    )
    num_chunks = math.ceil(m / PQ_CODE_CHUNK_BYTES)
    packed_codes = torch.zeros(
        (total_groups, num_chunks, INTERLEAVED_GROUP_SIZE, PQ_CODE_CHUNK_BYTES),
        dtype=torch.uint8,
        device=list_codes.device,
    )
    if num_vectors > 0:
        groups = torch.div(physical_positions, INTERLEAVED_GROUP_SIZE, rounding_mode="floor")
        lanes = physical_positions % INTERLEAVED_GROUP_SIZE
        for chunk_idx in range(num_chunks):
            start = chunk_idx * PQ_CODE_CHUNK_BYTES
            end = min(start + PQ_CODE_CHUNK_BYTES, m)
            packed_codes[groups, chunk_idx, lanes, : end - start] = list_codes[:, start:end]
    # Triton consumes the chunk as two aligned 64-bit words. ``view`` is a
    # reinterpretation, so this keeps the exact cuVS byte layout without a
    # second allocation or a packing copy.
    return packed_codes.contiguous().view(torch.uint64)


def build_interleaved_pq_scalar_cache(
    list_values: torch.Tensor,
    list_prefix: torch.Tensor,
    list_count: torch.Tensor,
    *,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Place one list-row scalar at the matching physical PQ-code position.

    ``list_values`` follows the row-major IVF-list order used by
    ``list_codes``.  The returned one-dimensional tensor follows the padded
    physical order of :func:`build_interleaved_pq_lists`, so a Triton scan can
    load it with the same ``physical_positions`` used for codes and IDs.
    """
    if list_values.ndim != 1:
        raise ValueError("PQ scalar cache values must have shape [num_vectors]")

    num_vectors = int(list_values.numel())
    physical_positions, _, total_groups = _build_physical_positions(
        list_prefix,
        list_count,
        num_vectors,
    )
    values = torch.full(
        (total_groups * INTERLEAVED_GROUP_SIZE,),
        fill_value,
        dtype=list_values.dtype,
        device=list_values.device,
    )
    if num_vectors > 0:
        values[physical_positions] = list_values
    return values.contiguous()
