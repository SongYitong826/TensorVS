"""Small Triton epilogues shared by IVF coarse selection paths."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - TensorVS CUDA search requires Triton
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _restore_selected_squared_l2_kernel(
        queries_ptr,
        selected_scores_ptr,
        dim,
        selected_count,
        BLOCK_D: tl.constexpr,
        BLOCK_SELECTED: tl.constexpr,
    ):
        query_idx = tl.program_id(0)
        dim_offsets = tl.arange(0, BLOCK_D)
        query_values = tl.load(
            queries_ptr + query_idx * dim + dim_offsets,
            mask=dim_offsets < dim,
            other=0.0,
        ).to(tl.float32)
        query_norm = tl.sum(query_values * query_values, axis=0)

        selected_offsets = tl.arange(0, BLOCK_SELECTED)
        mask = selected_offsets < selected_count
        scores = tl.load(
            selected_scores_ptr + query_idx * selected_count + selected_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        distances = tl.maximum(scores + query_norm, 0.0)
        tl.store(
            selected_scores_ptr + query_idx * selected_count + selected_offsets,
            distances,
            mask=mask,
        )


def restore_selected_squared_l2(
    queries: torch.Tensor,
    selected_scores: torch.Tensor,
) -> torch.Tensor:
    """Add each query norm to selected centroid scores in one GPU kernel."""
    if not queries.is_cuda:
        query_norms = (queries * queries).sum(dim=1, keepdim=True)
        return selected_scores.add_(query_norms).clamp_min_(0.0)
    if triton is None:  # pragma: no cover - guarded by the CUDA installation
        raise RuntimeError("Triton is required for CUDA coarse selection")
    if queries.ndim != 2 or selected_scores.ndim != 2:
        raise ValueError("queries and selected_scores must both be rank-2")
    if queries.shape[0] != selected_scores.shape[0]:
        raise ValueError("queries and selected_scores must have the same row count")
    if not queries.is_contiguous() or not selected_scores.is_contiguous():
        raise ValueError("queries and selected_scores must be contiguous")

    dim = int(queries.shape[1])
    selected_count = int(selected_scores.shape[1])
    _restore_selected_squared_l2_kernel[(int(queries.shape[0]),)](
        queries,
        selected_scores,
        dim,
        selected_count,
        BLOCK_D=triton.next_power_of_2(dim),
        BLOCK_SELECTED=triton.next_power_of_2(selected_count),
        num_warps=4,
    )
    return selected_scores
