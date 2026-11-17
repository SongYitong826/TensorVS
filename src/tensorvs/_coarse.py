"""PyTorch coarse-list scores shared by both IVF indexes."""

from __future__ import annotations

import torch


def _coarse_centroid_scores_addmm(
    queries: torch.Tensor,
    centroids: torch.Tensor,
    centroid_norms: torch.Tensor,
) -> torch.Tensor:
    """Compute ``||c||^2 - 2 q^T c`` for coarse-list selection.

    The omitted query norm is constant across every centroid in a query row,
    so it cannot change the selected lists. Keeping it out of the full
    ``[nq, nlist]`` matrix avoids two pointwise passes before Top-K.
    """
    centroid_norm_base = centroid_norms.unsqueeze(0)
    if not queries.is_cuda:
        distances = torch.addmm(
            centroid_norm_base,
            queries,
            centroids.transpose(0, 1),
            beta=1.0,
            alpha=-2.0,
        )
    else:
        previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            distances = torch.addmm(
                centroid_norm_base,
                queries,
                centroids.transpose(0, 1),
                beta=1.0,
                alpha=-2.0,
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
    return distances


def _coarse_squared_l2_distance_addmm(
    queries: torch.Tensor,
    centroids: torch.Tensor,
    centroid_norms: torch.Tensor,
) -> torch.Tensor:
    """Compute all coarse ``||q-c||^2`` distances with cached norms."""
    query_norms = (queries * queries).sum(dim=1, keepdim=True)
    distances = _coarse_centroid_scores_addmm(queries, centroids, centroid_norms)
    return distances.add_(query_norms).clamp_min_(0.0)
