from __future__ import annotations

import pytest
import torch

from tensorvs._coarse_triton import restore_selected_squared_l2
from tensorvs.index_ivfpq import (
    _coarse_centroid_scores_addmm,
    _coarse_squared_l2_distance_addmm,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimized_coarse_selection_matches_full_squared_l2() -> None:
    generator = torch.Generator(device="cuda").manual_seed(1907)
    queries = torch.randn(13, 200, generator=generator, device="cuda")
    centroids = torch.randn(1024, 200, generator=generator, device="cuda")
    centroid_norms = centroids.square().sum(dim=1)

    full_distances = _coarse_squared_l2_distance_addmm(
        queries, centroids, centroid_norms
    )
    expected_distances, expected_ids = torch.topk(
        full_distances, 120, dim=1, largest=False
    )

    scores = _coarse_centroid_scores_addmm(queries, centroids, centroid_norms)
    actual_distances, actual_ids = torch.topk(
        scores, 120, dim=1, largest=False, sorted=False
    )
    restore_selected_squared_l2(queries, actual_distances)

    expected_order = torch.argsort(expected_ids, dim=1)
    actual_order = torch.argsort(actual_ids, dim=1)
    torch.testing.assert_close(
        torch.gather(actual_ids, 1, actual_order),
        torch.gather(expected_ids, 1, expected_order),
    )
    torch.testing.assert_close(
        torch.gather(actual_distances, 1, actual_order),
        torch.gather(expected_distances, 1, expected_order),
        rtol=1e-5,
        atol=2e-5,
    )
