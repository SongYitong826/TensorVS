import pytest
import torch

from flash_kmeans import (
    KMEANS_ALGORITHM_BALANCED,
    KMEANS_ALGORITHM_LLOYD,
    fit_best_kmeans,
)


@pytest.mark.parametrize("algorithm", [KMEANS_ALGORITHM_LLOYD, KMEANS_ALGORITHM_BALANCED])
def test_flash_kmeans_training_exposes_both_algorithms(algorithm):
    generator = torch.Generator().manual_seed(1234)
    cluster_centers = torch.tensor(
        [[-4.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0],
         [0.0, -4.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]]
    )
    data = (
        cluster_centers[:, None, :]
        + 0.05 * torch.randn((4, 32, 4), generator=generator)
    ).reshape(-1, 4)

    result = fit_best_kmeans(
        data,
        d=4,
        k=4,
        niter=4,
        use_triton=False,
        device="cpu",
        seed=1234,
        n_init=1,
        max_points_per_centroid=None,
        min_points_per_centroid=0,
        algorithm=algorithm,
    )

    assert result.centroids.shape == (4, 4)
    assert result.labels.shape == (128,)
    assert torch.isfinite(result.centroids).all()
    assert int(result.labels.min()) >= 0
    assert int(result.labels.max()) < 4
