from __future__ import annotations

import torch


def squared_l2_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute squared L2 distance between query matrix x and base matrix y."""
    x = x.to(dtype=torch.float32, copy=False)
    y = y.to(dtype=torch.float32, copy=False)
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
    distances = x_norm + y_norm - 2 * (x @ y.transpose(0, 1))
    return distances.clamp_min_(0.0)


def rowwise_squared_l2_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute squared L2 distance for aligned row pairs x[i] and y[i]."""
    if x.shape != y.shape:
        raise ValueError(f"Expected x and y to have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    x = x.to(dtype=torch.float32, copy=False)
    y = y.to(dtype=torch.float32, copy=False)
    return ((x - y) ** 2).sum(dim=1)


def inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise inner products between query matrix x and base matrix y."""
    return x @ y.transpose(0, 1)


def inner_product_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Convert inner product similarity into a distance-like quantity."""
    return -inner_product(x, y)
