from __future__ import annotations

import torch


def refine_search_results(
    queries: torch.Tensor,
    candidate_ids: torch.Tensor,
    vectors: torch.Tensor,
    topk: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rerank ANN candidates with exact squared-L2 distances to raw vectors."""
    if queries.ndim != 2:
        raise ValueError(f"queries must be 2D, got shape {tuple(queries.shape)}")
    if candidate_ids.ndim != 2:
        raise ValueError(f"candidate_ids must be 2D, got shape {tuple(candidate_ids.shape)}")
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape {tuple(vectors.shape)}")
    if queries.shape[0] != candidate_ids.shape[0]:
        raise ValueError("queries and candidate_ids must contain the same number of rows")
    if queries.shape[1] != vectors.shape[1]:
        raise ValueError(
            f"query dimension {queries.shape[1]} does not match vector dimension {vectors.shape[1]}"
        )

    topk = int(topk)
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if topk > candidate_ids.shape[1]:
        raise ValueError(
            f"topk={topk} exceeds the candidate count {candidate_ids.shape[1]}"
        )

    if device is None:
        if queries.is_cuda:
            target_device = queries.device
        elif candidate_ids.is_cuda:
            target_device = candidate_ids.device
        else:
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)

    ids = candidate_ids.to(dtype=torch.int64, device=target_device, copy=False)
    if (ids < -1).any():
        raise ValueError("candidate_ids may only use -1 as the missing-id sentinel")
    if ((ids >= vectors.shape[0]) & (ids != -1)).any():
        raise ValueError("candidate_ids contains an id outside the raw vector table")

    valid = ids >= 0
    safe_ids = ids.clamp_min(0)
    if vectors.device == target_device:
        candidate_vectors = vectors[safe_ids]
    elif vectors.device.type == "cuda":
        candidate_vectors = vectors.to(
            dtype=torch.float32,
            device=target_device,
        )[safe_ids]
    else:
        candidate_vectors = vectors[safe_ids.cpu()].to(
            dtype=torch.float32,
            device=target_device,
        )

    q = queries.to(dtype=torch.float32, device=target_device, copy=False).contiguous()
    exact_distances = ((q[:, None, :] - candidate_vectors) ** 2).sum(dim=-1)
    exact_distances.masked_fill_(~valid, float("inf"))
    refined_distances, order = torch.topk(exact_distances, k=topk, dim=1, largest=False)
    refined_ids = torch.gather(ids, 1, order)
    return refined_distances, refined_ids
