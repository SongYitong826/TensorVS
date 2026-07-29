from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_fvecs(dataset_path: str | Path) -> torch.Tensor:
    """Load a Faiss-style .fvecs file into a 2D float32 tensor."""
    path = Path(dataset_path)
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"Dataset file is empty: {path}")

    dim = int(raw.view(np.int32)[0])
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension {dim} in {path}")

    row_width = dim + 1
    if raw.size % row_width != 0:
        raise ValueError(f"Malformed fvecs file: {path}")

    data = raw.reshape(-1, row_width)[:, 1:]
    return torch.from_numpy(np.ascontiguousarray(data))


def load_ivecs(dataset_path: str | Path) -> torch.Tensor:
    """Load a Faiss-style .ivecs file into a 2D int64 tensor."""
    path = Path(dataset_path)
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"Dataset file is empty: {path}")

    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension {dim} in {path}")

    row_width = dim + 1
    if raw.size % row_width != 0:
        raise ValueError(f"Malformed ivecs file: {path}")

    data = raw.reshape(-1, row_width)[:, 1:]
    return torch.from_numpy(np.ascontiguousarray(data)).to(torch.int64)


def load_dataset(dataset_path: str | Path) -> torch.Tensor:
    """Load a 2D tensor dataset from a .pt/.pth, .npy, or .fvecs file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
    elif path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".fvecs":
        data = load_fvecs(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    if isinstance(data, dict):
        for key in ("vectors", "embeddings", "data", "xb"):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError("Dataset dict must contain one of: vectors, embeddings, data, xb")

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif not isinstance(data, torch.Tensor):
        raise TypeError(f"Unsupported dataset type: {type(data)!r}")

    if data.ndim != 2:
        raise ValueError(f"Dataset must have shape (num_vectors, dim), got {tuple(data.shape)}")

    return data.to(dtype=torch.float32, copy=False).contiguous()


def compute_recall_at_k(retrieved_ids: torch.Tensor, groundtruth_ids: torch.Tensor, topk: int) -> float:
    """Compute strict average recall@k by overlap with ground-truth top-k ids."""
    eval_k = min(topk, retrieved_ids.shape[1], groundtruth_ids.shape[1])
    retrieved = retrieved_ids[:, :eval_k]
    groundtruth = groundtruth_ids[:, :eval_k]

    recall_sum = 0.0
    for query_idx in range(retrieved.shape[0]):
        retrieved_set = set(retrieved[query_idx].tolist()) - {-1}
        groundtruth_set = set(groundtruth[query_idx].tolist()) - {-1}
        if not groundtruth_set:
            continue
        recall_sum += len(retrieved_set & groundtruth_set) / len(groundtruth_set)

    return recall_sum / retrieved.shape[0]
