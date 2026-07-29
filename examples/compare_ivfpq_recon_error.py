from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_fvecs(path: str | Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")

    dim = int(raw[:1].view(np.int32)[0])
    row_width = dim + 1
    if raw.size % row_width != 0:
        raise ValueError(f"Invalid fvecs layout in {path}")

    rows = raw.reshape(-1, row_width)
    actual_dims = rows[:, 0].view(np.int32)
    if not np.all(actual_dims == dim):
        raise ValueError(f"Inconsistent fvecs dimensions in {path}")

    return np.asarray(rows[:, 1:], dtype=np.float32, order="C")


def sample_rows(vectors: np.ndarray, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = min(int(sample_size), int(vectors.shape[0]))
    sample_idx = rng.choice(vectors.shape[0], size=count, replace=False)
    return sample_idx.astype(np.int64, copy=False), vectors[sample_idx]


def summarize_errors(sq_l2: np.ndarray) -> dict[str, float | int]:
    return {
        "sample_size": int(sq_l2.shape[0]),
        "mean_squared_l2_error": float(np.mean(sq_l2)),
        "mean_l2_error": float(np.mean(np.sqrt(sq_l2))),
        "max_squared_l2_error": float(np.max(sq_l2)),
    }


def run_tensorvs(index_path: Path, dataset_path: Path, sample_size: int, seed: int) -> dict[str, float | int]:
    import torch

    payload = torch.load(index_path, map_location="cpu")
    labels = payload["assignments"].to(torch.int64)
    centroids = payload["centroids"].to(torch.float32)
    pq_codebooks = payload["pq_codebooks"].to(torch.float32)
    pq_codes = payload["pq_codes"].to(torch.int64)

    vectors = load_fvecs(dataset_path)
    sample_idx, sample_x_np = sample_rows(vectors, sample_size, seed)
    sample_idx_t = torch.from_numpy(sample_idx)
    sample_x = torch.from_numpy(sample_x_np)

    sample_labels = labels[sample_idx_t]
    sample_codes = pq_codes[sample_idx_t]
    coarse = centroids[sample_labels]

    m = int(pq_codebooks.shape[0])
    subdim = int(pq_codebooks.shape[2])
    subspace_idx = torch.arange(m, dtype=torch.int64).unsqueeze(0).expand(sample_codes.shape[0], -1)
    decoded = pq_codebooks[subspace_idx, sample_codes].reshape(sample_codes.shape[0], m * subdim)
    recon = coarse + decoded
    err = sample_x - recon
    sq_l2 = (err * err).sum(dim=1).numpy()

    out = summarize_errors(sq_l2)
    out["impl"] = "tensorvs"
    return out


def run_cuvs(index_path: Path, dataset_path: Path, sample_size: int, seed: int) -> dict[str, float | int]:
    from pylibraft.common import device_ndarray
    from cuvs.neighbors import ivf_pq

    index = ivf_pq.load(str(index_path))
    pq_centers_h = index.pq_centers.copy_to_host()
    centers_h = index.centers.copy_to_host()
    centers_rot_h = index.centers_rot.copy_to_host()
    rotation_h = index.rotation_matrix.copy_to_host()

    vectors = load_fvecs(dataset_path)
    _, sample_x = sample_rows(vectors, sample_size, seed)

    labels, codes = ivf_pq.transform(index, device_ndarray(sample_x))
    labels_h = labels.copy_to_host().astype(np.int64, copy=False)
    codes_h = codes.copy_to_host().astype(np.int64, copy=False)

    m, subdim, _ = pq_centers_h.shape
    decoded = np.empty((codes_h.shape[0], m, subdim), dtype=np.float32)
    for subspace_idx in range(m):
        decoded[:, subspace_idx, :] = pq_centers_h[subspace_idx, :, codes_h[:, subspace_idx]]

    recon_rot = centers_rot_h[labels_h] + decoded.reshape(codes_h.shape[0], m * subdim)
    recon = recon_rot @ rotation_h
    err = sample_x - recon
    sq_l2 = np.sum(err * err, axis=1)

    out = summarize_errors(sq_l2)
    out["impl"] = "cuvs"
    out["rotation_max_abs_from_identity"] = float(
        np.max(np.abs(rotation_h - np.eye(rotation_h.shape[0], rotation_h.shape[1], dtype=np.float32)))
    )
    out["centers_vs_centers_rot_max_abs"] = float(np.max(np.abs(centers_h - centers_rot_h)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare IVF-PQ reconstruction error for TensorVS and cuVS.")
    parser.add_argument("--impl", choices=("tensorvs", "cuvs"), required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.impl == "tensorvs":
        out = run_tensorvs(args.index_path, args.dataset_path, args.sample_size, args.seed)
    else:
        out = run_cuvs(args.index_path, args.dataset_path, args.sample_size, args.seed)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
