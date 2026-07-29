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


def sample_ids(num_rows: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = min(int(sample_size), int(num_rows))
    return np.asarray(rng.choice(num_rows, size=count, replace=False), dtype=np.int64)


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.shape[0], dtype=np.float64)

    start = 0
    while start < sorted_x.shape[0]:
        end = start + 1
        while end < sorted_x.shape[0] and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def summarize(true_dist: np.ndarray, approx_dist: np.ndarray, topk: int) -> dict[str, float | int]:
    if true_dist.shape != approx_dist.shape:
        raise ValueError("true_dist and approx_dist must share the same shape")

    k = min(int(topk), int(true_dist.shape[1]))
    spearman_values = []
    topk_hits = 0
    nq = true_dist.shape[0]
    for qid in range(nq):
        td = true_dist[qid]
        ad = approx_dist[qid]
        spearman_values.append(spearman_rho(td, ad))
        true_topk = np.argsort(td, kind="mergesort")[:k]
        approx_topk = np.argsort(ad, kind="mergesort")[:k]
        topk_hits += np.intersect1d(true_topk, approx_topk).size

    spearman_arr = np.asarray(spearman_values, dtype=np.float64)
    return {
        "num_queries": int(nq),
        "num_candidates": int(true_dist.shape[1]),
        "mean_spearman_rho": float(np.mean(spearman_arr)),
        "median_spearman_rho": float(np.median(spearman_arr)),
        "min_spearman_rho": float(np.min(spearman_arr)),
        "sample_topk_overlap": float(topk_hits / float(nq * k)),
    }


def compute_true_distances(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    q_norm = np.sum(queries * queries, axis=1, keepdims=True)
    c_norm = np.sum(candidates * candidates, axis=1, keepdims=True).T
    prod = queries @ candidates.T
    dist = q_norm + c_norm - 2.0 * prod
    np.maximum(dist, 0.0, out=dist)
    return dist


def compute_tensorvs_approx(
    *,
    index_path: Path,
    candidate_ids: np.ndarray,
    query_vectors: np.ndarray,
) -> np.ndarray:
    import torch

    payload = torch.load(index_path, map_location="cpu")
    labels = payload["assignments"].to(torch.int64)
    centroids = payload["centroids"].to(torch.float32)
    pq_codebooks = payload["pq_codebooks"].to(torch.float32)
    pq_codes = payload["pq_codes"].to(torch.int64)

    cand_ids_t = torch.from_numpy(candidate_ids)
    cand_labels = labels[cand_ids_t]
    cand_codes = pq_codes[cand_ids_t]

    queries_t = torch.from_numpy(np.asarray(query_vectors, dtype=np.float32))
    residual_q = queries_t[:, None, :] - centroids[cand_labels][None, :, :]
    nq, nc, dim = residual_q.shape
    m = int(pq_codebooks.shape[0])
    subdim = int(pq_codebooks.shape[2])
    residual_q = residual_q.view(nq, nc, m, subdim)

    approx = torch.zeros((nq, nc), dtype=torch.float32)
    for subspace_idx in range(m):
        code_idx = cand_codes[:, subspace_idx]
        codewords = pq_codebooks[subspace_idx][code_idx]
        diff = residual_q[:, :, subspace_idx, :] - codewords.unsqueeze(0)
        approx += (diff * diff).sum(dim=-1)
    return approx.numpy()


def compute_cuvs_approx(
    *,
    index_path: Path,
    candidate_vectors: np.ndarray,
    query_vectors: np.ndarray,
) -> np.ndarray:
    from pylibraft.common import device_ndarray
    from cuvs.neighbors import ivf_pq

    index = ivf_pq.load(str(index_path))
    pq_centers = index.pq_centers.copy_to_host()
    centers_rot = index.centers_rot.copy_to_host()

    cand_labels, cand_codes = ivf_pq.transform(index, device_ndarray(candidate_vectors))
    cand_labels_h = cand_labels.copy_to_host().astype(np.int64, copy=False)
    cand_codes_h = cand_codes.copy_to_host().astype(np.int64, copy=False)

    nq = int(query_vectors.shape[0])
    nc = int(candidate_vectors.shape[0])
    m, subdim, _ = pq_centers.shape
    residual_q = query_vectors[:, None, :] - centers_rot[cand_labels_h][None, :, :]
    residual_q = residual_q.reshape(nq, nc, m, subdim)

    approx = np.zeros((nq, nc), dtype=np.float32)
    for subspace_idx in range(m):
        codewords = pq_centers[subspace_idx, :, cand_codes_h[:, subspace_idx]]
        if codewords.shape[0] != nc:
            codewords = codewords.T
        diff = residual_q[:, :, subspace_idx, :] - codewords[None, :, :]
        approx += np.sum(diff * diff, axis=-1)
    return approx


def run_tensorvs(
    *,
    index_path: Path,
    dataset_path: Path,
    query_path: Path,
    num_queries: int,
    num_candidates: int,
    seed: int,
    topk: int,
) -> dict[str, float | int]:
    dataset = load_fvecs(dataset_path)
    queries = load_fvecs(query_path)
    query_vectors = queries[: min(int(num_queries), queries.shape[0])]
    candidate_ids = sample_ids(dataset.shape[0], num_candidates, seed)
    candidate_vectors = dataset[candidate_ids]

    true_dist = compute_true_distances(query_vectors, candidate_vectors)
    approx_dist = compute_tensorvs_approx(
        index_path=index_path,
        candidate_ids=candidate_ids,
        query_vectors=query_vectors,
    )
    out = summarize(true_dist, approx_dist, topk)
    out["impl"] = "tensorvs"
    return out


def run_cuvs(
    *,
    index_path: Path,
    dataset_path: Path,
    query_path: Path,
    num_queries: int,
    num_candidates: int,
    seed: int,
    topk: int,
) -> dict[str, float | int]:
    dataset = load_fvecs(dataset_path)
    queries = load_fvecs(query_path)
    query_vectors = queries[: min(int(num_queries), queries.shape[0])]
    candidate_ids = sample_ids(dataset.shape[0], num_candidates, seed)
    candidate_vectors = dataset[candidate_ids]

    true_dist = compute_true_distances(query_vectors, candidate_vectors)
    approx_dist = compute_cuvs_approx(
        index_path=index_path,
        candidate_vectors=candidate_vectors,
        query_vectors=query_vectors,
    )
    out = summarize(true_dist, approx_dist, topk)
    out["impl"] = "cuvs"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare IVF-PQ approximate-distance rank correlation.")
    parser.add_argument("--impl", choices=("tensorvs", "cuvs"), required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--query-path", type=Path, required=True)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-candidates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    kwargs = dict(
        index_path=args.index_path,
        dataset_path=args.dataset_path,
        query_path=args.query_path,
        num_queries=args.num_queries,
        num_candidates=args.num_candidates,
        seed=args.seed,
        topk=args.topk,
    )
    if args.impl == "tensorvs":
        out = run_tensorvs(**kwargs)
    else:
        out = run_cuvs(**kwargs)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
