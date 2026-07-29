from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_fvecs(path: str | Path, max_rows: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")
    dim = int(raw[:1].view(np.int32)[0])
    row_width = dim + 1
    if raw.size % row_width != 0:
        raise ValueError(f"Invalid fvecs layout in {path}")
    rows = raw.reshape(-1, row_width)
    if not np.all(rows[:, 0].view(np.int32) == dim):
        raise ValueError(f"Inconsistent fvecs dimensions in {path}")
    vectors = np.asarray(rows[:, 1:], dtype=np.float32, order="C")
    if max_rows is not None:
        vectors = vectors[:max_rows]
    return vectors


def sample_ids(num_rows: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = min(int(sample_size), int(num_rows))
    return np.asarray(rng.choice(num_rows, size=count, replace=False), dtype=np.int64)


def compute_true_distances(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    q_norm = np.sum(queries * queries, axis=1, keepdims=True)
    c_norm = np.sum(candidates * candidates, axis=1, keepdims=True).T
    prod = queries @ candidates.T
    dist = q_norm + c_norm - 2.0 * prod
    np.maximum(dist, 0.0, out=dist)
    return dist


def compute_tensorvs_approx(index_path: Path, candidate_ids: np.ndarray, query_vectors: np.ndarray) -> np.ndarray:
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
    nq, nc, _ = residual_q.shape
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


def compute_cuvs_approx(index_path: Path, candidate_vectors: np.ndarray, query_vectors: np.ndarray) -> np.ndarray:
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


def top_entries(
    score_matrix: np.ndarray,
    candidate_ids: np.ndarray,
    true_dist: np.ndarray,
    tv_approx: np.ndarray,
    cv_approx: np.ndarray,
    limit: int,
    largest: bool,
) -> list[dict[str, float | int]]:
    flat = score_matrix.reshape(-1)
    order = np.argsort(flat)
    if largest:
        order = order[::-1]
    out = []
    nq, nc = score_matrix.shape
    for flat_idx in order[:limit]:
        qid = int(flat_idx // nc)
        local_cid = int(flat_idx % nc)
        out.append(
            {
                "query_id": qid,
                "candidate_id": int(candidate_ids[local_cid]),
                "delta_error": float(score_matrix[qid, local_cid]),
                "true_distance": float(true_dist[qid, local_cid]),
                "tensorvs_approx": float(tv_approx[qid, local_cid]),
                "cuvs_approx": float(cv_approx[qid, local_cid]),
                "tensorvs_error": float(tv_approx[qid, local_cid] - true_dist[qid, local_cid]),
                "cuvs_error": float(cv_approx[qid, local_cid] - true_dist[qid, local_cid]),
            }
        )
    return out


def aggregate_per_candidate(
    candidate_ids: np.ndarray,
    delta_error: np.ndarray,
    true_dist: np.ndarray,
    tv_approx: np.ndarray,
    cv_approx: np.ndarray,
    limit: int,
    largest: bool,
) -> list[dict[str, float | int]]:
    mean_delta = delta_error.mean(axis=0)
    mean_tv_error = (tv_approx - true_dist).mean(axis=0)
    mean_cv_error = (cv_approx - true_dist).mean(axis=0)
    order = np.argsort(mean_delta)
    if largest:
        order = order[::-1]
    out = []
    for idx in order[:limit]:
        out.append(
            {
                "candidate_id": int(candidate_ids[idx]),
                "mean_delta_error": float(mean_delta[idx]),
                "mean_tensorvs_error": float(mean_tv_error[idx]),
                "mean_cuvs_error": float(mean_cv_error[idx]),
            }
        )
    return out


def save_partial(
    *,
    output_path: Path,
    impl: str,
    candidate_ids: np.ndarray,
    true_dist: np.ndarray,
    approx_dist: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        impl=impl,
        candidate_ids=np.asarray(candidate_ids, dtype=np.int64),
        true_dist=np.asarray(true_dist, dtype=np.float32),
        approx_dist=np.asarray(approx_dist, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TensorVS vs cuVS IVFPQ distance bias.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    for name in ("tensorvs", "cuvs"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--index", type=Path, required=True)
        sub.add_argument("--dataset-path", type=Path, required=True)
        sub.add_argument("--query-path", type=Path, required=True)
        sub.add_argument("--output-path", type=Path, required=True)
        sub.add_argument("--num-queries", type=int, default=100)
        sub.add_argument("--num-candidates", type=int, default=5000)
        sub.add_argument("--seed", type=int, default=0)

    cmp_p = subparsers.add_parser("compare")
    cmp_p.add_argument("--tensorvs-path", type=Path, required=True)
    cmp_p.add_argument("--cuvs-path", type=Path, required=True)
    cmp_p.add_argument("--report-limit", type=int, default=20)

    args = parser.parse_args()

    if args.cmd in ("tensorvs", "cuvs"):
        dataset = load_fvecs(args.dataset_path)
        queries = load_fvecs(args.query_path, max_rows=args.num_queries)
        candidate_ids = sample_ids(dataset.shape[0], args.num_candidates, args.seed)
        candidate_vectors = dataset[candidate_ids]
        true_dist = compute_true_distances(queries, candidate_vectors)
        if args.cmd == "tensorvs":
            approx_dist = compute_tensorvs_approx(args.index, candidate_ids, queries)
        else:
            approx_dist = compute_cuvs_approx(args.index, candidate_vectors, queries)
        save_partial(
            output_path=args.output_path,
            impl=args.cmd,
            candidate_ids=candidate_ids,
            true_dist=true_dist,
            approx_dist=approx_dist,
        )
        return

    tv = np.load(args.tensorvs_path)
    cv = np.load(args.cuvs_path)
    candidate_ids = tv["candidate_ids"]
    true_dist = tv["true_dist"]
    tv_approx = tv["approx_dist"]
    cv_approx = cv["approx_dist"]
    if not np.array_equal(candidate_ids, cv["candidate_ids"]):
        raise ValueError("Candidate sets differ between TensorVS and cuVS partial outputs")
    if not np.array_equal(true_dist, cv["true_dist"]):
        raise ValueError("True-distance matrices differ between TensorVS and cuVS partial outputs")

    tv_error = tv_approx - true_dist
    cv_error = cv_approx - true_dist
    delta_error = tv_error - cv_error

    summary = {
        "num_queries": int(true_dist.shape[0]),
        "num_candidates": int(true_dist.shape[1]),
        "tensorvs_mean_error": float(tv_error.mean()),
        "cuvs_mean_error": float(cv_error.mean()),
        "tensorvs_mean_abs_error": float(np.mean(np.abs(tv_error))),
        "cuvs_mean_abs_error": float(np.mean(np.abs(cv_error))),
        "mean_delta_error": float(delta_error.mean()),
        "mean_abs_delta_error": float(np.mean(np.abs(delta_error))),
        "tensorvs_underestimates_more_fraction": float(np.mean(delta_error < 0.0)),
        "tensorvs_overestimates_more_fraction": float(np.mean(delta_error > 0.0)),
    }

    report = {
        "summary": summary,
        "most_tensorvs_underestimated_pairs": top_entries(
            delta_error, candidate_ids, true_dist, tv_approx, cv_approx, args.report_limit, largest=False
        ),
        "most_tensorvs_overestimated_pairs": top_entries(
            delta_error, candidate_ids, true_dist, tv_approx, cv_approx, args.report_limit, largest=True
        ),
        "candidates_tensorvs_underestimates_more": aggregate_per_candidate(
            candidate_ids, delta_error, true_dist, tv_approx, cv_approx, args.report_limit, largest=False
        ),
        "candidates_tensorvs_overestimates_more": aggregate_per_candidate(
            candidate_ids, delta_error, true_dist, tv_approx, cv_approx, args.report_limit, largest=True
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
