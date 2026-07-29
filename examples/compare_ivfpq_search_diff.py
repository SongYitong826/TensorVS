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
    actual_dims = rows[:, 0].view(np.int32)
    if not np.all(actual_dims == dim):
        raise ValueError(f"Inconsistent fvecs dimensions in {path}")

    vectors = np.asarray(rows[:, 1:], dtype=np.float32, order="C")
    if max_rows is not None:
        vectors = vectors[:max_rows]
    return vectors


def load_ivecs(path: str | Path, max_rows: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"Empty ivecs file: {path}")

    dim = int(raw[0])
    row_width = dim + 1
    if raw.size % row_width != 0:
        raise ValueError(f"Invalid ivecs layout in {path}")

    rows = raw.reshape(-1, row_width)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"Inconsistent ivecs dimensions in {path}")

    vectors = np.asarray(rows[:, 1:], dtype=np.int64, order="C")
    if max_rows is not None:
        vectors = vectors[:max_rows]
    return vectors


def save_results(
    *,
    output_path: Path,
    ids: np.ndarray,
    distances: np.ndarray,
    impl: str,
    topk: int,
    nprobe: int,
    num_queries: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.asarray(ids, dtype=np.int64),
        distances=np.asarray(distances, dtype=np.float32),
        meta=json.dumps(
            {
                "impl": impl,
                "topk": int(topk),
                "nprobe": int(nprobe),
                "num_queries": int(num_queries),
            }
        ),
    )


def run_tensorvs_search(
    *,
    index_path: Path,
    query_path: Path,
    output_path: Path,
    topk: int,
    nprobe: int,
    num_queries: int,
    q_batch: int,
    device: str,
    selection_mode: str,
) -> None:
    import torch
    from tensorvs import load_dataset, load_ivfpq_index, search_ivfpq_index

    index = load_ivfpq_index(index_path, load_lists=True)
    device_obj = torch.device(device)
    index.reside_on_device(device_obj)
    queries = load_dataset(query_path)[:num_queries]

    all_ids = []
    all_distances = []
    for start in range(0, queries.shape[0], q_batch):
        end = min(start + q_batch, queries.shape[0])
        q = queries[start:end].to(device_obj)
        d, i = search_ivfpq_index(
            index=index,
            queries=q,
            topk=topk,
            nprobe=nprobe,
            selection_mode=selection_mode,
        )
        all_ids.append(i.cpu().numpy())
        all_distances.append(d.cpu().numpy())

    save_results(
        output_path=output_path,
        ids=np.concatenate(all_ids, axis=0),
        distances=np.concatenate(all_distances, axis=0),
        impl="tensorvs",
        topk=topk,
        nprobe=nprobe,
        num_queries=queries.shape[0],
    )


def run_cuvs_search(
    *,
    index_path: Path,
    query_path: Path,
    output_path: Path,
    topk: int,
    nprobe: int,
    num_queries: int,
    q_batch: int,
) -> None:
    from pylibraft.common import device_ndarray
    from cuvs.neighbors import ivf_pq

    index = ivf_pq.load(str(index_path))
    queries = load_fvecs(query_path, max_rows=num_queries)
    search_params = ivf_pq.SearchParams(
        n_probes=nprobe,
        lut_dtype=np.float32,
        internal_distance_dtype=np.float32,
    )

    all_ids = []
    all_distances = []
    for start in range(0, queries.shape[0], q_batch):
        end = min(start + q_batch, queries.shape[0])
        d, i = ivf_pq.search(
            search_params,
            index,
            device_ndarray(queries[start:end]),
            topk,
        )
        all_ids.append(i.copy_to_host())
        all_distances.append(d.copy_to_host())

    save_results(
        output_path=output_path,
        ids=np.concatenate(all_ids, axis=0),
        distances=np.concatenate(all_distances, axis=0),
        impl="cuvs",
        topk=topk,
        nprobe=nprobe,
        num_queries=queries.shape[0],
    )


def compute_true_distances_for_ids(
    base_vectors: np.ndarray,
    queries: np.ndarray,
    ids: np.ndarray,
) -> np.ndarray:
    nq, k = ids.shape
    out = np.full((nq, k), np.inf, dtype=np.float32)
    for qid in range(nq):
        row_ids = ids[qid]
        valid = row_ids >= 0
        if np.any(valid):
            vecs = base_vectors[row_ids[valid]]
            diff = vecs - queries[qid]
            out[qid, valid] = np.sum(diff * diff, axis=1)
    return out


def strict_recall_per_query(ids: np.ndarray, gt: np.ndarray, topk: int) -> np.ndarray:
    k = min(topk, ids.shape[1], gt.shape[1])
    out = np.zeros(ids.shape[0], dtype=np.float32)
    for qid in range(ids.shape[0]):
        out[qid] = np.intersect1d(ids[qid, :k], gt[qid, :k]).size / float(k)
    return out


def compare_results(
    *,
    tensorvs_path: Path,
    cuvs_path: Path,
    dataset_path: Path,
    query_path: Path,
    groundtruth_path: Path,
    report_limit: int,
) -> None:
    tv = np.load(tensorvs_path)
    cv = np.load(cuvs_path)
    tv_ids = tv["ids"]
    tv_dist = tv["distances"]
    cv_ids = cv["ids"]
    cv_dist = cv["distances"]

    if tv_ids.shape != cv_ids.shape:
        raise ValueError(f"Mismatched result shapes: {tv_ids.shape} vs {cv_ids.shape}")

    num_queries, topk = tv_ids.shape
    queries = load_fvecs(query_path, max_rows=num_queries)
    base = load_fvecs(dataset_path)
    gt = load_ivecs(groundtruth_path, max_rows=num_queries)

    tv_true = compute_true_distances_for_ids(base, queries, tv_ids)
    cv_true = compute_true_distances_for_ids(base, queries, cv_ids)
    gt_true = compute_true_distances_for_ids(base, queries, gt[:, :topk])

    tv_recall = strict_recall_per_query(tv_ids, gt, topk)
    cv_recall = strict_recall_per_query(cv_ids, gt, topk)
    diff_mask = np.any(tv_ids != cv_ids, axis=1)
    diff_qids = np.flatnonzero(diff_mask)

    summary = {
        "num_queries": int(num_queries),
        "topk": int(topk),
        "queries_with_different_top10": int(diff_qids.size),
        "tensorvs_strict_recall_at_k": float(np.mean(tv_recall)),
        "cuvs_strict_recall_at_k": float(np.mean(cv_recall)),
        "cuvs_better_queries": int(np.sum(cv_recall > tv_recall)),
        "tensorvs_better_queries": int(np.sum(tv_recall > cv_recall)),
        "same_recall_queries": int(np.sum(tv_recall == cv_recall)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    order = np.argsort(cv_recall - tv_recall)[::-1]
    shown = 0
    for qid in order:
        if not diff_mask[qid]:
            continue
        if shown >= report_limit:
            break
        shown += 1
        record = {
            "query_id": int(qid),
            "tensorvs_recall": float(tv_recall[qid]),
            "cuvs_recall": float(cv_recall[qid]),
            "groundtruth_ids": gt[qid, :topk].tolist(),
            "groundtruth_true_distances": [float(x) for x in gt_true[qid].tolist()],
            "tensorvs_ids": tv_ids[qid].tolist(),
            "tensorvs_reported_distances": [float(x) for x in tv_dist[qid].tolist()],
            "tensorvs_true_distances": [float(x) for x in tv_true[qid].tolist()],
            "cuvs_ids": cv_ids[qid].tolist(),
            "cuvs_reported_distances": [float(x) for x in cv_dist[qid].tolist()],
            "cuvs_true_distances": [float(x) for x in cv_true[qid].tolist()],
        }
        print(json.dumps(record, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TensorVS and cuVS IVF-PQ top10 differences.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    tv = subparsers.add_parser("tensorvs-search")
    tv.add_argument("--index-path", type=Path, required=True)
    tv.add_argument("--query-path", type=Path, required=True)
    tv.add_argument("--output-path", type=Path, required=True)
    tv.add_argument("--topk", type=int, default=10)
    tv.add_argument("--nprobe", type=int, default=1024)
    tv.add_argument("--num-queries", type=int, default=1000)
    tv.add_argument("--q-batch", type=int, default=64)
    tv.add_argument("--device", type=str, default="cuda:3")
    tv.add_argument(
        "--selection-mode",
        choices=("triton_topk", "tensor_topk"),
        default="tensor_topk",
    )

    cv = subparsers.add_parser("cuvs-search")
    cv.add_argument("--index-path", type=Path, required=True)
    cv.add_argument("--query-path", type=Path, required=True)
    cv.add_argument("--output-path", type=Path, required=True)
    cv.add_argument("--topk", type=int, default=10)
    cv.add_argument("--nprobe", type=int, default=1024)
    cv.add_argument("--num-queries", type=int, default=1000)
    cv.add_argument("--q-batch", type=int, default=64)

    cmp_p = subparsers.add_parser("compare")
    cmp_p.add_argument("--tensorvs-path", type=Path, required=True)
    cmp_p.add_argument("--cuvs-path", type=Path, required=True)
    cmp_p.add_argument("--dataset-path", type=Path, required=True)
    cmp_p.add_argument("--query-path", type=Path, required=True)
    cmp_p.add_argument("--groundtruth-path", type=Path, required=True)
    cmp_p.add_argument("--report-limit", type=int, default=20)

    args = parser.parse_args()
    if args.cmd == "tensorvs-search":
        run_tensorvs_search(
            index_path=args.index_path,
            query_path=args.query_path,
            output_path=args.output_path,
            topk=args.topk,
            nprobe=args.nprobe,
            num_queries=args.num_queries,
            q_batch=args.q_batch,
            device=args.device,
            selection_mode=args.selection_mode,
        )
    elif args.cmd == "cuvs-search":
        run_cuvs_search(
            index_path=args.index_path,
            query_path=args.query_path,
            output_path=args.output_path,
            topk=args.topk,
            nprobe=args.nprobe,
            num_queries=args.num_queries,
            q_batch=args.q_batch,
        )
    else:
        compare_results(
            tensorvs_path=args.tensorvs_path,
            cuvs_path=args.cuvs_path,
            dataset_path=args.dataset_path,
            query_path=args.query_path,
            groundtruth_path=args.groundtruth_path,
            report_limit=args.report_limit,
        )


if __name__ == "__main__":
    main()
