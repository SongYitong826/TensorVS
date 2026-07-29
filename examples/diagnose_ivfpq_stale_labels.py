from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flash_kmeans.kmeans_large import kmeans_largeN_assign
from tensorvs import load_dataset, load_ivfpq_index, search_ivfpq_index
from tensorvs.index_ivfpq import _encode_vectors_with_pq
from tensorvs.util import load_ivecs


def strict_recall_at_k(ids: torch.Tensor, groundtruth: torch.Tensor, topk: int) -> float:
    hits = 0
    for query_ids, gt_ids in zip(ids[:, :topk], groundtruth[:, :topk], strict=True):
        hits += torch.isin(query_ids, gt_ids).sum().item()
    return hits / float(ids.shape[0] * topk)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure and correct stale coarse labels in a TensorVS residual-PQ index."
    )
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--query-path", type=Path, required=True)
    parser.add_argument("--groundtruth-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--q-batch", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--nprobe", type=int, default=1024)
    parser.add_argument("--search-corrected", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    index = load_ivfpq_index(args.index_path, load_lists=True)
    vectors = load_dataset(args.dataset_path)
    old_labels = index.assignments.to(torch.int64)
    new_labels = kmeans_largeN_assign(
        vectors,
        index.centroids,
        device=device,
    ).to(torch.int64).cpu()

    mismatch = old_labels != new_labels
    groundtruth = load_ivecs(args.groundtruth_path)[: args.num_queries]
    gt_ids = groundtruth[:, : args.topk].reshape(-1)
    summary = {
        "num_vectors": int(vectors.shape[0]),
        "mismatch_count": int(mismatch.sum()),
        "mismatch_fraction": float(mismatch.float().mean()),
        "gt_topk_mismatch_count": int(mismatch[gt_ids].sum()),
        "gt_topk_mismatch_fraction": float(mismatch[gt_ids].float().mean()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.search_corrected:
        return

    centroids = index.centroids.to(device=device, dtype=torch.float32)
    codebooks = index.pq_codebooks.to(device=device, dtype=torch.float32)
    vectors_device = vectors.to(device=device, dtype=torch.float32)
    labels_device = new_labels.to(device=device)
    residuals = (vectors_device - centroids[labels_device]).contiguous()
    corrected_codes = _encode_vectors_with_pq(
        residuals,
        codebooks,
        int(index.metadata["m"]),
        code_dtype=index.list_codes.dtype,
    ).cpu()

    sort_perm = torch.argsort(new_labels, stable=True)
    labels_sorted = new_labels[sort_perm]
    list_count = torch.bincount(
        labels_sorted, minlength=index.centroids.shape[0]
    ).to(torch.int64)
    index.assignments = new_labels
    index.list_codes = corrected_codes[sort_perm]
    index.list_ids_flat = torch.arange(vectors.shape[0], dtype=torch.int64)[sort_perm]
    index.list_count = list_count
    index.list_prefix = torch.cumsum(list_count, dim=0) - list_count

    del residuals, corrected_codes, vectors_device, labels_device, centroids, codebooks
    torch.cuda.empty_cache()
    index.reside_on_device(device)

    queries = load_dataset(args.query_path)[: args.num_queries]
    result_ids = []
    for start in range(0, queries.shape[0], args.q_batch):
        query_batch = queries[start : start + args.q_batch].to(device)
        _, ids = search_ivfpq_index(
            index,
            query_batch,
            topk=args.topk,
            nprobe=args.nprobe,
        )
        result_ids.append(ids.cpu())

    ids = torch.cat(result_ids)
    print(
        json.dumps(
            {
                "corrected_label_reencode_strict_recall": strict_recall_at_k(
                    ids, groundtruth, args.topk
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
