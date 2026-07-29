from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorvs import (
    build_ivfpq_index,
    load_dataset,
    load_ivfpq_index,
    save_ivfpq_index,
    search_ivfpq_index,
)
from tensorvs.index_ivfpq import IVFPQBuildResult
from tensorvs.util import load_ivecs


def strict_recall_at_k(ids: torch.Tensor, groundtruth: torch.Tensor, topk: int) -> float:
    hits = 0
    for query_ids, gt_ids in zip(ids[:, :topk], groundtruth[:, :topk], strict=True):
        hits += torch.isin(query_ids, gt_ids).sum().item()
    return hits / float(ids.shape[0] * topk)


def build_with_final_labels(
    vectors: torch.Tensor,
    *,
    nlist: int,
    niter: int,
    m: int,
    nbits: int,
    seed: int,
    device: torch.device,
) -> IVFPQBuildResult:
    return build_ivfpq_index(
        vectors=vectors,
        nlist=nlist,
        niter=niter,
        use_triton=True,
        device=device,
        m=m,
        nbits=nbits,
        seed=seed,
        n_init=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild TensorVS IVF-PQ with final-label prediction and a chosen seed."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--query-path", type=Path, required=True)
    parser.add_argument("--groundtruth-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--niter", type=int, default=25)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--q-batch", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--nprobe", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(args.device)
    vectors = load_dataset(args.dataset_path)
    start = time.perf_counter()
    result = build_with_final_labels(
        vectors,
        nlist=args.nlist,
        niter=args.niter,
        m=args.m,
        nbits=args.nbits,
        seed=args.seed,
        device=device,
    )
    torch.cuda.synchronize(device)
    build_seconds = time.perf_counter() - start
    save_ivfpq_index(
        result,
        args.output_path,
        dataset_path=args.dataset_path,
        build_time_seconds=build_seconds,
    )

    del result
    torch.cuda.empty_cache()
    index = load_ivfpq_index(args.output_path, load_lists=True)
    index.reside_on_device(device)
    queries = load_dataset(args.query_path)[: args.num_queries]
    groundtruth = load_ivecs(args.groundtruth_path)[: args.num_queries]

    result_ids = []
    for start_row in range(0, queries.shape[0], args.q_batch):
        query_batch = queries[start_row : start_row + args.q_batch].to(device)
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
                "build_seconds": build_seconds,
                "seed": args.seed,
                "strict_recall": strict_recall_at_k(ids, groundtruth, args.topk),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
