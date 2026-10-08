"""Isolate Torch Flat scan strategies without changing production dispatch.

The diagnostic clones search_torch and changes only its use_dense assignment.
It compares the existing adaptive dispatch with forced list-grouped and dense
scanning, preserving all distance, Top-K, memory and precision settings.

    CUDA_VISIBLE_DEVICES=1 conda run -n tensorvs python -m plot.diagnose_ivfflat_torch
"""
from __future__ import annotations

import argparse
import ast
import csv
import inspect
import json
import os
from pathlib import Path
import statistics
import time

import torch

from tensorvs import compute_recall_at_k, load_dataset, load_ivecs, load_ivfflat_index
import tensorvs._torch_search as implementation
from bench._benchmark_utils import probe_visit_statistics


def scan_function(mode):
    if mode == "adaptive":
        return implementation.search_torch
    tree = ast.parse(inspect.getsource(implementation.search_torch))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "use_dense"
                           for target in node.targets)]
    assert len(assignments) == 1
    assignments[0].value = ast.Constant(mode == "dense")
    namespace = dict(vars(implementation))
    exec(compile(ast.fix_missing_locations(tree), f"<diagnostic_{mode}>", "exec"), namespace)
    return namespace["search_torch"]


def grouped_plan_statistics(index, queries, nprobe):
    layout = index.layout_cache.torch_layout
    from tensorvs.index_ivfflat import _coarse_centroid_scores_addmm
    scores = _coarse_centroid_scores_addmm(queries, layout.tensors["centroids"],
                                          layout.tensors["centroid_norms"])
    probes = scores.topk(nprobe, largest=False, sorted=False).indices
    schedule = implementation._schedule(probes, layout.host_counts, layout.host_offsets, 32, 2048)
    return dict(shape_buckets=len(schedule), tasks=sum(len(tasks) for tasks in schedule.values()),
                valid_distance_pairs=sum(len(task[0]) * task[2] for tasks in schedule.values() for task in tasks),
                padded_distance_pairs=sum(qw * cw * len(tasks) for (qw, cw), tasks in schedule.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("plot/results"))
    args = parser.parse_args()
    device = torch.device(args.device)
    records = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(device=str(device), gpu=torch.cuda.get_device_name(device),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  torch=str(torch.__version__), q_batch=128, K=10, use_compile=False,
                  memory_budget_mb=128, candidate_chunk_size=2048,
                  repetitions=args.repetitions, warmup="at least 2 full rounds and 0.5 seconds",
                  timing="median of complete resident-query rounds; CUDA synchronization after each batch",
                  intervention="diagnostic-only Python clones change only the use_dense assignment",
                  results=records)
    functions = {mode: scan_function(mode) for mode in ("adaptive", "grouped", "dense")}
    with torch.inference_mode():
        for dataset, probes in (("glove1.2m", (50, 60, 63, 64, 80, 120, 400)),
                                ("sift1m", (50, 60, 63, 64, 70, 80))):
            root = Path("/storage/vector_data") / dataset
            index = load_ivfflat_index(root / "GPUIndex" / "TenVS_IVFFlat_nlist1024.bin")
            queries = load_dataset(root / f"{dataset}_query.fvecs")[:1024].to(device)
            groundtruth = load_ivecs(root / f"{dataset}_groundtruth.ivecs")[:queries.shape[0]]
            batches = list(queries.split(128))
            if batches[-1].shape[0] < 128:
                last = batches[-1]
                batches[-1] = torch.cat((last, last[-1:].expand(128 - last.shape[0], -1)))
            index.reside_on_device(device, backend="torch")
            torch.cuda.synchronize(device)
            total = int(index.metadata["num_vectors"])
            print(f"Loaded {dataset}, N={total}, queries={queries.shape[0]}", flush=True)
            for nprobe in probes:
                visits = probe_visit_statistics(index, queries, nprobe, device, int(queries.shape[1]) * 4)
                plan = grouped_plan_statistics(index, batches[0], nprobe)
                reference = None
                for mode, function in functions.items():
                    kwargs = dict(kind="flat", device=device, topk=10, nprobe=nprobe,
                                  query_batch_size=128, candidate_chunk_size=2048,
                                  memory_budget_mb=128, use_compile=False)
                    def run(retain=False):
                        saved = []
                        for batch in batches:
                            output = function(index, batch, **kwargs)
                            torch.cuda.synchronize(device)
                            if retain:
                                saved.append(tuple(t.clone() for t in output))
                        if retain:
                            return tuple(torch.cat([pair[i] for pair in saved])[:queries.shape[0]]
                                         for i in range(2))
                    warmed, started = 0, time.perf_counter()
                    while warmed < 2 or time.perf_counter() - started < 0.5:
                        run()
                        warmed += 1
                    samples = []
                    for _ in range(args.repetitions):
                        torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        run()
                        samples.append(time.perf_counter() - started)
                    values, ids = run(retain=True)
                    recall = compute_recall_at_k(ids.cpu(), groundtruth, 10)
                    if reference is None:
                        reference = (values, ids, recall)
                    torch.testing.assert_close(values, reference[0], atol=2e-4, rtol=2e-5)
                    profile = {}
                    function(index, batches[0], profile=profile, **kwargs)
                    actual = "dense" if "dense_vector_distance_topk" in profile else "grouped"
                    expected = ("dense" if nprobe >= 64 else "grouped") if mode == "adaptive" else mode
                    assert actual == expected
                    elapsed = statistics.median(samples)
                    record = dict(dataset=dataset, num_vectors=total, query_count=queries.shape[0],
                                  nprobe=nprobe, mode=mode, executed_scan=actual, recall=recall,
                                  median_qps=queries.shape[0] / elapsed, samples_seconds=samples,
                                  minimum_qps=queries.shape[0] / max(samples),
                                  maximum_qps=queries.shape[0] / min(samples),
                                  logical_candidates_per_query=visits["visited_vectors_per_query"],
                                  first_batch_grouped_plan=plan,
                                  computed_vectors_per_query=total if actual == "dense" else None,
                                  first_batch_profile_seconds=profile,
                                  topk_set_agreement=(ids[:, :, None] == reference[1][:, None, :]).any(2).float().mean().item(),
                                  max_distance_difference=(values - reference[0]).abs().max().item(),
                                  recall_difference=recall - reference[2])
                    records.append(record)
                    print(f"{dataset} nprobe={nprobe} {mode} -> {actual}: "
                          f"QPS={record['median_qps']:.1f}, Recall={recall:.6f}", flush=True)
                    (args.output_dir / "ivfflat_torch_scan_diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
                del reference
            del index, queries, batches, values, ids
            torch.cuda.empty_cache()
    fields = ("dataset", "nprobe", "mode", "executed_scan", "median_qps", "recall",
              "logical_candidates_per_query", "computed_vectors_per_query", "recall_difference")
    with (args.output_dir / "ivfflat_torch_scan_diagnosis.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
