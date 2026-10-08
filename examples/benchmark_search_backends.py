"""Compare steady-state search backends using an existing index (no training).

Run with ``conda run -n tensorvs python examples/benchmark_search_backends.py
--algorithm flat --index ... --queries ... --device cuda:1``.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from tensorvs import (load_dataset, load_ivfflat_index, load_ivfpq_index,
                      search_ivfflat_index, search_ivfpq_index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("flat", "pq"), required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--nq", type=int, default=128)
    parser.add_argument("--query-batch", type=int, default=128)
    parser.add_argument("--nprobe", type=int, default=120)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--all-paths", action="store_true",
                        help="Include auto, compiled torch and PQ workspace/graph/autotune/unpacked variants")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warmup-seconds", type=float, default=0.5,
                        help="Minimum warming duration per path to stabilize GPU clocks")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.nq, args.query_batch, args.repetitions) <= 0 or args.warmup < 0 or args.warmup_seconds < 0:
        parser.error("nq, query-batch and repetitions must be positive; warmup must be nonnegative")
    load, search = (load_ivfflat_index, search_ivfflat_index) if args.algorithm == "flat" else (load_ivfpq_index, search_ivfpq_index)
    index = load(args.index)
    q = load_dataset(args.queries)[:args.nq].to(args.device)
    if q.shape[0] == 0:
        parser.error("queries must not be empty")
    batches = list(q.split(args.query_batch))
    if args.all_paths and args.algorithm == "pq" and batches[-1].shape[0] != args.query_batch:
        parser.error("PQ all-paths requires full fixed-size batches for CUDA Graph replay; choose a multiple of query-batch")
    device = q.device
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    modes = [("triton", "triton", {}), ("torch", "torch", {})]
    if args.all_paths:
        modes.insert(1, ("auto", "auto", {}))
    if args.compiled or args.all_paths:
        modes.append(("torch_compile", "torch", {"use_compile": True}))
    if args.all_paths and args.algorithm == "pq":
        modes.extend([
            ("triton_workspace", "triton", {"use_preallocated_workspace": True}),
            ("triton_cuda_graph", "triton", {"use_cuda_graph": True}),
            ("triton_autotune", "triton", {"triton_autotune": True}),
            ("triton_autotune_workspace", "triton", {"triton_autotune": True,
                                                        "use_preallocated_workspace": True}),
            ("triton_autotune_cuda_graph", "triton", {"triton_autotune": True,
                                                         "use_cuda_graph": True}),
            ("triton_unpacked", "triton", {"use_packed_16b_codes": False}),
        ])
    results = []
    reference = None
    with torch.inference_mode():
        for name, backend, options in modes:
            kwargs = dict(backend=backend, device=device, nprobe=args.nprobe, topk=args.topk,
                          **options)
            def run(retain=False):
                saved = []
                for batch in batches:
                    outputs = search(index, batch, **kwargs)
                    if retain:
                        # Workspace/graph outputs are overwritten by the next
                        # batch. Clone only in the untimed validation pass.
                        saved.append(tuple(t.clone() for t in outputs))
                if retain:
                    return tuple(torch.cat([item[i] for item in saved], 0) for i in range(2))
            started = time.perf_counter()
            search(index, batches[0], **kwargs)
            synchronize()
            first_call = time.perf_counter() - started
            warmed = 0
            warm_started = time.perf_counter()
            while warmed < args.warmup or time.perf_counter() - warm_started < args.warmup_seconds:
                run()
                synchronize()
                warmed += 1
            times = []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.cuda.memory_allocated(device)
            for _ in range(args.repetitions):
                synchronize()
                started = time.perf_counter()
                run()
                synchronize()
                times.append((time.perf_counter() - started) * 1e6 / q.shape[0])
            peak_mb = ((torch.cuda.max_memory_allocated(device) - baseline) / (1 << 20)
                       if device.type == "cuda" else None)
            outputs = run(retain=True)
            synchronize()
            if reference is None:
                reference = outputs
            matched = (outputs[1][:, :, None] == reference[1][:, None, :]).any(2)
            agreement = matched.sum().item() / matched.numel()
            finite = torch.isfinite(outputs[0]) & torch.isfinite(reference[0])
            difference = ((outputs[0][finite] - reference[0][finite]).abs().max().item()
                          if finite.any().item() else 0.0)
            if options.get("use_cuda_graph") and index.search_workspace.graph is None:
                raise RuntimeError("The CUDA Graph benchmark did not capture/replay a graph")
            median = statistics.median(times)
            result = dict(path=name, backend=backend, compiled=options.get("use_compile", False),
                          options=options, first_call_seconds=first_call,
                          actual_warmup_rounds=warmed,
                          median_us_per_query=median, queries_per_second=1e6 / median,
                          median_round_ms=median * q.shape[0] / 1000, samples_us_per_query=times,
                          topk_set_agreement=agreement,
                          mismatched_query_count=(~matched).any(1).sum().item(),
                          mismatched_result_count=(~matched).sum().item(),
                          maximum_distance_difference=difference,
                          finite_results_match=torch.equal(torch.isfinite(outputs[0]), torch.isfinite(reference[0])),
                          incremental_peak_mb=peak_mb)
            results.append(result)
            if args.profile and not options.get("use_cuda_graph"):
                profile = {}
                for batch in batches:
                    search(index, batch, profile=profile, **kwargs)
                result["profile_us_per_query"] = {name: seconds * 1e6 / q.shape[0] for name, seconds in profile.items()}
            print(json.dumps(result), flush=True)
    report = dict(algorithm=args.algorithm, index=args.index, device=str(device),
                  gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  autotune_cache=os.environ.get("TENSORVS_TRITON_AUTOTUNE_CACHE", "default user cache"),
                  torch_version=str(torch.__version__), runtime=torch.version.hip or torch.version.cuda,
                  query_count=q.shape[0], query_batch=args.query_batch, dimension=q.shape[1],
                  nprobe=args.nprobe, topk=args.topk, warmup_rounds=args.warmup,
                  minimum_warmup_seconds=args.warmup_seconds,
                  repetitions=args.repetitions, timing="host wall clock with device synchronization; resident inputs/index",
                  results=results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
