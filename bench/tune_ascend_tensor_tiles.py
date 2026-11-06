"""Tune grouped Tensor execution on a few queries; never rebuild the IVF index.

Writes a diagnostic report separately from experiment CSV files. Keep the NPU
idle while running this script. Both the candidate set and FP32/LUT precision
are fixed; only query tiles, candidate chunks, task caps and workspace vary.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from ascend_tensor_bench import configure_npu
from amd_tensor_bench import index_path_for
from tensorvs import (load_dataset, load_ivecs, load_ivfflat_index, load_ivfpq_index,
                      search_ivfflat_index, search_ivfpq_index)
from tensorvs import _torch_search as native

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("ivfflat", "ivfpq"))
    p.add_argument("--dataset", choices=("glove1.2m", "hunyuan"), default="glove1.2m")
    p.add_argument("--data-root", type=Path, default=Path("/home/ma-user/work/data"))
    p.add_argument("--q-batch", type=int, default=128)
    p.add_argument("--query-offset", type=int, default=0, help="Holdout query offset for diagnostic validation")
    p.add_argument("--reference-query-limit", type=int, default=0,
                   help="Validate sampled queries for large batches; 0 validates/times the full baseline")
    p.add_argument("--skip-stage-profile", action="store_true",
                   help="Skip the separate diagnostic baseline event-profiling pass")
    p.add_argument("--nprobe", type=int, default=80)
    p.add_argument("--scan", choices=("grouped", "dense", "gpu"), default="grouped",
                   help="Tensor execution strategy to compare with the default grouped baseline")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--query-tiles", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--candidate-chunks", type=int, nargs="+", default=[1024, 2048])
    p.add_argument("--task-caps", type=int, nargs="+", default=[64, 256])
    p.add_argument("--memory-budgets", type=float, nargs="+", default=[128])
    p.add_argument("--lut-dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--lut-chunks", type=int, nargs="+", default=[16])
    p.add_argument("--pq-lookup", choices=("flat", "gather"), default="flat")
    p.add_argument("--min-query-tile", type=int, default=1)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if min(args.q_batch, args.nprobe, args.topk, args.repetitions,
           *args.query_tiles, *args.candidate_chunks, *args.task_caps, *args.memory_budgets, *args.lut_chunks) <= 0:
        p.error("Counts and workspace budgets must be positive")
    if any(tile & (tile - 1) for tile in args.query_tiles):
        p.error("Query tiles must be powers of two")
    if args.query_offset < 0 or args.reference_query_limit < 0:
        p.error("Query offset/reference limit must be nonnegative")
    if args.min_query_tile <= 0 or args.min_query_tile & (args.min_query_tile - 1):
        p.error("Minimum query tile must be a positive power of two")
    if (args.kind == "ivfflat" and args.scan == "gpu") or (args.kind == "ivfpq" and args.scan == "dense"):
        p.error("Flat supports grouped/dense; PQ supports grouped/gpu")
    # Do not contaminate a concurrently running formal experiment.
    info = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True).stdout
    from ascend_tensor_bench import PROCESS_ROW
    if PROCESS_ROW.findall(info):
        raise RuntimeError("NPU already has a running process; run tuning after that experiment finishes")
    device = configure_npu(0)
    torch.set_num_threads(1)
    path = index_path_for(args.data_root / args.dataset, args.kind)
    load, search = ((load_ivfflat_index, search_ivfflat_index) if args.kind == "ivfflat"
                    else (load_ivfpq_index, search_ivfpq_index))
    print(f"Loading unchanged index: {path}", flush=True)
    index = load(path, load_lists=True)
    if args.nprobe > len(index.centroids):
        p.error("nprobe exceeds index nlist")
    query_slice = slice(args.query_offset, args.query_offset + args.q_batch)
    queries = load_dataset(args.data_root / args.dataset / f"{args.dataset}_query.fvecs")[query_slice]
    truth = load_ivecs(args.data_root / args.dataset / f"{args.dataset}_groundtruth.ivecs")[query_slice]
    if len(queries) != args.q_batch:
        p.error("Not enough queries for q-batch")
    if len(truth) != len(queries) or truth.shape[1] < args.topk:
        p.error("Insufficient ground truth")
    queries = queries.to(device)
    reference_count = min(args.reference_query_limit or len(queries), len(queries))
    reference_positions = torch.linspace(0, len(queries) - 1, reference_count, dtype=torch.int64)
    reference_queries = queries.index_select(0, reference_positions.to(device))
    index.reside_on_device(device, backend="torch", persist_layout=False)
    torch.npu.synchronize(device)
    source_paths = [Path(__file__), ROOT / "src/tensorvs/_torch_search.py",
                    ROOT / "src/tensorvs/index_ivfflat.py", ROOT / "src/tensorvs/index_ivfpq.py",
                    ROOT / "src/tensorvs/_layout_cache.py", ROOT / "src/tensorvs/_torch_pq_bucketed.py"]
    report = dict(kind=args.kind, dataset=args.dataset, q_batch=args.q_batch,
        nprobe=args.nprobe, topk=args.topk, lut_dtype=args.lut_dtype, query_offset=args.query_offset,
        index=str(path), index_size_bytes=path.stat().st_size,
        index_mtime_ns=path.stat().st_mtime_ns, index_metadata=index.metadata,
        torch_version=str(torch.__version__), device=torch.npu.get_device_name(device),
        allow_hf32=False, search_strategy="grouped", use_compile=False,
        baseline_query_count=reference_count, full_query_count=len(queries),
        validation_query_positions=reference_positions.tolist(),
        speedup_comparable=reference_count == len(queries),
        cpu_threads=torch.get_num_threads(), trial_strategy=args.scan,
        source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths},
        timing="median synchronized latency of one resident query batch; diagnostic only, not a full benchmark",
        results=[])
    output = args.output or ROOT / "diagnostics" / f"Ascend_Tensor_Tiles_{args.kind}_{args.dataset}_Q{args.q_batch}_P{args.nprobe}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = output.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2, default=str) + "\n")
        tmp.replace(output)

    configurations = [(None, 2048, 256, 128., 16)] + list(itertools.product(
        args.query_tiles, args.candidate_chunks, args.task_caps, args.memory_budgets, args.lut_chunks))
    reference = None
    source_stat = path.stat()
    try:
        with torch.inference_mode():
            for number, (tile, chunk, cap, budget, lut_chunk) in enumerate(configurations):
                scan = "grouped" if number == 0 else args.scan
                row = dict(query_tile_size=tile, candidate_chunk_size=chunk,
                           max_tasks_per_batch=cap, memory_budget_mb=budget, scan=scan, lut_chunk_size=lut_chunk,
                           pq_lookup="flat" if number == 0 else args.pq_lookup)
                row["min_query_tile_size"] = 1 if number == 0 else args.min_query_tile
                active_queries = reference_queries if number == 0 else queries
                row["queries_measured"] = len(active_queries)
                options = dict(device=device, backend="torch", topk=args.topk, nprobe=args.nprobe,
                    persist_layout=False, use_compile=False, torch_query_batch_size=args.q_batch,
                    torch_query_tile_size=tile, torch_candidate_chunk_size=chunk,
                    torch_max_tasks_per_batch=cap, torch_memory_budget_mb=budget)
                options["torch_min_query_tile_size"] = row["min_query_tile_size"]
                options.update(torch_flat_scan=scan) if args.kind == "ivfflat" else options.update(
                    torch_pq_scan=scan, lut_dtype=args.lut_dtype, torch_lut_chunk_size=lut_chunk,
                    torch_pq_lookup=row["pq_lookup"])
                schedules = []
                original = native._schedule

                def capture(probes, counts, offsets, qtile, cchunk, **kw):
                    tasks = original(probes, counts, offsets, qtile, cchunk, **kw)
                    schedules.append(dict(effective_query_tile=qtile, effective_candidate_chunk=cchunk,
                        tasks=sum(map(len, tasks.values())), shape_buckets=len(tasks),
                        padded_query_candidates=sum(q * c * len(items) for (q, c), items in tasks.items()),
                        actual_query_candidates=sum(len(t[0]) * t[2] for items in tasks.values() for t in items)))
                    return tasks

                try:
                    native._schedule = capture
                    d, ids = search(index, active_queries, **options)
                    torch.npu.synchronize(device)
                finally:
                    native._schedule = original
                d, ids = d.cpu(), ids.cpu()
                if reference is None:
                    reference = d, ids
                validation_d = d if number == 0 else d.index_select(0, reference_positions)
                validation_ids = ids if number == 0 else ids.index_select(0, reference_positions)
                torch.testing.assert_close(validation_d, reference[0], rtol=1e-4, atol=1e-3)
                agreement = sum(len(set(a.tolist()) & set(b.tolist()))
                    for a, b in zip(validation_ids, reference[1])) / validation_ids.numel()
                if agreement < .98:
                    raise RuntimeError(f"Tiling changes neighbors unexpectedly: {agreement}")
                samples = []
                for _ in range(args.repetitions):
                    torch.npu.synchronize(device)
                    begin = time.perf_counter()
                    distances, found = search(index, active_queries, **options)
                    torch.npu.synchronize(device)
                    samples.append(time.perf_counter() - begin)
                    del distances, found
                row.update(latency_ms=statistics.median(samples) * 1000,
                    qps=len(active_queries) / statistics.median(samples), samples_seconds=samples,
                    baseline_id_overlap=agreement, schedule=schedules[-1] if schedules else {})
                active_truth = truth.index_select(0, reference_positions) if number == 0 else truth
                row["recall"] = sum(len(set(a.tolist()) & set(b[:args.topk].tolist()))
                    for a, b in zip(ids, active_truth)) / ids.numel()
                validation_truth = truth.index_select(0, reference_positions)
                row["validation_recall"] = sum(len(set(a.tolist()) & set(b[:args.topk].tolist()))
                    for a, b in zip(validation_ids, validation_truth)) / validation_ids.numel()
                if number == 0 and not args.skip_stage_profile:
                    stages = {}
                    search(index, active_queries, profile=stages, **options)
                    torch.npu.synchronize(device)
                    row["stages_seconds"] = stages
                report["results"].append(row)
                save()
                print(f"scan={scan} tile={str(tile):>4} chunk={chunk} tasks={cap} budget={budget:g} lut={lut_chunk} "
                      f"latency={row['latency_ms']:.2f} ms QPS={row['qps']:.2f}", flush=True)
        baseline = report["results"][0]
        best = min(report["results"] if reference_count == len(queries) else report["results"][1:],
                   key=lambda r: r["latency_ms"])
        if (path.stat().st_size, path.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise RuntimeError("Source index changed while tuning")
        speedup = baseline["latency_ms"] / best["latency_ms"] if reference_count == len(queries) else None
        report.update(best=best, speedup_over_default=speedup)
        save()
        comparison = f"{speedup:.3f}x" if speedup is not None else "not compared (sampled baseline)"
        print(f"Best: {best}; speedup={comparison}; saved {output}", flush=True)
    finally:
        index.release_device_residency()
        torch.npu.synchronize(device)


if __name__ == "__main__":
    main()
