"""Tune portable Tensor execution without rebuilding or changing the IVF index.

Supports Ascend, Cambricon MLU and Kunlun through their PyTorch runtimes, plus
CPU/CUDA/MPS. Run on an idle device. Reports are separate from experiment CSVs;
index parameters, nprobe, K and precision stay fixed throughout each sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import torch

try:
    from .amd_tensor_bench import index_path_for
except ImportError:
    from amd_tensor_bench import index_path_for
from tensorvs import (load_dataset, load_ivecs, load_ivfflat_index, load_ivfpq_index,
                      search_ivfflat_index, search_ivfpq_index)
from tensorvs import _torch_search as native

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = {
    "ascend": ("npu:0", "torch_npu"),
    "mlu": ("mlu:0", "torch_mlu"),
    # xpytorch/torch_xmlir can expose Kunlun through the CUDA compatibility API.
    # --device and --device-module allow other vendor runtime registrations.
    "kunlun": ("cuda:0", "torch_xmlir"),
    "cpu": ("cpu", None),
    "cuda": ("cuda:0", None),
    "mps": ("mps", None),
}


@dataclass
class DeviceRuntime:
    device: torch.device
    api: object
    name: str
    modules: dict
    precision: dict

    def synchronize(self):
        # The selected device is made current at setup. Vendor APIs and MPS
        # all accept this form, unlike synchronize(device) on some runtimes.
        if self.api is not None:
            self.api.synchronize()


def configure_runtime(platform, device_name=None, device_modules=None):
    """Import only the selected vendor plugin and require real device timing."""
    default_device, default_module = PLATFORMS[platform]
    module_names = device_modules if device_modules is not None else ([default_module] if default_module else [])
    modules = {}
    for name in module_names:
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            raise RuntimeError(f"{platform} requires {name} in the current vendor PyTorch environment") from error
        modules[name] = str(getattr(module, "__version__", "unknown"))
    device = torch.device(device_name or default_device)
    if platform == "cpu":
        if device.type != "cpu":
            raise ValueError("--platform cpu requires a CPU device")
        return DeviceRuntime(device, None, "CPU", modules, {})
    expected_type = {"ascend": "npu", "mlu": "mlu", "cuda": "cuda", "mps": "mps"}.get(platform)
    if device.type == "cpu" or (expected_type and device.type != expected_type):
        raise ValueError(f"Device {device} does not match platform {platform}")
    api = getattr(torch, device.type, None)
    availability = torch.backends.mps if device.type == "mps" else api
    if availability is None or not availability.is_available():
        raise RuntimeError(f"{platform} device {device} is unavailable; CPU fallback is not used")
    if not callable(getattr(api, "synchronize", None)):
        raise RuntimeError(f"torch.{device.type}.synchronize is required for accurate timing")
    if hasattr(api, "device_count") and device.index is not None:
        if device.index >= api.device_count():
            raise ValueError(f"Device {device} is unavailable")
    if hasattr(api, "set_device"):
        api.set_device(device)

    # Fix precision for every trial, including the reference. Do not treat
    # reduced matmul precision as an execution-parameter speedup.
    torch.set_float32_matmul_precision("highest")
    precision = {"float32_matmul_precision": torch.get_float32_matmul_precision()}
    backend = getattr(torch.backends, device.type, None)
    for label, owner in ((f"torch.{device.type}", api), (f"torch.backends.{device.type}", backend)):
        matmul = getattr(owner, "matmul", None)
        for flag in ("allow_hf32", "allow_tf32"):
            if hasattr(matmul, flag):
                setattr(matmul, flag, False)
                precision[f"{label}.matmul.{flag}"] = bool(getattr(matmul, flag))
    name = str(api.get_device_name(device)) if hasattr(api, "get_device_name") else str(device)
    return DeviceRuntime(device, api, name, modules, precision)


def load_groundtruth(path):
    """Keep neighbor IDs as integers; the float dataset loader loses large IDs."""
    if path.suffix == ".ivecs":
        return load_ivecs(path)
    if path.suffix == ".npy":
        data = torch.from_numpy(np.load(path, allow_pickle=False))
    elif path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError("Ground truth must be .ivecs, .npy, or a .pt/.pth integer tensor")
    if not isinstance(data, torch.Tensor) or data.ndim != 2 or data.dtype not in (
            torch.int32, torch.int64):
        raise ValueError("Ground truth must be a 2D int32/int64 tensor of neighbor IDs")
    return data.to(dtype=torch.int64)


def index_signature(path):
    """Track file and directory indexes without hashing multi-GB payloads."""
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {str(p.relative_to(path) if path.is_dir() else p.name):
            (p.stat().st_size, p.stat().st_mtime_ns) for p in files}


def main(argv=None, *, default_platform="ascend"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("ivfflat", "ivfpq"))
    p.add_argument("--platform", choices=PLATFORMS, default=default_platform)
    p.add_argument("--device", help="Override the platform device, e.g. mlu:1 or cuda:0")
    p.add_argument("--device-module", action="append",
                   help="Import this vendor plugin instead of the platform default; repeat for multiple plugins")
    p.add_argument("--dataset", default="glove1.2m", help="Dataset label; also used for default input paths")
    p.add_argument("--data-root", type=Path, default=Path("/home/ma-user/work/data"))
    p.add_argument("--index", type=Path, help="Existing TensorVS index file or directory; never rebuilt")
    p.add_argument("--queries", type=Path, help="Query .fvecs/.npy/.pt/.pth file")
    p.add_argument("--groundtruth", type=Path, help="Neighbor IDs in .ivecs/.npy/.pt/.pth format")
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
    p.add_argument("--pq-lookup", choices=("flat", "gather"), nargs="+", default=["flat"])
    p.add_argument("--min-query-tile", type=int, default=1)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
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
    if args.platform == "ascend":
        info = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True).stdout
        try:
            from .ascend_tensor_bench import PROCESS_ROW
        except ImportError:
            from ascend_tensor_bench import PROCESS_ROW
        if PROCESS_ROW.findall(info):
            raise RuntimeError("NPU already has a running process; run tuning after that experiment finishes")
    runtime = configure_runtime(args.platform, args.device, args.device_module)
    device = runtime.device
    torch.set_num_threads(1)
    path = args.index or index_path_for(args.data_root / args.dataset, args.kind)
    query_path = args.queries or args.data_root / args.dataset / f"{args.dataset}_query.fvecs"
    truth_path = args.groundtruth or args.data_root / args.dataset / f"{args.dataset}_groundtruth.ivecs"
    source_signature = index_signature(path)
    load, search = ((load_ivfflat_index, search_ivfflat_index) if args.kind == "ivfflat"
                    else (load_ivfpq_index, search_ivfpq_index))
    print(f"Loading unchanged index: {path}", flush=True)
    index = load(path, load_lists=True)
    if args.nprobe > len(index.centroids):
        p.error("nprobe exceeds index nlist")
    query_slice = slice(args.query_offset, args.query_offset + args.q_batch)
    queries = load_dataset(query_path)[query_slice]
    truth = load_groundtruth(truth_path)[query_slice]
    if len(queries) != args.q_batch:
        p.error("Not enough queries for q-batch")
    if truth.ndim != 2 or len(truth) != len(queries) or truth.shape[1] < args.topk:
        p.error("Insufficient ground truth")
    if queries.shape[1] != index.centroids.shape[1]:
        p.error("Query dimension differs from index dimension")
    queries = queries.to(device)
    reference_count = min(args.reference_query_limit or len(queries), len(queries))
    reference_positions = torch.linspace(0, len(queries) - 1, reference_count, dtype=torch.int64)
    reference_queries = queries.index_select(0, reference_positions.to(device))
    index.reside_on_device(device, backend="torch", persist_layout=False)
    runtime.synchronize()
    source_paths = [Path(__file__), ROOT / "src/tensorvs/_torch_search.py",
                    ROOT / "src/tensorvs/index_ivfflat.py", ROOT / "src/tensorvs/index_ivfpq.py",
                    ROOT / "src/tensorvs/_layout_cache.py", ROOT / "src/tensorvs/_torch_pq_bucketed.py"]
    report = dict(kind=args.kind, dataset=args.dataset, q_batch=args.q_batch,
        nprobe=args.nprobe, topk=args.topk, lut_dtype=args.lut_dtype, query_offset=args.query_offset,
        index=str(path), index_size_bytes=sum(size for size, _mtime in source_signature.values()),
        index_mtime_ns=path.stat().st_mtime_ns, index_signature=source_signature, index_metadata=index.metadata,
        queries=str(query_path), groundtruth=str(truth_path),
        torch_version=str(torch.__version__), platform=args.platform,
        device=runtime.name, execution_device=str(device), device_modules=runtime.modules,
        precision=runtime.precision, search_strategy="grouped", use_compile=False,
        baseline_query_count=reference_count, full_query_count=len(queries),
        validation_query_positions=reference_positions.tolist(),
        speedup_comparable=reference_count == len(queries),
        cpu_threads=torch.get_num_threads(), trial_strategy=args.scan,
        source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths},
        timing="median synchronized latency of one resident query batch; diagnostic only, not a full benchmark",
        results=[])
    output = args.output or ROOT / "diagnostics" / f"{args.platform.capitalize()}_Tensor_Tiles_{args.kind}_{args.dataset}_Q{args.q_batch}_P{args.nprobe}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = output.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2, default=str) + "\n")
        tmp.replace(output)

    configurations = [(None, 2048, 256, 128., 16, "flat")] + list(itertools.product(
        args.query_tiles, args.candidate_chunks, args.task_caps, args.memory_budgets,
        args.lut_chunks if args.kind == "ivfpq" else [16],
        args.pq_lookup if args.kind == "ivfpq" else ["flat"]))
    reference = None
    try:
        with torch.inference_mode():
            for number, (tile, chunk, cap, budget, lut_chunk, lookup) in enumerate(configurations):
                scan = "grouped" if number == 0 else args.scan
                row = dict(query_tile_size=tile, candidate_chunk_size=chunk,
                           max_tasks_per_batch=cap, memory_budget_mb=budget, scan=scan, lut_chunk_size=lut_chunk,
                           pq_lookup=lookup)
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
                    runtime.synchronize()
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
                    runtime.synchronize()
                    begin = time.perf_counter()
                    distances, found = search(index, active_queries, **options)
                    runtime.synchronize()
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
                # The search profiler only has synchronized timers for these
                # backends. Other devices still get synchronized batch QPS.
                profile_supported = device.type in {"cpu", "mps", "cuda", "npu"} and args.platform != "kunlun"
                if number == 0 and not args.skip_stage_profile and profile_supported:
                    stages = {}
                    search(index, active_queries, profile=stages, **options)
                    runtime.synchronize()
                    row["stages_seconds"] = stages
                elif number == 0:
                    row["stage_profile_skipped"] = "requested" if args.skip_stage_profile else "unsupported device timer"
                report["results"].append(row)
                save()
                print(f"scan={scan} tile={str(tile):>4} chunk={chunk} tasks={cap} budget={budget:g} lut={lut_chunk} lookup={lookup} "
                      f"latency={row['latency_ms']:.2f} ms QPS={row['qps']:.2f}", flush=True)
        baseline = report["results"][0]
        best = min(report["results"] if reference_count == len(queries) else report["results"][1:],
                   key=lambda r: r["latency_ms"])
        if index_signature(path) != source_signature:
            raise RuntimeError("Source index changed while tuning")
        speedup = baseline["latency_ms"] / best["latency_ms"] if reference_count == len(queries) else None
        report.update(best=best, speedup_over_default=speedup)
        save()
        comparison = f"{speedup:.3f}x" if speedup is not None else "not compared (sampled baseline)"
        print(f"Best: {best}; speedup={comparison}; saved {output}", flush=True)
    finally:
        index.release_device_residency()
        runtime.synchronize()


if __name__ == "__main__":
    main()
