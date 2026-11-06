"""Benchmark the portable Tensor path on Ascend, using the AMD/CUDA CSV schema.

Run each dataset/index in a separate process. Indexes and queries reside on the
NPU; synchronized wall-clock QPS excludes recall and the separate memory sweep.
The process HBM peak is sampled with npu-smi, like AMD-SMI/NVML on other devices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

import torch

try:
    import torch_npu  # Registers the npu device; do not use transfer_to_npu.
except ImportError:
    torch_npu = None

from tensorvs import (
    compute_recall_at_k, load_dataset, load_ivecs, load_ivfflat_index,
    load_ivfpq_index, search_ivfflat_index, search_ivfpq_index,
)

try:
    from ._benchmark_utils import _make_padded_batch, probe_visit_statistics
    from .amd_tensor_bench import FIELDS, NPROBES, atomic_csv, atomic_json, index_path_for, output_name
except ImportError:
    from _benchmark_utils import _make_padded_batch, probe_visit_statistics
    from amd_tensor_bench import FIELDS, NPROBES, atomic_csv, atomic_json, index_path_for, output_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEBIBYTE = 1 << 20
# npu-smi's process table: NPU, Chip, PID, Process name, Process memory(MB).
PROCESS_ROW = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(\d+)\s*\|$", re.M)


def process_hbm_mib(output: str, pid: int) -> tuple[int, int, int]:
    """Require an actual current-process reading, never substitute total HBM."""
    matches = [(int(npu), int(chip), int(memory))
               for npu, chip, process, _name, memory in PROCESS_ROW.findall(output)
               if int(process) == pid]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one npu-smi process row for PID {pid}; found {matches}")
    return matches[0]


class NpuProcessMemorySampler:
    """Driver-reported current-process HBM sampled only in the memory sweep."""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.pid = os.getpid()
        self.peak_bytes = 0
        self.samples = 0
        self.physical_device = None
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._sample_lock = threading.Lock()

    def sample(self) -> None:
        # The sweep and background thread both sample; serialize driver CLI
        # calls instead of issuing overlapping npu-smi requests.
        with self._sample_lock:
            self._sample()

    def _sample(self) -> None:
        output = subprocess.run(["npu-smi", "info"], check=True, capture_output=True,
                                text=True, timeout=15).stdout
        npu, chip, memory = process_hbm_mib(output, self.pid)
        if self.physical_device is not None and self.physical_device != (npu, chip):
            raise RuntimeError("npu-smi process moved to a different physical NPU")
        self.physical_device = (npu, chip)
        self.peak_bytes = max(self.peak_bytes, memory * MEBIBYTE)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except BaseException as error:
                self._error = error
                self._stop.set()

    def __enter__(self):
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()
        if self._error is not None:
            raise RuntimeError("npu-smi background HBM sampling failed") from self._error


def configure_npu(device_id: int) -> torch.device:
    if torch_npu is None or not torch.npu.is_available():
        raise RuntimeError("An Ascend PyTorch environment with torch_npu is required")
    if not 0 <= device_id < torch.npu.device_count():
        raise ValueError(f"NPU {device_id} is unavailable")
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    # Match the strict FP32 coarse/Flat math used by the other Tensor scripts.
    torch.npu.matmul.allow_hf32 = False
    return device


def benchmark(args: argparse.Namespace) -> None:
    device = configure_npu(args.npu_id)
    dataset_dir = args.data_root / args.dataset
    index_path = index_path_for(dataset_dir, args.kind)
    query_path = dataset_dir / f"{args.dataset}_query.fvecs"
    truth_path = dataset_dir / f"{args.dataset}_groundtruth.ivecs"
    load, search = ((load_ivfflat_index, search_ivfflat_index) if args.kind == "ivfflat"
                    else (load_ivfpq_index, search_ivfpq_index))
    print(f"Loading {index_path}", flush=True)
    index = load(index_path, load_lists=True)
    queries, truth = load_dataset(query_path), load_ivecs(truth_path)
    nlist = int(index.centroids.shape[0])
    if len(queries) != len(truth) or queries.shape[1] != index.centroids.shape[1]:
        raise ValueError("Index, query and ground-truth dimensions/counts differ")
    if nlist != 1024:
        raise ValueError("Expected the same nlist=1024 indexes as the other figure benchmarks")
    nprobes = tuple(args.nprobes or NPROBES[args.dataset])
    if max(nprobes) > nlist:
        raise ValueError("nprobe exceeds index nlist")
    nq = min(args.nq, len(queries))
    queries, truth = queries[:nq], truth[:nq]
    if nq == 0 or truth.shape[1] < args.topk:
        raise ValueError("Empty queries or insufficient ground-truth neighbors")
    csv_path = args.output_dir / output_name(args.dataset, args.kind, index, "tensor", platform="Ascend")
    json_path = csv_path.with_suffix(".json")
    source_files = [Path(__file__), PROJECT_ROOT / "bench/_benchmark_utils.py",
                    PROJECT_ROOT / "bench/amd_tensor_bench.py",
                    *(PROJECT_ROOT / "src/tensorvs").glob("*.py")]
    config = dict(dataset=args.dataset, kind=args.kind, nq=nq, topk=args.topk,
                  q_batches=args.q_batches, nprobes=nprobes, npu_id=args.npu_id,
                  warmup=args.warmup, repetitions=args.repetitions,
                  flat_scan=args.flat_scan, pq_scan=args.pq_scan, lut_dtype=args.lut_dtype,
                  candidate_chunk_size=args.candidate_chunk_size,
                  query_tile_size=args.query_tile_size, max_tasks_per_batch=args.max_tasks_per_batch,
                  lut_chunk_size=args.lut_chunk_size,
                  pq_lookup=args.pq_lookup,
                  min_query_tile_size=args.min_query_tile_size,
                  memory_budget_mb=args.memory_budget_mb,
                  cpu_threads=torch.get_num_threads(),
                  torch_version=str(torch.__version__), torch_npu_version=str(torch_npu.__version__),
                  device_name=torch.npu.get_device_name(device), allow_hf32=False,
                  index=str(index_path.resolve()), index_size_bytes=index_path.stat().st_size,
                  index_mtime_ns=index_path.stat().st_mtime_ns,
                  query=str(query_path.resolve()), query_size_bytes=query_path.stat().st_size,
                  query_mtime_ns=query_path.stat().st_mtime_ns,
                  truth=str(truth_path.resolve()), truth_size_bytes=truth_path.stat().st_size,
                  truth_mtime_ns=truth_path.stat().st_mtime_ns,
                  source_sha256={str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(source_files)})
    completed, details = {}, {}
    ordered_keys = [(batch, probe) for batch in args.q_batches for probe in nprobes]
    if csv_path.exists() or json_path.exists():
        if args.overwrite:
            pass
        elif not args.resume or not csv_path.exists() or not json_path.exists():
            raise FileExistsError(f"Results exist; use --resume or --overwrite: {csv_path}")
        else:
            previous = json.loads(json_path.read_text())
            if previous.get("config") != json.loads(json.dumps(config)):
                raise ValueError("Existing results use different settings, sources, or input files")
            with csv_path.open(newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != list(FIELDS):
                    raise ValueError("Existing CSV has different columns")
                for row in reader:
                    key = (int(row["q_batch"]), int(row["nprobes"]))
                    if key in completed or key not in ordered_keys or int(row["K"]) != args.topk:
                        raise ValueError(f"Duplicate/unexpected result row: {row}")
                    if not all(math.isfinite(float(row[f])) for f in FIELDS):
                        raise ValueError(f"Non-finite metric: {row}")
                    completed[key] = row
            details = {(int(row["q_batch"]), int(row["nprobes"])): row
                       for row in previous.get("results", [])
                       if (int(row["q_batch"]), int(row["nprobes"])) in completed}
            if details.keys() != completed.keys():
                raise ValueError("Existing CSV rows are missing JSON measurement details")
    if len(completed) == len(ordered_keys):
        print(f"Already complete: {csv_path}", flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(device=device, backend="torch", persist_layout=False, use_compile=False,
                  torch_candidate_chunk_size=args.candidate_chunk_size,
                  torch_query_tile_size=args.query_tile_size,
                  torch_max_tasks_per_batch=args.max_tasks_per_batch,
                  torch_min_query_tile_size=args.min_query_tile_size,
                  torch_memory_budget_mb=args.memory_budget_mb)
    if args.kind == "ivfflat":
        kwargs["torch_flat_scan"] = args.flat_scan
        payload_bytes = queries.shape[1] * 4
    else:
        kwargs.update(torch_pq_scan=args.pq_scan, lut_dtype=args.lut_dtype,
                      torch_lut_chunk_size=args.lut_chunk_size, torch_pq_lookup=args.pq_lookup)
        payload_bytes = int(index.metadata["m"]) * index.persisted_list_codes_interleaved_by_subspace.element_size()
    index.reside_on_device(device, backend="torch", persist_layout=False)
    torch.npu.synchronize(device)
    started = time.perf_counter()
    metadata = dict(status="running", config=config, backend="torch", device=str(device),
                    implementation="Standard PyTorch Tensor operators through torch_npu; no custom kernels",
                    index_metadata=index.metadata, expected_rows=len(ordered_keys),
                    timing="median of synchronized full-query sweeps with resident padded query batches; "
                           "excludes recall and separate process-HBM sampling sweep",
                    recall="mean overlap with ground-truth top-K divided by K",
                    memory_unit="MiB", memory_metric="sampled npu-smi current-process HBM peak; "
                        "includes resident index, query batches, allocator cache and NPU runtime; "
                        "separate search sweep, not an instantaneous hardware peak",
                    query_dependent_cpu_scheduling=(args.kind == "ivfflat" and args.flat_scan != "dense")
                        or (args.kind == "ivfpq" and args.pq_scan != "gpu"),
                    results=[])

    def save():
        metadata.update(results=[details[key] for key in ordered_keys if key in details],
                        completed_rows=len(completed))
        atomic_json(json_path, metadata)
        atomic_csv(csv_path, [completed[key] for key in ordered_keys if key in completed])

    save()
    try:
        with torch.inference_mode():
            visit_cache = {}
            for q_batch, probe in ordered_keys:
                if (q_batch, probe) in completed:
                    continue
                if probe not in visit_cache:
                    visit_cache[probe] = probe_visit_statistics(index, queries, probe, device, payload_bytes)
                    torch.npu.synchronize(device)
                batches = [_make_padded_batch(queries, start, q_batch, device)
                           for start in range(0, nq, q_batch)]
                options = dict(kwargs, torch_query_batch_size=q_batch)

                def sweep(retain=False, sampler=None):
                    elapsed, saved = 0.0, []
                    for batch, valid in batches:
                        torch.npu.synchronize(device)
                        begin = time.perf_counter()
                        distances, ids = search(index, batch, topk=args.topk, nprobe=probe, **options)
                        torch.npu.synchronize(device)
                        elapsed += time.perf_counter() - begin
                        if distances.device != device or ids.device != device:
                            raise RuntimeError("Search output left the selected NPU")
                        if sampler is not None:
                            sampler.sample()
                        if retain:
                            if not torch.isfinite(distances[:valid]).all().item():
                                raise RuntimeError("Non-finite search distances")
                            saved.append(ids[:valid].cpu())
                        del distances, ids
                    return elapsed, saved

                print(f"{args.dataset} {args.kind} QBatch={q_batch} K={args.topk} nprobe={probe}", flush=True)
                for _ in range(args.warmup):
                    sweep()
                samples = []
                for repetition in range(args.repetitions):
                    elapsed, saved = sweep(retain=repetition == args.repetitions - 1)
                    samples.append(elapsed)
                ids = torch.cat(saved)
                if ids.shape != (nq, args.topk) or (ids < 0).any().item():
                    raise RuntimeError("Invalid or missing result IDs")
                recall = float(compute_recall_at_k(ids, truth, args.topk))
                del ids, saved
                torch.npu.synchronize(device)
                torch.npu.empty_cache()
                torch.npu.reset_peak_memory_stats(device)
                with NpuProcessMemorySampler() as sampler:
                    sweep(sampler=sampler)
                row = dict(nprobes=probe, q_batch=q_batch, K=args.topk,
                           qps=nq / statistics.median(samples), recall=recall,
                           visited_vectors_per_query=visit_cache[probe]["visited_vectors_per_query"],
                           peak_gpu_memory=sampler.peak_bytes / MEBIBYTE)
                if not all(math.isfinite(float(value)) for value in row.values()):
                    raise RuntimeError(f"Non-finite metric: {row}")
                key = (q_batch, probe)
                completed[key] = row
                details[key] = dict(row, samples_seconds=samples, memory_samples=sampler.samples,
                    physical_npu=sampler.physical_device,
                    peak_torch_allocated_mib=torch.npu.max_memory_allocated(device) / MEBIBYTE,
                    peak_torch_reserved_mib=torch.npu.max_memory_reserved(device) / MEBIBYTE,
                    visit_statistics=visit_cache[probe])
                save()
                print(f"  QPS={row['qps']:.2f} recall={recall:.6f} HBM={row['peak_gpu_memory']:.1f} MiB", flush=True)
                del batches
        metadata.update(status="complete", wall_seconds=time.perf_counter() - started)
        save()
    except BaseException as error:
        metadata.update(status="failed", error=repr(error), wall_seconds=time.perf_counter() - started)
        save()
        raise
    finally:
        index.release_device_residency()
        torch.npu.synchronize(device)
        torch.npu.empty_cache()
    print(f"Saved {csv_path} and {json_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(NPROBES), required=True)
    parser.add_argument("--kind", choices=("ivfflat", "ivfpq"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/home/ma-user/work/data"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "result")
    parser.add_argument("--npu-id", type=int, default=0)
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--q-batches", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--nprobes", type=int, nargs="+", help="Default: same dataset profile as AMD/CUDA")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--flat-scan", choices=("auto", "grouped", "dense"), default="auto")
    parser.add_argument("--pq-scan", choices=("auto", "grouped", "gpu", "bucketed"), default="auto")
    parser.add_argument("--lut-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--lut-chunk-size", type=int, default=16,
                        help="PQ subspaces per eager lookup/reduction (execution tiling only)")
    parser.add_argument("--pq-lookup", choices=("flat", "gather"), default="flat",
                        help="Grouped PQ lookup operator organization")
    parser.add_argument("--candidate-chunk-size", type=int, default=2048)
    parser.add_argument("--query-tile-size", type=int,
                        help="Grouped list-task query tile (power of two); explicit tiles honor candidate-chunk-size")
    parser.add_argument("--max-tasks-per-batch", type=int, default=256,
                        help="Grouped scan task cap, still limited by memory budget")
    parser.add_argument("--min-query-tile-size", type=int, default=1,
                        help="Minimum padded query lanes in a grouped list task (power of two)")
    parser.add_argument("--memory-budget-mb", type=float, default=128)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    values = (*args.q_batches, *(args.nprobes or NPROBES[args.dataset]), args.nq,
              args.topk, args.repetitions, args.candidate_chunk_size)
    if min(values) <= 0 or args.warmup < 0 or not math.isfinite(args.memory_budget_mb) or args.memory_budget_mb <= 0:
        parser.error("Counts, sizes and memory budget must be positive; warmup must be nonnegative")
    if len(set(args.q_batches)) != len(args.q_batches) or (args.nprobes and len(set(args.nprobes)) != len(args.nprobes)):
        parser.error("Duplicate query batch size or nprobe")
    if args.query_tile_size is not None and (args.query_tile_size <= 0 or args.query_tile_size & (args.query_tile_size - 1)):
        parser.error("query-tile-size must be a positive power of two")
    if min(args.max_tasks_per_batch, args.lut_chunk_size) <= 0:
        parser.error("max-tasks-per-batch and lut-chunk-size must be positive")
    if args.min_query_tile_size <= 0 or args.min_query_tile_size & (args.min_query_tile_size - 1):
        parser.error("min-query-tile-size must be a positive power of two")
    benchmark(args)


if __name__ == "__main__":
    main()
