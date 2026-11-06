"""Benchmark TensorVS IVF indexes on AMD ROCm via Tensor or Triton kernels.

Run one dataset and index type per process to isolate GPU memory measurements.
The CSV schema and result filenames follow the CUDA/MPS benchmark scripts.

Example:
    /root/tensorvs-venv/bin/python -u bench/amd_tensor_bench.py \
        --dataset glove1.2m --kind ivfflat
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
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

from tensorvs import (
    compute_recall_at_k,
    load_dataset,
    load_ivecs,
    load_ivfflat_index,
    load_ivfpq_index,
    search_ivfflat_index,
    search_ivfpq_index,
)

try:
    from ._benchmark_utils import _make_padded_batch, probe_visit_statistics
except ImportError:
    from _benchmark_utils import _make_padded_batch, probe_visit_statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("nprobes", "q_batch", "K", "qps", "recall",
          "visited_vectors_per_query", "peak_gpu_memory")
NPROBES = {
    "glove1.2m": (50, 80, 100, 120, 140, 160, 180, 200, 220, 240,
                   260, 280, 300, 320, 340, 360, 380, 400),
    "hunyuan": (350, 380, 400, 450, 500, 550, 600, 650, 700, 750,
                800, 850, 900, 950, 1000),
}
RESULT_TRAIN_LABEL = {"glove1.2m": 512, "hunyuan": 256}
MEBIBYTE = 1 << 20


class AmdProcessVramSampler:
    """Sample this process's AMD-SMI VRAM usage in a separate timing sweep."""

    def __init__(self, gpu_id: int, interval: float = 0.05) -> None:
        self.gpu_id = gpu_id
        self.interval = interval
        self.pid = os.getpid()
        self.peak_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def sample(self) -> None:
        command = ["amd-smi", "process", "-g", str(self.gpu_id),
                   "-p", str(self.pid), "--json"]
        output = subprocess.run(command, check=True, capture_output=True,
                                text=True, timeout=15).stdout
        for gpu in json.loads(output):
            for entry in gpu.get("process_list", []):
                info = entry.get("process_info")
                if isinstance(info, dict) and int(info.get("pid", -1)) == self.pid:
                    memory = info["memory_usage"]["vram_mem"]
                    if memory.get("unit") != "B":
                        raise ValueError(f"Unexpected AMD-SMI VRAM unit: {memory}")
                    self.peak_bytes = max(self.peak_bytes, int(memory["value"]))
                    self.samples += 1
                    return
        raise RuntimeError(f"AMD-SMI did not report process {self.pid} on GPU {self.gpu_id}")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except BaseException as error:
                self._error = error
                self._stop.set()

    def __enter__(self) -> AmdProcessVramSampler:
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()
        if self._error is not None:
            raise RuntimeError("AMD-SMI background VRAM sampling failed") from self._error


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def index_path_for(dataset_dir: Path, kind: str) -> Path:
    index_dir = dataset_dir / "GPUIndex"
    if kind == "ivfflat":
        path = index_dir / "TenVS_IVFFlat_nlist1024.bin"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    matches = sorted(index_dir.glob("TenVS_IVFPQResidual_nlist1024_m*_nbits*_train*.bin"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one IVF-PQ index in {index_dir}; found {matches}")
    return matches[0]


def output_name(dataset: str, kind: str, index: object, route: str,
                *, platform: str = "AMD") -> str:
    nlist = int(index.centroids.shape[0])
    prefix = "TensorVS_Tensor" if route == "tensor" else "TensorVS_Triton"
    if kind == "ivfflat":
        return f"{platform}_{prefix}_IVFFLAT_{dataset}_nlist{nlist}.csv"
    m = int(index.metadata["m"])
    nbits = int(index.metadata["nbits"])
    train_label = RESULT_TRAIN_LABEL[dataset]
    return f"{platform}_{prefix}_IVFPQ_{dataset}_nlist{nlist}_M{m}_nbits{nbits}_train{train_label}.csv"


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    values = (*args.q_batches, *(args.nprobes or NPROBES[args.dataset]),
              args.nq, args.topk, args.repetitions, args.candidate_chunk_size)
    if min(values) <= 0 or args.warmup < 0 or args.memory_budget_mb <= 0:
        parser.error("Batch sizes, probes, NQ, K, repetitions, chunk size and budget must be positive")
    if len(set(args.q_batches)) != len(args.q_batches):
        parser.error("Duplicate query batch size")
    if args.nprobes and len(set(args.nprobes)) != len(args.nprobes):
        parser.error("Duplicate nprobe")
    if args.resume and args.overwrite:
        parser.error("Choose either --resume or --overwrite")
    if not torch.cuda.is_available() or torch.version.hip is None:
        parser.error("An AMD ROCm GPU with PyTorch HIP support is required")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        parser.error(f"GPU {args.gpu_id} is unavailable")


def benchmark(args: argparse.Namespace) -> None:
    dataset_dir = args.data_root / args.dataset
    index_path = index_path_for(dataset_dir, args.kind)
    query_path = dataset_dir / f"{args.dataset}_query.fvecs"
    truth_path = dataset_dir / f"{args.dataset}_groundtruth.ivecs"
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    load = load_ivfflat_index if args.kind == "ivfflat" else load_ivfpq_index
    search = search_ivfflat_index if args.kind == "ivfflat" else search_ivfpq_index
    print(f"Loading {index_path}", flush=True)
    index = load(index_path, load_lists=True)
    queries = load_dataset(query_path)
    truth = load_ivecs(truth_path)
    if len(queries) != len(truth) or int(index.centroids.shape[1]) != int(queries.shape[1]):
        raise ValueError("Index, queries and ground truth have inconsistent dimensions/counts")
    if int(index.centroids.shape[0]) != 1024:
        raise ValueError("Expected an nlist=1024 index")
    nprobes = tuple(args.nprobes or NPROBES[args.dataset])
    if max(nprobes) > int(index.centroids.shape[0]):
        raise ValueError("nprobe exceeds index nlist")
    nq = min(args.nq, len(queries))
    queries, truth = queries[:nq], truth[:nq]
    if truth.shape[1] < args.topk:
        raise ValueError("Ground truth has fewer neighbors than requested K")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / output_name(args.dataset, args.kind, index, args.route)
    json_path = csv_path.with_suffix(".json")
    config = dict(dataset=args.dataset, kind=args.kind, nq=nq, topk=args.topk,
                  q_batches=args.q_batches, nprobes=nprobes, gpu_id=args.gpu_id,
                  warmup=args.warmup, repetitions=args.repetitions,
                  flat_scan=args.flat_scan, pq_scan=args.pq_scan,
                  lut_dtype=args.lut_dtype, candidate_chunk_size=args.candidate_chunk_size,
                  memory_budget_mb=args.memory_budget_mb,
                  index=str(index_path.resolve()), index_size_bytes=index_path.stat().st_size,
                  index_mtime_ns=index_path.stat().st_mtime_ns,
                  query=str(query_path.resolve()), query_size_bytes=query_path.stat().st_size,
                  truth=str(truth_path.resolve()), truth_size_bytes=truth_path.stat().st_size,
                  torch_version=str(torch.__version__), hip_version=str(torch.version.hip),
                  gpu_name=torch.cuda.get_device_name(device))
    completed: dict[tuple[int, int], dict] = {}
    details: dict[tuple[int, int], dict] = {}
    if csv_path.exists() or json_path.exists():
        if args.overwrite:
            pass
        elif not args.resume or not csv_path.exists() or not json_path.exists():
            raise FileExistsError(f"Results exist; use --resume or --overwrite: {csv_path}")
        else:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
            if previous.get("config") != json.loads(json.dumps(config)):
                raise ValueError("Existing result configuration differs from this run")
            with csv_path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != list(FIELDS):
                    raise ValueError("Existing CSV has a different column layout")
                for row in reader:
                    key = (int(row["q_batch"]), int(row["nprobes"]))
                    if key in completed or key[0] not in args.q_batches or key[1] not in nprobes:
                        raise ValueError(f"Duplicate/unexpected result row: {key}")
                    if int(row["K"]) != args.topk or not all(math.isfinite(float(row[f])) for f in FIELDS):
                        raise ValueError(f"Invalid result row: {row}")
                    completed[key] = row
            details = {}
            for row in previous.get("results", []):
                key = (int(row["q_batch"]), int(row["nprobes"]))
                if key in completed:
                    details[key] = row
    ordered_keys = [(batch, probe) for batch in args.q_batches for probe in nprobes]
    if len(completed) == len(ordered_keys):
        print(f"Already complete: {csv_path}", flush=True)
        return

    backend = "torch" if args.route == "tensor" else "triton"
    kwargs = dict(device=device, backend=backend, persist_layout=False)
    if args.route == "tensor":
        kwargs.update(use_compile=False,
                      torch_candidate_chunk_size=args.candidate_chunk_size,
                      torch_memory_budget_mb=args.memory_budget_mb)
    if args.kind == "ivfflat":
        if args.route == "tensor":
            kwargs["torch_flat_scan"] = args.flat_scan
        payload_bytes = int(queries.shape[1]) * 4
    else:
        kwargs["lut_dtype"] = args.lut_dtype
        if args.route == "tensor":
            kwargs["torch_pq_scan"] = args.pq_scan
        payload_bytes = (int(index.metadata["m"]) *
                         index.persisted_list_codes_interleaved_by_subspace.element_size())
    index.reside_on_device(device, backend=backend, persist_layout=False)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    metadata = dict(status="running", config=config, backend=backend,
                    implementation=("PyTorch tensor operations on AMD ROCm"
                                    if args.route == "tensor" else
                                    "TensorVS Triton kernels on AMD ROCm"),
                    index_metadata=index.metadata, expected_rows=len(ordered_keys),
                    memory_unit="MiB", memory_metric="sampled AMD-SMI current-process VRAM peak; "
                    "includes resident index, query batches and GPU runtime; sampled in a separate "
                    "search sweep, not an instantaneous hardware peak",
                    timing="median of synchronized full-query sweeps with resident padded query "
                    "batches; excludes recall and separate GPU memory sweep",
                    recall="mean overlap with ground-truth top-K divided by K",
                    results=[])

    def save() -> None:
        metadata["results"] = [details[key] for key in ordered_keys if key in details]
        metadata["completed_rows"] = len(completed)
        atomic_json(json_path, metadata)
        # Write the CSV last: an interrupted save can leave extra JSON detail,
        # which resume safely discards until that CSV row is measured again.
        atomic_csv(csv_path, [completed[key] for key in ordered_keys if key in completed])

    save()
    try:
        with torch.inference_mode():
            visit_cache = {}
            for q_batch, probe in ordered_keys:
                if (q_batch, probe) in completed:
                    continue
                if probe not in visit_cache:
                    visit_cache[probe] = probe_visit_statistics(
                        index, queries, probe, device, payload_bytes)
                batches = [_make_padded_batch(queries, start, q_batch, device)
                           for start in range(0, nq, q_batch)]
                options = dict(kwargs)
                if args.route == "tensor":
                    options["torch_query_batch_size"] = q_batch

                def sweep(retain: bool = False, sampler: AmdProcessVramSampler | None = None):
                    elapsed, saved = 0.0, []
                    for batch, valid in batches:
                        torch.cuda.synchronize(device)
                        begin = time.perf_counter()
                        distances, ids = search(index, batch, topk=args.topk,
                                                nprobe=probe, **options)
                        torch.cuda.synchronize(device)
                        elapsed += time.perf_counter() - begin
                        if distances.device.type != "cuda" or ids.device.type != "cuda":
                            raise RuntimeError("Search output left the GPU")
                        if sampler is not None:
                            sampler.sample()
                        if retain:
                            if not torch.isfinite(distances[:valid]).all().item():
                                raise RuntimeError("Non-finite search distance")
                            saved.append(ids[:valid].cpu())
                        del distances, ids
                    return elapsed, saved

                print(f"{args.dataset} {args.kind} QBatch={q_batch} K={args.topk} "
                      f"nprobe={probe}", flush=True)
                for _ in range(args.warmup):
                    sweep()
                samples = []
                for repetition in range(args.repetitions):
                    elapsed, saved = sweep(retain=repetition == args.repetitions - 1)
                    samples.append(elapsed)
                ids = torch.cat(saved)
                if ids.shape != (nq, args.topk) or (ids < 0).any().item():
                    raise RuntimeError("Invalid or missing search result IDs")
                recall = float(compute_recall_at_k(ids, truth, args.topk))
                del ids, saved
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                with AmdProcessVramSampler(args.gpu_id) as sampler:
                    sweep(sampler=sampler)
                peak_allocated = torch.cuda.max_memory_allocated(device) / MEBIBYTE
                peak_reserved = torch.cuda.max_memory_reserved(device) / MEBIBYTE
                row = dict(nprobes=probe, q_batch=q_batch, K=args.topk,
                           qps=nq / statistics.median(samples), recall=recall,
                           visited_vectors_per_query=visit_cache[probe]["visited_vectors_per_query"],
                           peak_gpu_memory=sampler.peak_bytes / MEBIBYTE)
                if not all(math.isfinite(float(value)) for value in row.values()):
                    raise RuntimeError(f"Non-finite benchmark metric: {row}")
                key = (q_batch, probe)
                completed[key] = row
                details[key] = dict(row, samples_seconds=samples,
                                    memory_samples=sampler.samples,
                                    peak_torch_allocated_mib=peak_allocated,
                                    peak_torch_reserved_mib=peak_reserved,
                                    visit_statistics=visit_cache[probe])
                save()
                print(f"  QPS={row['qps']:.2f} recall={recall:.6f} "
                      f"VRAM={row['peak_gpu_memory']:.1f} MiB", flush=True)
                del batches
        metadata.update(status="complete", wall_seconds=time.perf_counter() - started)
        save()
    except BaseException as error:
        metadata.update(status="failed", error=repr(error),
                        wall_seconds=time.perf_counter() - started)
        save()
        raise
    finally:
        index.release_device_residency()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    print(f"Saved {csv_path} and {json_path}", flush=True)


def main(default_route: str = "tensor") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("tensor", "triton"), default=default_route)
    parser.add_argument("--dataset", choices=tuple(NPROBES), required=True)
    parser.add_argument("--kind", choices=("ivfflat", "ivfpq"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/vector_data"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "result")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--q-batches", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--nprobes", type=int, nargs="+", help="Default: dataset CUDA figure profile")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--flat-scan", choices=("auto", "grouped", "dense"), default="auto")
    parser.add_argument("--pq-scan", choices=("auto", "grouped", "gpu", "bucketed"), default="auto")
    parser.add_argument("--lut-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--candidate-chunk-size", type=int, default=2048)
    parser.add_argument("--memory-budget-mb", type=float, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    validate_args(args, parser)
    benchmark(args)


if __name__ == "__main__":
    main()
