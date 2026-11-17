"""Run TensorVS IVF-Flat measurements needed by the CUDA-kernel figure.

Examples, run from the TensorVS repository root::

    conda run -n tensorvs python bench/run_ivfflat_figure.py --dataset sift10m
    conda run -n tensorvs python bench/run_ivfflat_figure.py --dataset hunyuan

The defaults cover K=10 and QBatch=32/128/512. Use ``--overwrite`` to
replace the selected experiment rows while preserving unrelated CSV rows.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import ivfflat_search_bench as benchmark
import ivfpq_search_bench as profile_source


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/storage/vector_data")
DEFAULT_QBATCHES = (32, 128, 512)
MIN_FREE_GPU_MIB = 8192


def progress(message: str) -> None:
    """Print one immediately visible, timestamped progress message."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def select_gpu(requested_gpu: int | None) -> int:
    """Select a usable GPU before index construction starts."""
    command = (
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    )
    try:
        output = subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout
        free_by_gpu = {
            int(index.strip()): int(free_mib.strip())
            for line in output.splitlines()
            if line.strip()
            for index, free_mib in [line.split(",", maxsplit=1)]
        }
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise RuntimeError(f"Unable to query free GPU memory with nvidia-smi: {error}") from error
    if not free_by_gpu:
        raise RuntimeError("nvidia-smi reported no GPUs")

    gpu_id = requested_gpu if requested_gpu is not None else max(
        free_by_gpu, key=free_by_gpu.get
    )
    if gpu_id not in free_by_gpu:
        raise ValueError(f"GPU {gpu_id} does not exist; available GPUs: {sorted(free_by_gpu)}")
    if free_by_gpu[gpu_id] < MIN_FREE_GPU_MIB:
        availability = ", ".join(
            f"GPU {index}: {free_mib} MiB" for index, free_mib in sorted(free_by_gpu.items())
        )
        raise RuntimeError(
            f"GPU {gpu_id} has only {free_by_gpu[gpu_id]} MiB free; "
            f"at least {MIN_FREE_GPU_MIB} MiB is required. Free memory: {availability}"
        )
    mode = "Requested" if requested_gpu is not None else "Auto-selected"
    progress(f"{mode} GPU {gpu_id} with {free_by_gpu[gpu_id]} MiB free")
    return gpu_id


@dataclass(frozen=True)
class Config:
    dataset: str
    dataset_dir: Path
    base_path: Path
    query_path: Path
    groundtruth_path: Path
    result_path: Path
    index_path: Path
    statistics_path: Path
    nlist: int
    topk: int
    q_batches: tuple[int, ...]
    nprobes: tuple[int, ...]
    nq: int
    warmup: int
    gpu_id: int


def resolve_input_file(dataset_dir: Path, preferred: str, pattern: str) -> Path:
    preferred_path = dataset_dir / preferred
    if preferred_path.is_file():
        return preferred_path
    candidates = sorted(dataset_dir.glob(pattern))
    if len(candidates) == 1:
        print(f"Using non-standard dataset filename: {candidates[0]}")
        return candidates[0]
    binary_name = {
        "*_base.fvecs": "base.bin",
        "*_query.fvecs": "query.bin",
        "*_groundtruth.ivecs": "groundtruth.bin",
    }.get(pattern)
    if not candidates and binary_name and (dataset_dir / binary_name).is_file():
        return dataset_dir / binary_name
    raise FileNotFoundError(
        f"Could not resolve {preferred_path}; {pattern!r} matched {len(candidates)} files"
    )


def make_config(args: argparse.Namespace) -> Config:
    dataset = args.dataset.casefold()
    dataset_dir = args.dataset_dir or DATA_ROOT / ("sift100M" if dataset == "sift100m" else dataset)
    if dataset not in profile_source.DATASET_PROFILES:
        raise ValueError(
            f"No dataset profile for {dataset!r} in bench/ivfpq_search_bench.py; "
            f"available profiles: {tuple(profile_source.DATASET_PROFILES)}"
        )
    dataset_profile = profile_source.DATASET_PROFILES[dataset]
    nlist = int(args.nlist or dataset_profile["nlist"])
    nprobes = tuple(args.nprobes or dataset_profile["nprobes"])
    q_batches = tuple(args.q_batches)
    if any(value <= 0 or value > nlist for value in nprobes):
        raise ValueError(f"Every nprobe must be in [1, {nlist}]: {nprobes}")
    if any(value <= 0 for value in q_batches) or len(set(q_batches)) != len(q_batches):
        raise ValueError(f"QBatch values must be positive and unique: {q_batches}")

    base_path = resolve_input_file(dataset_dir, f"{dataset}_base.fvecs", "*_base.fvecs")
    query_path = resolve_input_file(dataset_dir, f"{dataset}_query.fvecs", "*_query.fvecs")
    groundtruth_path = resolve_input_file(
        dataset_dir, f"{dataset}_groundtruth.ivecs", "*_groundtruth.ivecs"
    )
    gpu_index_dir = dataset_dir / "GPUIndex"
    return Config(
        dataset=dataset,
        dataset_dir=dataset_dir,
        base_path=base_path,
        query_path=query_path,
        groundtruth_path=groundtruth_path,
        result_path=PROJECT_ROOT / "result" / f"IVFFLAT_{dataset}_nlist{nlist}.csv",
        index_path=gpu_index_dir / f"TenVS_IVFFlat_nlist{nlist}.bin",
        statistics_path=PROJECT_ROOT / "result" / "Build" / f"CUDA_TensorVS_Triton_IVFFLAT_{dataset}_nlist{nlist}_build.csv",
        nlist=nlist,
        topk=args.topk,
        q_batches=q_batches,
        nprobes=nprobes,
        nq=args.nq,
        warmup=args.warmup,
        gpu_id=select_gpu(args.gpu_id),
    )


def prepare_result_file(config: Config, overwrite: bool) -> None:
    if not config.result_path.is_file():
        return
    with config.result_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        return
    k_column = "K" if "K" in fieldnames else "k"
    is_target = lambda row: (
        int(float(row["q_batch"])) in config.q_batches
        and int(float(row[k_column])) == config.topk
    )
    target_rows = [row for row in rows if is_target(row)]
    if not target_rows:
        return
    if not overwrite:
        raise FileExistsError(
            f"{config.result_path} contains {len(target_rows)} selected rows; "
            "rerun with --overwrite to replace them"
        )
    with config.result_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in rows if not is_target(row))
    print(f"Removed {len(target_rows)} selected rows from {config.result_path}")


def configure_benchmark(config: Config) -> None:
    benchmark.DIR_PATH = config.dataset_dir
    benchmark.DATASET_NAME = config.dataset
    benchmark.DATASET_PATH = config.base_path
    benchmark.QUERY_PATH = config.query_path
    benchmark.GROUNDTRUTH_PATH = config.groundtruth_path
    benchmark.RESULT_DIR = config.result_path.parent
    benchmark.NLIST = config.nlist
    benchmark.TOPKS = (config.topk,)
    benchmark.NPROBES = config.nprobes
    benchmark.NQ = config.nq
    benchmark.WARMUP_ITERATIONS = config.warmup
    benchmark.DEVICE = torch.device(f"cuda:{config.gpu_id}")
    benchmark.INDEX_PATH = config.index_path


def validate_outputs(config: Config) -> None:
    if not config.index_path.exists():
        raise FileNotFoundError(f"TensorVS index was not created: {config.index_path}")
    if not config.result_path.is_file():
        raise FileNotFoundError(f"TensorVS result was not created: {config.result_path}")
    with config.result_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    selected = [
        row for row in rows
        if int(float(row["q_batch"])) in config.q_batches
        and int(float(row["K"])) == config.topk
    ]
    expected = len(config.q_batches) * len(config.nprobes)
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} TensorVS rows, found {len(selected)}")
    for q_batch in config.q_batches:
        measured = {
            int(float(row["nprobes"])) for row in selected
            if int(float(row["q_batch"])) == q_batch
        }
        if measured != set(config.nprobes):
            raise RuntimeError(f"QBatch={q_batch} nprobe mismatch: {sorted(measured)}")
    for field in ("qps", "recall", "visited_vectors_per_query", "peak_gpu_memory"):
        if any(not row.get(field) for row in selected):
            raise RuntimeError(f"TensorVS result contains an empty {field} value")

    if not config.statistics_path.is_file():
        raise FileNotFoundError(f"Index statistics missing: {config.statistics_path}")
    with config.statistics_path.open(newline="", encoding="utf-8") as csv_file:
        statistics = list(csv.DictReader(csv_file))
    if not any(
        row.get("dataset") == config.dataset
        and row.get("method") == "TensorVS"
        and int(row.get("nlist", -1)) == config.nlist
        and row.get("build_seconds") and row.get("index_size_bytes")
        for row in statistics
    ):
        raise RuntimeError(f"TensorVS build statistics missing: {config.statistics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset directory/name, e.g. sift10m or hunyuan.")
    parser.add_argument("--dataset-dir", type=Path, help="Override /storage/vector_data/<dataset>.")
    parser.add_argument("--nlist", type=int, help="Override the dataset profile nlist.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--q-batches", type=int, nargs="+", default=list(DEFAULT_QBATCHES))
    parser.add_argument("--nprobes", type=int, nargs="+", help="Override the dataset nprobe profile.")
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    configured_gpu = os.environ.get("TENSORVS_BENCH_GPU")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=int(configured_gpu) if configured_gpu is not None else None,
        help="CUDA device id; default: auto-select the GPU with most free memory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    config = make_config(args)
    prepare_result_file(config, args.overwrite)
    configure_benchmark(config)
    progress(
        f"TensorVS {config.dataset} IVF-Flat: nlist={config.nlist}, K={config.topk}, "
        f"QBatch={list(config.q_batches)}, nprobe={list(config.nprobes)}, GPU={config.gpu_id}"
    )
    progress(
        f"Index action: {'load existing' if config.index_path.exists() else 'build new'}; "
        f"path={config.index_path}"
    )
    progress(f"Results will be written to {config.result_path}")
    total_start = time.perf_counter()
    for index, q_batch in enumerate(config.q_batches, start=1):
        batch_start = time.perf_counter()
        progress(
            f"[{index}/{len(config.q_batches)}] Starting QBatch={q_batch}; "
            f"{len(config.nprobes)} nprobe measurements"
        )
        benchmark.Q_BATCH = q_batch
        benchmark.main()
        torch.cuda.empty_cache()
        progress(
            f"[{index}/{len(config.q_batches)}] Finished QBatch={q_batch} "
            f"in {time.perf_counter() - batch_start:.1f}s"
        )
    progress("All searches finished; validating result rows and index statistics")
    validate_outputs(config)
    progress(f"Validated result: {config.result_path}")
    progress(f"Validated index: {config.index_path}")
    progress(f"Validated index statistics: {config.statistics_path}")
    progress(f"Experiment completed in {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
