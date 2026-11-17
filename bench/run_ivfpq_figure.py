"""Run TensorVS IVF-PQ measurements needed by the CUDA-kernel figure.

Examples, from the TensorVS repository root::

    conda run -n tensorvs python -u bench/run_ivfpq_figure.py --dataset sift10m
    conda run -n tensorvs python -u bench/run_ivfpq_figure.py --dataset hunyuan

Defaults: K=10, QBatch=32/128/512, and the dataset profile's IVF-PQ and
nprobe settings.  ``--overwrite`` replaces only those selected result rows.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import ivfpq_search_bench as benchmark
from run_ivfflat_figure import progress, resolve_input_file, select_gpu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/storage/vector_data")
DEFAULT_QBATCHES = (32, 128, 512)


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
    pq_dim: int
    pq_bits: int
    topk: int
    q_batches: tuple[int, ...]
    nprobes: tuple[int, ...]
    nq: int
    warmup: int
    gpu_id: int


def make_config(args: argparse.Namespace) -> Config:
    dataset = args.dataset.casefold()
    dataset_dir = args.dataset_dir or DATA_ROOT / ("sift100M" if dataset == "sift100m" else dataset)
    if dataset not in benchmark.DATASET_PROFILES:
        raise ValueError(
            f"No IVF-PQ profile for {dataset!r}; available profiles: "
            f"{tuple(benchmark.DATASET_PROFILES)}"
        )
    profile = benchmark.DATASET_PROFILES[dataset]
    nlist = int(args.nlist or profile["nlist"])
    pq_dim = int(args.pq_dim or profile["pq_dim"])
    pq_bits = int(args.pq_bits)
    nprobes = tuple(args.nprobes or profile["nprobes"])
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
    train_label = int(benchmark.RESULT_TRAIN_LABELS.get(dataset, 256))
    result_name = (
        f"IVFPQ_{dataset}_nlist{nlist}_M{pq_dim}_nbits{pq_bits}_"
        f"train{train_label}.csv"
    )
    gpu_index_dir = dataset_dir / "GPUIndex"
    return Config(
        dataset=dataset,
        dataset_dir=dataset_dir,
        base_path=base_path,
        query_path=query_path,
        groundtruth_path=groundtruth_path,
        result_path=PROJECT_ROOT / "result" / result_name,
        index_path=gpu_index_dir / (
            f"TenVS_IVFPQResidual_nlist{nlist}_m{pq_dim}_nbits{pq_bits}_train256.bin"
        ),
        statistics_path=PROJECT_ROOT / "result" / "Build" / f"CUDA_TensorVS_Triton_{result_name[:-4]}_build.csv",
        nlist=nlist,
        pq_dim=pq_dim,
        pq_bits=pq_bits,
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
    with config.result_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        return
    k_column = "K" if "K" in fieldnames else "k"
    is_target = lambda row: (
        int(float(row["q_batch"])) in config.q_batches
        and int(float(row[k_column])) == config.topk
        and int(float(row["nprobes"])) in config.nprobes
    )
    selected = [row for row in rows if is_target(row)]
    if not selected:
        return
    if not overwrite:
        raise FileExistsError(
            f"{config.result_path} contains {len(selected)} selected rows; "
            "rerun with --overwrite to replace them"
        )
    with config.result_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in rows if not is_target(row))
    progress(f"Removed {len(selected)} selected rows from {config.result_path}")


def configure_benchmark(config: Config) -> None:
    benchmark.DIR_PATH = config.dataset_dir
    benchmark.DATASET_NAME = config.dataset
    benchmark.DATASET_PATH = config.base_path
    benchmark.QUERY_PATH = config.query_path
    benchmark.GROUNDTRUTH_PATH = config.groundtruth_path
    benchmark.RESULT_DIR = config.result_path.parent
    benchmark.DATASET_PROFILE = benchmark.DATASET_PROFILES[config.dataset]
    benchmark.NLIST = config.nlist
    benchmark.M = config.pq_dim
    benchmark.NBITS = config.pq_bits
    benchmark.TOPK = config.topk
    benchmark.NPROBES = config.nprobes
    benchmark.NQ = config.nq
    benchmark.WARMUP_ITERATIONS = config.warmup
    benchmark.DEVICE = torch.device(f"cuda:{config.gpu_id}")
    benchmark.INDEX_PATH = config.index_path
    benchmark.LEGACY_RESULT_FILENAMES[config.dataset] = config.result_path.name
    benchmark.UPDATE_PEAK_MEMORY_ONLY = False
    os.environ["TENSORVS_BENCH_REPLACE_EXISTING"] = "0"


def validate_outputs(config: Config) -> None:
    if not config.index_path.is_file():
        raise FileNotFoundError(f"TensorVS IVF-PQ index was not created: {config.index_path}")
    if not config.result_path.is_file():
        raise FileNotFoundError(f"TensorVS IVF-PQ result was not created: {config.result_path}")
    with config.result_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    selected = [
        row for row in rows
        if int(float(row["q_batch"])) in config.q_batches
        and int(float(row.get("K", row.get("k", "-1")))) == config.topk
        and int(float(row["nprobes"])) in config.nprobes
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
    with config.statistics_path.open(newline="", encoding="utf-8") as source:
        statistics = list(csv.DictReader(source))
    if not any(
        row.get("dataset") == config.dataset
        and row.get("method", "").casefold() == "tensorvs"
        and int(row.get("nlist", -1)) == config.nlist
        and row.get("build_seconds") and row.get("index_size_bytes")
        for row in statistics
    ):
        raise RuntimeError(f"TensorVS IVF-PQ statistics missing: {config.statistics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. sift10m or hunyuan.")
    parser.add_argument("--dataset-dir", type=Path, help="Override /storage/vector_data/<dataset>.")
    parser.add_argument("--nlist", type=int, help="Override the dataset profile nlist.")
    parser.add_argument("--pq-dim", type=int, help="Override the dataset profile M/PQ dimension.")
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--q-batches", type=int, nargs="+", default=list(DEFAULT_QBATCHES))
    parser.add_argument("--nprobes", type=int, nargs="+", help="Override the dataset nprobe profile.")
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    configured_gpu = os.environ.get("TENSORVS_BENCH_GPU")
    parser.add_argument(
        "--gpu-id", type=int,
        default=int(configured_gpu) if configured_gpu is not None else None,
        help="CUDA device id; default: GPU with most free memory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--update-peak-memory-only",
        action="store_true",
        help="Rerun only QBatch=128, K=10 and update only peak_gpu_memory.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    if args.update_peak_memory_only:
        args.q_batches = [128]
        args.topk = 10
    config = make_config(args)
    if args.update_peak_memory_only:
        if not config.result_path.is_file():
            raise FileNotFoundError(
                f"Peak-memory update requires an existing result: {config.result_path}"
            )
    else:
        prepare_result_file(config, args.overwrite)
    configure_benchmark(config)
    benchmark.UPDATE_PEAK_MEMORY_ONLY = args.update_peak_memory_only
    progress(
        f"TensorVS {config.dataset} IVF-PQ: nlist={config.nlist}, M={config.pq_dim}, "
        f"nbits={config.pq_bits}, K={config.topk}, QBatch={list(config.q_batches)}, "
        f"nprobe={list(config.nprobes)}, GPU={config.gpu_id}"
    )
    if args.update_peak_memory_only:
        progress("Update mode: only peak_gpu_memory for QBatch=128, K=10 will change")
    progress(
        f"Index action: {'load existing' if config.index_path.exists() else 'build new'}; "
        f"path={config.index_path}"
    )
    progress(f"Results will be written to {config.result_path}")
    total_start = time.perf_counter()
    for position, q_batch in enumerate(config.q_batches, start=1):
        batch_start = time.perf_counter()
        progress(
            f"[{position}/{len(config.q_batches)}] Starting QBatch={q_batch}; "
            f"{len(config.nprobes)} nprobe measurements"
        )
        benchmark.Q_BATCH = q_batch
        benchmark.main()
        torch.cuda.empty_cache()
        progress(
            f"[{position}/{len(config.q_batches)}] Finished QBatch={q_batch} "
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
