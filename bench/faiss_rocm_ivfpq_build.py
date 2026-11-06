"""Build and measure GloVe1.2M IVF-PQ on the Faiss ROCm GPU backend.

GloVe is 200-dimensional. Faiss requires d % M == 0, so 24 zero-valued
coordinates are appended to every base vector before training and adding.
Queries for this saved index must receive the same padding.

Host .fvecs reading and zero padding are measured separately from Faiss build
time. The CSV has the same columns as amd_ivfflat_build.py; detailed settings
and phase timings are saved in JSON.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import platform
import subprocess
import time

import faiss
import numpy as np

from amd_ivfflat_build import MEBIBYTE, VramSampler, read_fvecs, save_csv, save_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "glove1.2m"
ORIGINAL_DIM = 200
M = 32
NBITS = 8
PADDED_DIM = 224


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    stem = f"ivfpq_nlist{args.nlist}_m{M}_nbits{NBITS}_pad{PADDED_DIM}_build"
    index_path = args.data_root / DATASET / "GPUIndex" / "FaissROCm" / f"{stem}.index"
    result_path = args.output_dir / f"AMD_FaissROCm_IVFPQ_{DATASET}_nlist{args.nlist}_m{M}_nbits{NBITS}_pad{PADDED_DIM}_build.csv"
    return index_path, index_path.with_suffix(".json"), result_path


def make_index(gpu_id: int, nlist: int):
    if not hasattr(faiss, "GpuIndexIVFPQ"):
        raise RuntimeError("This Faiss build does not include GPU IVF-PQ")
    if gpu_id >= faiss.get_num_gpus():
        raise ValueError(f"Faiss ROCm cannot see GPU {gpu_id}")
    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexIVFPQConfig()
    config.device = gpu_id
    config.indicesOptions = faiss.INDICES_64_BIT
    config.interleavedLayout = False
    config.useFloat16LookupTables = False
    index = faiss.GpuIndexIVFPQ(
        resources, PADDED_DIM, nlist, M, NBITS, faiss.METRIC_L2, config
    )
    return resources, index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(DATASET,), default=DATASET)
    parser.add_argument("--data-root", type=Path, default=Path("/root/vector_data"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "result" / "Build")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--kmeans-iters", type=int, default=20,
                        help="Coarse quantizer k-means iterations")
    parser.add_argument("--train-fraction", type=float, default=0.5,
                        help="Target fraction of base vectors used for coarse training")
    parser.add_argument("--pq-kmeans-iters", type=int, default=20)
    parser.add_argument("--pq-train-points-per-centroid", type=int, default=256)
    parser.add_argument("--add-batch-size", type=int, default=65536)
    parser.add_argument("--check-only", action="store_true",
                        help="Check input shape and GPU index construction without building")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.gpu_id < 0 or min(args.nlist, args.kmeans_iters,
                              args.pq_kmeans_iters, args.pq_train_points_per_centroid,
                              args.add_batch_size) <= 0:
        parser.error("GPU ID must be nonnegative and counts must be positive")
    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must be in (0, 1]")

    base_path = args.data_root / DATASET / f"{DATASET}_base.fvecs"
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    with base_path.open("rb") as stream:
        original_dim = int(np.fromfile(stream, dtype="<i4", count=1)[0])
    if original_dim != ORIGINAL_DIM or base_path.stat().st_size % (4 * (ORIGINAL_DIM + 1)):
        raise ValueError("Expected a complete 200-dimensional GloVe .fvecs file")
    expected_vectors = base_path.stat().st_size // (4 * (ORIGINAL_DIM + 1))
    if expected_vectors < args.nlist:
        raise ValueError("Base has fewer vectors than nlist")

    index_path, index_meta_path, csv_path = output_paths(args)
    result_meta_path = csv_path.with_suffix(".json")
    if not args.check_only:
        existing = [path for path in (index_path, index_meta_path, csv_path,
                                      result_meta_path) if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(f"Index or result already exists: {existing}; use --overwrite")
    resources, index = make_index(args.gpu_id, args.nlist)
    if args.check_only:
        print(f"CHECK OK: Faiss ROCm GPU {args.gpu_id}; {expected_vectors} GloVe vectors; "
              f"d={ORIGINAL_DIM}->{PADDED_DIM}; nlist={args.nlist}; M={M}; nbits={NBITS}",
              flush=True)
        return

    base = read_fvecs(base_path)
    if len(base) != expected_vectors or base.shape[1] != ORIGINAL_DIM:
        raise ValueError("GloVe .fvecs count or dimension is inconsistent")
    pad_started = time.perf_counter()
    padded = np.pad(base, ((0, 0), (0, PADDED_DIM - ORIGINAL_DIM)), mode="constant")
    padded = np.ascontiguousarray(padded, dtype=np.float32)
    pad_seconds = time.perf_counter() - pad_started
    del base
    if padded.shape != (expected_vectors, PADDED_DIM) or not np.isfinite(padded).all():
        raise ValueError("Padded GloVe array has an invalid shape or values")
    print(f"Faiss ROCm IVF-PQ {DATASET}: {expected_vectors} vectors, "
          f"d={ORIGINAL_DIM}->{PADDED_DIM}, nlist={args.nlist}, M={M}, nbits={NBITS}",
          flush=True)

    # Match the coarse training target used by the existing IVF-Flat build.
    coarse_max_points = math.ceil(expected_vectors * args.train_fraction / args.nlist)
    index.cp.niter = args.kmeans_iters
    index.cp.max_points_per_centroid = coarse_max_points
    index.pq.cp.niter = args.pq_kmeans_iters
    index.pq.cp.max_points_per_centroid = args.pq_train_points_per_centroid

    index_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Incomplete temporary index exists: {temporary}")
    try:
        with VramSampler(args.gpu_id) as sampler:
            started = time.perf_counter()
            index.train(padded)
            faiss.gpu_sync_all_devices()
            train_seconds = time.perf_counter() - started
            print(f"Trained in {train_seconds:.2f}s; adding vectors...", flush=True)
            for first in range(0, expected_vectors, args.add_batch_size):
                index.add(padded[first:first + args.add_batch_size])
            faiss.gpu_sync_all_devices()
            build_seconds = time.perf_counter() - started
        if int(index.ntotal) != expected_vectors:
            raise RuntimeError(f"Faiss indexed {index.ntotal} of {expected_vectors} vectors")

        save_started = time.perf_counter()
        cpu_index = faiss.index_gpu_to_cpu(index)
        if cpu_index.ntotal != expected_vectors or cpu_index.code_size != M:
            raise RuntimeError("Converted Faiss index has the wrong size or PQ code length")
        faiss.write_index(cpu_index, str(temporary))
        save_seconds = time.perf_counter() - save_started
        index_size_bytes = temporary.stat().st_size
        if index_size_bytes <= 0:
            raise RuntimeError("Serialized index is empty")
        temporary.replace(index_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    row = {
        "dataset": DATASET,
        "method": "faiss-rocm",
        "nlist": args.nlist,
        "n_vectors": expected_vectors,
        "dim": ORIGINAL_DIM,
        "build_seconds": build_seconds,
        "save_seconds": save_seconds,
        "index_size_bytes": index_size_bytes,
        "index_size_mib": index_size_bytes / MEBIBYTE,
        "peak_build_gpu_memory_mib": sampler.peak_bytes / MEBIBYTE,
    }
    if not all(math.isfinite(float(row[name])) for name in
               ("build_seconds", "save_seconds", "index_size_mib",
                "peak_build_gpu_memory_mib")):
        raise RuntimeError(f"Non-finite measurement: {row}")

    source = Path("/root/faiss-rocm-src")
    source_commit = None
    if (source / ".git").is_dir():
        source_commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    metadata = {
        **row,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "index_path": str(index_path.resolve()),
        "base_path": str(base_path.resolve()),
        "base_size_bytes": base_path.stat().st_size,
        "original_dim": ORIGINAL_DIM,
        "padded_dim": PADDED_DIM,
        "padding": "append 24 float32 zeros to each base vector; do the same for queries",
        "pad_seconds": pad_seconds,
        "total_build_with_pad_seconds": pad_seconds + build_seconds,
        "train_seconds": train_seconds,
        "add_seconds": build_seconds - train_seconds,
        "m": M,
        "nbits": NBITS,
        "pq_code_bytes_per_vector": M,
        "faiss_indices_option": "INDICES_64_BIT",
        "faiss_interleaved_layout": False,
        "faiss_float16_lookup_tables": False,
        "kmeans_n_iters": args.kmeans_iters,
        "kmeans_trainset_fraction_target": args.train_fraction,
        "max_points_per_centroid": coarse_max_points,
        "max_train_points": coarse_max_points * args.nlist,
        "pq_kmeans_n_iters": args.pq_kmeans_iters,
        "pq_train_points_per_centroid": args.pq_train_points_per_centroid,
        "add_batch_size": args.add_batch_size,
        "library_version": getattr(faiss, "__version__", "unknown"),
        "faiss_source_commit": source_commit,
        "serialization": "faiss.index_gpu_to_cpu + faiss.write_index",
        "timing": "Host .fvecs read and zero padding excluded from build_seconds; padding in pad_seconds; GPU input transfer, training and add included in build_seconds; save_seconds separate",
        "memory_metric": "sampled AMD-SMI current-process VRAM peak during build, MiB",
        "memory_samples": sampler.samples,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    save_metadata(index_meta_path, metadata)
    save_metadata(result_meta_path, metadata)
    save_csv(csv_path, row)
    print(f"Saved {index_path}: {row['index_size_mib']:.2f} MiB; "
          f"build {build_seconds:.2f}s; VRAM peak "
          f"{row['peak_build_gpu_memory_mib']:.2f} MiB", flush=True)
    print(f"Metrics: {csv_path} and {result_meta_path}", flush=True)


if __name__ == "__main__":
    main()
