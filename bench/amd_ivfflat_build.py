"""Build and measure IVF-Flat indexes on an AMD GPU with Faiss ROCm or hipVS.

The timer starts after the .fvecs file has been read into host memory and
ends once the full index is ready on the GPU. Thus input transfer is included
for both methods. Conversion and serialization are timed separately.

Examples:
    /root/faiss-rocm-venv/bin/python bench/amd_ivfflat_build.py --method faiss-rocm --dataset glove1.2m
    bash bench/hipvs_python.sh -u bench/amd_ivfflat_build.py --method hipvs --dataset hunyuan
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEBIBYTE = 1 << 20
FIELDS = (
    "dataset", "method", "nlist", "n_vectors", "dim", "build_seconds",
    "save_seconds", "index_size_bytes", "index_size_mib",
    "peak_build_gpu_memory_mib",
)


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i4")
    if raw.size == 0:
        raise ValueError(f"Empty vector file: {path}")
    dim = int(raw[0])
    if dim <= 0 or raw.size % (dim + 1):
        raise ValueError(f"Invalid fvecs records: {path}")
    records = raw.reshape(-1, dim + 1)
    if not np.all(records[:, 0] == dim):
        raise ValueError(f"Inconsistent fvecs dimensions: {path}")
    return np.ascontiguousarray(records[:, 1:].view("<f4"))


class VramSampler:
    """Sample current-process AMD-SMI VRAM while the index is built."""

    def __init__(self, gpu_id: int, interval: float = 0.05):
        self.gpu_id = gpu_id
        self.interval = interval
        self.pid = os.getpid()
        self.peak_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def sample(self) -> None:
        output = subprocess.run(
            ["amd-smi", "process", "-g", str(self.gpu_id), "-p", str(self.pid), "--json"],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout
        for gpu in json.loads(output):
            for entry in gpu.get("process_list", []):
                info = entry.get("process_info")
                if not isinstance(info, dict) or int(info.get("pid", -1)) != self.pid:
                    continue
                memory = info["memory_usage"]["vram_mem"]
                if memory.get("unit") != "B":
                    raise ValueError(f"Unexpected AMD-SMI VRAM unit: {memory}")
                self.peak_bytes = max(self.peak_bytes, int(memory["value"]))
                self.samples += 1
                return

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except BaseException as error:
                self._error = error
                self._stop.set()

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()
        if self._error is not None:
            raise RuntimeError("AMD-SMI VRAM sampling failed") from self._error
        if not self.samples:
            raise RuntimeError(f"AMD-SMI did not report PID {self.pid} on GPU {self.gpu_id}")


def paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    subdir = "FaissROCm" if args.method == "faiss-rocm" else "hipVS"
    suffix = ".index" if args.method == "faiss-rocm" else ".bin"
    # Keep these measurement-only indexes separate from the existing hipVS
    # search benchmark's index + list-count metadata pair.
    stem = f"ivfflat_nlist{args.nlist}_build"
    index = args.data_root / args.dataset / "GPUIndex" / subdir / f"{stem}{suffix}"
    method_label = "FaissROCm" if args.method == "faiss-rocm" else "hipVS"
    csv_path = args.output_dir / f"AMD_{method_label}_IVFFLAT_{args.dataset}_nlist{args.nlist}_build.csv"
    return index, index.with_suffix(".json"), csv_path


def save_metadata(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_csv(path: Path, row: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({field: row[field] for field in FIELDS})
    temporary.replace(path)


def build_faiss(args: argparse.Namespace, base: np.ndarray, temporary: Path):
    import faiss

    if not hasattr(faiss, "GpuIndexIVFFlat"):
        raise RuntimeError("This Faiss build has no GPU support; build with FAISS_ENABLE_ROCM=ON")
    if args.gpu_id >= faiss.get_num_gpus():
        raise ValueError(f"Faiss cannot see GPU {args.gpu_id}")
    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexIVFFlatConfig()
    config.device = args.gpu_id
    config.indicesOptions = faiss.INDICES_64_BIT
    index = faiss.GpuIndexIVFFlat(resources, base.shape[1], args.nlist, faiss.METRIC_L2, config)
    # Faiss defaults to 10 k-means iterations and at most 256 samples per
    # centroid. Match hipVS's requested iteration count and train fraction.
    max_points_per_centroid = math.ceil(len(base) * args.train_fraction / args.nlist)
    index.cp.niter = args.kmeans_iters
    index.cp.max_points_per_centroid = max_points_per_centroid
    with VramSampler(args.gpu_id) as sampler:
        started = time.perf_counter()
        index.train(base)
        faiss.gpu_sync_all_devices()
        train_seconds = time.perf_counter() - started
        index.add(base)
        faiss.gpu_sync_all_devices()
        build_seconds = time.perf_counter() - started
    if int(index.ntotal) != len(base):
        raise RuntimeError(f"Faiss indexed {index.ntotal} of {len(base)} vectors")
    save_started = time.perf_counter()
    cpu_index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(cpu_index, str(temporary))
    save_seconds = time.perf_counter() - save_started
    source = Path("/root/faiss-rocm-src")
    source_commit = None
    if (source / ".git").is_dir():
        source_commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    return build_seconds, save_seconds, sampler, {
        "library_version": getattr(faiss, "__version__", "unknown"),
        "faiss_source_commit": source_commit,
        "train_seconds": train_seconds,
        "add_seconds": build_seconds - train_seconds,
        "faiss_indices_option": "INDICES_64_BIT",
        "kmeans_n_iters": args.kmeans_iters,
        "kmeans_trainset_fraction_target": args.train_fraction,
        "max_points_per_centroid": max_points_per_centroid,
        "max_train_points": max_points_per_centroid * args.nlist,
        "serialization": "faiss.index_gpu_to_cpu + faiss.write_index",
    }


def build_hipvs(args: argparse.Namespace, base: np.ndarray, temporary: Path):
    import cupy as cp
    from cuvs.neighbors import ivf_flat

    cp.cuda.Device(args.gpu_id).use()
    params = ivf_flat.IndexParams(
        n_lists=args.nlist, metric="sqeuclidean",
        kmeans_n_iters=args.kmeans_iters,
        kmeans_trainset_fraction=args.train_fraction,
    )
    with VramSampler(args.gpu_id) as sampler:
        started = time.perf_counter()
        gpu_base = cp.asarray(base)
        index = ivf_flat.build(params, gpu_base)
        cp.cuda.get_current_stream().synchronize()
        build_seconds = time.perf_counter() - started
    save_started = time.perf_counter()
    ivf_flat.save(str(temporary), index)
    cp.cuda.get_current_stream().synchronize()
    save_seconds = time.perf_counter() - save_started
    return build_seconds, save_seconds, sampler, {
        "library_version": "amd-hipvs 1.0.0",
        "kmeans_n_iters": args.kmeans_iters,
        "kmeans_trainset_fraction": args.train_fraction,
        "serialization": "cuvs.neighbors.ivf_flat.save(include_dataset=True)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("faiss-rocm", "hipvs"), required=True)
    parser.add_argument("--dataset", choices=("glove1.2m", "hunyuan"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/vector_data"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "result" / "Build")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.gpu_id < 0 or args.nlist <= 0 or args.kmeans_iters <= 0:
        parser.error("GPU ID must be nonnegative and nlist/iterations positive")
    if not 0 < args.train_fraction <= 1:
        parser.error("train-fraction must be in (0, 1]")
    index_path, meta_path, csv_path = paths(args)
    existing = [p for p in (index_path, meta_path, csv_path) if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Results already exist: {existing}; use --overwrite to rebuild")
    base_path = args.data_root / args.dataset / f"{args.dataset}_base.fvecs"
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    base = read_fvecs(base_path)
    if len(base) < args.nlist:
        raise ValueError("Base contains fewer vectors than nlist")
    print(f"{args.method} IVF-Flat {args.dataset}: {base.shape}, nlist={args.nlist}", flush=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Incomplete temporary index exists: {temporary}")
    build = build_faiss if args.method == "faiss-rocm" else build_hipvs
    try:
        build_seconds, save_seconds, sampler, extra = build(args, base, temporary)
        index_size_bytes = temporary.stat().st_size
        if index_size_bytes <= 0:
            raise RuntimeError("Serialized index is empty")
        temporary.replace(index_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    row = {
        "dataset": args.dataset,
        "method": args.method,
        "nlist": args.nlist,
        "n_vectors": len(base),
        "dim": base.shape[1],
        "build_seconds": build_seconds,
        "save_seconds": save_seconds,
        "index_size_bytes": index_size_bytes,
        "index_size_mib": index_size_bytes / MEBIBYTE,
        "peak_build_gpu_memory_mib": sampler.peak_bytes / MEBIBYTE,
    }
    if not all(math.isfinite(float(row[key])) for key in
               ("build_seconds", "save_seconds", "index_size_mib", "peak_build_gpu_memory_mib")):
        raise RuntimeError(f"Non-finite measurement: {row}")
    metadata = {
        **row, **extra,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "index_path": str(index_path.resolve()),
        "base_path": str(base_path.resolve()),
        "base_size_bytes": base_path.stat().st_size,
        "timing": "Host .fvecs read excluded; GPU input transfer, training and add included in build_seconds; save_seconds separate",
        "memory_metric": "sampled AMD-SMI current-process VRAM peak during build, MiB",
        "memory_samples": sampler.samples,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    save_metadata(meta_path, metadata)
    save_metadata(csv_path.with_suffix(".json"), metadata)
    save_csv(csv_path, row)
    print(f"Saved {index_path}: {index_size_bytes / MEBIBYTE:.1f} MiB; "
          f"build {build_seconds:.2f}s; VRAM peak {row['peak_build_gpu_memory_mib']:.1f} MiB", flush=True)
    print(f"Metrics: {csv_path} and {meta_path}", flush=True)


if __name__ == "__main__":
    main()
