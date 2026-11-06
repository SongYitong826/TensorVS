"""Build and measure hipVS IVF-PQ indexes on the AMD GPU.

CSV columns match amd_ivfflat_build.py and faiss_rocm_ivfpq_build.py.
The timer excludes host .fvecs reading and includes host-to-GPU transfer,
hipVS training and insertion. Serialization is timed separately.

GloVe uses the original 200-dimensional vectors and M=32, matching the
existing hipVS search experiment. hipVS handles its required dimension
transformation internally; the input is not zero-padded by this script.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import threading
import time

import cupy as cp
import numpy as np
from cuvs.neighbors import ivf_pq

from amd_ivfflat_build import MEBIBYTE, read_fvecs, save_csv, save_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_M = {"glove1.2m": 32, "hunyuan": 384}


class DrmVramSampler:
    """Sample this process's AMD VRAM through Linux DRM fdinfo.

    This reads the same per-process amdgpu accounting exposed by amd-smi but
    avoids launching a subprocess during long GPU kernels. Each device is
    counted once even if the process holds multiple DRM file descriptors.
    """

    def __init__(self, interval: float = 0.02):
        self.pid = os.getpid()
        self.interval = interval
        self.peak_kib = 0
        self.peak_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def sample(self) -> None:
        by_device: dict[str, int] = {}
        for path in Path(f"/proc/{self.pid}/fdinfo").iterdir():
            try:
                lines = path.read_text(encoding="ascii").splitlines()
            except OSError:
                continue  # A file descriptor may close while it is read.
            if "drm-driver:\tamdgpu" not in lines:
                continue
            device = None
            vram_kib = None
            for line in lines:
                if line.startswith("drm-pdev:"):
                    device = line.partition(":")[2].strip()
                elif line.startswith("drm-memory-vram:"):
                    value = line.partition(":")[2].strip().split()
                    if len(value) != 2 or value[1] != "KiB":
                        raise ValueError(f"Unexpected DRM VRAM unit: {line}")
                    vram_kib = int(value[0])
            if device and vram_kib is not None:
                by_device[device] = max(by_device.get(device, 0), vram_kib)
        if by_device:
            self.peak_kib = max(self.peak_kib, sum(by_device.values()))
            # On this server AMD-SMI's process JSON reports B as 1000 times
            # the numeric DRM KiB counter. Match that conversion so the CSV
            # stays comparable with earlier AMD-SMI-based Build results.
            self.peak_bytes = self.peak_kib * 1000
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

    def __exit__(self, exc_type, _exc, _traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            self.sample()
        except BaseException as error:
            self._error = error
        if exc_type is not None:
            return False  # Do not hide an exception from hipVS build().
        if self._error is not None:
            raise RuntimeError("DRM process VRAM sampling failed") from self._error
        if not self.samples:
            raise RuntimeError(f"No AMD GPU VRAM accounting found for PID {self.pid}")


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    stem = (f"ivfpq_nlist{args.nlist}_M{args.pq_dim}_nbits{args.pq_bits}_"
            f"train{args.pq_train_points}_build")
    index_path = (args.data_root / args.dataset / "GPUIndex" / "hipVS" /
                  f"{stem}.bin")
    csv_path = (args.output_dir /
                f"AMD_hipVS_IVFPQ_{args.dataset}_nlist{args.nlist}_M{args.pq_dim}_"
                f"nbits{args.pq_bits}_train{args.pq_train_points}_build.csv")
    return index_path, index_path.with_suffix(".json"), csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DEFAULT_M), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/vector_data"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "result" / "Build")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--pq-dim", type=int, help="Default: 32 for GloVe; 384 for Hunyuan")
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--pq-train-points", type=int, default=256)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--check-only", action="store_true",
                        help="Check input shape and hipVS parameters without building")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.pq_dim = args.pq_dim or DEFAULT_M[args.dataset]
    if args.gpu_id < 0 or min(args.nlist, args.pq_dim, args.pq_bits,
                              args.pq_train_points, args.kmeans_iters) <= 0:
        parser.error("GPU ID must be nonnegative and counts must be positive")
    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must be in (0, 1]")
    if args.pq_bits > 8:
        parser.error("--pq-bits must be at most 8")

    base_path = args.data_root / args.dataset / f"{args.dataset}_base.fvecs"
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    with base_path.open("rb") as stream:
        dim = int(np.fromfile(stream, dtype="<i4", count=1)[0])
    if dim <= 0 or base_path.stat().st_size % (4 * (dim + 1)):
        raise ValueError("Invalid .fvecs dimensions or file length")
    expected_vectors = base_path.stat().st_size // (4 * (dim + 1))
    if expected_vectors < args.nlist or args.pq_dim > dim:
        raise ValueError("Base is too small for nlist or pq-dim exceeds dimension")

    index_path, index_meta_path, csv_path = output_paths(args)
    result_meta_path = csv_path.with_suffix(".json")
    if not args.check_only:
        existing = [path for path in (index_path, index_meta_path,
                                      csv_path, result_meta_path) if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(f"Index or result already exists: {existing}; use --overwrite")
    cp.cuda.Device(args.gpu_id).use()
    params = ivf_pq.IndexParams(
        n_lists=args.nlist,
        metric="sqeuclidean",
        kmeans_n_iters=args.kmeans_iters,
        kmeans_trainset_fraction=args.train_fraction,
        pq_dim=args.pq_dim,
        pq_bits=args.pq_bits,
        max_train_points_per_pq_code=args.pq_train_points,
        add_data_on_build=True,
    )
    if args.check_only:
        print(f"CHECK OK: hipVS GPU {args.gpu_id}; {expected_vectors} "
              f"{args.dataset} vectors; d={dim}; nlist={args.nlist}; "
              f"M={args.pq_dim}; nbits={args.pq_bits}", flush=True)
        return

    base = read_fvecs(base_path)
    if len(base) != expected_vectors or base.shape[1] != dim:
        raise ValueError("Dataset count or dimension is inconsistent")
    print(f"hipVS IVF-PQ {args.dataset}: {base.shape}, nlist={args.nlist}, "
          f"M={args.pq_dim}, nbits={args.pq_bits}", flush=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Incomplete temporary index exists: {temporary}")
    try:
        with DrmVramSampler() as sampler:
            started = time.perf_counter()
            gpu_base = cp.asarray(base)
            index = ivf_pq.build(params, gpu_base)
            cp.cuda.get_current_stream().synchronize()
            build_seconds = time.perf_counter() - started

        save_started = time.perf_counter()
        ivf_pq.save(str(temporary), index)
        cp.cuda.get_current_stream().synchronize()
        save_seconds = time.perf_counter() - save_started
        index_size_bytes = temporary.stat().st_size
        if index_size_bytes <= 0:
            raise RuntimeError("Serialized hipVS IVF-PQ index is empty")
        temporary.replace(index_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    row = {
        "dataset": args.dataset,
        "method": "hipvs",
        "nlist": args.nlist,
        "n_vectors": expected_vectors,
        "dim": dim,
        "build_seconds": build_seconds,
        "save_seconds": save_seconds,
        "index_size_bytes": index_size_bytes,
        "index_size_mib": index_size_bytes / MEBIBYTE,
        "peak_build_gpu_memory_mib": sampler.peak_bytes / MEBIBYTE,
    }
    if not all(math.isfinite(float(row[key])) for key in
               ("build_seconds", "save_seconds", "index_size_mib",
                "peak_build_gpu_memory_mib")):
        raise RuntimeError(f"Non-finite measurement: {row}")
    metadata = {
        **row,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "index_path": str(index_path.resolve()),
        "base_path": str(base_path.resolve()),
        "base_size_bytes": base_path.stat().st_size,
        "pq_dim": args.pq_dim,
        "pq_bits": args.pq_bits,
        "pq_train_points_per_code": args.pq_train_points,
        "kmeans_n_iters": args.kmeans_iters,
        "kmeans_trainset_fraction": args.train_fraction,
        "add_data_on_build": True,
        "input_padding": "none; hipVS handles nondivisible dimensions internally",
        "library_version": "amd-hipvs 1.0.0",
        "serialization": "cuvs.neighbors.ivf_pq.save(include_dataset=True)",
        "timing": "Host .fvecs read excluded; GPU input transfer, training and add included in build_seconds; save_seconds separate",
        "memory_metric": "sampled Linux DRM fdinfo current-process drm-memory-vram peak during build; numeric KiB counter multiplied by 1000 to match AMD-SMI process JSON bytes used by prior Build results; reported in MiB",
        "memory_raw_peak_kib": sampler.peak_kib,
        "memory_samples": sampler.samples,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    save_metadata(index_meta_path, metadata)
    save_metadata(result_meta_path, metadata)
    save_csv(csv_path, row)
    print(f"Saved {index_path}: {row['index_size_mib']:.2f} MiB; "
          f"build {build_seconds:.2f}s; "
          f"VRAM peak {row['peak_build_gpu_memory_mib']:.2f} MiB", flush=True)
    print(f"Metrics: {csv_path} and {result_meta_path}", flush=True)


if __name__ == "__main__":
    main()
