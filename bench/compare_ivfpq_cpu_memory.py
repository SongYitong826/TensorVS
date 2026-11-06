#!/usr/bin/env python3
"""Compare Faiss and TensorVS IVF-PQ CPU memory with identical index data.

Each backend runs in a fresh process.  The benchmark samples the current
process RSS through Mach ``task_info`` instead of using the historical
``ru_maxrss`` high-water mark.  Faiss imports the exact TensorVS centroids,
codebooks, list codes, and IDs; it does not retrain the index.

Run from the repository root, for example::

    conda run -n tensorvs python -u bench/compare_ivfpq_cpu_memory.py \
        --dataset /Users/yitongsong/Dataset/glove1.2m
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import gc
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

try:
    from ._mps_results import index_stem, save_source_index, write_result
except ImportError:
    from _mps_results import index_stem, save_source_index, write_result


ROOT = Path(__file__).resolve().parents[1]
TOPK = 10
Q_BATCH = 128
THREADS = 4
PROFILES = {
    "glove1.2m": (50, 80, 100, 120, 140, 160, 180, 200, 220, 240,
                   260, 280, 300, 320, 340, 360, 380, 400),
    "hunyuan": (350, 380, 400, 450, 500, 550, 600, 650, 700, 750,
                800, 850, 900, 950, 1000),
}
FIELDS = (
    "dataset", "method", "execution_device", "q_batch", "K", "nprobes",
    "nlist", "dim", "M", "nbits", "threads", "warmup_iterations",
    "baseline_rss_mib", "peak_rss_mib", "search_delta_rss_mib",
    "search_seconds", "qps", "memory_samples", "sample_interval_ms",
    "compact_cpu_residency", "index_path", "index_size_bytes", "query_path",
    "measured_at_utc",
)


class _MachTaskBasicInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time_seconds", ctypes.c_int32),
        ("user_time_microseconds", ctypes.c_int32),
        ("system_time_seconds", ctypes.c_int32),
        ("system_time_microseconds", ctypes.c_int32),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    ]


_LIBSYSTEM = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_LIBSYSTEM.task_info.argtypes = (
    ctypes.c_uint,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint),
)
_LIBSYSTEM.task_info.restype = ctypes.c_int
_MACH_TASK_SELF = ctypes.c_uint.in_dll(_LIBSYSTEM, "mach_task_self_").value
_MACH_TASK_BASIC_INFO = 20


def current_rss_bytes() -> int:
    """Return current resident bytes for this process on macOS."""
    info = _MachTaskBasicInfo()
    count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_int))
    result = _LIBSYSTEM.task_info(
        _MACH_TASK_SELF,
        _MACH_TASK_BASIC_INFO,
        ctypes.byref(info),
        ctypes.byref(count),
    )
    if result != 0:
        raise OSError(f"task_info(MACH_TASK_BASIC_INFO) failed: {result}")
    return int(info.resident_size)


def release_free_cpu_pages() -> None:
    """Ask macOS malloc zones to return unused pages before RSS sampling."""
    try:
        malloc_default_zone = _LIBSYSTEM.malloc_default_zone
        malloc_default_zone.argtypes = ()
        malloc_default_zone.restype = ctypes.c_void_p
        pressure_relief = _LIBSYSTEM.malloc_zone_pressure_relief
        pressure_relief.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        pressure_relief.restype = ctypes.c_size_t
        pressure_relief(malloc_default_zone(), 0)
    except AttributeError:
        pass


class PeakRssSampler:
    """Poll current RSS while one steady-state search is running."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.peak = 0
        self.samples = 0
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def sample(self) -> None:
        self.peak = max(self.peak, current_rss_bytes())
        self.samples += 1

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval_seconds):
                self.sample()
        except BaseException as error:  # pragma: no cover - diagnostic path
            self.error = error
            self._stop.set()

    def __enter__(self) -> PeakRssSampler:
        self.sample()
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join()
        self.sample()
        if self.error is not None:
            raise RuntimeError("RSS sampling failed") from self.error


def read_vectors(path: Path, dtype: object, limit: int):
    import numpy as np

    raw = np.memmap(path, dtype=np.int32, mode="r")
    dim = int(raw[0])
    if dim <= 0 or len(raw) % (dim + 1):
        raise ValueError(f"Invalid vector file: {path}")
    rows = raw.reshape(-1, dim + 1)[:limit]
    values = rows[:, 1:].view(np.float32) if np.dtype(dtype).kind == "f" else rows[:, 1:]
    return np.ascontiguousarray(values, dtype=dtype)


def write_rows(path: Path, rows: list[dict[str, object]], *, complete=False) -> None:
    write_result(path, rows, complete=complete)


def import_faiss_cpu(export_dir: Path):
    import faiss
    import numpy as np

    arrays = {
        name: np.load(export_dir / f"{name}.npy", mmap_mode="r")
        for name in ("centroids", "data", "ids", "counts", "offsets")
    }
    centers = arrays["centroids"]
    original_dim = int(centers.shape[1])
    rotation = None
    if (export_dir / "rotation.npy").exists():
        rotation = np.ascontiguousarray(np.load(export_dir / "rotation.npy"))
        centers = np.load(export_dir / "pq_centroids.npy", mmap_mode="r")
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    nlist, working_dim = centers.shape
    books = np.load(export_dir / "codebooks.npy", mmap_mode="r")
    m, ksub, subdim = books.shape
    if ksub != 256 or working_dim != m * subdim:
        raise ValueError("Exported PQ dimensions are inconsistent")

    quantizer = faiss.IndexFlatL2(working_dim)
    quantizer.add(centers)
    index = faiss.IndexIVFPQ(
        quantizer, working_dim, nlist, int(m), 8, faiss.METRIC_L2
    )
    faiss.copy_array_to_vector(
        np.ascontiguousarray(books).reshape(-1), index.pq.centroids
    )
    index.by_residual = True
    index.is_trained = True
    ids = arrays["ids"]
    data = arrays["data"]
    for list_id, count_value in enumerate(arrays["counts"]):
        count = int(count_value)
        if count == 0:
            continue
        start = int(arrays["offsets"][list_id])
        list_ids = np.ascontiguousarray(ids[start:start + count])
        codes = np.ascontiguousarray(data[start:start + count]).view(np.uint8).reshape(-1)
        index.invlists.add_entries(
            list_id, count, faiss.swig_ptr(list_ids), faiss.swig_ptr(codes)
        )
    index.ntotal = len(ids)
    index.precompute_table()
    configuration = (int(nlist), original_dim, int(m), 8)
    del arrays, books, centers, ids, data
    gc.collect()
    return index, rotation, configuration


def validate_result(distances, ids, topk: int) -> None:
    import numpy as np

    if hasattr(distances, "detach"):
        distances = distances.detach().cpu().numpy()
        ids = ids.detach().cpu().numpy()
    if distances.shape != ids.shape or distances.shape[1] != topk:
        raise RuntimeError(f"Unexpected result shape: {distances.shape}, {ids.shape}")
    if not np.isfinite(distances).all() or (ids < 0).any():
        raise RuntimeError("Search returned invalid distances or IDs")


def worker(args: argparse.Namespace) -> None:
    import numpy as np

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    queries = read_vectors(args.queries, np.float32, args.q_batch)
    if len(queries) != args.q_batch:
        raise ValueError(f"Need {args.q_batch} queries, found {len(queries)}")

    if args.worker == "faiss":
        import faiss

        faiss.omp_set_num_threads(args.threads)
        index, rotation, configuration = import_faiss_cpu(args.export_dir)
        query_input = (
            np.ascontiguousarray(queries @ rotation.T, dtype=np.float32)
            if rotation is not None
            else queries
        )

        def search(probe: int):
            params = faiss.SearchParametersIVF()
            params.nprobe = probe
            return index.search(query_input, args.topk, params=params)

        method = "Faiss CPU"
    else:
        import torch
        from tensorvs import load_ivfpq_index, search_ivfpq_index

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        index = load_ivfpq_index(args.index, load_lists=True)
        index.reside_on_device("cpu", backend="torch", persist_layout=True)
        if args.compact:
            index.compact_cpu_residency(
                discard_build_metadata=True,
                persist_layout=True,
            )
        query_input = torch.from_numpy(queries)
        configuration = (
            int(index.centroids.shape[0]),
            int(index.centroids.shape[1]),
            int(index.metadata["m"]),
            int(index.metadata["nbits"]),
        )

        def search(probe: int):
            with torch.inference_mode():
                return search_ivfpq_index(
                    index,
                    query_input,
                    topk=args.topk,
                    nprobe=probe,
                    device="cpu",
                    backend="torch",
                    torch_query_batch_size=args.q_batch,
                    torch_memory_budget_mb=args.memory_budget_mb,
                    use_compile=False,
                    persist_layout=True,
                    torch_pq_scan="bucketed",
                    lut_dtype="fp32",
                )

        method = "TensorVS CPU"

    nlist, dim, m, nbits = configuration
    if max(args.nprobes) > nlist:
        raise ValueError(f"nprobe exceeds nlist={nlist}: {args.nprobes}")
    rows: list[dict[str, object]] = []
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    rows = [row for row in rows if row["method"] != method]

    for probe in args.nprobes:
        for _ in range(args.warmup):
            distances, ids = search(probe)
            validate_result(distances, ids, args.topk)
            del distances, ids
        gc.collect()
        release_free_cpu_pages()
        baseline = current_rss_bytes()
        with PeakRssSampler(args.sample_interval_ms / 1000.0) as sampler:
            begin = time.perf_counter()
            distances, ids = search(probe)
            elapsed = time.perf_counter() - begin
            validate_result(distances, ids, args.topk)
            sampler.sample()
        peak = max(baseline, sampler.peak)
        del distances, ids
        gc.collect()
        row: dict[str, object] = {
            "dataset": args.dataset.name,
            "method": method,
            "execution_device": "cpu",
            "q_batch": args.q_batch,
            "K": args.topk,
            "nprobes": probe,
            "nlist": nlist,
            "dim": dim,
            "M": m,
            "nbits": nbits,
            "threads": args.threads,
            "warmup_iterations": args.warmup,
            "baseline_rss_mib": baseline / (1 << 20),
            "peak_rss_mib": peak / (1 << 20),
            "search_delta_rss_mib": max(0, peak - baseline) / (1 << 20),
            "search_seconds": elapsed,
            "qps": args.q_batch / elapsed,
            "memory_samples": sampler.samples,
            "sample_interval_ms": args.sample_interval_ms,
            "compact_cpu_residency": args.worker == "tensorvs" and args.compact,
            "index_path": str(args.index.resolve()),
            "index_size_bytes": args.index.stat().st_size,
            "query_path": str(args.queries.resolve()),
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        write_rows(args.output, rows)
        print(
            f"{args.dataset.name} {method} nprobe={probe} "
            f"baseline={row['baseline_rss_mib']:.2f} MiB "
            f"peak={row['peak_rss_mib']:.2f} MiB "
            f"delta={row['search_delta_rss_mib']:.2f} MiB "
            f"qps={row['qps']:.2f} "
            f"samples={sampler.samples}",
            flush=True,
        )
    write_rows(args.output, rows, complete=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--nprobes", type=int, nargs="+")
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--q-batch", type=int, default=Q_BATCH)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--memory-budget-mb", type=float, default=128.0)
    parser.add_argument("--sample-interval-ms", type=float, default=1.0)
    parser.add_argument("--no-compact", dest="compact", action="store_false")
    parser.set_defaults(compact=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", choices=("faiss", "tensorvs"), help=argparse.SUPPRESS)
    parser.add_argument("--index", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--export-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dataset.name not in PROFILES:
        parser.error(f"No CPU-memory profile for {args.dataset.name!r}")
    args.nprobes = tuple(args.nprobes or PROFILES[args.dataset.name])
    args.queries = args.queries or args.dataset / f"{args.dataset.name}_query.fvecs"
    if (
        min(args.topk, args.q_batch, args.threads, args.warmup, *args.nprobes) <= 0
        or not math.isfinite(args.memory_budget_mb)
        or args.memory_budget_mb <= 0
        or not math.isfinite(args.sample_interval_ms)
        or args.sample_interval_ms <= 0
    ):
        parser.error("Sizes, memory budget, and sample interval must be positive")
    if len(set(args.nprobes)) != len(args.nprobes):
        parser.error("nprobe values must be unique")
    if not args.queries.is_file():
        parser.error(f"Missing query file: {args.queries}")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        worker(args)
        return
    indexes = list((args.dataset / "GPUIndex").glob("TenVS_IVFPQResidual_*.bin"))
    if len(indexes) != 1:
        raise ValueError(f"Expected one IVF-PQ index, found: {indexes}")
    args.index = indexes[0]
    outputs = {
        backend: args.output or args.output_dir / "Build" / (
            index_stem(args.index, args.dataset.name, "ivfpq", method) + "_cpu_memory.csv")
        for backend, method in (("faiss", "Faiss CPU"), ("tensorvs", "TensorVS CPU"))
    }
    for output in set(outputs.values()):
        for path in (output, output.with_suffix(".json")):
            if path.exists():
                if not args.overwrite:
                    raise FileExistsError(f"Output exists; pass --overwrite: {path}")
                path.unlink()
    save_source_index(args.index, args.dataset.name, "ivfpq", args.output_dir)
    with tempfile.TemporaryDirectory(prefix="tensorvs-ivfpq-cpu-memory-") as directory:
        export_dir = Path(directory)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench" / "export_comparison_index.py"),
                "ivfpq",
                str(args.index),
                str(export_dir),
            ],
            check=True,
        )
        forwarded = [
            "--dataset", str(args.dataset),
            "--queries", str(args.queries),
            "--nprobes", *(str(value) for value in args.nprobes),
            "--topk", str(args.topk),
            "--q-batch", str(args.q_batch),
            "--threads", str(args.threads),
            "--warmup", str(args.warmup),
            "--memory-budget-mb", str(args.memory_budget_mb),
            "--sample-interval-ms", str(args.sample_interval_ms),
            "--index", str(args.index),
            "--export-dir", str(export_dir),
        ]
        if not args.compact:
            forwarded.append("--no-compact")
        for backend in ("faiss", "tensorvs"):
            subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()), *forwarded,
                 "--output", str(outputs[backend]), "--worker", backend],
                check=True,
            )


if __name__ == "__main__":
    main()
