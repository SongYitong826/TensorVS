"""Shared helpers for the configurable IVF benchmark scripts."""

from __future__ import annotations

import csv
import ctypes
import ctypes.util
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch



def _load_headered_binary(path: Path, dtype: str) -> torch.Tensor:
    """Read a uint32 (rows, columns) header followed by a dense matrix.

    SIFT100M base/query payloads are uint8; ground-truth IDs are uint32.
    Map the source so conversion does not allocate another full input copy.
    """
    import numpy as np

    header = np.fromfile(path, dtype="<u4", count=2)
    if header.size != 2:
        raise ValueError(f"Binary matrix header is truncated: {path}")
    rows, columns = map(int, header)
    if rows <= 0 or columns <= 0:
        raise ValueError(f"Binary matrix dimensions must be positive: {path}")
    expected = 8 + rows * columns * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"Binary matrix payload size does not match its header: {path}")
    matrix = np.memmap(path, dtype=dtype, mode="c", offset=8, shape=(rows, columns))
    return torch.from_numpy(matrix)


def load_benchmark_vectors(path: str | Path) -> torch.Tensor:
    """Load regular TensorVS vectors or headered uint8 .bin benchmark data."""
    from tensorvs import load_dataset

    path = Path(path)
    if path.suffix == ".bin":
        return _load_headered_binary(path, "u1").to(dtype=torch.float32)
    return load_dataset(path)


def load_benchmark_groundtruth(path: str | Path) -> torch.Tensor:
    """Load .ivecs or headered uint32 .bin neighbor IDs."""
    from tensorvs import load_ivecs

    path = Path(path)
    if path.suffix == ".bin":
        return _load_headered_binary(path, "<u4").to(dtype=torch.int64)
    return load_ivecs(path)


MEBIBYTE = 1024 * 1024
_NVML_SUCCESS = 0
_NVML_ERROR_ALREADY_INITIALIZED = 5
_NVML_ERROR_INSUFFICIENT_SIZE = 7
_NVML_VALUE_NOT_AVAILABLE = (1 << 64) - 1
_NVML: ctypes.CDLL | None = None


class _NvmlProcessInfo(ctypes.Structure):
    _fields_ = (
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    )


def _nvml_library() -> ctypes.CDLL:
    global _NVML
    if _NVML is not None:
        return _NVML
    library_name = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
    library = ctypes.CDLL(library_name)
    library.nvmlInit_v2.argtypes = []
    library.nvmlInit_v2.restype = ctypes.c_int
    library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
    library.nvmlDeviceGetHandleByUUID.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.nvmlDeviceGetHandleByUUID.restype = ctypes.c_int
    library.nvmlDeviceGetComputeRunningProcesses_v3.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(_NvmlProcessInfo),
    ]
    library.nvmlDeviceGetComputeRunningProcesses_v3.restype = ctypes.c_int
    status = library.nvmlInit_v2()
    if status not in (_NVML_SUCCESS, _NVML_ERROR_ALREADY_INITIALIZED):
        raise RuntimeError(f"nvmlInit_v2 failed with status {status}")
    _NVML = library
    return library


def _nvml_device_handle(cuda_device_id: int) -> ctypes.c_void_p:
    library = _nvml_library()
    handle = ctypes.c_void_p()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    identifier = str(cuda_device_id)
    if visible_devices:
        identifiers = [value.strip() for value in visible_devices.split(",")]
        if cuda_device_id >= len(identifiers):
            raise RuntimeError(
                f"CUDA device {cuda_device_id} is absent from CUDA_VISIBLE_DEVICES"
            )
        identifier = identifiers[cuda_device_id]
    if identifier.startswith(("GPU-", "MIG-")):
        status = library.nvmlDeviceGetHandleByUUID(
            identifier.encode("ascii"), ctypes.byref(handle)
        )
    elif identifier.isdigit():
        status = library.nvmlDeviceGetHandleByIndex_v2(
            int(identifier), ctypes.byref(handle)
        )
    else:
        raise RuntimeError(
            "Process GPU-memory tracking requires a numeric or GPU-UUID "
            f"CUDA_VISIBLE_DEVICES entry, got {identifier!r}"
        )
    if status != _NVML_SUCCESS:
        raise RuntimeError(f"NVML device lookup failed with status {status}")
    return handle


def process_gpu_memory_bytes(device: torch.device, *, pid: int | None = None) -> int:
    """Return one process's resident GPU memory using NVML (default: self)."""
    if device.type != "cuda":
        return 0
    device_id = device.index if device.index is not None else torch.cuda.current_device()
    library = _nvml_library()
    handle = _nvml_device_handle(int(device_id))
    count = ctypes.c_uint(0)
    status = library.nvmlDeviceGetComputeRunningProcesses_v3(
        handle, ctypes.byref(count), None
    )
    if status == _NVML_SUCCESS:
        return 0
    if status != _NVML_ERROR_INSUFFICIENT_SIZE:
        raise RuntimeError(
            f"nvmlDeviceGetComputeRunningProcesses_v3 failed with status {status}"
        )
    count.value = count.value * 2 + 5
    processes = (_NvmlProcessInfo * count.value)()
    status = library.nvmlDeviceGetComputeRunningProcesses_v3(
        handle, ctypes.byref(count), processes
    )
    if status != _NVML_SUCCESS:
        raise RuntimeError(
            f"nvmlDeviceGetComputeRunningProcesses_v3 failed with status {status}"
        )
    process_id = os.getpid() if pid is None else int(pid)
    memory_values = [
        int(processes[index].used_gpu_memory)
        for index in range(count.value)
        if int(processes[index].pid) == process_id
        and int(processes[index].used_gpu_memory) != _NVML_VALUE_NOT_AVAILABLE
    ]
    return sum(memory_values)


class ProcessGpuMemorySampler:
    """Continuously sample this process's NVML memory on one CUDA device."""

    def __init__(self, device: torch.device, interval_seconds: float = 0.001) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def _sample(self) -> None:
        try:
            self.peak_bytes = max(self.peak_bytes, process_gpu_memory_bytes(self.device))
        except BaseException as error:
            self._error = error
            self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        if self._error is not None:
            raise RuntimeError("NVML GPU-memory sampling failed") from self._error
        return self.peak_bytes


def _squared_l2_query_centroid_distances(
    queries: torch.Tensor,
    centroids: torch.Tensor,
    centroid_norms: torch.Tensor,
) -> torch.Tensor:
    """Match the strict-FP32 coarse distance calculation used by search."""
    query_norms = (queries * queries).sum(dim=1, keepdim=True)
    if not queries.is_cuda:
        distances = torch.addmm(
            centroid_norms.unsqueeze(0),
            queries,
            centroids.transpose(0, 1),
            beta=1.0,
            alpha=-2.0,
        )
    else:
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            distances = torch.addmm(
                centroid_norms.unsqueeze(0),
                queries,
                centroids.transpose(0, 1),
                beta=1.0,
                alpha=-2.0,
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
    return distances.add_(query_norms).clamp_min_(0.0)


def probe_visit_statistics(
    index: Any,
    queries: torch.Tensor,
    nprobe: int,
    device: torch.device,
    data_bytes_per_vector: int,
) -> dict[str, float | int]:
    """Return logical vectors and payload bytes visited by one nprobe run.

    The selected list counts are computed with the same coarse-distance
    formula as the public search functions.  This is done before the timed
    region, so the benchmark's QP/QPS does not include this accounting pass.
    """
    if index.list_count is None:
        raise ValueError("The index must be loaded with its IVF lists")

    nprobe = max(1, min(int(nprobe), int(index.centroids.shape[0])))
    with torch.inference_mode():
        query_device = queries.to(device=device, dtype=torch.float32, copy=False).contiguous()
        centroids = index.centroids.to(device=device, dtype=torch.float32, copy=False)
        centroid_norms = index.centroid_norms.to(
            device=device,
            dtype=torch.float32,
            copy=False,
        )
        list_count = index.list_count.to(device=device, dtype=torch.int64, copy=False)
        coarse_distances = _squared_l2_query_centroid_distances(
            query_device,
            centroids,
            centroid_norms,
        )
        _, probe_ids = torch.topk(coarse_distances, k=nprobe, dim=1, largest=False)
        visited_per_query = list_count[probe_ids].sum(dim=1)
        visited_total = int(visited_per_query.sum().item())
        visited_min = int(visited_per_query.min().item())
        visited_max = int(visited_per_query.max().item())

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    num_queries = int(queries.shape[0])
    visited_bytes = visited_total * int(data_bytes_per_vector)
    return {
        "visited_vectors_total": visited_total,
        "visited_vectors_per_query": visited_total / max(num_queries, 1),
        "visited_vectors_min_per_query": visited_min,
        "visited_vectors_max_per_query": visited_max,
        "visited_data_bytes_total": visited_bytes,
        "visited_data_mb_total": visited_bytes / MEBIBYTE,
        "visited_data_mb_per_query": visited_bytes / max(num_queries, 1) / MEBIBYTE,
    }


def _make_padded_batch(
    queries: torch.Tensor,
    start: int,
    q_batch: int,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, int]:
    end = min(start + q_batch, int(queries.shape[0]))
    batch_size = end - start
    query_batch = queries[start:end].to(device=device, dtype=dtype, copy=False)
    if batch_size < q_batch:
        query_batch = torch.cat(
            (query_batch, query_batch[-1:].expand(q_batch - batch_size, -1)),
            dim=0,
        )
    return query_batch, batch_size


def run_one_nprobe(
    *,
    queries: torch.Tensor,
    groundtruth: torch.Tensor,
    nprobe: int,
    topk: int,
    q_batch: int,
    warmup_iterations: int,
    device: torch.device,
    search_one: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]],
    visit_stats: dict[str, float | int],
    recall_fn: Callable[[torch.Tensor, torch.Tensor, int], float],
    memory_baseline_process_mb: float | None = None,
) -> dict[str, float | int | None]:
    """Warm up and measure all query batches for one nprobe value."""
    if queries.shape[0] == 0:
        raise ValueError("At least one query is required")
    if q_batch <= 0:
        raise ValueError(f"Q_BATCH must be positive, got {q_batch}")

    baseline_allocated: int | None = None
    baseline_reserved: int | None = None
    if device.type == "cuda" and memory_baseline_process_mb is not None:
        # IVFPQ reports logical query working memory.  Reset before warmup so
        # persistent LUT/workspace allocations made by warmup are included.
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))

    first_batch, _ = _make_padded_batch(queries, 0, q_batch, device)
    for _ in range(max(1, int(warmup_iterations))):
        warmup_distances, warmup_ids = search_one(first_batch, nprobe)
        del warmup_distances, warmup_ids
    del first_batch
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    baseline_process: int | None = None
    peak_process: int | None = None
    if device.type == "cuda":
        if baseline_allocated is None:
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
        baseline_process = process_gpu_memory_bytes(device)
        peak_process = baseline_process

    memory_sampler: ProcessGpuMemorySampler | None = None
    if device.type == "cuda":
        memory_sampler = ProcessGpuMemorySampler(device)
        memory_sampler.start()

    result_ids: list[torch.Tensor] = []
    search_seconds = 0.0
    for start in range(0, int(queries.shape[0]), q_batch):
        query_batch, batch_size = _make_padded_batch(queries, start, q_batch, device)
        search_start = time.perf_counter()
        batch_distances, batch_ids = search_one(query_batch, nprobe)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        search_seconds += time.perf_counter() - search_start
        if device.type == "cuda":
            peak_process = max(
                peak_process or 0,
                process_gpu_memory_bytes(device),
            )
        result_ids.append(batch_ids[:batch_size].cpu())
        del query_batch, batch_distances, batch_ids

    if memory_sampler is not None:
        peak_process = max(peak_process or 0, memory_sampler.stop())

    ids = torch.cat(result_ids, dim=0)
    q_count = int(queries.shape[0])
    elapsed = max(search_seconds, 1e-12)
    qps = q_count / elapsed
    recall = float(recall_fn(ids, groundtruth, topk))

    peak_allocated: int | None = None
    peak_reserved: int | None = None
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

    row: dict[str, float | int | None] = {
        "nprobe": int(nprobe),
        "qp": qps,
        "qps": qps,
        "search_seconds": search_seconds,
        "recall": recall,
        "recall_at_k": recall,
        "peak_gpu_memory_allocated_mb": (
            peak_allocated / MEBIBYTE if peak_allocated is not None else None
        ),
        "peak_gpu_memory_reserved_mb": (
            peak_reserved / MEBIBYTE if peak_reserved is not None else None
        ),
        "peak_gpu_memory_process_mb": (
            peak_process / MEBIBYTE if peak_process is not None else None
        ),
        "search_gpu_memory_delta_process_mb": (
            max(0.0, peak_process / MEBIBYTE - memory_baseline_process_mb)
            if peak_process is not None and memory_baseline_process_mb is not None
            else (peak_process - baseline_process) / MEBIBYTE
            if peak_process is not None and baseline_process is not None
            else None
        ),
        # Compatibility aliases for older benchmark callers. These now carry
        # the process-resident value, not device-wide total-minus-free memory.
        "peak_gpu_memory_driver_mb": (
            peak_process / MEBIBYTE if peak_process is not None else None
        ),
        "search_gpu_memory_delta_driver_mb": (
            (peak_process - baseline_process) / MEBIBYTE
            if peak_process is not None and baseline_process is not None
            else None
        ),
        "search_gpu_memory_delta_allocated_mb": (
            (peak_allocated - baseline_allocated) / MEBIBYTE
            if peak_allocated is not None and baseline_allocated is not None
            else None
        ),
        "search_gpu_memory_delta_reserved_mb": (
            (peak_reserved - baseline_reserved) / MEBIBYTE
            if peak_reserved is not None and baseline_reserved is not None
            else None
        ),
        "peak_gpu_memory": (
            peak_process / MEBIBYTE if peak_process is not None else None
        ),
    }
    row.update(visit_stats)
    return row


def write_results(
    *,
    result_dir: Path,
    file_stem: str,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write one experiment's table and complete metadata to CSV and JSON."""
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / f"{file_stem}.json"
    csv_path = result_dir / f"{file_stem}.csv"

    payload = dict(metadata)
    payload["results"] = rows
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    fieldnames: list[str] = []
    for key in (*metadata.keys(), *(rows[0].keys() if rows else ())):
        if key not in fieldnames:
            fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(metadata)
            csv_row.update(row)
            for key, value in list(csv_row.items()):
                if isinstance(value, (list, tuple, dict)):
                    csv_row[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(csv_row)
    return csv_path, json_path


def safe_filename(value: str) -> str:
    """Keep dataset names in result filenames portable."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._") or "dataset"


def print_result(row: dict[str, Any]) -> None:
    peak_memory = row.get(
        "peak_gpu_memory",
        row.get("peak_gpu_memory_driver_mb", row.get("peak_gpu_memory_allocated_mb")),
    )
    print(
        f"nprobe={row['nprobe']:>4}  "
        f"QP/QPS={row['qps']:.4f}  "
        f"Recall@K={row['recall']:.6f}  "
        f"visited={row['visited_vectors_per_query']:.2f} vectors/query  "
        f"peak_query_memory={peak_memory} MiB"
    )
