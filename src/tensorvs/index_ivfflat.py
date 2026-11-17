"""Build, persist, and search IVF-Flat indexes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from flash_kmeans import (
    FAISS_DEFAULT_MAX_POINTS_PER_CENTROID,
    FAISS_DEFAULT_MIN_POINTS_PER_CENTROID,
    FAISS_DEFAULT_NREDO,
    FAISS_DEFAULT_SEED,
    fit_best_kmeans,
)
from ._coarse import (
    _coarse_centroid_scores_addmm as _coarse_centroid_scores_addmm,
    _coarse_squared_l2_distance_addmm as _coarse_squared_l2_distance_addmm,
)
from ._ivf_interleaved import (
    INTERLEAVED_GROUP_SIZE,
    build_interleaved_flat_lists,
    build_interleaved_pq_scalar_cache,
    interleaved_vector_width,
)
from ._ivfflat_triton import can_run_topk as can_run_ivfflat_triton_topk
from ._ivfflat_triton import can_use_uint8_scan, prepare_uint8_centered_norms
from ._ivfflat_triton import triton_topk
from ._ivfflat_triton import triton as _triton
from ._layout_cache import (
    LayoutCache,
    logical_lists,
    normalize_device,
    prepare_torch_layout,
    record_triton_layout,
)
from ._torch_search import search_device, search_torch, validate_backend


def _get_interleaved_flat_lists(
    index: IVFFlatIndex,
    search_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retrieve the Triton-only interleaved IVF-Flat layout."""
    dim = int(index.centroids.shape[1])

    def layout_is_compatible(
        vectors: torch.Tensor | None,
        ids: torch.Tensor | None,
        group_prefix: torch.Tensor | None,
        vector_norms: torch.Tensor | None,
    ) -> bool:
        if vectors is None or ids is None or group_prefix is None or vector_norms is None:
            return False
        if vectors.ndim != 4 or vectors.shape[2] != INTERLEAVED_GROUP_SIZE:
            return False
        vector_width = interleaved_vector_width(dim, vectors.dtype)
        return (
            vectors.shape[1] == dim // vector_width
            and ids.numel() == vectors.shape[0] * INTERLEAVED_GROUP_SIZE
            and group_prefix.ndim == 1
            and vector_norms.numel() == ids.numel()
        )

    has_persisted_layout = (
        index.persisted_list_vectors_interleaved is not None
        and index.persisted_list_ids_interleaved is not None
        and index.persisted_list_group_prefix is not None
        and index.persisted_list_vector_norms_interleaved is not None
    )

    def persisted_layout() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert index.persisted_list_vectors_interleaved is not None
        assert index.persisted_list_ids_interleaved is not None
        assert index.persisted_list_group_prefix is not None
        assert index.persisted_list_vector_norms_interleaved is not None
        return (
            index.persisted_list_vectors_interleaved.to(
                device=search_device,
                copy=False,
            ),
            index.persisted_list_ids_interleaved.to(
                dtype=torch.int64,
                device=search_device,
                copy=False,
            ),
            index.persisted_list_group_prefix.to(
                dtype=torch.int64,
                device=search_device,
                copy=False,
            ),
            index.persisted_list_vector_norms_interleaved.to(
                dtype=torch.float32,
                device=search_device,
                copy=False,
            ),
        )

    if index.resident_device is not None and index.resident_device == search_device:
        if layout_is_compatible(
            index.list_vectors_interleaved,
            index.list_ids_interleaved,
            index.list_group_prefix,
            index.list_vector_norms_interleaved,
        ):
            return (
                index.list_vectors_interleaved,
                index.list_ids_interleaved,
                index.list_group_prefix,
                index.list_vector_norms_interleaved,
            )

    if has_persisted_layout and layout_is_compatible(
        index.persisted_list_vectors_interleaved,
        index.persisted_list_ids_interleaved,
        index.persisted_list_group_prefix,
        index.persisted_list_vector_norms_interleaved,
    ):
        return persisted_layout()

    if layout_is_compatible(
        index.list_vectors_interleaved,
        index.list_ids_interleaved,
        index.list_group_prefix,
        index.list_vector_norms_interleaved,
    ):
        return (
            index.list_vectors_interleaved.to(device=search_device, copy=False),
            index.list_ids_interleaved.to(dtype=torch.int64, device=search_device, copy=False),
            index.list_group_prefix.to(dtype=torch.int64, device=search_device, copy=False),
            index.list_vector_norms_interleaved.to(dtype=torch.float32, device=search_device, copy=False),
        )

    raise ValueError(
        "Triton IVF-Flat interleaved lists are not loaded. "
        "Call load_ivfflat_index(..., load_lists=True)."
    )


@dataclass
class IVFFlatIndex:
    """In-memory representation of IVF-Flat index."""
    index_dir: Path
    metadata: dict[str, Any]
    vectors: torch.Tensor | None
    assignments: torch.Tensor
    centroids: torch.Tensor
    ids: torch.Tensor
    lists: list[dict[str, Any]]
    list_count: torch.Tensor | None
    # Static coarse-centroid squared norms used by Triton TopK.
    centroid_norms: torch.Tensor | None = None
    # cuVS-style `[group, chunk, 32, veclen]` Triton layout.
    list_vectors_interleaved: torch.Tensor | None = None
    list_ids_interleaved: torch.Tensor | None = None
    list_group_prefix: torch.Tensor | None = None
    list_vector_norms_interleaved: torch.Tensor | None = None
    # CPU copies loaded from disk; ``reside_on_device`` transfers them once.
    persisted_list_vectors_interleaved: torch.Tensor | None = None
    persisted_list_ids_interleaved: torch.Tensor | None = None
    persisted_list_group_prefix: torch.Tensor | None = None
    persisted_list_vector_norms_interleaved: torch.Tensor | None = None
    # Cached host scalar used to choose the long-list Triton split plan.  It
    # avoids a device-to-host synchronization for every query batch.
    triton_max_list_group_count: int | None = None
    resident_device: torch.device | None = None
    layout_cache: LayoutCache = field(default_factory=LayoutCache, repr=False)
    _uint8_norm_cache: tuple[torch.Tensor, torch.Tensor] | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    @property
    def logical_lists(self):
        """Unpadded, device-independent list-major view, restored lazily."""
        return logical_lists(self, "flat")

    def __post_init__(self) -> None:
        self.centroids = self.centroids.to(dtype=torch.float32, copy=False)
        if self.centroid_norms is None:
            self.centroid_norms = (self.centroids * self.centroids).sum(dim=-1)
        else:
            self.centroid_norms = self.centroid_norms.to(dtype=torch.float32, copy=False)

    def reside_on_device(self, device: torch.device | str, *, backend: str = "triton",
                         persist_layout: bool = True) -> IVFFlatIndex:
        """Move search-critical tensors onto a target device and keep them resident."""
        validate_backend(backend)
        target_device = normalize_device(device)
        if backend == "torch" or (backend == "auto" and (target_device.type != "cuda" or _triton is None)):
            prepare_torch_layout(self, "flat", target_device, persist=persist_layout)
            return self
        self.layout_cache.torch_layout = None
        if self.list_count is None:
            raise ValueError("Index flat lists are not loaded. Call load_ivfflat_index(..., load_lists=True).")

        has_persisted_layout = (
            self.persisted_list_vectors_interleaved is not None
            and self.persisted_list_ids_interleaved is not None
            and self.persisted_list_group_prefix is not None
            and self.persisted_list_vector_norms_interleaved is not None
        )
        has_resident_layout = (
            self.list_vectors_interleaved is not None
            and self.list_ids_interleaved is not None
            and self.list_group_prefix is not None
            and self.list_vector_norms_interleaved is not None
        )
        if not has_persisted_layout and not has_resident_layout:
            raise ValueError(
                "Triton IVF-Flat interleaved lists are not loaded. "
                "Call load_ivfflat_index(..., load_lists=True)."
            )

        if persist_layout:
            record_triton_layout(self, target_device)
        if self.resident_device == target_device and has_resident_layout:
            return self
        self._uint8_norm_cache = None
        self.centroids = self.centroids.to(dtype=torch.float32, device=target_device, copy=False)
        self.centroid_norms = self.centroid_norms.to(
            dtype=torch.float32,
            device=target_device,
            copy=False,
        )
        self.list_count = self.list_count.to(dtype=torch.int64, device=target_device, copy=False)
        if has_persisted_layout:
            self.list_vectors_interleaved = self.persisted_list_vectors_interleaved.to(
                device=target_device,
                copy=False,
            )
            self.list_ids_interleaved = self.persisted_list_ids_interleaved.to(
                dtype=torch.int64,
                device=target_device,
                copy=False,
            )
            self.list_group_prefix = self.persisted_list_group_prefix.to(
                dtype=torch.int64,
                device=target_device,
                copy=False,
            )
            self.list_vector_norms_interleaved = self.persisted_list_vector_norms_interleaved.to(
                dtype=torch.float32,
                device=target_device,
                copy=False,
            )
        else:
            self.list_vectors_interleaved = self.list_vectors_interleaved.to(
                device=target_device,
                copy=False,
            )
            self.list_ids_interleaved = self.list_ids_interleaved.to(
                dtype=torch.int64,
                device=target_device,
                copy=False,
            )
            self.list_group_prefix = self.list_group_prefix.to(
                dtype=torch.int64,
                device=target_device,
                copy=False,
            )
            self.list_vector_norms_interleaved = self.list_vector_norms_interleaved.to(
                dtype=torch.float32,
                device=target_device,
                copy=False,
            )
        self.resident_device = target_device
        return self

    def release_device_residency(self) -> IVFFlatIndex:
        """Move resident search tensors back to CPU."""
        self.centroids = self.centroids.to(dtype=torch.float32, device="cpu", copy=False)
        self.centroid_norms = self.centroid_norms.to(dtype=torch.float32, device="cpu", copy=False)
        if self.list_count is not None:
            self.list_count = self.list_count.to(dtype=torch.int64, device="cpu", copy=False)
        self.list_vectors_interleaved = None
        self.list_ids_interleaved = None
        self.list_group_prefix = None
        self.list_vector_norms_interleaved = None
        self._uint8_norm_cache = None
        self.resident_device = None
        self.layout_cache.torch_layout = None
        self.layout_cache.triton_fingerprint = None
        return self


@dataclass
class IVFFlatBuildResult:
    """In-memory result of IVF-Flat training before serialization."""

    vectors: torch.Tensor
    labels: torch.Tensor
    centroids: torch.Tensor
    device: torch.device
    seed: int = 0
    n_init: int = 1
    coarse_inertia: float | None = None
    coarse_seed: int | None = None

def save_ivfflat_index(
    build_result: IVFFlatBuildResult,
    output_dir: str | Path,
    dataset_path: str | Path | None = None,
    build_time_seconds: float | None = None,
) -> Path:
    """Persist IVF-Flat with its default interleaved execution cache."""
    output_path = Path(output_dir)

    vectors_cpu = build_result.vectors.cpu()
    labels_cpu = build_result.labels.cpu().to(torch.int64)
    centroids_cpu = build_result.centroids.cpu().to(torch.float32)
    centroid_norms_cpu = (centroids_cpu * centroids_cpu).sum(dim=-1)
    ids = torch.arange(vectors_cpu.shape[0], dtype=torch.int64)

    if labels_cpu.numel() > 0:
        sort_perm = torch.argsort(labels_cpu, stable=True)
        labels_sorted = labels_cpu[sort_perm]
        list_vectors = vectors_cpu[sort_perm]
        list_ids_flat = ids[sort_perm]
        if list_vectors.dtype == torch.uint8:
            # Compute norms in FP32 without expanding the entire stored index.
            list_vector_norms = torch.empty(len(list_vectors), dtype=torch.float32)
            for offset in range(0, len(list_vectors), 65536):
                rows = list_vectors[offset:offset + 65536].to(torch.float32)
                list_vector_norms[offset:offset + len(rows)] = rows.square().sum(dim=1)
        else:
            list_vectors = list_vectors.to(torch.float32)
            list_vector_norms = list_vectors.square().sum(dim=1)
        list_count = torch.bincount(labels_sorted, minlength=centroids_cpu.shape[0]).to(torch.int64)
        list_prefix = torch.cumsum(list_count, dim=0) - list_count
    else:
        dim = int(centroids_cpu.shape[1])
        list_vectors = torch.empty((0, dim), dtype=torch.uint8 if vectors_cpu.dtype == torch.uint8 else torch.float32)
        list_ids_flat = torch.empty((0,), dtype=torch.int64)
        list_vector_norms = torch.empty((0,), dtype=torch.float32)
        list_count = torch.zeros((centroids_cpu.shape[0],), dtype=torch.int64)
        list_prefix = torch.zeros((centroids_cpu.shape[0],), dtype=torch.int64)

    interleaved_layout = build_interleaved_flat_lists(
        list_vectors,
        list_ids_flat,
        list_prefix,
        list_count,
    )
    interleaved_vector_norms = build_interleaved_pq_scalar_cache(
        list_vector_norms,
        list_prefix,
        list_count,
    )

    cluster_sizes = {
        str(cluster_id): int(count)
        for cluster_id, count in enumerate(list_count.tolist())
    }

    is_single_file = output_path.suffix != ""
    metadata_output = str(output_path)
    metadata: dict[str, Any] = {
        "dataset_path": str(Path(dataset_path))
        if dataset_path is not None
        else "in_memory_tensor",
        "output_dir": metadata_output,
        "num_vectors": int(vectors_cpu.shape[0]),
        "dim": int(vectors_cpu.shape[1]),
        "nlist": int(centroids_cpu.shape[0]),
        "device": str(build_result.device),
        "seed": int(build_result.seed),
        "n_init": int(build_result.n_init),
        "coarse_inertia": build_result.coarse_inertia,
        "coarse_seed": build_result.coarse_seed,
        "stores_full_vectors": False,
        "logical_index_schema": 1,
        "default_runtime_layout": "interleaved",
        "cluster_sizes": cluster_sizes,
        "storage_format": "single_file" if is_single_file else "directory_interleaved",
        "vector_storage_dtype": str(list_vectors.dtype).removeprefix("torch."),
        "interleaved_layout": {
            "available": interleaved_layout is not None,
            "group_size": INTERLEAVED_GROUP_SIZE,
            "vector_width": interleaved_layout.vector_width,
        },
    }
    if build_time_seconds is not None:
        metadata["build_time_seconds"] = build_time_seconds

    if is_single_file:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata,
            "assignments": labels_cpu,
            "centroids": centroids_cpu,
            "centroid_norms": centroid_norms_cpu,
            "ids": ids,
            "list_count": list_count,
            "list_vectors_interleaved": interleaved_layout.vectors,
            "list_ids_interleaved": interleaved_layout.ids,
            "list_group_prefix": interleaved_layout.group_prefix,
            "list_vector_norms_interleaved": interleaved_vector_norms,
        }
        torch.save(payload, output_path)
        index_size_bytes = output_path.stat().st_size
        metadata["index_size_bytes"] = index_size_bytes
        metadata["index_size_mb"] = round(index_size_bytes / (1024 * 1024), 4)
        payload["metadata"] = metadata
        torch.save(payload, output_path)
        return output_path

    out_dir = output_path
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(labels_cpu, out_dir / "assignments.pt")
    torch.save(centroids_cpu, out_dir / "centroids.pt")
    torch.save(centroid_norms_cpu, out_dir / "centroid_norms.pt")
    torch.save(ids, out_dir / "ids.pt")
    torch.save(list_count, out_dir / "list_count.pt")
    torch.save(interleaved_layout.vectors, out_dir / "list_vectors_interleaved.pt")
    torch.save(interleaved_layout.ids, out_dir / "list_ids_interleaved.pt")
    torch.save(interleaved_layout.group_prefix, out_dir / "list_group_prefix.pt")
    torch.save(interleaved_vector_norms, out_dir / "list_vector_norms_interleaved.pt")

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    index_size_bytes = sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())
    metadata["index_size_bytes"] = index_size_bytes
    metadata["index_size_mb"] = round(index_size_bytes / (1024 * 1024), 4)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def load_ivfflat_index(index_dir: str | Path, load_lists: bool = True) -> IVFFlatIndex:
    """Load IVF-Flat and its default cache; logical rows are restored lazily."""
    index_path = Path(index_dir)
    if not index_path.exists():
        raise FileNotFoundError(f"Index directory not found: {index_path}")

    if index_path.is_file():
        payload = torch.load(index_path, map_location="cpu")
        metadata = payload["metadata"]
        assignments = payload["assignments"]
        centroids = payload["centroids"].to(torch.float32)
        centroid_norms = payload.get("centroid_norms")
        if centroid_norms is None:
            centroid_norms = (centroids * centroids).sum(dim=-1)
        else:
            centroid_norms = centroid_norms.to(torch.float32)
        ids = payload["ids"]
        if load_lists:
            required_payload_keys = (
                "list_count",
                "list_vectors_interleaved",
                "list_ids_interleaved",
                "list_group_prefix",
            )
            missing_payload_keys = [key for key in required_payload_keys if key not in payload]
            if missing_payload_keys:
                raise FileNotFoundError(
                    "Triton IVF-Flat interleaved layout is missing from this archive; "
                    f"rebuild the index. Missing keys: {missing_payload_keys}"
                )
            list_count = payload["list_count"].to(torch.int64)
            persisted_list_vectors_interleaved = payload.get("list_vectors_interleaved")
            persisted_list_ids_interleaved = payload.get("list_ids_interleaved")
            persisted_list_group_prefix = payload.get("list_group_prefix")
            persisted_list_vector_norms_interleaved = payload.get("list_vector_norms_interleaved")
            if persisted_list_vectors_interleaved is not None:
                if persisted_list_vectors_interleaved.dtype != torch.uint8:
                    persisted_list_vectors_interleaved = persisted_list_vectors_interleaved.to(torch.float32)
                persisted_list_ids_interleaved = persisted_list_ids_interleaved.to(torch.int64)
                persisted_list_group_prefix = persisted_list_group_prefix.to(torch.int64)
                if persisted_list_vector_norms_interleaved is None:
                    # Older archives already had the interleaved vectors.  The
                    # norm cache can be reconstructed without restoring the
                    # removed row-major layout.
                    persisted_list_vector_norms_interleaved = (
                        persisted_list_vectors_interleaved.to(torch.float32).square().sum(dim=(1, 3)).reshape(-1)
                    )
                else:
                    persisted_list_vector_norms_interleaved = persisted_list_vector_norms_interleaved.to(
                        torch.float32
                    )
            lists = []
        else:
            list_count = None
            persisted_list_vectors_interleaved = None
            persisted_list_ids_interleaved = None
            persisted_list_group_prefix = None
            persisted_list_vector_norms_interleaved = None
            lists = []

        return IVFFlatIndex(
            index_dir=index_path,
            metadata=metadata,
            vectors=None,
            assignments=assignments,
            centroids=centroids,
            centroid_norms=centroid_norms,
            ids=ids,
            lists=lists,
            list_count=list_count,
            persisted_list_vectors_interleaved=persisted_list_vectors_interleaved,
            persisted_list_ids_interleaved=persisted_list_ids_interleaved,
            persisted_list_group_prefix=persisted_list_group_prefix,
            persisted_list_vector_norms_interleaved=persisted_list_vector_norms_interleaved,
        )

    metadata_path = index_path / "metadata.json"
    assignments_path = index_path / "assignments.pt"
    centroids_path = index_path / "centroids.pt"
    centroid_norms_path = index_path / "centroid_norms.pt"
    ids_path = index_path / "ids.pt"
    list_count_path = index_path / "list_count.pt"
    list_vectors_interleaved_path = index_path / "list_vectors_interleaved.pt"
    list_ids_interleaved_path = index_path / "list_ids_interleaved.pt"
    list_group_prefix_path = index_path / "list_group_prefix.pt"
    list_vector_norms_interleaved_path = index_path / "list_vector_norms_interleaved.pt"

    required_files = [metadata_path, assignments_path, centroids_path, ids_path]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing IVF-Flat index files: {missing_files}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assignments = torch.load(assignments_path, map_location="cpu")
    centroids = torch.load(centroids_path, map_location="cpu").to(torch.float32)
    if centroid_norms_path.exists():
        centroid_norms = torch.load(centroid_norms_path, map_location="cpu").to(torch.float32)
    else:
        centroid_norms = (centroids * centroids).sum(dim=-1)
    ids = torch.load(ids_path, map_location="cpu")

    if load_lists:
        required_layout_files = [
            list_count_path,
            list_vectors_interleaved_path,
            list_ids_interleaved_path,
            list_group_prefix_path,
        ]
        missing_layout_files = [str(path) for path in required_layout_files if not path.exists()]
        if missing_layout_files:
            raise FileNotFoundError(
                "Triton IVF-Flat interleaved layout is incomplete; rebuild the index. "
                f"Missing files: {missing_layout_files}"
            )

        list_count = torch.load(list_count_path, map_location="cpu").to(torch.int64)
        persisted_list_vectors_interleaved = torch.load(
            list_vectors_interleaved_path,
            map_location="cpu",
        )
        if persisted_list_vectors_interleaved.dtype != torch.uint8:
            persisted_list_vectors_interleaved = persisted_list_vectors_interleaved.to(torch.float32)
        persisted_list_ids_interleaved = torch.load(
            list_ids_interleaved_path,
            map_location="cpu",
        ).to(torch.int64)
        persisted_list_group_prefix = torch.load(
            list_group_prefix_path,
            map_location="cpu",
        ).to(torch.int64)
        if list_vector_norms_interleaved_path.exists():
            persisted_list_vector_norms_interleaved = torch.load(
                list_vector_norms_interleaved_path,
                map_location="cpu",
            ).to(torch.float32)
        else:
            persisted_list_vector_norms_interleaved = (
                persisted_list_vectors_interleaved.to(torch.float32).square().sum(dim=(1, 3)).reshape(-1)
            )
    else:
        list_count = None
        persisted_list_vectors_interleaved = None
        persisted_list_ids_interleaved = None
        persisted_list_group_prefix = None
        persisted_list_vector_norms_interleaved = None

    return IVFFlatIndex(
        index_dir=index_path,
        metadata=metadata,
        vectors=None,
        assignments=assignments,
        centroids=centroids,
        centroid_norms=centroid_norms,
        ids=ids,
        lists=[],
        list_count=list_count,
        persisted_list_vectors_interleaved=persisted_list_vectors_interleaved,
        persisted_list_ids_interleaved=persisted_list_ids_interleaved,
        persisted_list_group_prefix=persisted_list_group_prefix,
        persisted_list_vector_norms_interleaved=persisted_list_vector_norms_interleaved,
    )


def _search_ivfflat_index_triton(
    index: IVFFlatIndex,
    queries: torch.Tensor,
    topk: int = 10,
    nprobe: int = 10,
    device: torch.device | str | None = None,
    profile: dict[str, float] | None = None,
    selection_mode: str = "triton_topk",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search IVF-Flat using the Triton TopK implementation."""

    search_device: torch.device | None = None
    cuda_timing_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def start_timing() -> float | torch.cuda.Event:
        if profile is None:
            return 0.0
        if search_device is not None and search_device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(search_device))
            return start_event
        return time.perf_counter()

    def record_timing(name: str, start_time: float | torch.cuda.Event) -> None:
        if profile is None:
            return
        if search_device is not None and search_device.type == "cuda":
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream(search_device))
            cuda_timing_events.append((name, start_time, end_event))
            return
        profile[name] = profile.get(name, 0.0) + (time.perf_counter() - start_time)

    def flush_timing() -> None:
        if profile is None or not cuda_timing_events:
            return
        assert search_device is not None
        torch.cuda.synchronize(search_device)
        for name, start_event, end_event in cuda_timing_events:
            profile[name] = profile.get(name, 0.0) + (start_event.elapsed_time(end_event) / 1000.0)

    if queries.ndim != 2:
        raise ValueError(f"Query tensor must have shape (num_queries, dim), got {tuple(queries.shape)}")
    if queries.shape[1] != index.centroids.shape[1]:
        raise ValueError(
            f"Query dimension {queries.shape[1]} does not match index dimension {index.centroids.shape[1]}"
        )
    if index.list_count is None:
        raise ValueError("Index flat lists are not loaded. Call load_ivfflat_index(..., load_lists=True).")
    if selection_mode != "triton_topk":
        raise ValueError(
            "Unsupported IVFFlat selection_mode: "
            f"{selection_mode}; only 'triton_topk' is supported"
        )

    if device is None:
        if queries.is_cuda:
            search_device = queries.device
        elif torch.cuda.is_available():
            search_device = torch.device("cuda:0")
        else:
            search_device = torch.device("cpu")
    else:
        search_device = torch.device(device)

    step_start = start_timing()
    if queries.dtype == torch.uint8:
        queries = queries.to(device=search_device, copy=False).contiguous()
    Q = queries.to(dtype=torch.float32, device=search_device, copy=False).contiguous()
    if index.resident_device is not None and index.resident_device == search_device:
        centroids = index.centroids
        centroid_norms = index.centroid_norms
        list_count = index.list_count
    else:
        centroids = index.centroids.to(dtype=torch.float32, device=search_device, copy=False)
        centroid_norms = index.centroid_norms.to(
            dtype=torch.float32,
            device=search_device,
            copy=False,
        )
        list_count = index.list_count.to(dtype=torch.int64, device=search_device, copy=False)
    record_timing("prepare_tensors", step_start)

    nprobe = max(1, min(int(nprobe), centroids.shape[0]))
    topk = max(1, int(topk))
    num_queries = Q.shape[0]
    step_start = start_timing()
    result_distances = torch.full((num_queries, topk), float("inf"), dtype=torch.float32, device=search_device)
    result_ids = torch.full((num_queries, topk), -1, dtype=torch.int64, device=search_device)
    record_timing("init_outputs", step_start)

    if not can_run_ivfflat_triton_topk(
        device=search_device,
        topk=topk,
        num_queries=num_queries,
        nprobe=nprobe,
        dim=Q.shape[1],
    ):
        raise RuntimeError(
            "IVF-Flat search requires the Triton TopK path and a supported CUDA shape"
        )

    # Compute the query-invariant part of the full [query, nlist] strict-FP32
    # distance matrix. Query norms cannot affect list selection, so avoid
    # broadcasting them over all nlist entries.
    step_start = start_timing()
    coarse_scores = _coarse_centroid_scores_addmm(
        Q,
        centroids,
        centroid_norms,
    )
    record_timing("centroid_distance", step_start)
    step_start = start_timing()
    _, probe_cids = torch.topk(
        coarse_scores, k=nprobe, dim=1, largest=False, sorted=False
    )
    record_timing("centroid_topk", step_start)

    step_start = start_timing()
    (
        interleaved_vectors,
        interleaved_ids,
        list_group_prefix,
        interleaved_vector_norms,
    ) = _get_interleaved_flat_lists(index, search_device)
    scan_queries = Q
    uint8_centered_norms = None
    if can_use_uint8_scan(queries, interleaved_vectors, topk):
        cached = index._uint8_norm_cache
        if cached is None or cached[0] is not interleaved_vectors:
            uint8_centered_norms = prepare_uint8_centered_norms(interleaved_vectors)
            # Publish the cache only once its data is ready on any search stream.
            torch.cuda.current_stream(search_device).synchronize()
            index._uint8_norm_cache = (interleaved_vectors, uint8_centered_norms)
        else:
            uint8_centered_norms = cached[1]
        scan_queries = queries
    record_timing("triton_prepare_layout", step_start)

    step_start = start_timing()
    if index.triton_max_list_group_count is None:
        index.triton_max_list_group_count = int(
            torch.div(
                list_count.max() + (INTERLEAVED_GROUP_SIZE - 1),
                INTERLEAVED_GROUP_SIZE,
                rounding_mode="floor",
            ).item()
        )
    topk_distances, topk_ids = triton_topk(
        queries=scan_queries,
        list_vectors_interleaved=interleaved_vectors,
        list_ids_interleaved=interleaved_ids,
        list_group_prefix=list_group_prefix,
        list_vector_norms_interleaved=interleaved_vector_norms,
        list_count=list_count,
        probe_cids=probe_cids,
        topk=topk,
        max_list_group_count=index.triton_max_list_group_count,
        uint8_centered_norms=uint8_centered_norms,
    )
    result_distances.copy_(topk_distances)
    result_ids.copy_(topk_ids)
    record_timing("triton_topk_select", step_start)

    flush_timing()
    return result_distances, result_ids


def search_ivfflat_index(
    index: IVFFlatIndex,
    queries: torch.Tensor,
    topk: int = 10,
    nprobe: int = 10,
    device: torch.device | str | None = None,
    profile: dict[str, float] | None = None,
    selection_mode: str = "triton_topk",
    *,
    backend: str = "auto",
    torch_query_batch_size: int = 128,
    torch_candidate_chunk_size: int = 2048,
    torch_memory_budget_mb: float = 128,
    torch_flat_scan: str = "auto",
    torch_query_tile_size: int | None = None,
    torch_max_tasks_per_batch: int = 256,
    torch_min_query_tile_size: int = 1,
    use_compile: bool = False,
    persist_layout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search with the existing fused Triton path or bounded PyTorch blocks.

    ``auto`` prefers Triton for supported devices/shapes. ``use_compile``
    compiles the dense PyTorch scan and merges, leaving scheduling eager.
    Without ``device``, non-CPU queries keep their device; CPU queries use
    CUDA, then MPS, then CPU according to availability.
    Large GPU batches with broad probing use masked, contiguous GEMM blocks;
    small/sparse batches share candidates in list-grouped blocks.
    ``torch_flat_scan='grouped'`` always scans only the selected IVF lists;
    ``'dense'`` always computes all vector distances before IVF masking.
    ``'auto'`` retains the adaptive choice. This option applies to Torch.
    ``torch_query_tile_size`` overrides the grouped list-task query width
    (a power of two); ``torch_max_tasks_per_batch`` caps grouped tasks per
    call. Both remain subject to the workspace budget and do not alter IVF.
    The memory budget bounds scan/merge workspace, excluding resident index,
    input/output tensors, and compiler/library workspaces. The legacy
    ``selection_mode`` argument is retained for existing callers.
    """
    validate_backend(backend)
    if selection_mode != "triton_topk":
        raise ValueError(
            f"Unsupported IVFFlat selection_mode: {selection_mode!r}; "
            "expected 'triton_topk'; choose a backend with backend="
        )
    if queries.ndim != 2 or queries.shape[1] != index.centroids.shape[1]:
        raise ValueError("Queries must have shape [num_queries, index dimension]")
    target = search_device(queries, device)
    supported = can_run_ivfflat_triton_topk(
        device=target,
        topk=max(1, int(topk)),
        num_queries=queries.shape[0],
        nprobe=max(1, min(int(nprobe), index.centroids.shape[0])),
        dim=queries.shape[1],
    )
    if backend == "torch" or (backend == "auto" and not supported):
        return search_torch(
            index,
            queries,
            kind="flat",
            topk=topk,
            nprobe=nprobe,
            device=target,
            profile=profile,
            query_batch_size=torch_query_batch_size,
            candidate_chunk_size=torch_candidate_chunk_size,
            memory_budget_mb=torch_memory_budget_mb,
            use_compile=use_compile,
            persist_layout=persist_layout,
            flat_scan=torch_flat_scan,
            query_tile_size=torch_query_tile_size,
            max_tasks_per_batch=torch_max_tasks_per_batch,
            min_query_tile_size=torch_min_query_tile_size,
        )
    if not supported:
        raise RuntimeError("The requested Triton IVF-Flat backend does not support this device or shape")
    index.reside_on_device(target, persist_layout=persist_layout)
    return _search_ivfflat_index_triton(
        index,
        queries,
        topk=topk,
        nprobe=nprobe,
        device=target,
        profile=profile,
        selection_mode=selection_mode,
    )


def build_ivfflat_index(
    vectors: torch.Tensor,
    nlist: int,
    niter: int,
    use_triton: bool,
    device: torch.device | str,
    seed: int = FAISS_DEFAULT_SEED,
    n_init: int = FAISS_DEFAULT_NREDO,
    *,
    max_points_per_centroid: int | None = FAISS_DEFAULT_MAX_POINTS_PER_CENTROID,
    min_points_per_centroid: int = FAISS_DEFAULT_MIN_POINTS_PER_CENTROID,
    store_uint8: bool = False,
) -> IVFFlatBuildResult:
    """Train an IVF-Flat coarse quantizer.

    When set, ``max_points_per_centroid`` limits KMeans training to at most
    ``nlist * max_points_per_centroid`` sampled vectors. The returned labels
    are still assigned for the complete input dataset.
    """
    if vectors.ndim != 2:
        raise ValueError(f"Dataset tensor must have shape (num_vectors, dim), got {tuple(vectors.shape)}")
    if store_uint8 and vectors.dtype != torch.uint8:
        raise ValueError("UInt8 IVF-Flat storage requires UInt8 source vectors")
    raw_vectors = vectors if store_uint8 else None
    vectors = vectors.to(dtype=torch.float32, copy=False).contiguous()
    dim = int(vectors.shape[1])
    target_device = torch.device(device)

    trained = fit_best_kmeans(
        vectors,
        d=dim,
        k=nlist,
        niter=niter,
        use_triton=use_triton,
        device=target_device,
        seed=seed,
        n_init=n_init,
        stage=0,
        max_points_per_centroid=max_points_per_centroid,
        min_points_per_centroid=min_points_per_centroid,
    )

    return IVFFlatBuildResult(
        vectors=raw_vectors if raw_vectors is not None else vectors,
        labels=trained.labels,
        centroids=trained.centroids,
        device=target_device,
        seed=int(seed),
        n_init=int(n_init),
        coarse_inertia=trained.inertia,
        coarse_seed=trained.seed,
    )
