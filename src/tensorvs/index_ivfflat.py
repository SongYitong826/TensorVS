from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ._kmeans import fit_best_kmeans
from ._ivf_interleaved import (
    INTERLEAVED_GROUP_SIZE,
    build_interleaved_flat_lists,
    interleaved_vector_width,
)
from ._ivfflat_triton import can_run_topk as can_run_ivfflat_triton_topk
from ._ivfflat_triton import triton_topk
from .distance import squared_l2_distance


def _decode_pairs_and_distance_impl(
    q: torch.Tensor,
    q_norm: torch.Tensor,
    list_vectors: torch.Tensor,
    list_vector_norms: torch.Tensor,
    list_ids_flat: torch.Tensor,
    q_prefix: torch.Tensor,
    l_prefix: torch.Tensor,
    l_count: torch.Tensor,
    edge_qid_sorted: torch.Tensor,
    pair_prefix: torch.Tensor,
    pair_cluster_idx: torch.Tensor,
    pair_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed ``(query_id, distance)`` keys and vector IDs for all pairs."""
    local_pair_pos = pair_pos - pair_prefix[pair_cluster_idx]
    cur_l_count = l_count[pair_cluster_idx]
    query_local_id = torch.div(local_pair_pos, cur_l_count, rounding_mode="floor")
    cand_local_id = local_pair_pos % cur_l_count

    pair_qid = edge_qid_sorted[q_prefix[pair_cluster_idx] + query_local_id]
    pair_vec_pos = l_prefix[pair_cluster_idx] + cand_local_id
    pair_vid = list_ids_flat[pair_vec_pos]
    pair_q = q[pair_qid]
    pair_v = list_vectors[pair_vec_pos]
    pair_score = (pair_q * pair_v).sum(dim=1)
    pair_dist = q_norm[pair_qid] + list_vector_norms[pair_vec_pos] - 2.0 * pair_score
    pair_dist.clamp_min_(0.0)
    # All squared L2 distances are non-negative, so their FP32 IEEE bit order
    # is numeric order. Packing the query ID above the distance lets one sort
    # replace the old distance sort followed by a stable query-ID sort.
    distance_bits = pair_dist.view(torch.int32).to(torch.int64)
    pair_keys = (pair_qid << 32) | distance_bits
    return pair_keys, pair_vid


@lru_cache(maxsize=1)
def _get_decode_pairs_and_distance_fn():
    try:
        return torch.compile(_decode_pairs_and_distance_impl, fullgraph=False, dynamic=True)
    except Exception:
        return _decode_pairs_and_distance_impl


def _decode_pairs_and_distance_mode() -> str:
    fn = _get_decode_pairs_and_distance_fn()
    return "compiled" if fn is not _decode_pairs_and_distance_impl else "eager"


def _get_interleaved_flat_lists(
    index: IVFFlatIndex,
    list_vectors: torch.Tensor,
    list_ids_flat: torch.Tensor,
    list_prefix: torch.Tensor,
    list_count: torch.Tensor,
    search_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retrieve the persisted cache or build the resident interleaved layout."""
    has_persisted_layout = (
        index.persisted_list_vectors_interleaved is not None
        and index.persisted_list_ids_interleaved is not None
        and index.persisted_list_group_prefix is not None
    )

    def layout_matches_current_interleaving(vectors: torch.Tensor | None) -> bool:
        """Reject legacy padded-float4 caches for scalar-tail dimensions."""
        if vectors is None or vectors.ndim != 4:
            return False
        dim = int(list_vectors.shape[1])
        vector_width = interleaved_vector_width(dim, list_vectors.dtype)
        return (
            vectors.shape[1] == dim // vector_width
            and vectors.shape[2] == INTERLEAVED_GROUP_SIZE
            and vectors.shape[3] == vector_width
        )

    has_compatible_persisted_layout = has_persisted_layout and layout_matches_current_interleaving(
        index.persisted_list_vectors_interleaved
    )

    def persisted_layout() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert index.persisted_list_vectors_interleaved is not None
        assert index.persisted_list_ids_interleaved is not None
        assert index.persisted_list_group_prefix is not None
        return (
            index.persisted_list_vectors_interleaved.to(
                dtype=torch.float32,
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
        )

    if index.resident_device is not None and index.resident_device == search_device:
        if (
            index.list_vectors_interleaved is None
            or index.list_ids_interleaved is None
            or index.list_group_prefix is None
            or not layout_matches_current_interleaving(index.list_vectors_interleaved)
        ):
            if has_compatible_persisted_layout:
                (
                    index.list_vectors_interleaved,
                    index.list_ids_interleaved,
                    index.list_group_prefix,
                ) = persisted_layout()
            else:
                layout = build_interleaved_flat_lists(list_vectors, list_ids_flat, list_prefix, list_count)
                index.list_vectors_interleaved = layout.vectors
                index.list_ids_interleaved = layout.ids
                index.list_group_prefix = layout.group_prefix
        return index.list_vectors_interleaved, index.list_ids_interleaved, index.list_group_prefix

    if has_compatible_persisted_layout:
        return persisted_layout()

    layout = build_interleaved_flat_lists(list_vectors, list_ids_flat, list_prefix, list_count)
    return layout.vectors, layout.ids, layout.group_prefix


_DECODE_MODE_LOGGED = False


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
    list_vectors: torch.Tensor | None
    list_vector_norms: torch.Tensor | None
    list_ids_flat: torch.Tensor | None
    list_prefix: torch.Tensor | None
    list_count: torch.Tensor | None
    # Optional resident-only cuVS-style `[group, chunk, 32, veclen]` cache.
    list_vectors_interleaved: torch.Tensor | None = None
    list_ids_interleaved: torch.Tensor | None = None
    list_group_prefix: torch.Tensor | None = None
    # Optional CPU copy loaded from a dual-layout index. It is transferred only
    # when ``triton_topk`` requests the interleaved layout.
    persisted_list_vectors_interleaved: torch.Tensor | None = None
    persisted_list_ids_interleaved: torch.Tensor | None = None
    persisted_list_group_prefix: torch.Tensor | None = None
    resident_device: torch.device | None = None

    def reside_on_device(self, device: torch.device | str) -> IVFFlatIndex:
        """Move search-critical tensors onto a target device and keep them resident."""
        if self.list_vectors is None or self.list_ids_flat is None or self.list_prefix is None or self.list_count is None:
            raise ValueError("Index flat lists are not loaded. Call load_ivfflat_index(..., load_lists=True).")

        target_device = torch.device(device)
        self.centroids = self.centroids.to(dtype=torch.float32, device=target_device, copy=False)
        self.list_vectors = self.list_vectors.to(dtype=torch.float32, device=target_device, copy=False)
        self.list_vector_norms = self.list_vector_norms.to(dtype=torch.float32, device=target_device, copy=False)
        self.list_ids_flat = self.list_ids_flat.to(dtype=torch.int64, device=target_device, copy=False)
        self.list_prefix = self.list_prefix.to(dtype=torch.int64, device=target_device, copy=False)
        self.list_count = self.list_count.to(dtype=torch.int64, device=target_device, copy=False)
        self.list_vectors_interleaved = None
        self.list_ids_interleaved = None
        self.list_group_prefix = None
        self.resident_device = target_device
        return self

    def release_device_residency(self) -> IVFFlatIndex:
        """Move resident search tensors back to CPU."""
        if self.list_vectors is not None:
            self.centroids = self.centroids.to(dtype=torch.float32, device="cpu", copy=False)
            self.list_vectors = self.list_vectors.to(dtype=torch.float32, device="cpu", copy=False)
            self.list_vector_norms = self.list_vector_norms.to(dtype=torch.float32, device="cpu", copy=False)
            self.list_ids_flat = self.list_ids_flat.to(dtype=torch.int64, device="cpu", copy=False)
            self.list_prefix = self.list_prefix.to(dtype=torch.int64, device="cpu", copy=False)
            self.list_count = self.list_count.to(dtype=torch.int64, device="cpu", copy=False)
        self.list_vectors_interleaved = None
        self.list_ids_interleaved = None
        self.list_group_prefix = None
        self.resident_device = None
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
    """Persist a simple IVF-Flat index layout to a directory or single file."""
    output_path = Path(output_dir)

    vectors_cpu = build_result.vectors.cpu()
    labels_cpu = build_result.labels.cpu().to(torch.int64)
    centroids_cpu = build_result.centroids.cpu()
    ids = torch.arange(vectors_cpu.shape[0], dtype=torch.int64)

    if labels_cpu.numel() > 0:
        sort_perm = torch.argsort(labels_cpu, stable=True)
        labels_sorted = labels_cpu[sort_perm]
        list_vectors = vectors_cpu[sort_perm].to(torch.float32)
        list_ids_flat = ids[sort_perm]
        list_vector_norms = (list_vectors ** 2).sum(dim=1)
        list_count = torch.bincount(labels_sorted, minlength=centroids_cpu.shape[0]).to(torch.int64)
        list_prefix = torch.cumsum(list_count, dim=0) - list_count
    else:
        dim = int(centroids_cpu.shape[1])
        list_vectors = torch.empty((0, dim), dtype=torch.float32)
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
        "cluster_sizes": cluster_sizes,
        "storage_format": "single_file" if is_single_file else "directory_flat",
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
            "ids": ids,
            "list_vectors": list_vectors,
            "list_vector_norms": list_vector_norms,
            "list_ids_flat": list_ids_flat,
            "list_prefix": list_prefix,
            "list_count": list_count,
        }
        if interleaved_layout is not None:
            payload.update(
                {
                    "list_vectors_interleaved": interleaved_layout.vectors,
                    "list_ids_interleaved": interleaved_layout.ids,
                    "list_group_prefix": interleaved_layout.group_prefix,
                }
            )
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
    torch.save(ids, out_dir / "ids.pt")
    torch.save(list_vectors, out_dir / "list_vectors.pt")
    torch.save(list_vector_norms, out_dir / "list_vector_norms.pt")
    torch.save(list_ids_flat, out_dir / "list_ids_flat.pt")
    torch.save(list_prefix, out_dir / "list_prefix.pt")
    torch.save(list_count, out_dir / "list_count.pt")
    if interleaved_layout is not None:
        torch.save(interleaved_layout.vectors, out_dir / "list_vectors_interleaved.pt")
        torch.save(interleaved_layout.ids, out_dir / "list_ids_interleaved.pt")
        torch.save(interleaved_layout.group_prefix, out_dir / "list_group_prefix.pt")

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    index_size_bytes = sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())
    metadata["index_size_bytes"] = index_size_bytes
    metadata["index_size_mb"] = round(index_size_bytes / (1024 * 1024), 4)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def load_ivfflat_index(index_dir: str | Path, load_lists: bool = True
) -> IVFFlatIndex:
    """Load a saved IVF-Flat index directory or single-file archive."""
    index_path = Path(index_dir)
    if not index_path.exists():
        raise FileNotFoundError(f"Index directory not found: {index_path}")

    if index_path.is_file():
        payload = torch.load(index_path, map_location="cpu")
        metadata = payload["metadata"]
        assignments = payload["assignments"]
        centroids = payload["centroids"]
        ids = payload["ids"]
        if load_lists:
            list_vectors = payload["list_vectors"].to(torch.float32)
            list_vector_norms = payload["list_vector_norms"].to(torch.float32)
            list_ids_flat = payload["list_ids_flat"].to(torch.int64)
            list_prefix = payload["list_prefix"].to(torch.int64)
            list_count = payload["list_count"].to(torch.int64)
            persisted_list_vectors_interleaved = payload.get("list_vectors_interleaved")
            persisted_list_ids_interleaved = payload.get("list_ids_interleaved")
            persisted_list_group_prefix = payload.get("list_group_prefix")
            if persisted_list_vectors_interleaved is not None:
                persisted_list_vectors_interleaved = persisted_list_vectors_interleaved.to(torch.float32)
                persisted_list_ids_interleaved = persisted_list_ids_interleaved.to(torch.int64)
                persisted_list_group_prefix = persisted_list_group_prefix.to(torch.int64)
            lists = []
        else:
            list_vectors = None
            list_vector_norms = None
            list_ids_flat = None
            list_prefix = None
            list_count = None
            persisted_list_vectors_interleaved = None
            persisted_list_ids_interleaved = None
            persisted_list_group_prefix = None
            lists = []

        return IVFFlatIndex(
            index_dir=index_path,
            metadata=metadata,
            vectors=None,
            assignments=assignments,
            centroids=centroids,
            ids=ids,
            lists=lists,
            list_vectors=list_vectors,
            list_vector_norms=list_vector_norms,
            list_ids_flat=list_ids_flat,
            list_prefix=list_prefix,
            list_count=list_count,
            persisted_list_vectors_interleaved=persisted_list_vectors_interleaved,
            persisted_list_ids_interleaved=persisted_list_ids_interleaved,
            persisted_list_group_prefix=persisted_list_group_prefix,
        )

    metadata_path = index_path / "metadata.json"
    assignments_path = index_path / "assignments.pt"
    centroids_path = index_path / "centroids.pt"
    ids_path = index_path / "ids.pt"
    list_vectors_path = index_path / "list_vectors.pt"
    list_vector_norms_path = index_path / "list_vector_norms.pt"
    list_ids_flat_path = index_path / "list_ids_flat.pt"
    list_prefix_path = index_path / "list_prefix.pt"
    list_count_path = index_path / "list_count.pt"
    list_vectors_interleaved_path = index_path / "list_vectors_interleaved.pt"
    list_ids_interleaved_path = index_path / "list_ids_interleaved.pt"
    list_group_prefix_path = index_path / "list_group_prefix.pt"
    lists_dir = index_path / "lists"

    required_files = [
        metadata_path,
        assignments_path,
        centroids_path,
        ids_path,
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing IVF-Flat index files: {missing_files}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vectors = None
    assignments = torch.load(assignments_path, map_location="cpu")
    centroids = torch.load(centroids_path, map_location="cpu")
    ids = torch.load(ids_path, map_location="cpu")

    if load_lists:
        flat_layout_paths = [
            list_vectors_path,
            list_vector_norms_path,
            list_ids_flat_path,
            list_prefix_path,
            list_count_path,
        ]
        if all(path.exists() for path in flat_layout_paths):
            list_vectors = torch.load(list_vectors_path, map_location="cpu").to(torch.float32)
            list_vector_norms = torch.load(list_vector_norms_path, map_location="cpu").to(torch.float32)
            list_ids_flat = torch.load(list_ids_flat_path, map_location="cpu").to(torch.int64)
            list_prefix = torch.load(list_prefix_path, map_location="cpu").to(torch.int64)
            list_count = torch.load(list_count_path, map_location="cpu").to(torch.int64)
            lists = []
        else:
            if not lists_dir.exists():
                raise FileNotFoundError(
                    f"Index list payload not found: expected flat layout files in {index_path} "
                    f"or legacy list directory {lists_dir}"
                )
            lists = [torch.load(path, map_location="cpu") for path in sorted(lists_dir.glob("list_*.pt"))]
            list_count = torch.tensor([item["ids"].numel() for item in lists], dtype=torch.int64)
            list_prefix = torch.cumsum(list_count, dim=0) - list_count
            non_empty_lists = [item for item in lists if item["ids"].numel() > 0]
            if non_empty_lists:
                list_vectors = torch.cat([item["vectors"] for item in non_empty_lists], dim=0).to(torch.float32)
                if all("vector_norms" in item for item in non_empty_lists):
                    list_vector_norms = torch.cat(
                        [item["vector_norms"] for item in non_empty_lists], dim=0
                    ).to(torch.float32)
                else:
                    list_vector_norms = (list_vectors ** 2).sum(dim=1)
                list_ids_flat = torch.cat([item["ids"] for item in non_empty_lists], dim=0).to(torch.int64)
            else:
                list_vectors = torch.empty((0, centroids.shape[1]), dtype=torch.float32)
                list_vector_norms = torch.empty((0,), dtype=torch.float32)
                list_ids_flat = torch.empty((0,), dtype=torch.int64)

        interleaved_layout_paths = [
            list_vectors_interleaved_path,
            list_ids_interleaved_path,
            list_group_prefix_path,
        ]
        if all(path.exists() for path in interleaved_layout_paths):
            persisted_list_vectors_interleaved = torch.load(
                list_vectors_interleaved_path,
                map_location="cpu",
            ).to(torch.float32)
            persisted_list_ids_interleaved = torch.load(
                list_ids_interleaved_path,
                map_location="cpu",
            ).to(torch.int64)
            persisted_list_group_prefix = torch.load(
                list_group_prefix_path,
                map_location="cpu",
            ).to(torch.int64)
        else:
            persisted_list_vectors_interleaved = None
            persisted_list_ids_interleaved = None
            persisted_list_group_prefix = None
    else:
        lists = []
        list_vectors = None
        list_vector_norms = None
        list_ids_flat = None
        list_prefix = None
        list_count = None
        persisted_list_vectors_interleaved = None
        persisted_list_ids_interleaved = None
        persisted_list_group_prefix = None

    return IVFFlatIndex(
        index_dir=index_path,
        metadata=metadata,
        vectors=vectors,
        assignments=assignments,
        centroids=centroids,
        ids=ids,
        lists=lists,
        list_vectors=list_vectors,
        list_vector_norms=list_vector_norms,
        list_ids_flat=list_ids_flat,
        list_prefix=list_prefix,
        list_count=list_count,
        persisted_list_vectors_interleaved=persisted_list_vectors_interleaved,
        persisted_list_ids_interleaved=persisted_list_ids_interleaved,
        persisted_list_group_prefix=persisted_list_group_prefix,
    )


def search_ivfflat_index(
    index: IVFFlatIndex,
    queries: torch.Tensor,
    topk: int = 10,
    nprobe: int = 10,
    device: torch.device | str | None = None,
    profile: dict[str, float] | None = None,
    selection_mode: str = "tensor_topk",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search IVF-Flat with the Triton or Tensor selection path."""

    search_device: torch.device | None = None
    cuda_timing_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    global _DECODE_MODE_LOGGED
    debug_compile = os.environ.get("TENSORVS_DEBUG_COMPILE") == "1"
    if debug_compile and not _DECODE_MODE_LOGGED:
        mode = _decode_pairs_and_distance_mode()
        print(f"[tensorvs] decode_pairs_and_distance mode: {mode}")
        _DECODE_MODE_LOGGED = True

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
    if index.list_vectors is None or index.list_ids_flat is None or index.list_prefix is None or index.list_count is None:
        raise ValueError("Index flat lists are not loaded. Call load_ivfflat_index(..., load_lists=True).")
    if selection_mode not in {
        "triton_topk",
        "tensor_topk",
    }:
        raise ValueError(f"Unsupported IVFFlat selection_mode: {selection_mode}")

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
    Q = queries.to(dtype=torch.float32, device=search_device, copy=False).contiguous()
    if index.resident_device is not None and index.resident_device == search_device:
        centroids = index.centroids
        list_vectors = index.list_vectors
        list_vector_norms = index.list_vector_norms
        list_ids_flat = index.list_ids_flat
        list_prefix = index.list_prefix
        list_count = index.list_count
    else:
        centroids = index.centroids.to(dtype=torch.float32, device=search_device, copy=False)
        list_vectors = index.list_vectors.to(dtype=torch.float32, device=search_device, copy=False)
        list_vector_norms = index.list_vector_norms.to(dtype=torch.float32, device=search_device, copy=False)
        list_ids_flat = index.list_ids_flat.to(dtype=torch.int64, device=search_device, copy=False)
        list_prefix = index.list_prefix.to(dtype=torch.int64, device=search_device, copy=False)
        list_count = index.list_count.to(dtype=torch.int64, device=search_device, copy=False)
    record_timing("prepare_tensors", step_start)

    nprobe = max(1, min(int(nprobe), centroids.shape[0]))
    topk = max(1, int(topk))
    num_queries = Q.shape[0]
    step_start = start_timing()
    result_distances = torch.full((num_queries, topk), float("inf"), dtype=torch.float32, device=search_device)
    result_ids = torch.full((num_queries, topk), -1, dtype=torch.int64, device=search_device)
    record_timing("init_outputs", step_start)

    # Step 1: 计算查询向量与所有质心的平方L2距离矩阵
    # Compute the full query-centroid squared L2 distance matrix:
    #   dist_qc[b, c] = ||Q[b] - C[c]||^2
    # Then keep only the nearest `nprobe` centroids per query.
    step_start = start_timing()
    dist_qc = squared_l2_distance(Q, centroids)
    record_timing("centroid_distance", step_start)
    step_start = start_timing()
    _, probe_cids = torch.topk(dist_qc, k=nprobe, dim=1, largest=False)
    record_timing("centroid_topk", step_start)

    active_selection_mode = selection_mode
    if selection_mode == "triton_topk":
        if can_run_ivfflat_triton_topk(
            device=search_device,
            topk=topk,
            num_queries=num_queries,
            nprobe=nprobe,
            dim=Q.shape[1],
        ):
            step_start = start_timing()
            interleaved_vectors, interleaved_ids, list_group_prefix = _get_interleaved_flat_lists(
                index,
                list_vectors,
                list_ids_flat,
                list_prefix,
                list_count,
                search_device,
            )
            record_timing("triton_prepare_layout", step_start)

            step_start = start_timing()
            topk_distances, topk_ids = triton_topk(
                queries=Q,
                list_vectors_interleaved=interleaved_vectors,
                list_ids_interleaved=interleaved_ids,
                list_group_prefix=list_group_prefix,
                list_count=list_count,
                probe_cids=probe_cids,
                topk=topk,
            )
            result_distances.copy_(topk_distances)
            result_ids.copy_(topk_ids)
            record_timing("triton_topk_select", step_start)
        else:
            # The Tensor path is the portable fallback for unsupported
            # Triton devices or shapes.
            active_selection_mode = "tensor_topk"

    if active_selection_mode == "tensor_topk":
        # Expand the selected IVF lists in cluster-major order, then globally
        # sort packed (query, distance) keys to select each query's top-k.
        step_start = start_timing()
        edge_qid = torch.arange(num_queries, device=search_device, dtype=torch.int64).repeat_interleave(nprobe)
        edge_cid = probe_cids.reshape(-1).to(torch.int64)
        record_timing("expand_probe_edges", step_start)

        step_start = start_timing()
        edge_cid_sorted, perm = torch.sort(edge_cid)
        edge_qid_sorted = edge_qid[perm]
        unique_clusters, q_count = torch.unique_consecutive(edge_cid_sorted, return_counts=True)
        q_prefix = torch.cumsum(q_count, dim=0) - q_count
        record_timing("group_probes_by_cluster", step_start)

        step_start = start_timing()
        l_prefix = list_prefix[unique_clusters]
        l_count = list_count[unique_clusters]
        active_mask = l_count > 0
        if active_mask.any():
            q_count = q_count[active_mask]
            q_prefix = q_prefix[active_mask]
            l_prefix = l_prefix[active_mask]
            l_count = l_count[active_mask]
        record_timing("lookup_active_lists", step_start)

        if active_mask.any():
            _search_ivfflat_pair_expand_impl(
                Q=Q,
                list_vectors=list_vectors,
                list_vector_norms=list_vector_norms,
                list_ids_flat=list_ids_flat,
                q_count=q_count,
                q_prefix=q_prefix,
                l_prefix=l_prefix,
                l_count=l_count,
                edge_qid_sorted=edge_qid_sorted,
                result_distances=result_distances,
                result_ids=result_ids,
                topk=topk,
                search_device=search_device,
                start_timing=start_timing,
                record_timing=record_timing,
            )

    flush_timing()
    return result_distances, result_ids


def _search_ivfflat_pair_expand_impl(
    *,
    Q: torch.Tensor,
    list_vectors: torch.Tensor,
    list_vector_norms: torch.Tensor,
    list_ids_flat: torch.Tensor,
    q_count: torch.Tensor,
    q_prefix: torch.Tensor,
    l_prefix: torch.Tensor,
    l_count: torch.Tensor,
    edge_qid_sorted: torch.Tensor,
    result_distances: torch.Tensor,
    result_ids: torch.Tensor,
    topk: int,
    search_device: torch.device,
    start_timing,
    record_timing,
) -> None:
    step_start = start_timing()
    pair_count = q_count * l_count
    M = int(pair_count.sum().item())
    record_timing("count_pairs", step_start)
    if M == 0:
        return

    step_start = start_timing()
    pair_prefix = torch.cumsum(pair_count, dim=0) - pair_count
    pair_pos = torch.arange(M, device=search_device, dtype=torch.int64)
    pair_cluster_idx = torch.bucketize(pair_pos, pair_prefix[1:], right=True)
    record_timing("expand_candidate_pairs", step_start)

    step_start = start_timing()
    q_norm = (Q ** 2).sum(dim=1)
    decode_pairs_and_distance = _get_decode_pairs_and_distance_fn()
    pair_keys, pair_vid = decode_pairs_and_distance(
        Q,
        q_norm,
        list_vectors,
        list_vector_norms,
        list_ids_flat,
        q_prefix,
        l_prefix,
        l_count,
        edge_qid_sorted,
        pair_prefix,
        pair_cluster_idx,
        pair_pos,
    )
    record_timing("decode_pairs_and_distance", step_start)

    step_start = start_timing()
    sorted_keys, perm = torch.sort(pair_keys)
    qid_sorted = sorted_keys >> 32
    distance_bits_sorted = (sorted_keys & 0xFFFFFFFF).to(torch.int32)
    dist_sorted = distance_bits_sorted.view(torch.float32)
    vid_sorted = pair_vid[perm]
    record_timing("sort_by_query_and_distance", step_start)

    step_start = start_timing()
    _, q_pair_count = torch.unique_consecutive(qid_sorted, return_counts=True)
    q_pair_prefix = torch.cumsum(q_pair_count, dim=0) - q_pair_count
    record_timing("group_sorted_queries", step_start)

    step_start = start_timing()
    pos = torch.arange(M, device=search_device, dtype=torch.int64)
    record_timing("topk_segment_positions", step_start)

    step_start = start_timing()
    segment_idx = torch.bucketize(pos, q_pair_prefix[1:], right=True)
    record_timing("topk_bucketize_segments", step_start)

    step_start = start_timing()
    rank_in_query = pos - q_pair_prefix[segment_idx]
    keep = rank_in_query < topk
    record_timing("topk_rank_and_mask", step_start)
    if not keep.any():
        return

    step_start = start_timing()
    topk_qid = qid_sorted[keep]
    topk_rank = rank_in_query[keep]
    result_ids[topk_qid, topk_rank] = vid_sorted[keep]
    result_distances[topk_qid, topk_rank] = dist_sorted[keep]
    record_timing("topk_writeback", step_start)


def build_ivfflat_index(
    vectors: torch.Tensor,
    nlist: int,
    niter: int,
    use_triton: bool,
    device: torch.device | str,
    seed: int = 0,
    n_init: int = 2,
) -> IVFFlatBuildResult:
    """Train a reproducible, multi-initialization IVF-Flat coarse quantizer."""
    if vectors.ndim != 2:
        raise ValueError(f"Dataset tensor must have shape (num_vectors, dim), got {tuple(vectors.shape)}")
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
    )

    return IVFFlatBuildResult(
        vectors=vectors,
        labels=trained.labels,
        centroids=trained.centroids,
        device=target_device,
        seed=int(seed),
        n_init=int(n_init),
        coarse_inertia=trained.inertia,
        coarse_seed=trained.seed,
    )
