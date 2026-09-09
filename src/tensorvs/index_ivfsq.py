from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from .index_ivfflat import _coarse_squared_l2_distance_addmm
from ._ivf_interleaved import (
    build_interleaved_pq_lists,
    build_interleaved_pq_scalar_cache,
)
from flash_kmeans import (
    FAISS_DEFAULT_MAX_POINTS_PER_CENTROID,
    FAISS_DEFAULT_MIN_POINTS_PER_CENTROID,
    FAISS_DEFAULT_NREDO,
    FAISS_DEFAULT_SEED,
    fit_best_kmeans,
)
from .refine import refine_search_results
from ._ivfsq_triton import (
    can_run_query_major_topk as can_run_ivfsq_query_major_topk,
    can_run_topk as can_run_ivfsq_triton_topk,
    is_available as ivfsq_triton_available,
    triton_query_major_topk as triton_ivfsq_query_major_topk,
    triton_topk as triton_ivfsq_topk,
)


def _validate_sq_bits(bits: int) -> int:
    bits = int(bits)
    if bits not in (4, 8):
        raise ValueError(f"IVFSQ bits must be 4 or 8, got {bits}")
    return bits


def _pack_sq_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit scalar codes into each uint8 byte."""
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2D, got shape {tuple(codes.shape)}")
    num_vectors, dim = codes.shape
    packed_dim = (dim + 1) // 2
    packed = torch.zeros((num_vectors, packed_dim), dtype=torch.uint8, device=codes.device)
    low = codes[:, 0::2].to(torch.uint8)
    packed[:, : low.shape[1]] = low
    high = codes[:, 1::2]
    if high.shape[1] > 0:
        packed[:, : high.shape[1]] |= high.to(torch.uint8) << 4
    return packed


def _unpack_sq_codes(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack a row-major uint8 nibble layout back to ``[..., dim]`` codes."""
    if packed.ndim < 1:
        raise ValueError("packed codes must have at least one dimension")
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    return codes[..., :dim]


def _build_sq_triton_layout(
    *,
    list_codes: torch.Tensor,
    list_ids_flat: torch.Tensor,
    list_prefix: torch.Tensor,
    list_count: torch.Tensor,
    centroids: torch.Tensor,
    sq_min: torch.Tensor,
    sq_scale: torch.Tensor,
    bits: int,
    dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    torch.Tensor,
]:
    """Build a resident code-by-dimension layout for the Triton scan."""
    scalar_codes = _unpack_sq_codes(list_codes, dim) if bits == 4 else list_codes
    scalar_codes = scalar_codes.to(dtype=torch.uint8, copy=False).contiguous()
    list_ids = torch.repeat_interleave(
        torch.arange(list_count.numel(), device=list_count.device, dtype=torch.int64),
        list_count,
    )
    sq_base = centroids + sq_min
    scaled_residuals = scalar_codes.to(torch.float32) * sq_scale[list_ids]
    codes_interleaved = build_interleaved_pq_lists(
        scalar_codes,
        list_ids_flat,
        list_prefix,
        list_count,
    )
    residuals_interleaved = build_interleaved_pq_lists(
        scaled_residuals,
        list_ids_flat,
        list_prefix,
        list_count,
    )
    vector_norms = (scaled_residuals * scaled_residuals).sum(dim=1)
    interleaved_norms = build_interleaved_pq_scalar_cache(
        vector_norms,
        list_prefix,
        list_count,
    )
    max_list_group_count = int(
        torch.div(
            list_count.max() + 31,
            32,
            rounding_mode="floor",
        ).item()
    ) if list_count.numel() > 0 else 0
    return (
        codes_interleaved.codes_by_subspace,
        codes_interleaved.ids,
        codes_interleaved.group_prefix,
        interleaved_norms,
        max_list_group_count,
        sq_base,
        residuals_interleaved.codes_by_subspace,
    )


def _tensor_topk_sq_distance_impl(
    queries: torch.Tensor,
    candidate_codes: torch.Tensor,
    candidate_list_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    centroids: torch.Tensor,
    sq_min: torch.Tensor,
    sq_scale: torch.Tensor,
    list_ids_flat: torch.Tensor,
    candidate_positions: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode residual SQ candidates, compute L2, and reduce per query.

    The candidate matrix is query-major and padded only to the largest
    selected-list length in the batch.  Invalid columns are mapped to a safe
    row and masked before top-k, so empty IVF lists need no special kernel.
    """
    list_min = sq_min[candidate_list_ids]
    list_scale = sq_scale[candidate_list_ids]
    candidate_vectors = (
        centroids[candidate_list_ids]
        + list_min
        + candidate_codes.to(torch.float32) * list_scale
    )
    distances = ((queries[:, None, :] - candidate_vectors) ** 2).sum(dim=2)
    distances.masked_fill_(~valid_mask, float("inf"))
    topk_distances, topk_positions = torch.topk(distances, k=topk, dim=1, largest=False)
    topk_vector_positions = torch.gather(candidate_positions, 1, topk_positions)
    topk_ids = list_ids_flat[topk_vector_positions]
    topk_valid = torch.gather(valid_mask, 1, topk_positions)
    topk_ids.masked_fill_(~topk_valid, -1)
    return topk_distances, topk_ids


@lru_cache(maxsize=1)
def _get_tensor_topk_sq_distance_fn():
    try:
        return torch.compile(_tensor_topk_sq_distance_impl, fullgraph=True, dynamic=True)
    except Exception:
        return _tensor_topk_sq_distance_impl


@dataclass
class IVFSQIndex:
    """In-memory residual IVF scalar-quantized index."""

    index_dir: Path
    metadata: dict[str, Any]
    vectors: torch.Tensor | None
    assignments: torch.Tensor
    centroids: torch.Tensor
    ids: torch.Tensor
    sq_min: torch.Tensor
    sq_scale: torch.Tensor
    list_codes: torch.Tensor | None
    list_ids_flat: torch.Tensor | None
    list_prefix: torch.Tensor | None
    list_count: torch.Tensor | None
    centroid_norms: torch.Tensor | None = None
    # Resident-only layouts used by the Triton scans. The quantized code cache
    # is uint8; the decoded FP32 residual cache is used by query-major scan.
    # Neither cache is persisted.
    sq_codes_interleaved: torch.Tensor | None = None
    sq_ids_interleaved: torch.Tensor | None = None
    sq_group_prefix: torch.Tensor | None = None
    sq_vector_norms_interleaved: torch.Tensor | None = None
    sq_max_list_group_count: int | None = None
    sq_base: torch.Tensor | None = None
    sq_residuals_interleaved: torch.Tensor | None = None
    resident_device: torch.device | None = None

    def __post_init__(self) -> None:
        self.centroids = self.centroids.to(dtype=torch.float32, copy=False)
        self.sq_min = self.sq_min.to(dtype=torch.float32, copy=False)
        self.sq_scale = self.sq_scale.to(dtype=torch.float32, copy=False)
        if self.centroid_norms is None:
            self.centroid_norms = (self.centroids * self.centroids).sum(dim=-1)
        else:
            self.centroid_norms = self.centroid_norms.to(dtype=torch.float32, copy=False)
        _validate_sq_bits(self.bits)

    @property
    def bits(self) -> int:
        return _validate_sq_bits(int(self.metadata["sq_bits"]))

    @property
    def dim(self) -> int:
        return int(self.metadata["dim"])

    def reside_on_device(self, device: torch.device | str) -> IVFSQIndex:
        """Move search-critical tensors onto a target device and keep them resident."""
        if (
            self.list_codes is None
            or self.list_ids_flat is None
            or self.list_prefix is None
            or self.list_count is None
        ):
            raise ValueError("Index SQ lists are not loaded. Call load_ivfsq_index(..., load_lists=True).")
        target_device = torch.device(device)
        self.centroids = self.centroids.to(dtype=torch.float32, device=target_device, copy=False)
        self.centroid_norms = self.centroid_norms.to(dtype=torch.float32, device=target_device, copy=False)
        self.sq_min = self.sq_min.to(dtype=torch.float32, device=target_device, copy=False)
        self.sq_scale = self.sq_scale.to(dtype=torch.float32, device=target_device, copy=False)
        self.list_codes = self.list_codes.to(device=target_device, copy=False)
        self.list_ids_flat = self.list_ids_flat.to(dtype=torch.int64, device=target_device, copy=False)
        self.list_prefix = self.list_prefix.to(dtype=torch.int64, device=target_device, copy=False)
        self.list_count = self.list_count.to(dtype=torch.int64, device=target_device, copy=False)
        (
            self.sq_codes_interleaved,
            self.sq_ids_interleaved,
            self.sq_group_prefix,
            self.sq_vector_norms_interleaved,
            self.sq_max_list_group_count,
            self.sq_base,
            self.sq_residuals_interleaved,
        ) = _build_sq_triton_layout(
            list_codes=self.list_codes,
            list_ids_flat=self.list_ids_flat,
            list_prefix=self.list_prefix,
            list_count=self.list_count,
            centroids=self.centroids,
            sq_min=self.sq_min,
            sq_scale=self.sq_scale,
            bits=self.bits,
            dim=self.dim,
        ) if ivfsq_triton_available(target_device) else (None, None, None, None, None, None, None)
        self.resident_device = target_device
        return self

    def release_device_residency(self) -> IVFSQIndex:
        """Move resident search tensors back to CPU."""
        self.centroids = self.centroids.to(dtype=torch.float32, device="cpu", copy=False)
        self.centroid_norms = self.centroid_norms.to(dtype=torch.float32, device="cpu", copy=False)
        self.sq_min = self.sq_min.to(dtype=torch.float32, device="cpu", copy=False)
        self.sq_scale = self.sq_scale.to(dtype=torch.float32, device="cpu", copy=False)
        if self.list_codes is not None:
            self.list_codes = self.list_codes.to(device="cpu", copy=False)
        if self.list_ids_flat is not None:
            self.list_ids_flat = self.list_ids_flat.to(dtype=torch.int64, device="cpu", copy=False)
        if self.list_prefix is not None:
            self.list_prefix = self.list_prefix.to(dtype=torch.int64, device="cpu", copy=False)
        if self.list_count is not None:
            self.list_count = self.list_count.to(dtype=torch.int64, device="cpu", copy=False)
        self.sq_codes_interleaved = None
        self.sq_ids_interleaved = None
        self.sq_group_prefix = None
        self.sq_vector_norms_interleaved = None
        self.sq_max_list_group_count = None
        self.sq_base = None
        self.sq_residuals_interleaved = None
        self.resident_device = None
        return self


@dataclass
class IVFSQBuildResult:
    """In-memory result of IVF-SQ training before serialization."""

    vectors: torch.Tensor
    labels: torch.Tensor
    centroids: torch.Tensor
    sq_min: torch.Tensor
    sq_scale: torch.Tensor
    sq_codes: torch.Tensor
    bits: int
    device: torch.device
    seed: int = 0
    n_init: int = 1
    coarse_inertia: float | None = None
    coarse_seed: int | None = None


def build_ivfsq_index(
    vectors: torch.Tensor,
    nlist: int,
    niter: int,
    use_triton: bool,
    device: torch.device | str,
    bits: int = 8,
    seed: int = FAISS_DEFAULT_SEED,
    n_init: int = FAISS_DEFAULT_NREDO,
    *,
    nbits: int | None = None,
    max_points_per_centroid: int | None = FAISS_DEFAULT_MAX_POINTS_PER_CENTROID,
    min_points_per_centroid: int = FAISS_DEFAULT_MIN_POINTS_PER_CENTROID,
) -> IVFSQBuildResult:
    """Build a residual IVF-SQ index with 4-bit or 8-bit scalar codes.

    Each coarse list owns a per-dimension affine quantizer for its residuals:
    ``x ~= centroid[list] + sq_min[list] + code * sq_scale[list]``.
    The stored code tensor is uint8; 4-bit codes are packed two per byte.
    """
    if nbits is not None:
        if int(bits) != 8 and int(bits) != int(nbits):
            raise ValueError(f"bits={bits} conflicts with nbits={nbits}")
        bits = int(nbits)
    bits = _validate_sq_bits(bits)
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
        max_points_per_centroid=max_points_per_centroid,
        min_points_per_centroid=min_points_per_centroid,
    )
    labels = trained.labels.to(dtype=torch.int64, device=target_device, copy=False)
    centroids = trained.centroids.to(dtype=torch.float32, device=target_device, copy=False)
    device_vectors = vectors.to(device=target_device, dtype=torch.float32, copy=False)
    residuals = device_vectors - centroids[labels]

    qmax = float((1 << bits) - 1)
    sq_min = torch.zeros((nlist, dim), dtype=torch.float32, device=target_device)
    sq_scale = torch.ones((nlist, dim), dtype=torch.float32, device=target_device)
    for list_id in range(nlist):
        member_mask = labels == list_id
        if not bool(member_mask.any()):
            continue
        list_residuals = residuals[member_mask]
        list_min = list_residuals.amin(dim=0)
        list_max = list_residuals.amax(dim=0)
        list_width = list_max - list_min
        list_scale = list_width / qmax
        list_scale = torch.where(list_width > 0.0, list_scale, torch.ones_like(list_scale))
        sq_min[list_id] = list_min
        sq_scale[list_id] = list_scale

    codes = ((residuals - sq_min[labels]) / sq_scale[labels]).round()
    codes = codes.clamp_(0.0, qmax).to(torch.uint8)
    sq_codes = _pack_sq_codes(codes) if bits == 4 else codes

    return IVFSQBuildResult(
        vectors=vectors,
        labels=labels,
        centroids=centroids,
        sq_min=sq_min,
        sq_scale=sq_scale,
        sq_codes=sq_codes,
        bits=bits,
        device=target_device,
        seed=int(seed),
        n_init=int(n_init),
        coarse_inertia=trained.inertia,
        coarse_seed=trained.seed,
    )


def save_ivfsq_index(
    build_result: IVFSQBuildResult,
    output_path: str | Path,
    dataset_path: str | Path | None = None,
    build_time_seconds: float | None = None,
) -> Path:
    """Persist an IVF-SQ index to a directory or a single archive file."""
    output_path = Path(output_path)
    bits = _validate_sq_bits(build_result.bits)
    vectors_cpu = build_result.vectors.cpu().to(torch.float32)
    labels_cpu = build_result.labels.cpu().to(torch.int64)
    centroids_cpu = build_result.centroids.cpu().to(torch.float32)
    sq_min_cpu = build_result.sq_min.cpu().to(torch.float32)
    sq_scale_cpu = build_result.sq_scale.cpu().to(torch.float32)
    sq_codes_cpu = build_result.sq_codes.cpu().to(torch.uint8)
    ids = torch.arange(vectors_cpu.shape[0], dtype=torch.int64)
    sort_perm = torch.argsort(labels_cpu, stable=True)
    labels_sorted = labels_cpu[sort_perm]
    list_codes = sq_codes_cpu[sort_perm]
    list_ids_flat = ids[sort_perm]
    list_count = torch.bincount(labels_sorted, minlength=centroids_cpu.shape[0]).to(torch.int64)
    list_prefix = torch.cumsum(list_count, dim=0) - list_count
    centroid_norms_cpu = (centroids_cpu * centroids_cpu).sum(dim=-1)
    is_single_file = output_path.suffix != ""

    metadata: dict[str, Any] = {
        "dataset_path": str(Path(dataset_path)) if dataset_path is not None else "in_memory_tensor",
        "output_dir": str(output_path),
        "num_vectors": int(vectors_cpu.shape[0]),
        "dim": int(vectors_cpu.shape[1]),
        "nlist": int(centroids_cpu.shape[0]),
        "device": str(build_result.device),
        "seed": int(build_result.seed),
        "n_init": int(build_result.n_init),
        "coarse_inertia": build_result.coarse_inertia,
        "coarse_seed": build_result.coarse_seed,
        "sq_bits": bits,
        "sq_qmax": (1 << bits) - 1,
        "sq_code_dtype": "uint8",
        "sq_storage": "packed_nibble" if bits == 4 else "one_byte_per_code",
        "sq_mode": "residual_per_list_affine",
        "stores_full_vectors": False,
        "storage_format": "single_file" if is_single_file else "directory_sq",
        "cluster_sizes": {str(i): int(c) for i, c in enumerate(list_count.tolist())},
    }
    if build_time_seconds is not None:
        metadata["build_time_seconds"] = float(build_time_seconds)

    payload = {
        "metadata": metadata,
        "assignments": labels_cpu,
        "centroids": centroids_cpu,
        "centroid_norms": centroid_norms_cpu,
        "ids": ids,
        "sq_min": sq_min_cpu,
        "sq_scale": sq_scale_cpu,
        "list_codes": list_codes,
        "list_ids_flat": list_ids_flat,
        "list_prefix": list_prefix,
        "list_count": list_count,
    }
    if is_single_file:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        metadata["index_size_bytes"] = output_path.stat().st_size
        metadata["index_size_mb"] = round(metadata["index_size_bytes"] / (1024 * 1024), 4)
        payload["metadata"] = metadata
        torch.save(payload, output_path)
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(labels_cpu, output_path / "assignments.pt")
    torch.save(centroids_cpu, output_path / "centroids.pt")
    torch.save(centroid_norms_cpu, output_path / "centroid_norms.pt")
    torch.save(ids, output_path / "ids.pt")
    torch.save(sq_min_cpu, output_path / "sq_min.pt")
    torch.save(sq_scale_cpu, output_path / "sq_scale.pt")
    torch.save(list_codes, output_path / "list_codes.pt")
    torch.save(list_ids_flat, output_path / "list_ids_flat.pt")
    torch.save(list_prefix, output_path / "list_prefix.pt")
    torch.save(list_count, output_path / "list_count.pt")
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    index_size_bytes = sum(path.stat().st_size for path in output_path.rglob("*") if path.is_file())
    metadata["index_size_bytes"] = index_size_bytes
    metadata["index_size_mb"] = round(index_size_bytes / (1024 * 1024), 4)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def load_ivfsq_index(index_path: str | Path, load_lists: bool = True) -> IVFSQIndex:
    """Load an IVF-SQ directory or single-file archive."""
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index path not found: {index_path}")

    if index_path.is_file():
        payload = torch.load(index_path, map_location="cpu")
        metadata = payload["metadata"]
        assignments = payload["assignments"].to(torch.int64)
        centroids = payload["centroids"].to(torch.float32)
        centroid_norms = payload.get("centroid_norms")
        if centroid_norms is None:
            centroid_norms = (centroids * centroids).sum(dim=-1)
        ids = payload["ids"].to(torch.int64)
        sq_min = payload["sq_min"].to(torch.float32)
        sq_scale = payload["sq_scale"].to(torch.float32)
        if load_lists:
            list_codes = payload["list_codes"].to(torch.uint8)
            list_ids_flat = payload["list_ids_flat"].to(torch.int64)
            list_prefix = payload["list_prefix"].to(torch.int64)
            list_count = payload["list_count"].to(torch.int64)
        else:
            list_codes = list_ids_flat = list_prefix = list_count = None
        return IVFSQIndex(
            index_dir=index_path,
            metadata=metadata,
            vectors=None,
            assignments=assignments,
            centroids=centroids,
            ids=ids,
            sq_min=sq_min,
            sq_scale=sq_scale,
            list_codes=list_codes,
            list_ids_flat=list_ids_flat,
            list_prefix=list_prefix,
            list_count=list_count,
            centroid_norms=centroid_norms,
        )

    required = [
        index_path / "metadata.json",
        index_path / "assignments.pt",
        index_path / "centroids.pt",
        index_path / "ids.pt",
        index_path / "sq_min.pt",
        index_path / "sq_scale.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing IVF-SQ index files: {missing}")
    metadata = json.loads((index_path / "metadata.json").read_text(encoding="utf-8"))
    assignments = torch.load(index_path / "assignments.pt", map_location="cpu").to(torch.int64)
    centroids = torch.load(index_path / "centroids.pt", map_location="cpu").to(torch.float32)
    centroid_norm_path = index_path / "centroid_norms.pt"
    if centroid_norm_path.exists():
        centroid_norms = torch.load(centroid_norm_path, map_location="cpu").to(torch.float32)
    else:
        centroid_norms = (centroids * centroids).sum(dim=-1)
    ids = torch.load(index_path / "ids.pt", map_location="cpu").to(torch.int64)
    sq_min = torch.load(index_path / "sq_min.pt", map_location="cpu").to(torch.float32)
    sq_scale = torch.load(index_path / "sq_scale.pt", map_location="cpu").to(torch.float32)
    if load_lists:
        list_files = ["list_codes.pt", "list_ids_flat.pt", "list_prefix.pt", "list_count.pt"]
        missing_lists = [str(index_path / name) for name in list_files if not (index_path / name).exists()]
        if missing_lists:
            raise FileNotFoundError(f"Missing IVF-SQ list files: {missing_lists}")
        list_codes = torch.load(index_path / "list_codes.pt", map_location="cpu").to(torch.uint8)
        list_ids_flat = torch.load(index_path / "list_ids_flat.pt", map_location="cpu").to(torch.int64)
        list_prefix = torch.load(index_path / "list_prefix.pt", map_location="cpu").to(torch.int64)
        list_count = torch.load(index_path / "list_count.pt", map_location="cpu").to(torch.int64)
    else:
        list_codes = list_ids_flat = list_prefix = list_count = None
    return IVFSQIndex(
        index_dir=index_path,
        metadata=metadata,
        vectors=None,
        assignments=assignments,
        centroids=centroids,
        ids=ids,
        sq_min=sq_min,
        sq_scale=sq_scale,
        list_codes=list_codes,
        list_ids_flat=list_ids_flat,
        list_prefix=list_prefix,
        list_count=list_count,
        centroid_norms=centroid_norms,
    )


@torch.no_grad()
def search_ivfsq_index(
    index: IVFSQIndex,
    queries: torch.Tensor,
    topk: int = 10,
    nprobe: int = 10,
    device: torch.device | str | None = None,
    profile: dict[str, float] | None = None,
    selection_mode: str = "triton_topk",
    refine_vectors: torch.Tensor | None = None,
    refine_factor: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search an IVF-SQ index using Tensor or Triton scanning.

    ``triton_topk`` keeps SQ codes quantized in a resident dimension-major
    cache and decodes them inside list-grouped dot-product tiles.  It avoids
    materializing the query-major ``[Q, candidates, D]`` tensor used by the
    portable ``tensor_topk`` path. ``query_major`` launches one program per
    query/probe partition and scans the resident decoded residual cache.
    """
    if queries.ndim != 2:
        raise ValueError(f"Query tensor must have shape (num_queries, dim), got {tuple(queries.shape)}")
    if queries.shape[1] != index.dim:
        raise ValueError(f"Query dimension {queries.shape[1]} does not match index dimension {index.dim}")
    if (
        index.list_codes is None
        or index.list_ids_flat is None
        or index.list_prefix is None
        or index.list_count is None
    ):
        raise ValueError("Index SQ lists are not loaded. Call load_ivfsq_index(..., load_lists=True).")
    if selection_mode not in {"tensor_topk", "triton_topk", "query_major"}:
        raise ValueError(f"Unsupported IVFSQ selection_mode: {selection_mode}")

    search_device = torch.device(device) if device is not None else (
        queries.device if queries.is_cuda else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    cuda_timing_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def start_timing() -> float | torch.cuda.Event:
        if profile is None:
            return 0.0
        if search_device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(search_device))
            return start_event
        return time.perf_counter()

    def record_timing(name: str, start_time: float | torch.cuda.Event) -> None:
        if profile is None:
            return
        if search_device.type == "cuda":
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream(search_device))
            cuda_timing_events.append((name, start_time, end_event))
        else:
            profile[name] = profile.get(name, 0.0) + (time.perf_counter() - start_time)

    def flush_timing() -> None:
        if profile is None or not cuda_timing_events:
            return
        torch.cuda.synchronize(search_device)
        for name, start_event, end_event in cuda_timing_events:
            profile[name] = profile.get(name, 0.0) + (start_event.elapsed_time(end_event) / 1000.0)

    step_start = start_timing()
    q = queries.to(dtype=torch.float32, device=search_device, copy=False).contiguous()
    if index.resident_device is not None and index.resident_device == search_device:
        centroids = index.centroids
        centroid_norms = index.centroid_norms
        sq_min = index.sq_min
        sq_scale = index.sq_scale
        list_codes = index.list_codes
        list_ids_flat = index.list_ids_flat
        list_prefix = index.list_prefix
        list_count = index.list_count
    else:
        centroids = index.centroids.to(dtype=torch.float32, device=search_device, copy=False)
        centroid_norms = index.centroid_norms.to(dtype=torch.float32, device=search_device, copy=False)
        sq_min = index.sq_min.to(dtype=torch.float32, device=search_device, copy=False)
        sq_scale = index.sq_scale.to(dtype=torch.float32, device=search_device, copy=False)
        list_codes = index.list_codes.to(device=search_device, copy=False)
        list_ids_flat = index.list_ids_flat.to(dtype=torch.int64, device=search_device, copy=False)
        list_prefix = index.list_prefix.to(dtype=torch.int64, device=search_device, copy=False)
        list_count = index.list_count.to(dtype=torch.int64, device=search_device, copy=False)
    record_timing("prepare_tensors", step_start)

    topk = max(1, int(topk))
    refine_factor = int(refine_factor)
    if refine_factor < 1:
        raise ValueError(f"refine_factor must be >= 1, got {refine_factor}")
    if refine_vectors is None and refine_factor != 1:
        raise ValueError("refine_factor requires refine_vectors with the raw dataset")
    search_topk = topk * refine_factor if refine_vectors is not None else topk
    search_topk = max(topk, min(search_topk, int(index.metadata.get("num_vectors", search_topk))))
    nprobe = max(1, min(int(nprobe), centroids.shape[0]))
    num_queries = q.shape[0]
    step_start = start_timing()
    result_distances = torch.full((num_queries, search_topk), float("inf"), dtype=torch.float32, device=search_device)
    result_ids = torch.full((num_queries, search_topk), -1, dtype=torch.int64, device=search_device)
    record_timing("init_outputs", step_start)

    step_start = start_timing()
    dist_qc = _coarse_squared_l2_distance_addmm(q, centroids, centroid_norms)
    record_timing("centroid_distance", step_start)
    step_start = start_timing()
    # Probe order is irrelevant to the list-grouped scan; avoid sorting the
    # selected centroids and only retain the nearest set.
    _, probe_cids = torch.topk(dist_qc, k=nprobe, dim=1, largest=False, sorted=False)
    record_timing("centroid_topk", step_start)
    step_start = start_timing()
    probe_list_prefix = list_prefix[probe_cids]
    probe_list_count = list_count[probe_cids]
    record_timing("lookup_probe_lists", step_start)
    step_start = start_timing()
    # The list-grouped Triton path never materializes a candidate matrix, so
    # do not force a device synchronization just to size the Tensor fallback.
    # ``max_candidates`` is computed lazily below if that fallback is needed.
    max_candidates: int | None = None
    if selection_mode == "tensor_topk" or search_device.type != "cuda":
        max_candidates = int(probe_list_count.sum(dim=1).max().item()) if num_queries else 0
        record_timing("max_candidates", step_start)

    active_selection_mode = selection_mode
    if selection_mode in {"triton_topk", "query_major"} and search_device.type != "cuda":
        active_selection_mode = "tensor_topk"
    elif selection_mode in {"triton_topk", "query_major"} and search_device.type == "cuda":
        step_start = start_timing()
        if (
            index.sq_codes_interleaved is not None
            and index.sq_codes_interleaved.device == search_device
            and index.sq_ids_interleaved is not None
            and index.sq_group_prefix is not None
            and index.sq_vector_norms_interleaved is not None
            and index.sq_max_list_group_count is not None
            and index.sq_base is not None
            and index.sq_residuals_interleaved is not None
        ):
            sq_codes_interleaved = index.sq_codes_interleaved
            sq_ids_interleaved = index.sq_ids_interleaved
            sq_group_prefix = index.sq_group_prefix
            sq_vector_norms_interleaved = index.sq_vector_norms_interleaved
            max_list_group_count = index.sq_max_list_group_count
            sq_base = index.sq_base
            sq_residuals_interleaved = index.sq_residuals_interleaved
        else:
            (
                sq_codes_interleaved,
                sq_ids_interleaved,
                sq_group_prefix,
                sq_vector_norms_interleaved,
                max_list_group_count,
                sq_base,
                sq_residuals_interleaved,
            ) = _build_sq_triton_layout(
                list_codes=list_codes,
                list_ids_flat=list_ids_flat,
                list_prefix=list_prefix,
                list_count=list_count,
                centroids=centroids,
                sq_min=sq_min,
                sq_scale=sq_scale,
                bits=index.bits,
                dim=index.dim,
            )
            index.sq_codes_interleaved = sq_codes_interleaved
            index.sq_ids_interleaved = sq_ids_interleaved
            index.sq_group_prefix = sq_group_prefix
            index.sq_vector_norms_interleaved = sq_vector_norms_interleaved
            index.sq_max_list_group_count = max_list_group_count
            index.sq_base = sq_base
            index.sq_residuals_interleaved = sq_residuals_interleaved
        record_timing("triton_prepare_layout", step_start)

        if selection_mode == "query_major" and can_run_ivfsq_query_major_topk(
            device=search_device,
            topk=search_topk,
            num_queries=num_queries,
            nprobe=nprobe,
            dim=index.dim,
        ):
            step_start = start_timing()
            topk_distances, topk_ids = triton_ivfsq_query_major_topk(
                queries=q,
                residuals_by_dim=sq_residuals_interleaved,
                vector_norms_interleaved=sq_vector_norms_interleaved,
                sq_base=sq_base,
                list_ids_interleaved=sq_ids_interleaved,
                list_group_prefix=sq_group_prefix,
                list_count=list_count,
                probe_cids=probe_cids,
                topk=search_topk,
            )
            result_distances.copy_(topk_distances)
            result_ids.copy_(topk_ids)
            record_timing("query_major_topk_select", step_start)
            active_selection_mode = "done"
        elif selection_mode == "query_major":
            active_selection_mode = "triton_topk"

        if active_selection_mode == "triton_topk" and can_run_ivfsq_triton_topk(
            device=search_device,
            topk=search_topk,
            nprobe=nprobe,
            max_list_group_count=max_list_group_count,
        ):
            step_start = start_timing()
            topk_distances, topk_ids = triton_ivfsq_topk(
                queries=q,
                codes_by_dim=sq_codes_interleaved,
                vector_norms_interleaved=sq_vector_norms_interleaved,
                sq_base=sq_base,
                sq_scale=sq_scale,
                residuals_by_dim=sq_residuals_interleaved,
                list_ids_interleaved=sq_ids_interleaved,
                list_group_prefix=sq_group_prefix,
                list_count=list_count,
                probe_cids=probe_cids,
                topk=search_topk,
                max_list_group_count=max_list_group_count,
            )
            result_distances.copy_(topk_distances)
            result_ids.copy_(topk_ids)
            record_timing("triton_topk_select", step_start)
            active_selection_mode = "done"
        elif active_selection_mode == "triton_topk":
            active_selection_mode = "tensor_topk"

    if active_selection_mode == "tensor_topk":
        if max_candidates is None:
            step_start = start_timing()
            max_candidates = int(probe_list_count.sum(dim=1).max().item()) if num_queries else 0
            record_timing("max_candidates", step_start)
        if max_candidates <= 0:
            flush_timing()
            return result_distances, result_ids
        step_start = start_timing()
        candidate_offsets = torch.arange(max_candidates, device=search_device, dtype=torch.int64)
        candidate_positions = candidate_offsets.unsqueeze(0).expand(num_queries, -1).contiguous()
        candidate_count = probe_list_count.sum(dim=1, keepdim=True)
        valid_mask = candidate_positions < candidate_count
        probe_end = torch.cumsum(probe_list_count, dim=1)
        probe_idx = torch.searchsorted(probe_end, candidate_positions, right=True)
        probe_idx = probe_idx.clamp_max(nprobe - 1)
        probe_start = probe_end - probe_list_count
        list_start = torch.gather(probe_list_prefix, 1, probe_idx)
        local_start = torch.gather(probe_start, 1, probe_idx)
        vector_positions = list_start + candidate_positions - local_start
        safe_vector_positions = vector_positions.masked_fill(~valid_mask, 0)
        candidate_list_ids = torch.gather(probe_cids, 1, probe_idx)
        candidate_list_codes = list_codes[safe_vector_positions]
        record_timing("candidate_map", step_start)

        step_start = start_timing()
        if index.bits == 4:
            candidate_codes = _unpack_sq_codes(candidate_list_codes, index.dim)
        else:
            candidate_codes = candidate_list_codes
        record_timing("sq_code_unpack", step_start)

        selected_topk = min(search_topk, max_candidates)
        distance_fn = _get_tensor_topk_sq_distance_fn()
        step_start = start_timing()
        topk_distances, topk_ids = distance_fn(
            q,
            candidate_codes,
            candidate_list_ids,
            valid_mask,
            centroids,
            sq_min,
            sq_scale,
            list_ids_flat,
            safe_vector_positions,
            selected_topk,
        )
        result_distances[:, :selected_topk].copy_(topk_distances)
        result_ids[:, :selected_topk].copy_(topk_ids)
        record_timing("sq_distance_topk", step_start)

    if refine_vectors is not None:
        step_start = start_timing()
        refined_distances, refined_ids = refine_search_results(
            q,
            result_ids,
            refine_vectors,
            topk,
            device=search_device,
        )
        record_timing("exact_refine", step_start)
        flush_timing()
        return refined_distances, refined_ids
    flush_timing()
    return result_distances, result_ids
