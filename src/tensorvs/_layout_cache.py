"""Logical IVF lists and disposable, device-specific execution layouts.

The archive's interleaved encoding remains the default cache (and old archives
remain readable). Logical lists contain no padding or lane/warp assumptions.
Converted layouts are sidecars; they never modify the source index.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

LAYOUT_VERSION = 1


def normalize_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def device_fingerprint(device: torch.device | str) -> dict[str, Any]:
    device = normalize_device(device)
    key: dict[str, Any] = {
        "device": str(device), "torch": str(torch.__version__), "layout_version": LAYOUT_VERSION,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        key.update(name=props.name, capability=[props.major, props.minor],
                   processors=props.multi_processor_count,
                   runtime=torch.version.hip or torch.version.cuda)
    return key


@dataclass
class LogicalLists:
    """Contiguous list-major rows; offsets/counts describe variable-length lists."""
    kind: str
    data: torch.Tensor
    ids: torch.Tensor
    auxiliary: torch.Tensor  # squared vector norms or residual-PQ corrections
    counts: torch.Tensor
    offsets: torch.Tensor

    def to(self, device: torch.device) -> LogicalLists:
        return LogicalLists(self.kind, *(getattr(self, name).to(device=device) for name in
                            ("data", "ids", "auxiliary", "counts", "offsets")))


@dataclass
class TorchLayout:
    lists: LogicalLists
    host_counts: tuple[int, ...]
    host_offsets: tuple[int, ...]
    tensors: dict[str, torch.Tensor]
    fingerprint: dict[str, Any]
    row_list_ids: torch.Tensor | None = None
    pq_scan_upper_bounds: tuple[int, ...] = ()
    pq_anchor_grain: int = 0
    pq_has_empty_lists: bool = False
    pq_adc_words: torch.Tensor | None = None


@dataclass
class LayoutCache:
    logical: LogicalLists | None = None
    torch_layout: TorchLayout | None = None
    triton_fingerprint: dict[str, Any] | None = None
    persisted_keys: set[str] = field(default_factory=set)


def _cache(index: Any) -> LayoutCache:
    return index.layout_cache


def _source_signature(index: Any) -> dict[str, Any]:
    source = Path(index.index_dir)
    paths = [source] if source.is_file() else sorted(source.glob("*.pt"))
    if source.is_dir() and (source / "metadata.json").exists():
        paths.append(source / "metadata.json")
    return {"path": str(source.resolve()),
            "files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths],
            "vectors": index.metadata.get("num_vectors"),
            "dim": index.metadata.get("dim")}


def _cache_path(index: Any, backend: str, fingerprint: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    signature = {"source": _source_signature(index), "backend": backend,
                 "fingerprint": fingerprint}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(os.environ.get("TENSORVS_LAYOUT_CACHE_DIR", Path.home() / ".cache" / "tensorvs" / "layouts"))
    return root / f"{digest}.pt", signature


def _load(path: Path, signature: dict[str, Any]) -> Any | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["signature"] == signature:
            return payload["layout"]
    except (OSError, RuntimeError, EOFError, ValueError, KeyError, pickle.UnpicklingError):
        pass  # A partial or obsolete cache is rebuilt from the source archive.
    return None


def _store(path: Path, signature: dict[str, Any], layout: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save({"signature": signature, "layout": layout}, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def logical_lists(index: Any, kind: str) -> LogicalLists:
    cache = _cache(index)
    if cache.logical is not None:
        return cache.logical
    if index.list_count is None:
        raise ValueError("IVF lists are not loaded; load the index with load_lists=True")
    counts = index.list_count.detach().cpu().to(torch.int64)
    offsets = counts.cumsum(0) - counts
    prefix = index.persisted_list_group_prefix
    ids = index.persisted_list_ids_interleaved
    if prefix is None:
        prefix, ids = index.list_group_prefix, index.list_ids_interleaved
    if prefix is None or ids is None:
        raise ValueError("The persisted IVF list encoding is incomplete")
    prefix, ids = prefix.detach().cpu(), ids.detach().cpu()
    total = int(counts.sum())
    groups = torch.repeat_interleave(prefix, counts, output_size=total)
    row_offsets = torch.arange(total) - torch.repeat_interleave(offsets, counts, output_size=total)
    group_size = int(index.metadata.get("interleaved_layout", {}).get("group_size", 32))
    physical = groups * group_size + row_offsets
    if kind == "flat":
        data = index.persisted_list_vectors_interleaved
        auxiliary = index.persisted_list_vector_norms_interleaved
        if data is None:
            data, auxiliary = index.list_vectors_interleaved, index.list_vector_norms_interleaved
        if data is None:
            raise ValueError("IVF-Flat vectors are not loaded")
        # Decode in bounded chunks, avoiding a second full padded-vector copy.
        source = data.detach().cpu()
        decoded = torch.empty((total, index.centroids.shape[1]), dtype=source.dtype)
        chunk = max(1, (32 << 20) // max(1, source.shape[1] * source.shape[3] * source.element_size()))
        for start in range(0, total, chunk):
            p = physical[start:start + chunk]
            decoded[start:start + chunk] = source[p // group_size, :, p % group_size, :].reshape(-1, decoded.shape[1])
        aux = auxiliary.detach().cpu()[physical] if auxiliary is not None else decoded.square().sum(1)
    else:
        source = index.persisted_list_codes_interleaved_by_subspace
        auxiliary = index.persisted_list_pq_static_corrections_interleaved
        if source is None or auxiliary is None:
            raise ValueError("IVF-PQ codes or residual corrections are not loaded")
        decoded = source.detach().cpu()[:, physical].T.contiguous()
        aux = auxiliary.detach().cpu()[physical]
    cache.logical = LogicalLists(kind, decoded, ids[physical].contiguous(), aux.contiguous(), counts, offsets)
    return cache.logical


def prepare_torch_layout(index: Any, kind: str, device: torch.device | str,
                         *, persist: bool = True) -> TorchLayout:
    device = normalize_device(device)
    fingerprint = device_fingerprint(device)
    cache = _cache(index)
    if index.resident_device is not None:
        # Only one execution layout is held internally on GPUs. Keep CPU
        # sources/caches so switching back can upload without rebuilding.
        cached_torch = cache.torch_layout
        index.release_device_residency()
        cache.torch_layout = cached_torch
    if cache.torch_layout is not None and cache.torch_layout.fingerprint == fingerprint:
        return cache.torch_layout
    path, signature = _cache_path(index, "torch", fingerprint)
    payload = _load(path, signature) if persist else None
    if payload is not None:
        logical = LogicalLists(kind, **payload)
        cache.logical = logical
    else:
        logical = logical_lists(index, kind)
        if persist:
            _store(path, signature, {name: getattr(logical, name) for name in
                                    ("data", "ids", "auxiliary", "counts", "offsets")})
    names = ["centroids", "centroid_norms"]
    if kind == "pq":
        names += ["pq_codebooks", "pq_codebook_norms", "pq_rotation_matrix", "pq_centroids"]
    tensors = {name: getattr(index, name).to(device=device) for name in names
               if getattr(index, name, None) is not None}
    layout = TorchLayout(logical.to(device), tuple(logical.counts.tolist()),
                         tuple(logical.offsets.tolist()), tensors, fingerprint)
    if kind == "pq":
        # Query-independent bounds: any nprobe distinct lists contain at most
        # the sum of the nprobe longest lists. Build once from host metadata,
        # so a query never needs to read its GPU prefix sums back to the CPU.
        bounds, total = [0], 0
        for count in sorted(layout.host_counts, reverse=True):
            total += count
            bounds.append(total)
        layout.pq_scan_upper_bounds = tuple(bounds)
        positive = [count for count in layout.host_counts if count]
        minimum = min(positive, default=0)
        # One grain can cross at most one positive-length list boundary.
        # Very short lists retain general binary-search positioning.
        layout.pq_anchor_grain = (1 << min(10, minimum.bit_length() - 1)
                                  if minimum >= 32 else 0)
        layout.pq_has_empty_lists = 0 in layout.host_counts
    cache.torch_layout = layout  # Keep only the most recent torch device resident.
    return layout


def record_triton_layout(index: Any, device: torch.device | str) -> None:
    """Persist a device manifest referencing the unchanged default encoding."""
    fingerprint = device_fingerprint(device)
    cache = _cache(index)
    if cache.triton_fingerprint == fingerprint:
        return
    path, signature = _cache_path(index, "triton", fingerprint)
    key = str(path)
    if key not in cache.persisted_keys:
        if _load(path, signature) is None:
            _store(path, signature, {"encoding": "archive_interleaved", "group_size": 32})
        cache.persisted_keys.add(key)
    cache.triton_fingerprint = fingerprint


def prepare_pq_adc_words(layout: TorchLayout) -> torch.Tensor | None:
    """Read four byte codes per int32 word; retain logical rows for other scans.

    Divisible, contiguous rows on little-endian hosts can alias the original
    bytes. Other row widths use explicit, portable shifts once per layout.
    Signed words are decoded with a byte mask, including the high byte.
    """
    data = layout.lists.data
    if data.dtype != torch.uint8 or data.shape[1] < 4:
        return None
    if layout.pq_adc_words is None:
        if data.is_contiguous() and data.shape[1] % 4 == 0 and sys.byteorder == "little":
            try:
                layout.pq_adc_words = data.view(torch.int32)
            except RuntimeError:
                pass  # A backend without dtype views can still pack tensors.
        if layout.pq_adc_words is None:
            words = []
            for start in range(0, data.shape[1], 4):
                word = data[:, start].to(torch.int32)
                for byte in range(1, min(4, data.shape[1] - start)):
                    word = word | (data[:, start + byte].to(torch.int32) << (8 * byte))
                words.append(word)
            layout.pq_adc_words = torch.stack(words, 1).contiguous()
    return layout.pq_adc_words


def packed_pq_layout(index: Any, device: torch.device, pack: Any) -> torch.Tensor:
    """Reuse the converted packed cache after reload or a device change."""
    path, signature = _cache_path(index, "triton_packed_pq", device_fingerprint(device))
    packed = _load(path, signature)
    if packed is None:
        packed = pack(index.persisted_list_codes_interleaved_by_subspace)
        _store(path, signature, packed)
    return packed
