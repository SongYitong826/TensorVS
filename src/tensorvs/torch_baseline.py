"""Independent, conventional PyTorch IVF search baselines for comparisons.

Queries are processed one at a time: coarse selection, candidate collection,
distance/ADC evaluation, then a single global Top-K. Ordinary chunks bound
temporary tensors; they do not perform local Top-K or specialized scheduling.
PQ builds squared-L2 LUTs for each probed list's residual query directly.

Only standard torch operators are used. There is no Triton dependency, full
database masked scan, static PQ correction, compiler or execution-layout cache.
Existing indexes can supply their logical, unpadded lists through from_index;
their normal search dispatch and device residency are left untouched.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch

from .util import get_default_device

if TYPE_CHECKING:
    from .index_ivfflat import IVFFlatIndex
    from .index_ivfpq import IVFPQIndex

__all__ = ["TorchIVFFlatBaseline", "TorchIVFPQBaseline"]


def _integer_cpu(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise ValueError(f"{name} must contain integers")
    return tensor.detach().to(device="cpu", dtype=torch.int64)


def _group_rows(data, assignments, centroids, ids):
    """Build ordinary list-major rows once, using portable CPU torch ops."""
    if data.ndim != 2 or centroids.ndim != 2 or min(centroids.shape) < 1:
        raise ValueError("data and nonempty centroids must be 2D")
    labels = _integer_cpu(assignments, "assignments")
    if labels.shape != (data.shape[0],):
        raise ValueError("assignments must have one entry per data row")
    if ((labels < 0) | (labels >= centroids.shape[0])).any():
        raise ValueError("assignments contain an invalid list ID")
    if ids is None:
        ids = torch.arange(data.shape[0])
    ids = _integer_cpu(ids, "ids")
    if ids.shape != labels.shape or (ids < 0).any():
        raise ValueError("ids must contain one nonnegative integer per data row")
    order = labels.argsort(stable=True)
    counts = torch.bincount(labels, minlength=centroids.shape[0])
    return data.detach().cpu()[order], ids[order], counts


class _TorchIVF:
    def _initialize(self, data, ids, counts, centroids, device):
        target = (torch.device(device) if device is not None else
                  data.device if data.device.type != "cpu" else get_default_device())
        self.data = data.detach().to(target).contiguous()
        self.ids = ids.detach().to(device=target, dtype=torch.int64)
        self.counts = counts.detach().to(device=target, dtype=torch.int64)
        self.offsets = self.counts.cumsum(0) - self.counts
        self.centroids = centroids.detach().to(device=target, dtype=torch.float32).contiguous()

    @property
    def device(self) -> torch.device:
        return self.centroids.device

    def to(self, device: torch.device | str):
        """Return a baseline on another device, without changing this instance."""
        result = copy.copy(self)
        for name, tensor in vars(self).items():
            if isinstance(tensor, torch.Tensor):
                setattr(result, name, tensor.to(device))
        return result

    def _candidates(self, probes):
        # Expand just the selected lists, never all database rows. Dynamic
        # repeat_interleave can synchronize internally on accelerators.
        lengths = self.counts[probes]
        ranks = torch.repeat_interleave(torch.arange(probes.numel(), device=self.device), lengths)
        starts = lengths.cumsum(0) - lengths
        positions = (self.offsets[probes][ranks] + torch.arange(ranks.numel(), device=self.device)
                     - starts[ranks])
        return positions, ranks

    def _validate_search(self, queries, topk, nprobe, candidate_chunk_size):
        if queries.ndim != 2 or queries.shape[1] != self.centroids.shape[1]:
            raise ValueError("queries must have shape [num_queries, index dimension]")
        if topk <= 0 or nprobe <= 0 or candidate_chunk_size <= 0:
            raise ValueError("topk, nprobe and candidate_chunk_size must be positive")
        return queries.to(device=self.device, dtype=torch.float32), min(nprobe, self.centroids.shape[0])

    def _topk(self, distances, positions, topk):
        k = min(topk, distances.numel())
        values, order = distances.topk(k, largest=False, sorted=True)
        ids = self.ids[positions[order]]
        if k < topk:
            values = torch.cat((values, values.new_full((topk - k,), float("inf"))))
            ids = torch.cat((ids, ids.new_full((topk - k,), -1)))
        return values, ids


class TorchIVFFlatBaseline(_TorchIVF):
    """Exact squared-L2 search within the traditionally selected IVF lists.

    Supply vectors, their coarse assignments and trained centroids, or use
    from_index to compare against the same existing TensorVS index.
    """

    def __init__(self, vectors: torch.Tensor, assignments: torch.Tensor,
                 centroids: torch.Tensor, ids: torch.Tensor | None = None, *, device=None):
        if vectors.ndim != 2 or centroids.ndim != 2 or vectors.shape[1] != centroids.shape[1]:
            raise ValueError("vectors and centroids must be 2D with the same dimension")
        target = device if device is not None else vectors.device if vectors.device.type != "cpu" else None
        data, ids, counts = _group_rows(vectors, assignments, centroids, ids)
        self._initialize(data.to(torch.float32), ids, counts, centroids, target)

    @classmethod
    def from_index(cls, index: IVFFlatIndex, *, device=None):
        """Read the logical list view; no optimized execution layout is used."""
        lists = index.logical_lists
        result = cls.__new__(cls)
        result._initialize(lists.data.to(torch.float32), lists.ids, lists.counts, index.centroids, device)
        return result

    @torch.inference_mode()
    def search(self, queries: torch.Tensor, topk: int = 10, nprobe: int = 10, *,
               candidate_chunk_size: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
        queries, nprobe = self._validate_search(queries, topk, nprobe, candidate_chunk_size)
        output = queries.new_full((queries.shape[0], topk), float("inf"))
        output_ids = torch.full(output.shape, -1, dtype=torch.int64, device=self.device)
        for row, query in enumerate(queries):
            coarse = (self.centroids - query).square().sum(-1)
            probes = coarse.topk(nprobe, largest=False, sorted=True).indices
            positions, _ = self._candidates(probes)
            distances = query.new_empty(positions.numel())
            for start in range(0, positions.numel(), candidate_chunk_size):
                block = positions[start:start + candidate_chunk_size]
                distances[start:start + block.numel()] = (self.data[block] - query).square().sum(-1)
            output[row], output_ids[row] = self._topk(distances, positions, topk)
        return output, output_ids


class TorchIVFPQBaseline(_TorchIVF):
    """Residual IVF-PQ using per-probe residual LUTs and conventional ADC.

    codes describe residuals relative to each assigned coarse centroid.
    An optional [PQ dimension, input dimension] rotation supports existing
    TensorVS indexes whose PQ working dimension differs from the input.
    """

    def __init__(self, codes: torch.Tensor, assignments: torch.Tensor,
                 centroids: torch.Tensor, codebooks: torch.Tensor,
                 ids: torch.Tensor | None = None, *, rotation: torch.Tensor | None = None,
                 device=None):
        target = device if device is not None else codes.device if codes.device.type != "cpu" else None
        data, ids, counts = _group_rows(codes, assignments, centroids, ids)
        self._initialize_pq(data, ids, counts, centroids, codebooks, rotation, target)

    def _initialize_pq(self, codes, ids, counts, centroids, codebooks, rotation, device):
        if codebooks.ndim != 3 or min(codebooks.shape) < 1 or codes.shape[1] != codebooks.shape[0]:
            raise ValueError("codebooks must have shape [M, codewords, subdimension]; codes must have M columns")
        codes_dtype = codes.dtype
        normalized = _integer_cpu(codes, "codes")
        if codes_dtype == torch.int16:
            normalized = normalized & 65535  # Existing 16-bit archives store unsigned bits in int16.
        if ((normalized < 0) | (normalized >= codebooks.shape[1])).any():
            raise ValueError("codes contain an invalid codeword ID")
        storage = torch.uint8 if codebooks.shape[1] <= 256 else torch.int32
        self._initialize(normalized.to(storage), ids, counts, centroids, device)
        self.codebooks = codebooks.detach().to(device=self.device, dtype=torch.float32).contiguous()
        dim = codebooks.shape[0] * codebooks.shape[2]
        if rotation is None:
            if dim != centroids.shape[1]:
                raise ValueError("PQ working dimension requires a rotation matrix")
            self.rotation = None
            self.pq_centroids = self.centroids
        else:
            if rotation.shape != (dim, centroids.shape[1]):
                raise ValueError("rotation must have shape [PQ dimension, input dimension]")
            self.rotation = rotation.detach().to(device=self.device, dtype=torch.float32)
            self.pq_centroids = self.centroids @ self.rotation.T

    @classmethod
    def from_index(cls, index: IVFPQIndex, *, device=None):
        lists = index.logical_lists
        result = cls.__new__(cls)
        result._initialize_pq(lists.data, lists.ids, lists.counts, index.centroids,
                              index.pq_codebooks, index.pq_rotation_matrix, device)
        if index.pq_centroids is not None:
            result.pq_centroids = index.pq_centroids.detach().to(device=result.device, dtype=torch.float32)
        return result

    @torch.inference_mode()
    def search(self, queries: torch.Tensor, topk: int = 10, nprobe: int = 10, *,
               candidate_chunk_size: int = 4096, lut_probe_chunk_size: int = 8,
               lut_dtype: str = "fp32") -> tuple[torch.Tensor, torch.Tensor]:
        queries, nprobe = self._validate_search(queries, topk, nprobe, candidate_chunk_size)
        if lut_probe_chunk_size <= 0 or lut_dtype not in {"fp32", "fp16"}:
            raise ValueError("lut_probe_chunk_size must be positive; lut_dtype must be 'fp32' or 'fp16'")
        output = queries.new_full((queries.shape[0], topk), float("inf"))
        output_ids = torch.full(output.shape, -1, dtype=torch.int64, device=self.device)
        m, _, subdim = self.codebooks.shape
        subspaces = torch.arange(m, device=self.device)[None, :]
        for row, query in enumerate(queries):
            coarse = (self.centroids - query).square().sum(-1)
            probes = coarse.topk(nprobe, largest=False, sorted=True).indices
            positions, ranks = self._candidates(probes)
            pq_query = query if self.rotation is None else query @ self.rotation.T
            residuals = (pq_query - self.pq_centroids[probes]).reshape(nprobe, m, subdim)
            lut = query.new_empty((nprobe, m, self.codebooks.shape[1]),
                                  dtype=torch.float16 if lut_dtype == "fp16" else torch.float32)
            for start in range(0, nprobe, lut_probe_chunk_size):
                residual = residuals[start:start + lut_probe_chunk_size]
                table = (residual[:, :, None, :] - self.codebooks[None, :, :, :]).square().sum(-1)
                lut[start:start + residual.shape[0]] = table
            distances = query.new_empty(positions.numel())
            for start in range(0, positions.numel(), candidate_chunk_size):
                block = positions[start:start + candidate_chunk_size]
                codes = self.data[block].to(torch.int64)
                values = lut[ranks[start:start + block.numel(), None], subspaces, codes]
                distances[start:start + block.numel()] = values.to(torch.float32).sum(-1)
            output[row], output_ids[row] = self._topk(distances, positions, topk)
        return output, output_ids
