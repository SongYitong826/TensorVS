"""Bounded-memory, list-grouped PyTorch IVF search.

Scheduling stays outside compiled regions. Large GPU batches use dense masked
GEMMs for Flat or query-major packed list scans for PQ. Small/sparse searches
share candidate rows across queries visiting a list. Only block Top-Ks enter
the bounded final/online merge.
"""
from __future__ import annotations

import math
import time
import types
from functools import lru_cache
from typing import Any

import torch

from ._layout_cache import normalize_device, prepare_torch_layout
from .refine import refine_search_results
from .util import get_default_device


def search_device(queries: torch.Tensor, device: torch.device | str | None) -> torch.device:
    """Honor explicit devices and non-CPU queries before selecting a device."""
    if device is None:
        device = queries.device if queries.device.type != "cpu" else get_default_device()
    return normalize_device(device)


def validate_backend(backend: str) -> str:
    if backend not in ("auto", "torch", "triton"):
        raise ValueError(f"Unsupported backend {backend!r}; expected 'auto', 'torch', or 'triton'")
    return backend


def _flat_block(q, data, ids, norms, query_ids, starts, counts, valid_queries,
                candidate_offsets, k: int):
    positions = starts[:, None] + candidate_offsets[None, :]
    valid = candidate_offsets[None, :] < counts[:, None]
    safe_positions = positions.clamp_max(data.shape[0] - 1)
    x = data[safe_positions]
    queries = q[query_ids]
    base = queries.square().sum(-1, keepdim=True) + norms[safe_positions][:, None, :]
    distances = torch.baddbmm(base, queries, x.transpose(1, 2), beta=1.0, alpha=-2.0).clamp_min_(0.0)
    distances.masked_fill_(~(valid[:, None, :] & valid_queries[:, :, None]), float("inf"))
    values, order = distances.topk(k, dim=-1, largest=False, sorted=False)
    selected = ids[safe_positions].unsqueeze(1).expand(-1, query_ids.shape[1], -1).gather(2, order)
    return values, selected.masked_fill(~torch.isfinite(values), -1)


def _flat_dense_block(q, data, ids, norms, labels, allowed, k: int):
    base = q.square().sum(1, keepdim=True) + norms[None, :]
    distances = torch.addmm(base, q, data.T, beta=1.0, alpha=-2.0).clamp_min_(0.0)
    distances.masked_fill_(~allowed[:, labels], float("inf"))
    values, order = distances.topk(min(k, data.shape[0]), largest=False, sorted=False)
    return values, ids[order].masked_fill(~torch.isfinite(values), -1)


def _pq_packed_block(luts, data, ids, corrections, bias, probes, offsets,
                     prefix, previous, ordinals, k: int, subspace_chunk: int):
    # Pack each query's selected lists into a virtual contiguous candidate
    # stream. searchsorted maps stream positions to lists without copying
    # probes to the CPU or building one task per (query, list).
    ranks = torch.searchsorted(
        prefix, ordinals[None, :].expand(prefix.shape[0], -1).contiguous(),
        right=True).clamp_max(probes.shape[1] - 1)
    positions = (offsets[probes.gather(1, ranks)] + ordinals[None, :]
                 - previous.gather(1, ranks)).clamp_max(data.shape[0] - 1)
    distances = torch.zeros_like(positions, dtype=torch.float32)
    for subspace in range(0, data.shape[1], subspace_chunk):
        codes = data[positions, subspace:subspace + subspace_chunk].to(torch.int64)
        if data.dtype == torch.int16:
            codes = codes & 65535
        values = luts[:, subspace:subspace + subspace_chunk].gather(2, codes.transpose(1, 2))
        distances.add_(values.to(torch.float32).sum(1))
        del codes, values
    distances.add_(corrections[positions]).add_(bias.gather(1, ranks)).clamp_min_(0.0)
    distances.masked_fill_(ordinals[None, :] >= prefix[:, -1:], float("inf"))
    values, order = distances.topk(k, largest=False, sorted=False)
    selected = ids[positions.gather(1, order)]
    return values, selected.masked_fill(~torch.isfinite(values), -1)


def _merge_rows(best_values, best_ids, values, ids):
    all_values = torch.cat((best_values, values), 1)
    all_ids = torch.cat((best_ids, ids), 1)
    best, order = all_values.topk(best_values.shape[1], largest=False, sorted=False)
    return best, all_ids.gather(1, order)


def _chunked_topk(scan, size, chunk, best, best_ids, budget, timer, scan_name,
                  use_compile, specialization):
    # Keep only K results per block. Fall back to online merging when even
    # these small partial results would exceed the reserved budget.
    partial_bytes = best.shape[0] * best.shape[1] * (math.ceil(size / chunk) + 1) * 48
    collect = partial_bytes <= budget // 8
    values, ids = [best], [best_ids]
    merge = _compute(_merge_rows, use_compile, specialization)
    for start in range(0, size, chunk):
        begin = timer.begin()
        local_values, local_ids = scan(start)
        timer.end(scan_name, begin)
        if collect:
            values.append(local_values)
            ids.append(local_ids)
        else:
            begin = timer.begin()
            best, best_ids = merge(best, best_ids, local_values, local_ids)
            timer.end("topk_selection", begin)
    begin = timer.begin()
    if collect:
        candidates = torch.cat(values, 1)
        candidate_ids = torch.cat(ids, 1)
        best, order = candidates.topk(best.shape[1], largest=False, sorted=False)
        best_ids = candidate_ids.gather(1, order)
    timer.end("topk_selection", begin)
    return best, best_ids


def _dense_flat_search(batch, layout, probes, best, best_ids, budget, chunk, timer, use_compile):
    begin = timer.begin()
    if layout.row_list_ids is None:
        layout.row_list_ids = torch.repeat_interleave(
            torch.arange(len(layout.host_counts), device=batch.device), layout.lists.counts,
            output_size=layout.lists.data.shape[0])
    allowed = torch.zeros((batch.shape[0], len(layout.host_counts)),
                          dtype=torch.bool, device=batch.device).scatter_(1, probes, True)
    timer.end("list_scheduling_preparation", begin)
    specialization = (str(batch.device), batch.shape[0], batch.shape[1], chunk, best.shape[1])
    fn = _compute(_flat_dense_block, use_compile, specialization)
    def scan(start):
        end = start + chunk
        return fn(batch, layout.lists.data[start:end], layout.lists.ids[start:end],
                  layout.lists.auxiliary[start:end], layout.row_list_ids[start:end],
                  allowed, best.shape[1])
    return _chunked_topk(scan, layout.lists.data.shape[0], chunk, best, best_ids, budget,
                         timer, "dense_vector_distance_topk", use_compile, specialization)


def _packed_pq_search(batch, layout, lut, bias, probes, best, best_ids,
                      budget, chunk, timer, use_compile):
    begin = timer.begin()
    prefix = layout.lists.counts[probes].cumsum(1)
    previous = torch.cat((torch.zeros_like(prefix[:, :1]), prefix[:, :-1]), 1)
    # One scalar synchronization per batch replaces probe-ID download,
    # CPU sorting, ragged task construction and plan uploads.
    size = int(prefix[:, -1].max().item())
    ordinals = torch.arange(chunk, device=batch.device)
    bias = bias.gather(1, probes)
    timer.end("list_scheduling_preparation", begin)
    specialization = (str(batch.device), layout.lists.data.dtype, lut.dtype,
                      batch.shape[0], lut.shape[1], lut.shape[2], chunk, best.shape[1])
    fn = _compute(_pq_packed_block, use_compile, specialization)
    def scan(start):
        return fn(lut, layout.lists.data, layout.lists.ids, layout.lists.auxiliary,
                  bias, probes, layout.lists.offsets, prefix, previous, ordinals + start,
                  min(best.shape[1], chunk), 1 if use_compile else 4)
    return _chunked_topk(scan, size, chunk, best, best_ids, budget, timer,
                         "packed_lut_distance_topk", use_compile, specialization)


def _pq_block(luts, data, ids, corrections, probe_bias, query_ids, list_ids,
              starts, counts, valid_queries, candidate_offsets, k: int, subspace_chunk: int):
    positions = starts[:, None] + candidate_offsets[None, :]
    valid = candidate_offsets[None, :] < counts[:, None]
    safe_positions = positions.clamp_max(data.shape[0] - 1)
    # Signed int16 storage represents unsigned 16-bit PQ codes.
    codes = data[safe_positions].to(torch.int64)
    if data.dtype == torch.int16:
        codes = codes & 65535
    tables = luts[query_ids]  # [tasks, queries, m, ksub]
    # Reuse one [tasks, queries, candidates] accumulation buffer instead of
    # materializing [tasks, queries, subspaces, candidates] in eager mode.
    # Inductor can fuse these gathers and additions into a single scan.
    distances = torch.zeros((query_ids.shape[0], query_ids.shape[1], candidate_offsets.shape[0]),
                            device=luts.device, dtype=torch.float32)
    for subspace in range(0, data.shape[1], subspace_chunk):
        code_indices = codes[:, :, subspace:subspace + subspace_chunk].transpose(1, 2)[:, None, :, :].expand(-1, query_ids.shape[1], -1, -1)
        values = tables[:, :, subspace:subspace + subspace_chunk, :].gather(3, code_indices)
        distances = distances + values.to(torch.float32).sum(2)
    distances = (distances + corrections[safe_positions][:, None, :]
                 + probe_bias[query_ids, list_ids[:, None]][:, :, None]).clamp_min_(0.0)
    distances.masked_fill_(~(valid[:, None, :] & valid_queries[:, :, None]), float("inf"))
    values, order = distances.topk(k, dim=-1, largest=False, sorted=False)
    selected = ids[safe_positions].unsqueeze(1).expand(-1, query_ids.shape[1], -1).gather(2, order)
    return values, selected.masked_fill(~torch.isfinite(values), -1)


def _merge(best_values, best_ids, values, ids, query_ids, valid_queries):
    # A query can occur in several tasks. Give each task a disjoint slot;
    # scatter assignment is unique even when task query IDs repeat.
    tasks, width, k = values.shape
    slots = torch.arange(tasks, device=values.device)[:, None, None] * k
    ranks = torch.arange(k, device=values.device)[None, None, :]
    indices = query_ids[:, :, None] * (tasks * k) + slots + ranks
    # Padded query lanes must also have unique destinations, outside real rows.
    dummy = best_values.shape[0] * tasks * k + torch.arange(tasks * width * k, device=values.device).reshape(tasks, width, k)
    indices = torch.where(valid_queries[:, :, None], indices, dummy)
    size = best_values.shape[0] * tasks * k + tasks * width * k
    partial_values = torch.full((size,), float("inf"), device=values.device, dtype=values.dtype)
    partial_ids = torch.full((size,), -1, device=ids.device, dtype=ids.dtype)
    partial_values.scatter_(0, indices.flatten(), values.flatten())
    partial_ids.scatter_(0, indices.flatten(), ids.flatten())
    partial_values = partial_values[:best_values.shape[0] * tasks * k].view(best_values.shape[0], -1)
    partial_ids = partial_ids[:best_values.shape[0] * tasks * k].view(best_values.shape[0], -1)
    all_values = torch.cat((best_values, partial_values), dim=1)
    all_ids = torch.cat((best_ids, partial_ids), dim=1)
    result, order = all_values.topk(best_values.shape[1], dim=1, largest=False, sorted=False)
    return result, all_ids.gather(1, order)


def _collect(partial_values, partial_ids, values, ids, query_ids, valid_queries,
             row_slots, width: int, num_queries: int):
    tasks, queries, k = values.shape
    ranks = torch.arange(k, device=values.device)[None, None, :]
    indices = query_ids[:, :, None] * width + row_slots[:, :, None] + ranks
    dummy = num_queries * width + torch.arange(tasks * queries * k, device=values.device).reshape(tasks, queries, k)
    indices = torch.where(valid_queries[:, :, None], indices, dummy)
    partial_values.scatter_(0, indices.flatten(), values.flatten())
    partial_ids.scatter_(0, indices.flatten(), ids.flatten())


def _partial_slots(schedule, num_queries: int, k: int):
    slots = [0] * num_queries
    for (_, cw), tasks in schedule.items():
        local_k = min(k, cw)
        for task in tasks:
            row_slots = [slots[q] for q in task[0]]
            for q in task[0]:
                slots[q] += local_k
            task.append(row_slots)
    return max(slots, default=0)


def _flat_collect(q, data, ids, norms, query_ids, starts, counts, valid_queries,
                  candidate_offsets, k, partial_values, partial_ids, row_slots,
                  width, num_queries):
    values, selected = _flat_block(q, data, ids, norms, query_ids, starts, counts,
                                   valid_queries, candidate_offsets, k)
    _collect(partial_values, partial_ids, values, selected, query_ids, valid_queries,
             row_slots, width, num_queries)


def _pq_collect(luts, data, ids, corrections, probe_bias, query_ids, list_ids,
                starts, counts, valid_queries, candidate_offsets, k, subspace_chunk,
                partial_values, partial_ids, row_slots, width, num_queries):
    values, selected = _pq_block(luts, data, ids, corrections, probe_bias, query_ids,
                                 list_ids, starts, counts, valid_queries, candidate_offsets,
                                 k, subspace_chunk)
    _collect(partial_values, partial_ids, values, selected, query_ids, valid_queries,
             row_slots, width, num_queries)


@lru_cache(maxsize=128)
def _compute(function, use_compile: bool, specialization: tuple = ()):
    if not use_compile:
        return function
    if not hasattr(torch, "compile"):
        raise RuntimeError("use_compile=True requires PyTorch with torch.compile support")
    # Dynamo guards belong to a code object. Isolate device/dtype/tile/K
    # families so a legitimate CPU/GPU or shape change does not exhaust the
    # default recompilation limit of one shared Python function.
    code = function.__code__.replace(co_name=function.__name__ + "_specialized")
    isolated = types.FunctionType(code, function.__globals__, function.__name__, function.__defaults__)
    return torch.compile(isolated, fullgraph=True, dynamic=True)


def _schedule(probes: torch.Tensor, counts: tuple[int, ...], offsets: tuple[int, ...],
              query_tile: int, candidate_chunk: int):
    """Bucket tasks by dense shape, using CPU metadata and one probe transfer."""
    host = probes.detach().cpu()
    edges = host.flatten()
    sorted_lists, order = edges.sort(stable=True)
    visiting = (order // host.shape[1]).tolist()
    unique, edge_counts = sorted_lists.unique_consecutive(return_counts=True)
    buckets: dict[tuple[int, int], list[list]] = {}
    cursor = 0
    for list_id, visits in zip(unique.tolist(), edge_counts.tolist()):
        query_ids = visiting[cursor:cursor + visits]
        cursor += visits
        count = counts[list_id]
        if not count:
            continue
        for qs in range(0, visits, query_tile):
            rows = query_ids[qs:qs + query_tile]
            # Avoid dozens of tiny shape families: their launch/guard costs
            # outweigh the small amount of padding in fringe tasks.
            qwidth = max(min(8, query_tile), 1 << (len(rows) - 1).bit_length())
            for vs in range(0, count, candidate_chunk):
                length = min(candidate_chunk, count - vs)
                cwidth = max(min(128, candidate_chunk), 1 << (length - 1).bit_length())
                buckets.setdefault((qwidth, cwidth), []).append([rows, offsets[list_id] + vs, length, list_id])
    return buckets


def _task_tensors(tasks, qwidth: int, device: torch.device):
    # One small upload per task batch, then views for every plan column.
    records = [rows + [0] * (qwidth - len(rows)) + slots + [0] * (qwidth - len(rows))
               + [start, count, list_id, len(rows)]
               for rows, start, count, list_id, slots in tasks]
    plan = torch.tensor(records, dtype=torch.int64)
    if device.type == "cuda":
        plan = plan.pin_memory()
    plan = plan.to(device=device, non_blocking=True)
    query_ids = plan[:, :qwidth]
    valid = torch.arange(qwidth, device=device)[None, :] < plan[:, -1:]
    return (query_ids, plan[:, 2 * qwidth], plan[:, 2 * qwidth + 1],
            plan[:, 2 * qwidth + 2], valid, plan[:, qwidth:2 * qwidth])


class _Timer:
    def __init__(self, device, profile):
        self.device, self.profile = device, profile
        self.events = []

    def begin(self):
        if self.profile is None:
            return None
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return time.perf_counter()

    def end(self, name, begin):
        if begin is None:
            return
        if self.device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(self.device))
            self.events.append((name, begin, end))
        else:
            self.profile[name] = self.profile.get(name, 0.0) + time.perf_counter() - begin

    def finish(self):
        if self.events:
            torch.cuda.synchronize(self.device)
            for name, begin, end in self.events:
                self.profile[name] = self.profile.get(name, 0.0) + begin.elapsed_time(end) / 1000


def search_torch(index: Any, queries: torch.Tensor, *, kind: str, topk: int, nprobe: int,
                 device=None, profile=None, query_batch_size: int = 128,
                 candidate_chunk_size: int = 2048, memory_budget_mb: float = 128,
                 use_compile: bool = False, persist_layout: bool = True,
                 lut_dtype: str = "fp32", refine_vectors=None, refine_factor: int = 1,
                 flat_scan: str = "auto"):
    if flat_scan not in {"auto", "grouped", "dense"}:
        raise ValueError("flat_scan must be 'auto', 'grouped', or 'dense'")
    if queries.ndim != 2 or queries.shape[1] != index.centroids.shape[1]:
        raise ValueError("Queries must have shape [num_queries, index dimension]")
    if index.list_count is None:
        raise ValueError("IVF lists are not loaded; load the index with load_lists=True")
    if query_batch_size <= 0 or candidate_chunk_size <= 0 or not math.isfinite(memory_budget_mb) or memory_budget_mb <= 0:
        raise ValueError("Torch query/candidate chunk sizes and memory budget must be positive")
    device = search_device(queries, device)
    timer = _Timer(device, profile)
    begin = timer.begin()
    layout = prepare_torch_layout(index, kind, device, persist=persist_layout)
    q = queries.to(device=device, dtype=torch.float32).contiguous()
    timer.end("prepare_tensors", begin)
    topk = max(1, int(topk))
    nprobe = max(1, min(int(nprobe), index.centroids.shape[0]))
    refine_factor = int(refine_factor)
    if refine_factor < 1 or (refine_vectors is None and refine_factor != 1):
        raise ValueError("refine_factor must be >= 1 and requires refine_vectors when greater than 1")
    search_k = max(topk, min(topk * refine_factor, layout.lists.data.shape[0])) if refine_vectors is not None else topk
    distances = torch.full((q.shape[0], search_k), float("inf"), device=device)
    result_ids = torch.full((q.shape[0], search_k), -1, dtype=torch.int64, device=device)
    if q.shape[0] == 0 or layout.lists.data.shape[0] == 0:
        timer.finish()
        return distances[:, :topk], result_ids[:, :topk]
    if kind == "flat":
        from .index_ivfflat import _coarse_centroid_scores_addmm
        block_function = _flat_block
    else:
        from .index_ivfpq import (_coarse_centroid_scores_addmm, _apply_pq_rotation,
                                 _build_global_query_luts_bmm, _normalize_lut_dtype)
        lut_dtype = _normalize_lut_dtype(lut_dtype)
        block_function = _pq_block
    budget = int(memory_budget_mb * (1 << 20))
    # IEEE FP32 for Flat and coarse selection, matching the existing path.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        for qs in range(0, q.shape[0], query_batch_size):
            batch = q[qs:qs + query_batch_size]
            begin = timer.begin()
            scores = _coarse_centroid_scores_addmm(batch, layout.tensors["centroids"], layout.tensors["centroid_norms"])
            timer.end("centroid_distance", begin)
            begin = timer.begin()
            probes = scores.topk(nprobe, dim=1, largest=False, sorted=False).indices
            timer.end("centroid_topk", begin)
            if kind == "pq":
                begin = timer.begin()
                pq_q = _apply_pq_rotation(batch, layout.tensors.get("pq_rotation_matrix"))
                m, _, subdim = layout.tensors["pq_codebooks"].shape
                lut = _build_global_query_luts_bmm(pq_q, layout.tensors["pq_codebooks"],
                                                 layout.tensors["pq_codebook_norms"], m=m, subdim=subdim, use_tf32=True)
                lut = lut.permute(1, 0, 2).contiguous().to(torch.float16 if lut_dtype == "fp16" else torch.float32)
                bias = (scores + batch.square().sum(1, keepdim=True)).clamp_min_(0)
                timer.end("LUT_prepare", begin)
            best, best_ids = distances[qs:qs + batch.shape[0]], result_ids[qs:qs + batch.shape[0]]
            # Larger matrix/ADC blocks amortize generic PyTorch launches. Use
            # the existing grouped schedule for small batches, sparse probes,
            # or budgets that cannot hold a reasonably sized dense block.
            reserve = max(budget // 8, 48 * batch.shape[0] * search_k)
            workspace = budget - reserve
            preferred_chunk = candidate_chunk_size * (32 if kind == "flat" else 4)
            bytes_per_candidate = batch.shape[0] * (12 if kind == "flat" else (32 if use_compile else 100)) + 16
            available = workspace - (batch.shape[0] * len(layout.host_counts) if kind == "flat" else 0)
            capacity = min(65536 if kind == "flat" else 8192,
                           preferred_chunk, max(0, available // bytes_per_candidate))
            stream_chunk = 1 << (capacity.bit_length() - 1) if capacity else 0
            if kind == "flat" and flat_scan == "dense" and stream_chunk < 1:
                raise ValueError("Torch memory budget is too small for dense Flat scanning")
            use_dense = kind == "flat" and (flat_scan == "dense" or (
                flat_scan == "auto" and device.type == "cuda" and batch.shape[0] >= 64
                and nprobe * 16 >= len(layout.host_counts) and stream_chunk >= 4096))
            use_packed = (kind == "pq" and device.type == "cuda"
                          and batch.shape[0] >= 32 and stream_chunk >= 256)
            if use_dense or use_packed:
                if use_dense:
                    best, best_ids = _dense_flat_search(batch, layout, probes, best, best_ids,
                                                       budget, stream_chunk, timer, use_compile)
                else:
                    best, best_ids = _packed_pq_search(batch, layout, lut, bias, probes,
                                                      best, best_ids, budget, stream_chunk,
                                                      timer, use_compile)
                begin = timer.begin()
                best, order = best.sort(dim=1)
                distances[qs:qs + batch.shape[0]] = best
                result_ids[qs:qs + batch.shape[0]] = best_ids.gather(1, order)
                timer.end("topk_selection", begin)
                continue
            begin = timer.begin()
            # Shrink the vector tile if one task alone would exceed the budget.
            tile = min(32, query_batch_size)
            tile = 1 << (tile.bit_length() - 1)
            cchunk = 1 << (candidate_chunk_size.bit_length() - 1)
            def estimate(qwidth, cwidth):
                dim = batch.shape[1]
                if kind == "flat":
                    block = 4 * (cwidth * dim + qwidth * dim + 4 * qwidth * cwidth)
                else:
                    if use_compile:
                        # Fused indexing/ADC avoids both expanded code indices
                        # and a task-local copy of every query LUT.
                        block = 4 * cwidth * m + 8 * qwidth * cwidth
                    else:
                        block = 16 * cwidth * m + (20 + 8 * min(m, 8)) * qwidth * cwidth + 4 * qwidth * m * lut.shape[-1]
                merge = 48 * batch.shape[0] * search_k + 48 * qwidth * search_k
                return block + merge
            while estimate(tile, cchunk) > budget and (cchunk > 1 or tile > 1):
                if cchunk > 1:
                    cchunk //= 2
                else:
                    tile //= 2
            if estimate(tile, cchunk) > budget:
                raise ValueError("Torch memory budget is too small for one scan task and its Top-K merge")
            schedule = _schedule(probes, layout.host_counts, layout.host_offsets, tile, cchunk)
            partial_width = _partial_slots(schedule, batch.shape[0], search_k)
            max_tasks = 256
            reservoir_size = batch.shape[0] * partial_width + max_tasks * tile * min(search_k, cchunk)
            reservoir_bytes = reservoir_size * 12
            collect = (partial_width > 0 and reservoir_bytes <= budget // 3
                       and reservoir_bytes + estimate(tile, cchunk) <= budget)
            workspace_budget = budget - reservoir_bytes if collect else budget
            if collect:
                partial_values = torch.full((reservoir_size,), float("inf"), device=device)
                partial_ids = torch.full((reservoir_size,), -1, dtype=torch.int64, device=device)
            timer.end("list_scheduling_preparation", begin)
            for (qw, cw), tasks in schedule.items():
                tasks_per_batch = max(1, min(max_tasks, workspace_budget // estimate(qw, cw)))
                bucket_plan = _task_tensors(tasks, qw, device)
                candidate_offsets = torch.arange(cw, device=device)
                local_k = min(search_k, cw)
                for ts in range(0, len(tasks), tasks_per_batch):
                    query_ids, starts, counts, lists, valid, row_slots = (
                        column[ts:ts + tasks_per_batch] for column in bucket_plan)
                    specialization = (str(device), layout.lists.data.dtype, lut_dtype, batch.shape[1],
                                      layout.lists.data.shape[1],
                                      lut.shape[-1] if kind == "pq" else 0,
                                      qw, cw, local_k, search_k, batch.shape[0], query_ids.shape[0] == 1)
                    begin = timer.begin()
                    if kind == "flat":
                        block_args = (batch, layout.lists.data, layout.lists.ids, layout.lists.auxiliary,
                                      query_ids, starts, counts, valid, candidate_offsets, local_k)
                    else:
                        block_args = (lut, layout.lists.data, layout.lists.ids, layout.lists.auxiliary,
                                      bias, query_ids, lists, starts, counts, valid, candidate_offsets,
                                      local_k, 1 if use_compile else 8)
                    if collect and use_compile:
                        combined = _compute(_flat_collect if kind == "flat" else _pq_collect, True, specialization)
                        combined(*block_args, partial_values, partial_ids, row_slots, partial_width, batch.shape[0])
                        timer.end("vector_read_distance_topk_fused" if kind == "flat" else "lut_lookup_topk_fused", begin)
                        continue
                    block_fn = _compute(block_function, use_compile, specialization)
                    values, ids = block_fn(*block_args)
                    timer.end("vector_read_distance" if kind == "flat" else "lut_lookup_accumulation", begin)
                    begin = timer.begin()
                    if collect:
                        collect_fn = _compute(_collect, use_compile, specialization)
                        collect_fn(partial_values, partial_ids, values, ids, query_ids, valid,
                                   row_slots, partial_width, batch.shape[0])
                    else:
                        merge_fn = _compute(_merge, use_compile, specialization)
                        best, best_ids = merge_fn(best, best_ids, values, ids, query_ids, valid)
                    timer.end("topk_selection", begin)
            begin = timer.begin()
            if collect:
                partial_values = partial_values[:batch.shape[0] * partial_width].view(batch.shape[0], partial_width)
                partial_ids = partial_ids[:batch.shape[0] * partial_width].view(batch.shape[0], partial_width)
                all_values = torch.cat((best, partial_values), 1)
                all_ids = torch.cat((best_ids, partial_ids), 1)
                best, order = all_values.topk(search_k, dim=1, largest=False, sorted=False)
                best_ids = all_ids.gather(1, order)
            best, order = best.sort(dim=1)
            distances[qs:qs + batch.shape[0]] = best
            result_ids[qs:qs + batch.shape[0]] = best_ids.gather(1, order)
            timer.end("topk_selection", begin)
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    if refine_vectors is not None:
        begin = timer.begin()
        distances, result_ids = refine_search_results(q, result_ids, refine_vectors, topk, device=device)
        timer.end("exact_refine", begin)
    timer.finish()
    return distances, result_ids
