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

from ._coarse import _coarse_centroid_scores_addmm
from ._layout_cache import normalize_device, prepare_pq_adc_words, prepare_torch_layout
from ._pq_math import _apply_pq_rotation, _build_global_query_luts_bmm, _normalize_lut_dtype
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
                candidate_offsets, k: int, query_norms=None):
    positions = starts[:, None] + candidate_offsets[None, :]
    valid = candidate_offsets[None, :] < counts[:, None]
    safe_positions = positions.clamp_max(data.shape[0] - 1)
    # Flatten row gathers so backends can use index_select directly instead
    # of general multidimensional advanced-indexing implementations.
    positions_flat = safe_positions.flatten()
    query_ids_flat = query_ids.flatten()
    x = data.index_select(0, positions_flat).reshape(*safe_positions.shape, data.shape[1])
    queries = q.index_select(0, query_ids_flat).reshape(*query_ids.shape, q.shape[1])
    if query_norms is None:
        qnorms = queries.square().sum(-1, keepdim=True)
    else:
        qnorms = query_norms.index_select(0, query_ids_flat).reshape(*query_ids.shape, 1)
    vector_norms = norms.index_select(0, positions_flat).reshape_as(safe_positions)
    # A query's squared norm is constant across its candidates. Select using
    # ||x||^2 - 2 q.x, then restore ||q||^2 on only K values. This avoids
    # materializing the full broadcast bias and clamping the full score block.
    # Keep FP32 throughout; addition order can change floating-point near-ties.
    distances = torch.baddbmm(vector_norms[:, None, :], queries, x.transpose(1, 2),
                             beta=1.0, alpha=-2.0)
    distances.masked_fill_(~(valid[:, None, :] & valid_queries[:, :, None]), float("inf"))
    values, order = distances.topk(k, dim=-1, largest=False, sorted=False)
    values = (values + qnorms).clamp_min_(0.0)
    # Read IDs only for the selected rows, rather than copying every
    # candidate's ID and broadcasting it over all query lanes.
    selected_positions = (starts[:, None, None] + order).clamp_max(data.shape[0] - 1)
    selected = ids.index_select(0, selected_positions.flatten()).reshape_as(order)
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
    ranks = (torch.zeros((prefix.shape[0], ordinals.shape[0]), device=prefix.device, dtype=torch.int64)
             if probes.shape[1] == 1 else torch.searchsorted(
        prefix, ordinals[None, :].expand(prefix.shape[0], -1).contiguous(),
        right=True).clamp_max(probes.shape[1] - 1))
    valid = ordinals[None, :] < prefix[:, -1:]
    positions = torch.where(valid, offsets[probes.gather(1, ranks)] + ordinals[None, :]
                            - previous.gather(1, ranks), 0)
    distances = torch.zeros_like(positions, dtype=torch.float32)
    for subspace in range(0, data.shape[1], subspace_chunk):
        codes = data[positions, subspace:subspace + subspace_chunk].to(torch.int64)
        if data.dtype == torch.int16:
            codes = codes & 65535
        values = luts[:, subspace:subspace + subspace_chunk].gather(2, codes.transpose(1, 2))
        distances.add_(values.to(torch.float32).sum(1))
        del codes, values
    distances.add_(corrections[positions]).add_(bias.gather(1, ranks)).clamp_min_(0.0)
    distances.masked_fill_(~valid, float("inf"))
    values, order = distances.topk(k, largest=False, sorted=False)
    selected = ids[positions.gather(1, order)]
    return values, selected.masked_fill(~torch.isfinite(values), -1)


def _pq_packed_block_eager(luts, data, ids, corrections, bias, probes, offsets,
                         prefix, previous, ordinals, k: int, subspace_chunk: int):
    """GPU candidate mapping and contiguous ADC using only tensor operators."""
    ranks = (torch.zeros((prefix.shape[0], ordinals.shape[0]), device=prefix.device, dtype=torch.int64)
             if probes.shape[1] == 1 else torch.searchsorted(
                 prefix, ordinals[None, :].expand(prefix.shape[0], -1).contiguous(),
                 right=True).clamp_max(probes.shape[1] - 1))
    lists = probes.gather(1, ranks)
    valid = ordinals[None, :] < prefix[:, -1:]
    positions = torch.where(valid, offsets.index_select(0, lists.flatten()).reshape_as(lists)
                            + ordinals[None, :] - previous.gather(1, ranks), 0)
    index_dtype = torch.int32 if luts.numel() <= 2147483647 else torch.int64
    codes = data.index_select(0, positions.flatten()).reshape(*positions.shape, data.shape[1]).to(index_dtype)
    if data.dtype == torch.int16:
        codes = codes & 65535
    query_offsets = torch.arange(luts.shape[0], device=luts.device, dtype=index_dtype)[:, None, None]
    query_offsets = query_offsets * (luts.shape[1] * luts.shape[2])
    distances = None
    for start in range(0, data.shape[1], subspace_chunk):
        end = min(start + subspace_chunk, data.shape[1])
        subspace = torch.arange(start, end, device=luts.device, dtype=index_dtype) * luts.shape[2]
        indices = query_offsets + subspace[None, None, :] + codes[:, :, start:end]
        values = luts.flatten().index_select(0, indices.flatten()).reshape_as(indices)
        summed = values.sum(-1, dtype=torch.float32)
        if distances is None:
            distances = summed
        else:
            distances.add_(summed)
        del indices, values, summed
    distances.add_(corrections.index_select(0, positions.flatten()).reshape_as(positions))
    # A virtual block can span several lists, whose biases differ, so these
    # must be restored before its Top-K (unlike a single-list grouped block).
    distances.add_(bias.gather(1, ranks)).clamp_min_(0.)
    distances.masked_fill_(~valid, float("inf"))
    values, order = distances.topk(k, largest=False, sorted=False)
    selected = ids.index_select(0, positions.gather(1, order).flatten()).reshape_as(order)
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


def _pq_scan_plan(layout, queries, nprobe, k, budget, chunk, use_compile, query_tile):
    """Choose solely from immutable index statistics and invocation shapes.

    This is a bounded-memory heuristic, not dataset-specific tuning or a
    prediction that a given backend will always run one strategy faster.
    """
    upper = layout.pq_scan_upper_bounds[nprobe]
    grain = ((1 << (max(1, upper) - 1).bit_length()) if nprobe == 1
             else layout.pq_anchor_grain)
    # A single scan block does not amortize constructing a separate map.
    if not use_compile or upper <= chunk:
        grain = 0
    selection, span, selection_bytes = _pq_selection_plan(upper, k, chunk)
    # Anchor maps have five columns per grain, plus construction temporaries.
    mapping_span = max(span, math.ceil(upper / chunk) * chunk)
    mapping_bytes = 64 * math.ceil(mapping_span / grain) if grain else 0
    reserve = max(budget // 8, 48 * queries * k)
    metadata_bytes = 64 * queries * nprobe
    available = budget - reserve - metadata_bytes
    if grain and mapping_bytes * query_tile > max(0, available * 3 // 4):
        grain, mapping_bytes = 0, 0
    mapping_capacity = min(queries, max(1, available // max(1, mapping_bytes))) if grain else queries
    anchor_batch = 1 << (mapping_capacity.bit_length() - 1)
    # Construct a map once for several query tiles. Construction needs more
    # space than the five retained columns (32 bytes per anchor).
    retained_mapping = mapping_bytes * anchor_batch // 2
    aligned = math.ceil(upper / chunk) * chunk
    lane_bytes = 8 if grain else 32  # binary-search ranks need materialization
    if span != aligned and selection == "blocked":
        lane_bytes += 8  # exact scores + the inf-padded selection buffer
    per_query = lane_bytes * span + selection_bytes
    capacity = min(queries, anchor_batch,
                   max(0, (available - retained_mapping - 8 * span) // max(1, per_query)))
    # Whole-span selection needs reasonable parallelism. Otherwise use the
    # bounded streaming schedule, including for large K or tiny budgets.
    tiled = use_compile and capacity >= min(queries, 32)
    width = 1 << (capacity.bit_length() - 1) if tiled else query_tile
    if not tiled:
        selection, span = "stream", upper
        if grain:
            # Account for the map as well as the conservative stream scratch.
            capacity = min(query_tile, anchor_batch, max(1,
                (available - retained_mapping - 8 * chunk) // (32 * chunk)))
            width = 1 << (capacity.bit_length() - 1)
    return dict(mapping="anchors" if grain else "binary", anchor_grain=grain,
                selection=selection, scan_upper_bound=upper, score_span=span,
                query_tile=width, anchor_batch=anchor_batch, candidate_block=chunk,
                estimated_workspace_bytes=metadata_bytes + max(mapping_bytes * anchor_batch + reserve,
                    8 * span + per_query * width + retained_mapping + reserve)
                    if tiled else None)


def _pq_anchor_mapping(layout, probes, prefix, previous, bias, span, grain):
    deltas = layout.lists.offsets[probes] - previous
    starts = torch.arange(math.ceil(span / grain), device=probes.device) * grain
    if probes.shape[1] == 1:
        ranks = torch.zeros((probes.shape[0], starts.shape[0]), dtype=torch.int64,
                            device=probes.device)
    else:
        ranks = torch.searchsorted(prefix, starts[None, :].expand(prefix.shape[0], -1).contiguous(),
                                   right=True).clamp_max(probes.shape[1] - 1)
    following = (ranks + 1).clamp_max(probes.shape[1] - 1)
    return (deltas.gather(1, ranks), deltas.gather(1, following), prefix.gather(1, ranks),
            bias.gather(1, ranks), bias.gather(1, following), prefix[:, -1])


def _pq_narrow_addresses(rows: int, span: int) -> bool:
    # Offsets, selected-list prefixes and their differences are bounded by
    # the index size. Include the ordinal span for an aligned/streaming tail.
    # Shape metadata suffices; no query-dependent device scalar is read back.
    return rows + span <= 2147483647


def _pq_adc_lookup(luts, data, positions, packed_codes: bool):
    distances = torch.zeros_like(positions, dtype=torch.float32)
    if packed_codes:
        for start in range(0, luts.shape[1], 4):
            word = data[positions, start // 4]
            for byte in range(min(4, luts.shape[1] - start)):
                codes = ((word >> (8 * byte)) & 255).to(torch.int64)
                distances = distances + luts[:, start + byte, :].gather(1, codes).to(torch.float32)
    else:
        for subspace in range(luts.shape[1]):
            codes = data[positions, subspace].to(torch.int64)
            if data.dtype == torch.int16:
                codes = codes & 65535
            distances = distances + luts[:, subspace, :].gather(1, codes).to(torch.float32)
    return distances


def _pq_anchor_scores(luts, data, corrections, delta0, delta1, boundaries,
                      bias0, bias1, total, ordinals, grain: int, packed_codes: bool = False):
    if _pq_narrow_addresses(data.shape[0], ordinals.shape[0]):
        delta0, delta1 = delta0.to(torch.int32), delta1.to(torch.int32)
        boundaries, total = boundaries.to(torch.int32), total.to(torch.int32)
        ordinals = ordinals.to(torch.int32)
    bins = ordinals // grain
    following = ordinals[None, :] >= boundaries[:, bins]
    valid = ordinals[None, :] < total[:, None]
    positions = torch.where(valid, ordinals[None, :] + torch.where(
        following, delta1[:, bins], delta0[:, bins]), 0)
    distances = _pq_adc_lookup(luts, data, positions, packed_codes)
    distances = (distances + corrections[positions]
                 + torch.where(following, bias1[:, bins], bias0[:, bins])).clamp_min_(0.)
    return distances.masked_fill(~valid, float("inf"))


def _pq_anchor_scores_packed(luts, data, corrections, delta0, delta1, boundaries,
                             bias0, bias1, total, ordinals, grain: int):
    return _pq_anchor_scores(luts, data, corrections, delta0, delta1, boundaries,
                             bias0, bias1, total, ordinals, grain, packed_codes=True)


def _pq_anchor_ids(ids, delta0, delta1, boundaries, ordinals, grain: int, values):
    valid = torch.isfinite(values)
    ordinals = torch.where(valid, ordinals, 0)
    bins = ordinals // grain
    positions = ordinals + torch.where(ordinals >= boundaries.gather(1, bins),
                                      delta1.gather(1, bins), delta0.gather(1, bins))
    return ids[torch.where(valid, positions, 0)].masked_fill(~valid, -1)


def _pq_binary_scores(luts, data, corrections, bias, probes, offsets,
                      prefix, previous, ordinals):
    ranks = torch.searchsorted(prefix, ordinals[None, :].expand(prefix.shape[0], -1).contiguous(),
                               right=True).clamp_max(probes.shape[1] - 1)
    valid = ordinals[None, :] < prefix[:, -1:]
    positions = torch.where(valid, offsets[probes.gather(1, ranks)] + ordinals[None, :]
                            - previous.gather(1, ranks), 0)
    distances = torch.zeros_like(positions, dtype=torch.float32)
    for subspace in range(data.shape[1]):
        codes = data[positions, subspace:subspace + 1].to(torch.int64)
        if data.dtype == torch.int16:
            codes = codes & 65535
        values = luts[:, subspace:subspace + 1].gather(2, codes.transpose(1, 2))
        distances = distances + values.to(torch.float32).sum(1)
    distances = (distances + corrections[positions] + bias.gather(1, ranks)).clamp_min_(0.)
    return distances.masked_fill(~valid, float("inf"))


def _pq_binary_ids(ids, probes, offsets, prefix, previous, ordinals, values):
    valid = torch.isfinite(values)
    ordinals = torch.where(valid, ordinals, 0)
    ranks = torch.searchsorted(prefix, ordinals.contiguous(), right=True).clamp_max(probes.shape[1] - 1)
    positions = offsets[probes.gather(1, ranks)] + ordinals - previous.gather(1, ranks)
    return ids[torch.where(valid, positions, 0)].masked_fill(~valid, -1)


def _pq_region_width(size, k):
    # Bound the minima array while keeping the final gather small. These are
    # invocation shapes, not query contents or dataset/backend identifiers.
    width = min(1024, max(32, 1 << max(0, ((size + 2047) // 2048 - 1).bit_length())))
    return width if size >= 2048 and k * width * 4 < size else 0


def _pq_selection_plan(size, k, chunk):
    region = _pq_region_width(size, k)
    if region:
        span = math.ceil(size / region) * region
        # Minima, selected regions, int64 gather addresses, masks, gathered
        # scores and Top-K temporaries. Include copies made by eager backends.
        scratch = 16 * (span // region) + 32 * k * region + 48 * k
        return "regions", span, scratch
    selection = "global" if size <= chunk or k * 4 >= chunk else "blocked"
    blocks = math.ceil(size / chunk)
    aligned = blocks * chunk
    span = size if selection == "global" or aligned - size > size // 8 else aligned
    return selection, span, 48 * k * (blocks if selection == "blocked" else 1)


def _pq_select_scores(distances, k, chunk, selection):
    if selection == "global":
        return distances.topk(min(k, distances.shape[1]), largest=False, sorted=False)
    size = distances.shape[1]
    if selection == "regions":
        region = _pq_region_width(size, k)
        if not region:
            return distances.topk(min(k, size), largest=False, sorted=False)
        head = size // region * region
        minima = distances[:, :head].reshape(distances.shape[0], -1, region).amin(-1)
        if head < size:
            minima = torch.cat((minima, distances[:, head:].amin(-1, keepdim=True)), 1)
        # The K regions with smallest minima contain K distinct candidates
        # no worse than ANY excluded region's minimum. Keeping these entire
        # regions therefore preserves exact Top-K distances (ties may pick
        # different IDs). This is not an approximate candidate filter.
        regions = minima.topk(k, largest=False, sorted=False).indices
        ordinals = (regions[:, :, None] * region
                    + torch.arange(region, device=distances.device)[None, None, :]).flatten(1)
        scores = distances.gather(1, ordinals.clamp_max(size - 1))
        scores = scores.masked_fill(ordinals >= size, float("inf"))
        values, which = scores.topk(k, largest=False, sorted=False)
        return values, ordinals.gather(1, which)
    blocks = math.ceil(size / chunk)
    padding = blocks * chunk - size
    if padding:
        # Concatenation is portable; constant_pad_nd currently corrupts wide
        # 2-D tensors on the tested MPS runtime. Only pad scores, never ADC.
        distances = torch.cat((distances, distances.new_full((distances.shape[0], padding),
                                                             float("inf"))), 1)
    values, order = distances.reshape(distances.shape[0], blocks, chunk).topk(
        min(k, chunk), largest=False, sorted=False)
    offsets = torch.arange(blocks, device=distances.device)[None, :, None] * chunk
    ordinals = (order + offsets).flatten(1)
    best, which = values.flatten(1).topk(min(k, size), largest=False, sorted=False)
    return best, ordinals.gather(1, which)


def _packed_pq_search(batch, layout, lut, bias, probes, best, best_ids,
                      budget, chunk, timer, use_compile, query_tile=None, lut_chunk_size=16, pq_lookup="flat"):
    begin = timer.begin()
    adc_data = prepare_pq_adc_words(layout) if use_compile and layout.pq_anchor_grain else None
    adc_function = _pq_anchor_scores_packed if adc_data is not None else _pq_anchor_scores
    if adc_data is None:
        adc_data = layout.lists.data
    query_tile = batch.shape[0] if query_tile is None else query_tile
    plan = _pq_scan_plan(layout, batch.shape[0], probes.shape[1], best.shape[1],
                         budget, chunk, use_compile, query_tile)
    counts = layout.lists.counts[probes]
    if plan["mapping"] == "anchors" and layout.pq_has_empty_lists:
        # Zero-length lists must be last so one grain crosses <= 1 boundary.
        # This GPU permutation preserves the selected candidate set.
        order = counts.argsort(1, descending=True)
        probes, counts = probes.gather(1, order), counts.gather(1, order)
    prefix = counts.cumsum(1)
    previous = torch.cat((torch.zeros_like(prefix[:, :1]), prefix[:, :-1]), 1)
    # Fixed index-only upper bound avoids even a scalar GPU-to-CPU readback.
    # Invalid tail positions are masked on the device; selected lists are
    # scanned in full, including unusually long lists.
    size = plan["scan_upper_bound"]
    ordinals = torch.arange(chunk, device=batch.device)
    bias = bias.gather(1, probes)
    timer.end("list_scheduling_preparation", begin)
    query_tile = plan["query_tile"]
    # Fixed query tiling lets long candidate blocks fit the workspace budget;
    # all slices stay on-device, and neither bound depends on query contents.
    mapping_group = None
    for qs in range(0, batch.shape[0], query_tile):
        qe = min(qs + query_tile, batch.shape[0])
        specialization = (str(batch.device), layout.lists.data.dtype, lut.dtype,
                          qe - qs, lut.shape[1], lut.shape[2], chunk, best.shape[1],
                          plan["selection"], plan["anchor_grain"], plan["score_span"])
        tables, selected_probes = lut[qs:qe], probes[qs:qe]
        selected_bias, ends, starts = bias[qs:qe], prefix[qs:qe], previous[qs:qe]
        if plan["selection"] == "global" and size <= chunk:
            begin = timer.begin()
            fn = _compute(_pq_packed_block, True, specialization)
            values, ids = fn(tables, layout.lists.data, layout.lists.ids, layout.lists.auxiliary,
                selected_bias, selected_probes, layout.lists.offsets, ends, starts,
                torch.arange(size, device=batch.device), min(best.shape[1], size), 1)
            if values.shape[1] < best.shape[1]:
                values, ids = _merge_rows(best[qs:qe], best_ids[qs:qe], values, ids)
            best[qs:qe], best_ids[qs:qe] = values, ids
            timer.end("packed_lut_distance_topk", begin)
            continue
        mapping = None
        if use_compile and plan["mapping"] == "anchors":
            within_group = qs % plan["anchor_batch"]
            if within_group == 0:
                begin = timer.begin()
                group_end = min(qs + plan["anchor_batch"], batch.shape[0])
                mapping_span = (math.ceil(size / chunk) * chunk if plan["selection"] == "stream"
                                else plan["score_span"])
                mapping_group = _pq_anchor_mapping(layout, probes[qs:group_end], prefix[qs:group_end],
                    previous[qs:group_end], bias[qs:group_end], mapping_span, plan["anchor_grain"])
                timer.end("list_scheduling_preparation", begin)
            mapping = tuple(column[within_group:within_group + qe - qs] for column in mapping_group)
        if plan["selection"] != "stream":
            begin = timer.begin()
            span_ordinals = torch.arange(plan["score_span"], device=batch.device)
            if mapping is not None:
                scores_fn = _compute(adc_function, True, specialization)
                scores = scores_fn(tables, adc_data, layout.lists.auxiliary,
                                    *mapping, span_ordinals, plan["anchor_grain"])
            else:
                scores_fn = _compute(_pq_binary_scores, True, specialization)
                scores = scores_fn(tables, layout.lists.data, layout.lists.auxiliary,
                                    selected_bias, selected_probes, layout.lists.offsets,
                                    ends, starts, span_ordinals)
            timer.end("packed_lut_distance", begin)
            begin = timer.begin()
            values, selected_ordinals = _pq_select_scores(scores, best.shape[1], chunk, plan["selection"])
            del scores
            if mapping is not None:
                ids = _compute(_pq_anchor_ids, True, specialization)(layout.lists.ids,
                    *mapping[:3], selected_ordinals, plan["anchor_grain"], values)
            else:
                ids = _compute(_pq_binary_ids, True, specialization)(layout.lists.ids,
                    selected_probes, layout.lists.offsets, ends, starts, selected_ordinals, values)
            if values.shape[1] < best.shape[1]:
                values, ids = _merge_rows(best[qs:qe], best_ids[qs:qe], values, ids)
            best[qs:qe], best_ids[qs:qe] = values, ids
            timer.end("topk_selection", begin)
            continue
        fn = _compute(_pq_packed_block if use_compile or pq_lookup == "gather" else _pq_packed_block_eager, use_compile, specialization)
        def scan(start):
            # A short last block scans exactly the remaining index bound.
            lanes = ordinals[:min(chunk, size - start)] + start
            if mapping is not None:
                values = _compute(adc_function, True, specialization)(tables,
                    adc_data, layout.lists.auxiliary, *mapping, lanes, plan["anchor_grain"])
                values, order = values.topk(min(best.shape[1], lanes.shape[0]), largest=False, sorted=False)
                ids = _compute(_pq_anchor_ids, True, specialization)(layout.lists.ids,
                    *mapping[:3], order + start, plan["anchor_grain"], values)
                return values, ids
            return fn(tables, layout.lists.data, layout.lists.ids, layout.lists.auxiliary,
                      selected_bias, selected_probes, layout.lists.offsets, ends, starts,
                      lanes, min(best.shape[1], lanes.shape[0]), 1 if use_compile else lut_chunk_size)
        values, ids = _chunked_topk(scan, size, chunk, best[qs:qe], best_ids[qs:qe], budget, timer,
                                   "packed_lut_distance_topk", use_compile, specialization)
        best[qs:qe], best_ids[qs:qe] = values, ids
    return best, best_ids


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


def _pq_block_eager(luts, data, ids, corrections, probe_bias, query_ids, list_ids,
                    starts, counts, valid_queries, candidate_offsets, k: int,
                    subspace_chunk: int):
    """ADC with flat LUT indices, avoiding broadcasted four-dimensional gathers."""
    positions = starts[:, None] + candidate_offsets[None, :]
    safe_positions = positions.clamp_max(data.shape[0] - 1)
    positions_flat = safe_positions.flatten()
    # LUT addresses fit in int32 for ordinary batches; retain int64 when
    # the global table exceeds its range. index_select accepts both types.
    index_dtype = torch.int32 if luts.numel() <= 2147483647 else torch.int64
    codes = data.index_select(0, positions_flat).reshape(*positions.shape, data.shape[1]).to(index_dtype)
    if data.dtype == torch.int16:
        codes = codes & 65535
    # A single global LUT is shared by all list tasks. Construct its direct
    # addresses rather than copying whole query tables into every task and
    # gathering from a strided [task, query, subspace, codeword] tensor.
    ksub = luts.shape[2]
    query_offsets = query_ids.to(index_dtype)[:, :, None, None] * (luts.shape[1] * ksub)
    flat_luts = luts.flatten()
    distances = None
    for subspace in range(0, data.shape[1], subspace_chunk):
        end = min(subspace + subspace_chunk, data.shape[1])
        subspace_offsets = torch.arange(subspace, end, device=luts.device, dtype=index_dtype) * ksub
        # Keep a candidate's subspaces contiguous so the reduction reads
        # adjacent LUT values rather than reducing a strided candidate axis.
        indices = (query_offsets + subspace_offsets[None, None, None, :]
                   + codes[:, None, :, subspace:end])
        values = flat_luts.index_select(0, indices.flatten()).reshape_as(indices)
        summed = values.sum(-1, dtype=torch.float32)
        if distances is None:
            distances = summed
        else:
            distances.add_(summed)
        del indices, values, summed
    auxiliary = corrections.index_select(0, positions_flat).reshape_as(positions)
    bias_positions = query_ids * probe_bias.shape[1] + list_ids[:, None]
    biases = probe_bias.flatten().index_select(0, bias_positions.flatten()).reshape_as(query_ids)
    distances.add_(auxiliary[:, None, :])
    valid = candidate_offsets[None, :] < counts[:, None]
    distances.masked_fill_(~(valid[:, None, :] & valid_queries[:, :, None]), float("inf"))
    values, order = distances.topk(k, dim=-1, largest=False, sorted=False)
    # The probe bias is constant for every candidate in this (query,list)
    # task. Restore it on K selected values, preserving FP32 ADC and avoiding
    # a broadcast addition/clamp over the whole candidate block. Floating
    # point near-ties can change order when bias is added after selection.
    values.add_(biases[:, :, None]).clamp_min_(0.)
    selected_positions = (starts[:, None, None] + order).clamp_max(data.shape[0] - 1)
    selected = ids.index_select(0, selected_positions.flatten()).reshape_as(order)
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
              query_tile: int, candidate_chunk: int, *, minimum_query_width: int = 8):
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
            qwidth = max(min(minimum_query_width, query_tile), 1 << (len(rows) - 1).bit_length())
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
        if self.device.type == "npu":
            event = torch.npu.Event(enable_timing=True)
            event.record(torch.npu.current_stream(self.device))
            return event
        if self.device.type == "mps":
            torch.mps.synchronize()
        return time.perf_counter()

    def end(self, name, begin):
        if begin is None:
            return
        if self.device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(self.device))
            self.events.append((name, begin, end))
        elif self.device.type == "npu":
            end = torch.npu.Event(enable_timing=True)
            end.record(torch.npu.current_stream(self.device))
            self.events.append((name, begin, end))
        else:
            if self.device.type == "mps":
                torch.mps.synchronize()
            self.profile[name] = self.profile.get(name, 0.0) + time.perf_counter() - begin

    def finish(self):
        if self.events:
            (torch.npu if self.device.type == "npu" else torch.cuda).synchronize(self.device)
            for name, begin, end in self.events:
                self.profile[name] = self.profile.get(name, 0.0) + begin.elapsed_time(end) / 1000


def _pq_query_lut(batch, layout, lut_dtype):
    pq_q = _apply_pq_rotation(batch, layout.tensors.get("pq_rotation_matrix"))
    m, _, subdim = layout.tensors["pq_codebooks"].shape
    lut = _build_global_query_luts_bmm(pq_q, layout.tensors["pq_codebooks"],
        layout.tensors["pq_codebook_norms"], m=m, subdim=subdim, use_tf32=True)
    lut = lut.permute(1, 0, 2)
    if lut_dtype == "fp16":
        # Cast and reorder in one copy, avoiding an intermediate FP32 LUT.
        return lut.to(dtype=torch.float16, memory_format=torch.contiguous_format)
    # A same-dtype to() can return the original noncontiguous view.
    return lut.contiguous()


def search_torch(index: Any, queries: torch.Tensor, *, kind: str, topk: int, nprobe: int,
                 device=None, profile=None, query_batch_size: int = 128,
                 candidate_chunk_size: int = 2048, memory_budget_mb: float = 128,
                 use_compile: bool = False, persist_layout: bool = True,
                 lut_dtype: str = "fp32", refine_vectors=None, refine_factor: int = 1,
                 flat_scan: str = "auto", pq_scan: str = "auto",
                 query_tile_size: int | None = None, max_tasks_per_batch: int = 256,
                 lut_chunk_size: int = 16, pq_lookup: str = "flat", min_query_tile_size: int = 1):
    if flat_scan not in {"auto", "grouped", "dense"}:
        raise ValueError("flat_scan must be 'auto', 'grouped', or 'dense'")
    if pq_scan not in {"auto", "grouped", "gpu", "bucketed"}:
        raise ValueError("pq_scan must be 'auto', 'grouped', 'gpu', or 'bucketed'")
    if queries.ndim != 2 or queries.shape[1] != index.centroids.shape[1]:
        raise ValueError("Queries must have shape [num_queries, index dimension]")
    if index.list_count is None:
        raise ValueError("IVF lists are not loaded; load the index with load_lists=True")
    if query_batch_size <= 0 or candidate_chunk_size <= 0 or not math.isfinite(memory_budget_mb) or memory_budget_mb <= 0:
        raise ValueError("Torch query/candidate chunk sizes and memory budget must be positive")
    if query_tile_size is not None and (query_tile_size <= 0 or query_tile_size & (query_tile_size - 1)):
        raise ValueError("Torch query tile size must be a positive power of two")
    if max_tasks_per_batch <= 0:
        raise ValueError("Torch max tasks per batch must be positive")
    if lut_chunk_size <= 0:
        raise ValueError("Torch LUT chunk size must be positive")
    if pq_lookup not in {"flat", "gather"}:
        raise ValueError("Torch PQ lookup must be 'flat' or 'gather'")
    if min_query_tile_size <= 0 or min_query_tile_size & (min_query_tile_size - 1):
        raise ValueError("Torch minimum query tile size must be a positive power of two")
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
        block_function = _flat_block
    else:
        lut_dtype = _normalize_lut_dtype(lut_dtype)
        block_function = _pq_block if use_compile or pq_lookup == "gather" else _pq_block_eager
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
                m, ksub, _ = layout.tensors["pq_codebooks"].shape
                # Bucketed scans plan and reorder the small query matrix
                # before building a LUT; never gather the full lookup table.
                lut = None if pq_scan == "bucketed" else _pq_query_lut(batch, layout, lut_dtype)
                lut_numel = batch.shape[0] * m * ksub
                lut_bytes = 2 if lut_dtype == "fp16" else 4
                bias = (scores + batch.square().sum(1, keepdim=True)).clamp_min_(0)
                timer.end("probe_bias_prepare" if pq_scan == "bucketed" else "LUT_prepare", begin)
            best, best_ids = distances[qs:qs + batch.shape[0]], result_ids[qs:qs + batch.shape[0]]
            # Larger matrix/ADC blocks amortize generic PyTorch launches. Use
            # the existing grouped schedule for small batches, sparse probes,
            # or budgets that cannot hold a reasonably sized dense block.
            reserve = max(budget // 8, 48 * batch.shape[0] * search_k)
            workspace = budget - reserve
            preferred_chunk = candidate_chunk_size * (32 if kind == "flat" else 4)
            address_bytes = 8 if pq_lookup == "gather" else (4 if kind == "pq" and lut_numel <= 2147483647 else 8)
            packed_lane_bytes = (32 if use_compile else
                64 + address_bytes * m
                + (address_bytes + lut_bytes + 4) * min(m, lut_chunk_size)) if kind == "pq" else 0
            bytes_per_candidate = batch.shape[0] * (12 if kind == "flat" else packed_lane_bytes) + 16
            available = workspace - (batch.shape[0] * len(layout.host_counts) if kind == "flat" else 0)
            packed_query_tile = batch.shape[0]
            if kind == "pq" and pq_scan in {"gpu", "bucketed"}:
                # Prefer longer candidate blocks, reducing searchsorted,
                # block Top-K and merge calls; trade query width for length.
                desired_chunk = min(8192, preferred_chunk)
                capacity_queries = min(batch.shape[0], max(1,
                    (available // desired_chunk - 16) // packed_lane_bytes))
                packed_query_tile = 1 << (capacity_queries.bit_length() - 1)
                bytes_per_candidate = packed_query_tile * packed_lane_bytes + 16
            capacity = min(65536 if kind == "flat" else 8192,
                           preferred_chunk, max(0, available // bytes_per_candidate))
            stream_chunk = 1 << (capacity.bit_length() - 1) if capacity else 0
            if kind == "flat" and flat_scan == "dense" and stream_chunk < 1:
                raise ValueError("Torch memory budget is too small for dense Flat scanning")
            dense_shape = (device.type == "cuda" and batch.shape[0] >= 64
                           and nprobe * 16 >= len(layout.host_counts)) or (
                device.type == "npu" and batch.shape[0] >= 128
                and nprobe * 16 >= len(layout.host_counts))
            use_dense = kind == "flat" and (flat_scan == "dense" or (
                flat_scan == "auto" and dense_shape and stream_chunk >= 4096))
            use_packed = kind == "pq" and (pq_scan in {"gpu", "bucketed"} or (
                pq_scan == "auto" and device.type == "cuda"
                and batch.shape[0] >= 32 and stream_chunk >= 256))
            if use_packed and stream_chunk < 1:
                raise ValueError("Torch memory budget is too small for GPU PQ scanning")
            if use_dense or use_packed:
                if use_dense:
                    best, best_ids = _dense_flat_search(batch, layout, probes, best, best_ids,
                                                       budget, stream_chunk, timer, use_compile)
                else:
                    if pq_scan == "bucketed":
                        from ._torch_pq_bucketed import search_bucketed_pq
                        best, best_ids = search_bucketed_pq(batch, layout, lut, bias, probes,
                            best, best_ids, budget, stream_chunk, timer, use_compile,
                            lut_dtype=lut_dtype, lut_chunk_size=lut_chunk_size, pq_lookup=pq_lookup)
                    else:
                        best, best_ids = _packed_pq_search(batch, layout, lut, bias, probes,
                                                      best, best_ids, budget, stream_chunk,
                                                      timer, use_compile, packed_query_tile, lut_chunk_size, pq_lookup)
                begin = timer.begin()
                best, order = best.sort(dim=1)
                distances[qs:qs + batch.shape[0]] = best
                result_ids[qs:qs + batch.shape[0]] = best_ids.gather(1, order)
                timer.end("topk_selection", begin)
                continue
            if kind == "flat" and not use_compile:
                # Reuse one norm per query across all lists and scan chunks.
                begin = timer.begin()
                query_norms = batch.square().sum(1)
                timer.end("query_norm_preparation", begin)
            begin = timer.begin()
            # Shrink the vector tile if one task alone would exceed the budget.
            tile = min(query_tile_size or (8 if kind == "pq" and not use_compile else 32),
                       query_batch_size)
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
                        # Bound materialized flat addresses, LUT values and
                        # their FP32 reduction, not just the broadcast views.
                        address_bytes = 8 if pq_lookup == "gather" else (4 if lut.numel() <= 2147483647 else 8)
                        lane_bytes = 32 + (address_bytes + lut.element_size() + 4) * min(m, lut_chunk_size)
                        block = 16 * cwidth * m + lane_bytes * qwidth * cwidth
                        if pq_lookup == "gather":
                            block += qwidth * m * lut.shape[2] * lut.element_size()
                merge = 48 * batch.shape[0] * search_k + 48 * qwidth * search_k
                return block + merge
            while estimate(tile, cchunk) > budget and (cchunk > 1 or tile > 1):
                if cchunk > 1:
                    cchunk //= 2
                else:
                    tile //= 2
            if estimate(tile, cchunk) > budget:
                raise ValueError("Torch memory budget is too small for one scan task and its Top-K merge")
            max_tasks = max_tasks_per_batch
            host_probes = probes.detach().cpu()
            def plan(width):
                schedule = _schedule(host_probes, layout.host_counts, layout.host_offsets, tile, width,
                                     minimum_query_width=min_query_tile_size if not use_compile else 8)
                partial_width = _partial_slots(schedule, batch.shape[0], search_k)
                reservoir_size = batch.shape[0] * partial_width + max_tasks * tile * min(search_k, width)
                reservoir_bytes = reservoir_size * 12
                collect = (partial_width > 0 and reservoir_bytes <= budget // 3
                           and reservoir_bytes + estimate(tile, width) <= budget)
                return schedule, partial_width, reservoir_size, reservoir_bytes, collect
            # Smaller eager blocks reduce candidate padding. Retain the
            # requested larger blocks if extra block Top-Ks would force online
            # merging; large K/probe combinations benefit from the reservoir.
            # Explicit execution tiling also honors the requested candidate
            # chunk instead of silently applying the default eager 1024 cap.
            preferred = min(cchunk, 1024) if not use_compile and query_tile_size is None else cchunk
            schedule, partial_width, reservoir_size, reservoir_bytes, collect = plan(preferred)
            if not collect and preferred < cchunk:
                schedule, partial_width, reservoir_size, reservoir_bytes, collect = plan(cchunk)
            else:
                cchunk = preferred
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
                        if not use_compile:
                            block_args += (query_norms,)
                    else:
                        block_args = (lut, layout.lists.data, layout.lists.ids, layout.lists.auxiliary,
                                      bias, query_ids, lists, starts, counts, valid, candidate_offsets,
                                      local_k, 1 if use_compile else lut_chunk_size)
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
