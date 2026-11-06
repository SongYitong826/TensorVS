"""CPU length buckets over tight IVF-PQ storage, with Torch-only GPU ADC.

One vector of actual candidate counts is downloaded per API query batch.
Queries of similar length share bounded scan tiles. This retains some local
padding, unlike the slower experimental one-dimensional compact scan.
"""
import math
import torch
from ._torch_search import (_compute, _chunked_topk, _pq_anchor_mapping,
    _pq_anchor_scores, _pq_anchor_scores_packed, _pq_anchor_ids, _pq_binary_scores, _pq_binary_ids,
    _pq_select_scores, _pq_selection_plan, _pq_packed_block, _pq_packed_block_eager,
    _pq_query_lut)
from ._layout_cache import prepare_pq_adc_words


def _pq_length_buckets(lengths, k, available, chunk, grain, use_compile):
    """Plan all widths from host lengths and shapes before any LUT is built."""
    start = 0
    while start < len(lengths):
        width = min(128, len(lengths) - start)
        width = 1 << (width.bit_length() - 1)
        while True:
            size = lengths[start + width - 1]
            aligned = math.ceil(size / chunk) * chunk
            selection, span, partial_bytes = _pq_selection_plan(size, k, chunk)
            mapping_bytes = 64 * math.ceil(span / grain) if grain else 0
            lane_bytes = 8 if grain else 32
            if span != aligned and selection == "blocked":
                lane_bytes += 8
            fits = use_compile and width * (lane_bytes * span + mapping_bytes + partial_bytes) + 8 * span <= available
            if fits or width == 1:
                break
            width //= 2
        end = start + width
        yield start, end, size, span, selection, fits
        start = end


def search_bucketed_pq(batch, layout, lut, bias, probes, best, best_ids,
                       budget, chunk, timer, use_compile, *, lut_dtype="fp32", lut_chunk_size=16, pq_lookup="flat"):
    begin = timer.begin()
    adc_data = prepare_pq_adc_words(layout) if use_compile and layout.pq_anchor_grain else None
    adc_function = _pq_anchor_scores_packed if adc_data is not None else _pq_anchor_scores
    if adc_data is None:
        adc_data = layout.lists.data
    counts = layout.lists.counts[probes]
    lengths, host_order = counts.sum(1).detach().cpu().sort()
    lengths = lengths.tolist()
    order = host_order.to(batch.device)
    probes, counts = probes.index_select(0, order), counts.index_select(0, order)
    grain = layout.pq_anchor_grain if use_compile else 0
    if grain and layout.pq_has_empty_lists:
        permutation = counts.argsort(1, descending=True)
        probes, counts = probes.gather(1, permutation), counts.gather(1, permutation)
    prefix = counts.cumsum(1)
    previous = torch.cat((torch.zeros_like(prefix[:, :1]), prefix[:, :-1]), 1)
    selected_bias = bias.index_select(0, order).gather(1, probes)
    reserve = max(budget // 8, 48 * batch.shape[0] * best.shape[1])
    available = budget - reserve - 64 * batch.shape[0] * probes.shape[1]
    buckets = list(_pq_length_buckets(lengths, best.shape[1], available, chunk, grain, use_compile))
    timer.end("bucket_host_length_planning", begin)
    begin = timer.begin()
    if lut is None:
        # Generate once for the reordered API batch, preserving GEMM batch
        # size (not one small GEMM per scan bucket). Rotation is also deferred.
        tables = _pq_query_lut(batch.index_select(0, order), layout, lut_dtype)
    else:
        # Compatibility for internal callers supplying an existing table.
        tables = lut.index_select(0, order)
    lut = tables
    timer.end("LUT_prepare", begin)
    for start, end, size, span, selection, fits in buckets:
        width = end - start
        if not size:
            continue
        key = (str(batch.device), layout.lists.data.dtype, lut.dtype, lut.shape[1], lut.shape[2],
               best.shape[1], chunk, grain, selection, "cpu_bucket")
        begin = timer.begin()
        if fits:
            mapping = _pq_anchor_mapping(layout, probes[start:end], prefix[start:end], previous[start:end],
                                         selected_bias[start:end], span, grain) if grain else None
            ordinals = torch.arange(span, device=batch.device)
            timer.end("bucket_mapping", begin)
            begin = timer.begin()
            if mapping is not None:
                scores = _compute(adc_function, True, key)(tables[start:end], adc_data,
                    layout.lists.auxiliary, *mapping, ordinals, grain)
            else:
                scores = _compute(_pq_binary_scores, True, key)(tables[start:end], layout.lists.data,
                    layout.lists.auxiliary, selected_bias[start:end], probes[start:end], layout.lists.offsets,
                    prefix[start:end], previous[start:end], ordinals)
            timer.end("bucket_adc", begin)
            begin = timer.begin()
            values, ords = _pq_select_scores(scores, best.shape[1], chunk, selection)
            del scores
            if mapping is not None:
                ids = _compute(_pq_anchor_ids, True, key)(layout.lists.ids, *mapping[:3], ords, grain, values)
            else:
                ids = _compute(_pq_binary_ids, True, key)(layout.lists.ids, probes[start:end],
                    layout.lists.offsets, prefix[start:end], previous[start:end], ords, values)
            if values.shape[1] < best.shape[1]:
                values = torch.cat((values, values.new_full((width, best.shape[1] - values.shape[1]), float("inf"))), 1)
                ids = torch.cat((ids, ids.new_full((width, best.shape[1] - ids.shape[1]), -1)), 1)
            timer.end("bucket_topk", begin)
        else:
            # Exact-length streaming avoids full-score allocation for huge
            # lists, large K, small budgets, and uncompiled execution.
            address_bytes = 8 if pq_lookup == "gather" else (4 if tables.numel() <= 2147483647 else 8)
            lane_bytes = (32 if use_compile else 64 + address_bytes * lut.shape[1]
                          + (address_bytes + lut.element_size() + 4) * min(lut.shape[1], lut_chunk_size))
            capacity = max(0, available // (lane_bytes * width + 16))
            if capacity < 1:
                raise ValueError("Torch memory budget is too small for bucketed PQ scanning")
            stream_chunk = min(chunk, 1 << (capacity.bit_length() - 1))
            ordinals = torch.arange(stream_chunk, device=batch.device)
            fn = _compute(_pq_packed_block if use_compile or pq_lookup == "gather" else _pq_packed_block_eager, use_compile, key)
            def scan(offset):
                lanes = ordinals[:min(stream_chunk, size - offset)] + offset
                return fn(tables[start:end], layout.lists.data, layout.lists.ids, layout.lists.auxiliary,
                    selected_bias[start:end], probes[start:end], layout.lists.offsets,
                    prefix[start:end], previous[start:end], lanes,
                    min(best.shape[1], lanes.shape[0]), 1 if use_compile else lut_chunk_size)
            timer.end("bucket_mapping", begin)
            values, ids = _chunked_topk(scan, size, stream_chunk, best[order[start:end]], best_ids[order[start:end]],
                budget, timer, "bucket_adc_topk", use_compile, key)
        best.index_copy_(0, order[start:end], values)
        best_ids.index_copy_(0, order[start:end], ids)
    return best, best_ids
