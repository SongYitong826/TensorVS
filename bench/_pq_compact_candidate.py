"""Experimental compact ADC: CPU length planning, ordinary Torch GPU operators.

ADC visits only real candidates. ``row`` performs exact-length per-query
Top-K; ``blocked`` batches full blocks and pads only tail score selection.
"""
import math
import torch
from tensorvs._torch_search import _compute
from tensorvs._torch_search import (_pq_anchor_mapping, _pq_anchor_scores, _pq_anchor_ids,
    _pq_binary_scores, _pq_binary_ids, _pq_select_scores)


def anchor_scores(luts, data, corrections, d0, d1, ends, q0, q1, b0, b1, ordinals, grain):
    bins = ordinals // grain
    following = ordinals >= ends[bins]
    positions = ordinals + torch.where(following, d1[bins], d0[bins])
    queries = torch.where(following, q1[bins], q0[bins])
    distances = torch.zeros_like(positions, dtype=torch.float32)
    for subspace in range(data.shape[1]):
        codes = data[positions, subspace].to(torch.int64)
        if data.dtype == torch.int16:
            codes = codes & 65535
        addresses = (queries * luts.shape[1] + subspace) * luts.shape[2] + codes
        distances = distances + luts.flatten()[addresses].to(torch.float32)
    return (distances + corrections[positions] + torch.where(following, b1[bins], b0[bins])).clamp_min_(0.)


def binary_scores(luts, data, corrections, starts, previous, prefix, queries, biases, ordinals):
    ranks = torch.searchsorted(prefix, ordinals, right=True)
    positions = starts[ranks] + ordinals - previous[ranks]
    rows = queries[ranks]
    distances = torch.zeros_like(positions, dtype=torch.float32)
    for subspace in range(data.shape[1]):
        codes = data[positions, subspace].to(torch.int64)
        if data.dtype == torch.int16:
            codes = codes & 65535
        addresses = (rows * luts.shape[1] + subspace) * luts.shape[2] + codes
        distances = distances + luts.flatten()[addresses].to(torch.float32)
    return (distances + corrections[positions] + biases[ranks]).clamp_min_(0.)


def full_topk(scores, starts, width, k):
    values, order = scores[starts[:, None] + torch.arange(width, device=scores.device)[None, :]].topk(
        min(k, width), largest=False, sorted=False)
    return values, starts[:, None] + order


def tail_topk(scores, starts, lengths, width, k):
    lanes = torch.arange(width, device=scores.device)[None, :]
    valid = lanes < lengths[:, None]
    positions = torch.where(valid, starts[:, None] + lanes, 0)
    values, order = scores[positions].masked_fill(~valid, float("inf")).topk(
        min(k, width), largest=False, sorted=False)
    return values, positions.gather(1, order)


def result_ids(ids, starts, previous, prefix, ordinals, values):
    valid = torch.isfinite(values)
    ordinals = torch.where(valid, ordinals, 0)
    ranks = torch.searchsorted(prefix, ordinals.contiguous(), right=True)
    positions = starts[ranks] + ordinals - previous[ranks]
    return ids[positions].masked_fill(~valid, -1)


def select(scores, sizes, mode, width, k, use_compile, key):
    device = scores.device
    if mode == "row":
        values, ordinals, cursor = [], [], 0
        for size in sizes:
            if size:
                v, order = scores[cursor:cursor + size].topk(min(k, size), largest=False, sorted=False)
                order = order + cursor
            else:
                v = scores.new_empty(0)
                order = torch.empty(0, dtype=torch.int64, device=device)
            if v.numel() < k:
                v = torch.cat((v, scores.new_full((k - v.numel(),), float("inf"))))
                order = torch.cat((order, torch.zeros(k - order.numel(), dtype=torch.int64, device=device)))
            values.append(v); ordinals.append(order); cursor += size
        return torch.stack(values), torch.stack(ordinals)
    width = max(width, k)
    blocks = [size // width for size in sizes]
    partial_width = (max(blocks, default=0) + 1) * k
    partial = scores.new_full((len(sizes), partial_width), float("inf"))
    partial_ids = torch.zeros(partial.shape, dtype=torch.int64, device=device)
    full_starts, full_rows, full_slots, tail_starts, tail_lengths = [], [], [], [], []
    cursor = 0
    for row, (size, count) in enumerate(zip(sizes, blocks)):
        full_starts.extend(cursor + block * width for block in range(count))
        full_rows.extend([row] * count)
        full_slots.extend(block * k for block in range(count))
        tail_starts.append(cursor + count * width)
        tail_lengths.append(size % width)
        cursor += size
    integer = lambda values: torch.tensor(values, dtype=torch.int64, device=device)
    if full_starts:
        v, order = _compute(full_topk, use_compile, key)(scores, integer(full_starts), width, k)
        columns = integer(full_slots)[:, None] + torch.arange(v.shape[1], device=device)[None, :]
        rows = integer(full_rows)[:, None]
        partial[rows, columns] = v
        partial_ids[rows, columns] = order
    tail_width = max(tail_lengths, default=0)
    if tail_width:
        v, order = _compute(tail_topk, use_compile, key)(scores, integer(tail_starts), integer(tail_lengths), tail_width, k)
        columns = integer(blocks)[:, None] * k + torch.arange(v.shape[1], device=device)[None, :]
        rows = torch.arange(len(sizes), device=device)[:, None]
        partial[rows, columns] = v
        partial_ids[rows, columns] = order
    values, order = partial.topk(k, largest=False, sorted=False)
    return values, partial_ids.gather(1, order)


def compact_search(batch, layout, lut, bias, probes, best, best_ids,
                   budget, chunk, timer, use_compile, query_tile=None, *, mode="blocked"):
    if mode.startswith("bucket"):
        return bucket_search(batch, layout, lut, bias, probes, best, best_ids,
                             budget, chunk, timer, use_compile, int(mode[6:]))
    begin = timer.begin()
    host_probes = probes.detach().cpu()
    host_counts = torch.tensor(layout.host_counts, dtype=torch.int64)[host_probes]
    host_starts = torch.tensor(layout.host_offsets, dtype=torch.int64)[host_probes]
    sizes = host_counts.sum(1).tolist()
    timer.end("compact_host_length_planning", begin)
    # Real candidate counts determine microbatches; no dataset assumptions.
    limit = max(1, (budget * 7 // 8) // 32)
    groups, start, total = [], 0, 0
    for row, size in enumerate(sizes):
        if row > start and total + size > limit:
            groups.append((start, row)); start, total = row, 0
        total += size
    groups.append((start, len(sizes)))
    key = (str(batch.device), layout.lists.data.dtype, lut.dtype, lut.shape[1], lut.shape[2], best.shape[1], mode)
    for qs, qe in groups:
        begin = timer.begin()
        counts_cpu = host_counts[qs:qe].flatten()
        positive_cpu = counts_cpu.nonzero().flatten()
        size = sum(sizes[qs:qe])
        if not size:
            continue
        positive = positive_cpu.to(batch.device)
        counts = counts_cpu[positive_cpu].to(batch.device)
        starts = host_starts[qs:qe].flatten()[positive_cpu].to(batch.device)
        prefix = counts.cumsum(0)
        previous = torch.cat((prefix.new_zeros(1), prefix[:-1]))
        query_ids = positive // probes.shape[1]
        biases = bias[qs:qe].gather(1, probes[qs:qe]).flatten()[positive]
        grain = layout.pq_anchor_grain
        mapping = None
        if grain:
            ranks = torch.searchsorted(prefix, torch.arange(math.ceil(size / grain), device=batch.device) * grain,
                                       right=True)
            following = (ranks + 1).clamp_max(prefix.shape[0] - 1)
            deltas = starts - previous
            mapping = (deltas[ranks], deltas[following], prefix[ranks], query_ids[ranks],
                       query_ids[following], biases[ranks], biases[following])
        timer.end("compact_mapping", begin)
        begin = timer.begin()
        distances = torch.empty(size, dtype=torch.float32, device=batch.device)
        # Eager needs additional code/address/LUT temporaries; compiled gathers
        # fuse. Framework-internal workspaces are outside this estimate.
        lane_bytes = 16 if use_compile and grain else 64 + 24 * layout.lists.data.shape[1]
        available = max(1, budget * 7 // 8 - 4 * size)
        span = max(1, min(size, available // lane_bytes))
        ordinals = torch.arange(span, device=batch.device)
        for start in range(0, size, span):
            lanes = ordinals[:min(span, size - start)] + start
            if mapping is not None:
                values = _compute(anchor_scores, use_compile, key)(lut[qs:qe], layout.lists.data,
                    layout.lists.auxiliary, *mapping, lanes, grain)
            else:
                values = _compute(binary_scores, use_compile, key)(lut[qs:qe], layout.lists.data,
                    layout.lists.auxiliary, starts, previous, prefix, query_ids, biases, lanes)
            distances[start:start + lanes.shape[0]] = values
        timer.end("compact_adc", begin)
        begin = timer.begin()
        values, order = select(distances, sizes[qs:qe], mode, chunk, best.shape[1], use_compile, key)
        selected = _compute(result_ids, use_compile, key)(layout.lists.ids, starts, previous, prefix, order, values)
        best[qs:qe], best_ids[qs:qe] = values, selected
        timer.end("compact_segmented_topk", begin)
    return best, best_ids


def bucket_search(batch, layout, lut, bias, probes, best, best_ids,
                  budget, chunk, timer, use_compile, maximum_width=128):
    begin = timer.begin()
    host_probes = probes.detach().cpu()
    host_counts = torch.tensor(layout.host_counts, dtype=torch.int64)[host_probes]
    lengths, host_order = host_counts.sum(1).sort()
    lengths = lengths.tolist()
    order = host_order.to(batch.device)
    probes = probes.index_select(0, order)
    tables = lut.index_select(0, order)
    counts = layout.lists.counts[probes]
    grain = layout.pq_anchor_grain
    if grain and layout.pq_has_empty_lists:
        permutation = counts.argsort(1, descending=True)
        probes = probes.gather(1, permutation); counts = counts.gather(1, permutation)
    prefix = counts.cumsum(1)
    previous = torch.cat((torch.zeros_like(prefix[:, :1]), prefix[:, :-1]), 1)
    selected_bias = bias.index_select(0, order).gather(1, probes)
    reserve = max(budget // 8, 48 * batch.shape[0] * best.shape[1])
    available = budget - reserve
    timer.end("bucket_host_length_planning", begin)
    start = 0
    while start < batch.shape[0]:
        width = min(maximum_width, batch.shape[0] - start)
        width = 1 << (width.bit_length() - 1)
        while True:
            size = lengths[start + width - 1]
            aligned = math.ceil(size / chunk) * chunk
            span = size if aligned - size > size // 8 else aligned
            selection = "global" if size <= chunk or best.shape[1] * 4 >= chunk else "blocked"
            if selection == "global": span = size
            lanes = 16 if grain else 40
            per_query = lanes * span + 48 * best.shape[1] * math.ceil(max(1, span) / chunk)
            if width == 1 or per_query * width + 8 * span <= available:
                break
            width //= 2
        end = start + width
        if not size:
            start = end
            continue
        key = (str(batch.device), layout.lists.data.dtype, lut.dtype, lut.shape[1], lut.shape[2],
               best.shape[1], chunk, grain, selection, "cpu_bucket")
        begin = timer.begin()
        mapping = _pq_anchor_mapping(layout, probes[start:end], prefix[start:end], previous[start:end],
                                     selected_bias[start:end], span, grain) if grain else None
        ordinals = torch.arange(span, device=batch.device)
        timer.end("bucket_mapping", begin)
        begin = timer.begin()
        if mapping is not None:
            scores = _compute(_pq_anchor_scores, use_compile, key)(tables[start:end], layout.lists.data,
                layout.lists.auxiliary, *mapping, ordinals, grain)
        else:
            scores = _compute(_pq_binary_scores, use_compile, key)(tables[start:end], layout.lists.data,
                layout.lists.auxiliary, selected_bias[start:end], probes[start:end], layout.lists.offsets,
                prefix[start:end], previous[start:end], ordinals)
        timer.end("bucket_adc", begin)
        begin = timer.begin()
        values, ords = _pq_select_scores(scores, best.shape[1], chunk, selection)
        if mapping is not None:
            ids = _compute(_pq_anchor_ids, use_compile, key)(layout.lists.ids, *mapping[:3], ords, grain, values)
        else:
            ids = _compute(_pq_binary_ids, use_compile, key)(layout.lists.ids, probes[start:end],
                layout.lists.offsets, prefix[start:end], previous[start:end], ords, values)
        if values.shape[1] < best.shape[1]:
            values = torch.cat((values, values.new_full((width, best.shape[1] - values.shape[1]), float("inf"))), 1)
            ids = torch.cat((ids, ids.new_full((width, best.shape[1] - ids.shape[1]), -1)), 1)
        best.index_copy_(0, order[start:end], values)
        best_ids.index_copy_(0, order[start:end], ids)
        timer.end("bucket_topk", begin)
        start = end
    return best, best_ids
