"""Conservative CUDA IVF-PQ filtering for one 16-byte code per vector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

if not hasattr(getattr(gl, "shared_memory_descriptor", None), "gather"):
    raise ImportError("IVF-PQ filtering requires Gluon shared-memory gather support")

_LOCAL_K = 16


@dataclass(frozen=True)
class PruneLayout:
    packed: torch.Tensor
    list_corrections: torch.Tensor
    list_gap: torch.Tensor
    correction_magnitude: float


def prepare_layout(
    codes: torch.Tensor,
    gamma: torch.Tensor,
    counts: torch.Tensor,
    groups: torch.Tensor,
    centroids: torch.Tensor,
    codebooks: torch.Tensor,
    max_list_count: int,
) -> PruneLayout | None:
    """Build the search-only layout before any CUDA Graph capture."""
    books = codebooks.transpose(1, 2).contiguous()
    centers = centroids.reshape(-1, 16, books.shape[1]).transpose(0, 1).contiguous()
    corrections = (2.0 * torch.bmm(centers, books)).transpose(0, 1).contiguous()
    max_h, max_gamma = torch.stack((corrections.abs().amax(), gamma.abs().amax())).tolist()
    if not (max_h < 1e28 and max_gamma < 1e28):
        return None

    gap = torch.full(counts.shape, torch.inf, device=codes.device, dtype=torch.float32)
    block = 512
    _correction_gap[(triton.cdiv(max_list_count, block), counts.numel())](
        codes, gamma, counts, groups, corrections, gap, VECTORS=codes.shape[1], BLOCK=block,
    )
    return PruneLayout(
        codes.T.contiguous().view(torch.int32), corrections, gap, max_gamma + 32 * max_h,
    )

@gluon.jit
def xor_pair(a, b):
    return a ^ b


@gluon.jit
def swap_partners(keys, offsets, width: gl.constexpr, stage: gl.constexpr,
                  bit: gl.constexpr, alternating: gl.constexpr, layout: gl.constexpr):
    stride: gl.constexpr = 1 << bit
    shaped = gl.reshape(keys, (width // (2 * stride), 2, stride))
    pair_xor = gl.reduce(shaped, 1, xor_pair)
    partners = gl.reshape(shaped ^ gl.expand_dims(pair_xor, 1), (width,))
    partners = gl.convert_layout(partners, layout)
    descending = ((offsets >> stage) & 1) != 0 if alternating else False
    choose_max = (((offsets >> bit) & 1) != 0) ^ descending
    return gl.where(choose_max, gl.maximum(keys, partners), gl.minimum(keys, partners))


@gluon.jit
def bitonic_merge(keys, width: gl.constexpr, stage: gl.constexpr,
                  alternating: gl.constexpr, layout: gl.constexpr):
    offsets = gl.arange(0, width, layout=layout)
    for bit in gl.static_range(stage - 1, -1, -1):
        keys = swap_partners(keys, offsets, width, stage, bit, alternating, layout)
    return keys


@gluon.jit
def halve_and_merge(keys, width: gl.constexpr, k: gl.constexpr, log_k: gl.constexpr,
                    alternating: gl.constexpr, layout: gl.constexpr):
    keys = gl.reshape(keys, (width // k, 2, k))
    keys = gl.convert_layout(gl.reshape(gl.min(keys, 1), (width,)), layout)
    return bitonic_merge(keys, width, log_k, alternating, layout)


@gluon.jit
def topk_min(keys, width: gl.constexpr, k: gl.constexpr, layout: gl.constexpr):
    log_k: gl.constexpr = triton.next_power_of_2(k).bit_length() - 1
    log_width: gl.constexpr = triton.next_power_of_2(width).bit_length() - 1
    for stage in gl.static_range(1, log_k + 1):
        keys = bitonic_merge(keys, width, stage, stage < log_width, layout)
    for stage in gl.static_range(log_k + 1, log_width + 1):
        keys = halve_and_merge(keys, width >> (stage - log_k), k, log_k,
                               stage < log_width, layout)
    return keys


@gluon.jit
def load_packed(codes_ptr, positions, valid):
    safe_positions = gl.where(valid, positions, 0)
    address = codes_ptr + 4 * safe_positions
    return gl.inline_asm_elementwise(
        "ld.global.ca.v4.b32 {$0, $1, $2, $3}, [$4];",
        constraints="=r,=r,=r,=r,l", args=[address],
        dtype=(gl.uint32, gl.uint32, gl.uint32, gl.uint32), is_pure=True, pack=1,
    )


@gluon.jit
def update_best(best, scores, bias, positions, valid, layout: gl.constexpr,
                TILE: gl.constexpr, INF: gl.constexpr):
    distances = gl.maximum(scores + bias, 0.0)
    bits = distances.to(gl.uint32, bitcast=True).to(gl.int64)
    keys = gl.where(valid, (bits << 32) | positions, INF)
    tile_best = topk_min(keys, TILE, 16, layout)
    merged = gl.convert_layout(gl.reshape(gl.join(best, tile_best), (32,)), layout)
    return topk_min(merged, 32, 16, layout)


@gluon.jit
def rerank_partial_keys(partial_ptr, codes_ptr, gamma_ptr, lut_ptr, bias_ptr,
                        nq, nprobe, INF: gl.constexpr, TILE: gl.constexpr):
    """Rescore a short candidate pool with the original FP32 accumulation."""
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    ranks = gl.program_id(0) * TILE + gl.arange(0, TILE, layout=layout)
    live = ranks < nq * nprobe * 16
    keys = gl.load(partial_ptr + ranks, mask=live, other=INF)
    valid = keys < INF
    physical = keys & 0xFFFFFFFF
    edge = ranks // 16
    query = edge // nprobe
    words = load_packed(codes_ptr, physical, valid)
    score = gl.load(gamma_ptr + physical, mask=valid, other=0.0, cache_modifier=".cg")
    for w in gl.static_range(4):
        word = words[w]
        for b in gl.static_range(4):
            code = ((word >> (8 * b)) & 255).to(gl.int32)
            m = w * 4 + b
            value = gl.load(lut_ptr + (m * nq + query) * 256 + code, mask=valid, other=0.0,
                            cache_modifier=".ca")
            score += value.to(gl.float32)
    bias = gl.load(bias_ptr + edge, mask=valid, other=0.0)
    bits = gl.maximum(score + bias, 0.0).to(gl.uint32, bitcast=True).to(gl.int64)
    exact_keys = gl.where(valid, (bits << 32) | physical, INF)
    gl.store(partial_ptr + ranks, exact_keys, mask=live)


@gluon.jit
def _exact_list(lut_ptr, gamma_ptr, bias_ptr, codes_ptr, partial_ptr, groups_ptr,
                count_ptr, nq, nprobe, edge, cid, layout: gl.constexpr,
                TILE: gl.constexpr, INF: gl.constexpr):
    query = edge // nprobe
    base = gl.load(groups_ptr + cid).to(gl.int64) * 32
    count = gl.load(count_ptr + cid).to(gl.int32)
    bias = gl.load(bias_ptr + edge)
    offsets = gl.arange(0, TILE, layout=layout)
    rank = gl.arange(0, 16, layout=layout)
    best = gl.full((16,), INF, gl.int64, layout)
    if count > 0:
        entries = gl.arange(0, 4096, layout=layout)
        addresses = (entries // 256 * nq + query) * 256 + entries % 256
        lut = gl.allocate_shared_memory(
            lut_ptr.dtype.element_ty, [4096], gl.SwizzledSharedLayout(1, 1, 1, [0]),
            value=gl.load(lut_ptr + addresses, cache_modifier=".ca"),
        )
        for start in range(0, count, TILE):
            positions = base + start + offsets
            valid = start + offsets < count
            words = load_packed(codes_ptr, positions, valid)
            scores = gl.load(gamma_ptr + positions, mask=valid, other=0.0, cache_modifier=".cg")
            for w in gl.static_range(4):
                for b in gl.static_range(4):
                    code = ((words[w] >> (8 * b)) & 255).to(gl.int32)
                    scores += lut.gather((w * 4 + b) * 256 + code, 0).to(gl.float32)
            best = update_best(best, scores, bias, positions, valid, layout, TILE, INF)
    gl.store(partial_ptr + edge * 16 + rank, best)


@gluon.jit
def _seed_scan(lut_ptr, gamma_ptr, bias_ptr, codes_ptr, partial_ptr, groups_ptr, count_ptr,
               cids_ptr, nq, nprobe, INF: gl.constexpr):
    edge = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    _exact_list(lut_ptr, gamma_ptr, bias_ptr, codes_ptr, partial_ptr, groups_ptr, count_ptr,
                nq, nprobe, edge, gl.load(cids_ptr + edge), layout, 512, INF)

@triton.jit
def _correction_gap(codes_ptr, gamma_ptr, list_count_ptr, list_group_ptr,
                   correction_ptr, output_ptr, VECTORS: tl.constexpr, BLOCK: tl.constexpr):
    cid = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(list_count_ptr + cid).to(tl.int32)
    if tl.program_id(0) * BLOCK < count:
        positions = tl.load(list_group_ptr + cid).to(tl.int64) * 32 + offsets
        valid = offsets < count
        estimate = tl.full((BLOCK,), 0.0, tl.float32)
        for m in tl.static_range(16):
            code = tl.load(codes_ptr + m * VECTORS + positions, mask=valid, other=0).to(tl.int32)
            estimate += tl.load(correction_ptr + cid * 4096 + m * 256 + code)
        exact = tl.load(gamma_ptr + positions, mask=valid, other=0.0)
        worst = tl.min(tl.where(valid, exact - estimate, float('inf')), 0)
        tl.atomic_min(output_ptr + cid, worst, sem='relaxed')


@gluon.jit
def quantize_table(values, layout: gl.constexpr, SCALE: gl.constexpr):
    table = gl.reshape(values, (16, 256))
    minimum = gl.min(table, 1)
    relative = table - gl.expand_dims(minimum, 1)
    relative = gl.convert_layout(gl.reshape(relative, (4096,)), layout)
    quant = gl.minimum(gl.maximum(relative * (1.0 / SCALE), 0.0), 4095.0).to(gl.uint32)
    return quant, gl.sum(minimum, 0)


@gluon.jit
def stage_bounds(lut_ptr, correction_ptr, bias_ptr, limit_ptr, magnitude_ptr, gap_ptr,
                 nq, nprobe, edge0, edge1, cid, correction_magnitude, layout: gl.constexpr,
                 PAIR: gl.constexpr, SCALE: gl.constexpr):
    offsets = gl.arange(0, 4096, layout=layout)
    q0 = edge0 // nprobe
    q1 = edge1 // nprobe
    correction = gl.load(correction_ptr + cid * 4096 + offsets, cache_modifier='.ca')
    gap = gl.load(gap_ptr + cid)
    bias0 = gl.load(bias_ptr + edge0)
    limit0 = gl.load(limit_ptr + q0)
    addresses0 = (offsets // 256 * nq + q0) * 256 + offsets % 256
    values0 = gl.load(lut_ptr + addresses0, cache_modifier=".ca").to(gl.float32) + correction
    quant0, min0 = quantize_table(values0, layout, SCALE)
    # Cover the FP32 error in the original ADC sum, correction gap, and
    # folded table arithmetic before comparing a quantized lower bound.
    margin0 = 4.0 + 4e-6 * (correction_magnitude + 16 * gl.load(magnitude_ptr + q0) + gl.abs(bias0))
    limit0 = (limit0 - (bias0 + min0 + gap) + margin0) * (1.0 / SCALE)
    if PAIR:
        bias1 = gl.load(bias_ptr + edge1)
        limit1 = gl.load(limit_ptr + q1)
        addresses1 = (offsets // 256 * nq + q1) * 256 + offsets % 256
        values1 = gl.load(lut_ptr + addresses1, cache_modifier=".ca").to(gl.float32) + correction
        quant1, min1 = quantize_table(values1, layout, SCALE)
        margin1 = 4.0 + 4e-6 * (correction_magnitude + 16 * gl.load(magnitude_ptr + q1) + gl.abs(bias1))
        limit1 = (limit1 - (bias1 + min1 + gap) + margin1) * (1.0 / SCALE)
        packed = quant0 | (quant1 << 16)
    else:
        limit1 = -1.0
        packed = quant0
    lut = gl.allocate_shared_memory(gl.uint32, [4096], gl.SwizzledSharedLayout(1, 1, 1, [0]),
                                    value=packed)
    return lut, limit0, limit1


@gluon.jit
def accumulate_word(smem, word, total, part: gl.constexpr):
    for b in gl.static_range(4):
        code = ((word >> (8 * b)) & 255).to(gl.int32)
        total += smem.gather((4 * part + b) * 256 + code, 0)
    return total


@gluon.jit
def shortlist(total, valid, limit0, limit1, PAIR: gl.constexpr):
    pass0 = valid & ((total & 65535).to(gl.float32) <= limit0)
    if PAIR:
        pass1 = valid & ((total >> 16).to(gl.float32) <= limit1)
    else:
        pass1 = gl.full(valid.shape, False, gl.int1, valid.type.layout)
    return pass0, pass1


@gluon.jit
def add_int(a, b):
    return a + b


@gluon.jit
def enqueue(partial_ptr, edge, physical, selected, start):
    flags = selected.to(gl.int32)
    count = gl.sum(flags, 0)
    if count > 0:
        order = gl.associative_scan(flags, 0, add_int) + start - 1
        gl.store(partial_ptr + edge * 16 + order, physical, mask=selected & (order < 16))
    return start + count


@gluon.jit
def prune_list(lut_ptr, correction_ptr, gamma_ptr, bias_ptr, limit_ptr, magnitude_ptr, gap_ptr, codes_ptr,
               partial_ptr, groups_ptr, count_ptr, nq, nprobe, cid, edge0, edge1, correction_magnitude,
               layout: gl.constexpr, TILE: gl.constexpr, INF: gl.constexpr, PAIR: gl.constexpr,
               CHECK: gl.constexpr, SCALE: gl.constexpr):
    base = gl.load(groups_ptr + cid).to(gl.int64) * 32
    count = gl.load(count_ptr + cid).to(gl.int32)
    offsets = gl.arange(0, TILE, layout=layout)
    ranks = gl.arange(0, 16, layout=layout)
    gl.store(partial_ptr + edge0 * 16 + ranks, INF)
    if PAIR:
        gl.store(partial_ptr + edge1 * 16 + ranks, INF)
    q0 = edge0 // nprobe
    safe = ((gl.load(magnitude_ptr + q0) < 1e28) & (gl.abs(gl.load(bias_ptr + edge0)) < 1e28)
            & (gl.load(limit_ptr + q0) >= 0))
    if PAIR:
        q1 = edge1 // nprobe
        safe &= ((gl.load(magnitude_ptr + q1) < 1e28) & (gl.abs(gl.load(bias_ptr + edge1)) < 1e28)
                 & (gl.load(limit_ptr + q1) >= 0))
    written0 = 0
    written1 = 0
    if (count > 0) & safe:
        smem, limit0, limit1 = stage_bounds(lut_ptr, correction_ptr, bias_ptr, limit_ptr, magnitude_ptr, gap_ptr,
                                           nq, nprobe, edge0, edge1, cid, correction_magnitude, layout, PAIR, SCALE)
        if (limit0 >= 0) | (limit1 >= 0):
            for start in range(0, count, TILE):
                physical = base + start + offsets
                valid = start + offsets < count
                words = load_packed(codes_ptr, physical, valid)
                total = gl.full((TILE,), 0, gl.uint32, layout)
                for part in gl.static_range(CHECK):
                    total = accumulate_word(smem, words[part], total, part)
                pass0, pass1 = shortlist(total, valid, limit0, limit1, PAIR)
                if gl.max((pass0 | pass1).to(gl.int32), 0) != 0:
                    if CHECK == 2:
                        total = accumulate_word(smem, words[2], total, 2)
                        pass0, pass1 = shortlist(total, valid, limit0, limit1, PAIR)
                    if CHECK == 3 or gl.max((pass0 | pass1).to(gl.int32), 0) != 0:
                        total = accumulate_word(smem, words[3], total, 3)
                        pass0, pass1 = shortlist(total, valid, limit0, limit1, PAIR)
                        written0 = enqueue(partial_ptr, edge0, physical, pass0, written0)
                        if PAIR:
                            written1 = enqueue(partial_ptr, edge1, physical, pass1, written1)

    if (count > 0) & ((not safe) | (written0 > 16)):
        _exact_list(lut_ptr, gamma_ptr, bias_ptr, codes_ptr, partial_ptr, groups_ptr, count_ptr,
                    nq, nprobe, edge0, cid, layout, TILE, INF)
    if PAIR:
        if (count > 0) & ((not safe) | (written1 > 16)):
            _exact_list(lut_ptr, gamma_ptr, bias_ptr, codes_ptr, partial_ptr, groups_ptr, count_ptr,
                        nq, nprobe, edge1, cid, layout, TILE, INF)


@gluon.jit
def prune_scan(lut_ptr, correction_ptr, gamma_ptr, bias_ptr, limit_ptr, magnitude_ptr, gap_ptr,
               codes_ptr, partial_ptr, groups_ptr, count_ptr, cids_ptr, sorted_edges_ptr,
               nq, nprobe, tasks, correction_magnitude, TILE: gl.constexpr, WARPS: gl.constexpr, INF: gl.constexpr,
               CHECK: gl.constexpr, SCALE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [WARPS], [0])
    task = gl.program_id(0)
    edge0 = gl.load(sorted_edges_ptr + 2 * task).to(gl.int32)
    valid1 = 2 * task + 1 < tasks
    edge1 = gl.load(sorted_edges_ptr + 2 * task + 1, mask=valid1, other=0).to(gl.int32)
    cid0 = gl.load(cids_ptr + edge0).to(gl.int32)
    cid1 = gl.load(cids_ptr + edge1).to(gl.int32)
    if valid1 & (cid0 == cid1):
        prune_list(lut_ptr, correction_ptr, gamma_ptr, bias_ptr, limit_ptr, magnitude_ptr, gap_ptr, codes_ptr,
                   partial_ptr, groups_ptr, count_ptr, nq, nprobe, cid0, edge0, edge1, correction_magnitude,
                   layout, TILE, INF, True, CHECK, SCALE)
    else:
        prune_list(lut_ptr, correction_ptr, gamma_ptr, bias_ptr, limit_ptr, magnitude_ptr, gap_ptr, codes_ptr,
                   partial_ptr, groups_ptr, count_ptr, nq, nprobe, cid0, edge0, edge0, correction_magnitude,
                   layout, TILE, INF, False, CHECK, SCALE)
        if valid1:
            prune_list(lut_ptr, correction_ptr, gamma_ptr, bias_ptr, limit_ptr, magnitude_ptr, gap_ptr, codes_ptr,
                       partial_ptr, groups_ptr, count_ptr, nq, nprobe, cid1, edge1, edge1, correction_magnitude,
                       layout, TILE, INF, False, CHECK, SCALE)


def launch_pruned_scan(
    layout: PruneLayout,
    lut: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor,
    cids: torch.Tensor,
    groups: torch.Tensor,
    counts: torch.Tensor,
    topk: int,
    partial_keys: torch.Tensor | None,
    inf_key: int,
) -> torch.Tensor:
    """Write original-score candidate keys for the shared Triton Top-K merge."""
    nq, nprobe = cids.shape
    shape = (nq, nprobe, _LOCAL_K)
    if (
        partial_keys is None
        or partial_keys.shape != shape
        or partial_keys.device != lut.device
        or partial_keys.dtype != torch.int64
    ):
        partial_keys = torch.empty(shape, dtype=torch.int64, device=lut.device)

    seed_count = 1 if nprobe <= 10 else 2
    probe_order = torch.argsort(bias, dim=1, stable=True)
    seed_indices = probe_order[:, :seed_count].contiguous()
    seed_keys = torch.empty((nq, seed_count, _LOCAL_K), dtype=torch.int64, device=lut.device)
    _seed_scan[(nq * seed_count,)](
        lut, gamma, bias.gather(1, seed_indices), layout.packed, seed_keys, groups, counts,
        cids.gather(1, seed_indices), nq, seed_count, INF=inf_key, num_warps=4,
    )
    partial_keys.scatter_(1, seed_indices[:, :, None].expand_as(seed_keys), seed_keys)
    kth = seed_keys.flatten(1).topk(topk, dim=1, largest=False).values[:, -1]
    upper = (kth >> 32).to(torch.int32).view(torch.float32).contiguous()

    query_offsets = torch.arange(nq, device=lut.device, dtype=torch.int64)[:, None] * nprobe
    remaining = (query_offsets + probe_order[:, seed_count:]).flatten()
    list_ids = cids.flatten()[remaining].to(torch.int32)
    ordered = remaining[torch.argsort(list_ids, stable=True)]
    magnitudes = lut.abs().amax(dim=(0, 2)).float()
    prune_scan[(triton.cdiv(ordered.numel(), 2),)](
        lut, layout.list_corrections, gamma, bias, upper, magnitudes, layout.list_gap,
        layout.packed, partial_keys, groups, counts, cids, ordered, nq, nprobe, ordered.numel(),
        layout.correction_magnitude, TILE=1024, WARPS=4, INF=inf_key, CHECK=2, SCALE=16.0, num_warps=4,
    )
    tile = 256
    rerank_partial_keys[(triton.cdiv(partial_keys.numel(), tile),)](
        partial_keys, layout.packed, gamma, lut, bias, nq, nprobe, INF=inf_key, TILE=tile, num_warps=4,
    )
    return partial_keys
