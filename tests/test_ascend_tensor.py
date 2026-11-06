"""Real-NPU correctness checks against independent decoded-vector distances."""
from __future__ import annotations

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("requires an Ascend NPU", allow_module_level=True)

from tensorvs import (IVFFlatBuildResult, IVFPQBuildResult, load_ivfflat_index,
                      load_ivfpq_index, save_ivfflat_index, save_ivfpq_index,
                      search_ivfflat_index, search_ivfpq_index)
from tensorvs.index_ivfpq import _make_pq_rotation_matrix


@pytest.mark.parametrize("kind,scan", [("flat", "grouped"), ("flat", "dense"),
                                     ("pq", "grouped"), ("pq", "gpu"), ("pq", "bucketed")])
@pytest.mark.parametrize("topk", [5, 40])
def test_npu_search_matches_decoded_oracle(tmp_path, kind, scan, topk):
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    gen = torch.Generator().manual_seed(103)
    vectors = torch.randn(257, 7, generator=gen)
    centers = torch.randn(6, 7, generator=gen)
    labels = torch.randint(0, 5, (257,), generator=gen)
    labels[:130] = 0
    queries = torch.randn(9, 7, generator=gen)
    queries[-1] = centers[5]  # The closest list is empty.
    rotation = None
    options = dict(device="npu:0", backend="torch", topk=topk, persist_layout=False,
                   torch_query_batch_size=4, torch_candidate_chunk_size=32,
                   torch_memory_budget_mb=2)
    if kind == "flat":
        path = save_ivfflat_index(IVFFlatBuildResult(vectors, labels, centers, torch.device("cpu")),
                                 tmp_path / "flat.bin")
        index, search = load_ivfflat_index(path), search_ivfflat_index
        decoded = vectors
        options["torch_flat_scan"] = scan
    else:
        rotation = _make_pq_rotation_matrix(7, 9, device=torch.device("cpu"), seed=4)
        books = torch.randn(3, 16, 3, generator=gen)
        codes = torch.randint(0, 16, (257, 3), generator=gen, dtype=torch.uint8)
        path = save_ivfpq_index(IVFPQBuildResult(vectors, labels, centers, books, codes,
            torch.device("cpu"), 3, 4, pq_rotation_matrix=rotation), tmp_path / "pq.bin")
        index, search = load_ivfpq_index(path), search_ivfpq_index
        decoded = centers[labels] @ rotation.T + torch.stack(
            [books[m, codes[:, m].long()] for m in range(3)], 1).flatten(1)
        options.update(torch_pq_scan=scan, lut_dtype="fp32")
    for probe in (1, 3, 6):
        distances, ids = search(index, queries.to("npu:0"), nprobe=probe, **options)
        torch.npu.synchronize()
        assert distances.device.type == ids.device.type == "npu"
        distances, ids = distances.cpu(), ids.cpu()
        selected = ((queries[:, None] - centers).square().sum(-1)).topk(probe, largest=False).indices
        allowed = (labels[None, :, None] == selected[:, None, :]).any(-1)
        q = queries if rotation is None else queries @ rotation.T
        oracle = (q[:, None] - decoded).square().sum(-1).masked_fill(~allowed, float("inf"))
        expected = oracle.topk(topk, largest=False).values
        torch.testing.assert_close(distances, expected, atol=3e-4, rtol=3e-5)
        valid = ids >= 0
        torch.testing.assert_close(distances[valid], oracle.gather(1, ids.clamp_min(0))[valid],
                                   atol=3e-4, rtol=3e-5)
        assert torch.equal(valid, torch.isfinite(distances))
        for row in ids:
            neighbors = row[row >= 0]
            assert neighbors.unique().numel() == neighbors.numel()


@pytest.mark.parametrize("lut_dtype", ["fp16", "fp32"])
def test_npu_pq_384_subspaces_matches_cpu(tmp_path, lut_dtype):
    """Cover Hunyuan's large PQ configuration and FP16 LUT storage."""
    torch.npu.matmul.allow_hf32 = False
    gen = torch.Generator().manual_seed(809)
    vectors = torch.randn(97, 768, generator=gen)
    centers = torch.randn(4, 768, generator=gen)
    labels = torch.randint(0, 4, (97,), generator=gen)
    books = torch.randn(384, 16, 2, generator=gen)
    codes = torch.randint(0, 16, (97, 384), generator=gen, dtype=torch.uint8)
    path = save_ivfpq_index(IVFPQBuildResult(vectors, labels, centers, books, codes,
        torch.device("cpu"), 384, 4), tmp_path / "many.bin")
    index = load_ivfpq_index(path)
    queries = torch.randn(3, 768, generator=gen)
    kwargs = dict(backend="torch", topk=10, nprobe=4, lut_dtype=lut_dtype,
                  torch_pq_scan="grouped", torch_candidate_chunk_size=32,
                  torch_memory_budget_mb=4, persist_layout=False)
    expected, expected_ids = search_ivfpq_index(index, queries, device="cpu", **kwargs)
    actual, ids = search_ivfpq_index(index, queries.to("npu:0"), device="npu:0", **kwargs)
    tolerance = .2 if lut_dtype == "fp16" else .003
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=3e-5)
    assert torch.equal(ids.cpu().sort(1).values, expected_ids.sort(1).values)
