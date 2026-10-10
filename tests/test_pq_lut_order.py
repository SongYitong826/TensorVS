"""Deferred LUTs must preserve row identity, precision and device execution."""
import pytest
import torch

from tensorvs import IVFPQBuildResult, load_ivfpq_index, save_ivfpq_index, search_ivfpq_index
import tensorvs._torch_pq_bucketed as bucket
import tensorvs._torch_search as native
from tensorvs.index_ivfpq import _apply_pq_rotation, _build_global_query_luts_bmm


@pytest.mark.parametrize("device,compiled", [("cpu", False), ("mps", False), ("mps", True)])
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("rotated", [False, True])
def test_bucket_plan_precedes_lut_without_table_reordering(tmp_path, monkeypatch, device, compiled, precision, rotated):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    gen = torch.Generator().manual_seed(710)
    counts = torch.tensor([39, 0, 71, 53])
    labels = torch.repeat_interleave(torch.arange(4), counts)
    centers = torch.zeros(4, 8); centers[:, 0] = torch.arange(4).float() * 2
    books = torch.randn(4, 16, 2, generator=gen) * .1
    codes = torch.randint(0, 16, (len(labels), 4), generator=gen, dtype=torch.uint8)
    rotation = torch.eye(8).flip(0) if rotated else None
    residuals = torch.stack([books[m, codes[:, m].long()] for m in range(4)], 1).flatten(1)
    vectors = centers[labels] + (residuals @ rotation if rotated else residuals)
    build = IVFPQBuildResult(vectors, labels, centers, books, codes, torch.device("cpu"), 4, 4,
        pq_rotation_matrix=rotation, pq_centroids=centers @ rotation.T if rotated else None)
    index = load_ivfpq_index(save_ivfpq_index(build, tmp_path / "lut_order.bin"))
    queries = centers[torch.tensor([2, 0, 3, 1, 2, 1, 0, 3, 1, 2, 0])]
    queries = (queries + torch.randn(11, 8, generator=gen) * .03).to(device)
    options = dict(device=device, backend="torch", nprobe=1, topk=5, use_compile=compiled,
        lut_dtype=precision, torch_query_batch_size=4, torch_candidate_chunk_size=64,
        torch_memory_budget_mb=2, persist_layout=False)
    with torch.inference_mode():
        expected, expected_ids = search_ivfpq_index(index, queries, torch_pq_scan="gpu", **options)
        # Warm compilation outside the instrumentation below.
        search_ivfpq_index(index, queries, torch_pq_scan="bucketed", **options)
        layout = index.layout_cache.torch_layout
        # Preserve the previous two-copy conversion as the numeric/layout
        # oracle rather than comparing the new builder against itself.
        expected_luts = []
        for q in queries.split(4):
            working = _apply_pq_rotation(q, layout.tensors.get("pq_rotation_matrix"))
            raw = _build_global_query_luts_bmm(working, layout.tensors["pq_codebooks"],
                layout.tensors["pq_codebook_norms"], m=4, subdim=2, use_tf32=True)
            expected_luts.append(raw.permute(1, 0, 2).contiguous().to(
                torch.float16 if precision == "fp16" else torch.float32))
    original_plan = bucket._pq_length_buckets
    original_lut = bucket._pq_query_lut
    original_index_select = torch.Tensor.index_select
    events, lut_batches = [], []
    def plan(*args, **kwargs):
        result = list(original_plan(*args, **kwargs))
        events.append("plan")
        return iter(result)
    def lut(batch, layout, dtype):
        assert events[-1] == "plan", "bucket widths must be planned before LUT generation"
        row = len(lut_batches)
        source = queries.split(4)[row]
        selected = (source.cpu()[:, None] - centers).square().sum(-1).argmin(1)
        order = counts[selected].sort().indices.to(device)
        torch.testing.assert_close(batch, original_index_select(source, 0, order), atol=0, rtol=0)
        assert dtype == precision and batch.shape == source.shape
        table = original_lut(batch, layout, dtype)
        assert table.is_contiguous()
        torch.testing.assert_close(table, original_index_select(expected_luts[row], 0, order), atol=0, rtol=0)
        lut_batches.append(batch.shape[0]); events.append("lut")
        return table
    def no_lut_gather(tensor, dim, indices):
        assert not (tensor.ndim == 3 and tuple(tensor.shape[1:]) == (4, 16)), "full LUT must not be reordered"
        return original_index_select(tensor, dim, indices)
    monkeypatch.setattr(bucket, "_pq_length_buckets", plan)
    monkeypatch.setattr(bucket, "_pq_query_lut", lut)
    monkeypatch.setattr(torch.Tensor, "index_select", no_lut_gather)
    with torch.inference_mode():
        actual, ids = search_ivfpq_index(index, queries, torch_pq_scan="bucketed", **options)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
    assert events == ["plan", "lut"] * 3
    assert lut_batches == [4, 4, 3]
