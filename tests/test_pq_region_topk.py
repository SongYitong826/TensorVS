"""Exact selection oracles, including ties and portable full-search execution."""
from types import SimpleNamespace

import pytest
import torch

from tensorvs import IVFPQBuildResult, load_ivfpq_index, save_ivfpq_index, search_ivfpq_index
from tensorvs._torch_search import _pq_scan_plan, _pq_selection_plan, _pq_select_scores


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("size,k", [(2048, 10), (4099, 17), (8197, 3), (257, 300), (4099, 1000)])
def test_region_selection_matches_exhaustive_topk(device, size, k):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    gen = torch.Generator().manual_seed(407)
    scores = torch.rand(6, size, generator=gen) + 100
    count = min(k, size)
    # Best candidates clustered into one region, then spread widely, plus
    # entirely tied scores, too few finite scores and an all-invalid query.
    scores[0, -count:] = torch.arange(count).float()
    scores[1, torch.linspace(0, size - 1, count).long()] = torch.arange(count).float()
    scores[2].fill_(1.)
    scores[3].fill_(float("inf")); scores[3, -7:] = torch.arange(7).float()
    scores[4].fill_(float("inf"))
    scores = scores.to(device)
    # Explicit regions exercises odd tails; planner-selected fallback also
    # covers K >= N and large K without manufacturing a tiny final gather.
    for selection in ("regions", _pq_selection_plan(size, k, 1024)[0]):
        values, ids = _pq_select_scores(scores, k, 1024, selection)
        values, ids = values.cpu(), ids.cpu()
        oracle = scores.cpu().topk(count, largest=False).values
        torch.testing.assert_close(values.sort(1).values, oracle, atol=0, rtol=0)
        for row in range(len(scores)):
            valid = torch.isfinite(values[row])
            selected = ids[row, valid]
            assert selected.unique().numel() == selected.numel()
            torch.testing.assert_close(values[row, valid], scores.cpu()[row, selected], atol=0, rtol=0)


def test_region_plan_accounts_for_gather_workspace_and_falls_back():
    layout = SimpleNamespace(pq_scan_upper_bounds=(0, 100003), pq_anchor_grain=128)
    plan = _pq_scan_plan(layout, 128, 1, 10, 32 << 20, 8192, True, 128)
    assert plan["selection"] == "regions"
    assert 100003 <= plan["score_span"] < 100003 + 1024
    assert plan["estimated_workspace_bytes"] <= 32 << 20
    small = _pq_scan_plan(layout, 128, 1, 10, 128 << 10, 8192, True, 128)
    assert small["selection"] == "stream"
    assert _pq_selection_plan(100003, 10000, 8192)[0] == "global"


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("scan", ["gpu", "bucketed"])
@pytest.mark.parametrize("m,k", [(4, 10), (5, 17), (5, 137)])
def test_region_search_matches_decoded_vectors(tmp_path, monkeypatch, device, scan, m, k):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    if device == "cpu":
        import tensorvs._torch_search as native
        import tensorvs._torch_pq_bucketed as bucket
        monkeypatch.setattr(native, "_compute", lambda fn, use_compile, specialization=(): fn)
        monkeypatch.setattr(bucket, "_compute", lambda fn, use_compile, specialization=(): fn)
    gen = torch.Generator().manual_seed(2019)
    counts = torch.tensor([2101, 93, 0, 1107])
    labels = torch.repeat_interleave(torch.arange(4), counts)
    centers = torch.randn(4, m * 2, generator=gen)
    books = torch.randn(m, 256, 2, generator=gen)
    codes = torch.randint(0, 256, (len(labels), m), generator=gen, dtype=torch.uint8)
    decoded = centers[labels] + torch.stack([books[s, codes[:, s].long()] for s in range(m)], 1).flatten(1)
    build = IVFPQBuildResult(decoded, labels, centers, books, codes, torch.device("cpu"), m, 8)
    index = load_ivfpq_index(save_ivfpq_index(build, tmp_path / "regions.bin"))
    queries = torch.randn(5, m * 2, generator=gen)
    queries[-1] = centers[2]  # Empty selected list, including all-inf outputs.
    with torch.inference_mode():
        for probe in (1, 4):
            values, ids = search_ivfpq_index(index, queries, device=device, backend="torch",
                nprobe=probe, topk=k, use_compile=True, torch_pq_scan=scan,
                torch_query_batch_size=4, torch_candidate_chunk_size=1024,
                torch_memory_budget_mb=8, lut_dtype="fp32", persist_layout=False)
            selected = (queries[:, None] - centers).square().sum(-1).topk(probe, largest=False).indices
            allowed = (labels[None, :, None] == selected[:, None]).any(-1)
            distances = (queries[:, None] - decoded).square().sum(-1).masked_fill(~allowed, float("inf"))
            oracle = distances.topk(k, largest=False).values
            torch.testing.assert_close(values.cpu(), oracle, atol=3e-4, rtol=3e-5)
            valid = ids.cpu() >= 0
            assert torch.equal(valid, torch.isfinite(values.cpu()))
            for row in range(len(queries)):
                chosen = ids.cpu()[row, valid[row]]
                assert chosen.unique().numel() == chosen.numel()
                torch.testing.assert_close(values.cpu()[row, valid[row]], distances[row, chosen],
                                           atol=3e-4, rtol=3e-5)
    if device == "mps":
        transfers = []
        original_cpu = torch.Tensor.cpu
        def checked_cpu(tensor, *args, **kwargs):
            if tensor.device.type == "mps":
                assert scan == "bucketed" and tensor.ndim == 1 and tensor.dtype == torch.int64
                transfers.append(tensor.shape[0])
            return original_cpu(tensor, *args, **kwargs)
        def guard(original):
            def checked(tensor, *args, **kwargs):
                assert tensor.device.type != "mps", "selection must not read query data on CPU"
                return original(tensor, *args, **kwargs)
            return checked
        with monkeypatch.context() as patch, torch.inference_mode():
            patch.setattr(torch.Tensor, "cpu", checked_cpu)
            for name in ("item", "tolist", "numpy", "__int__", "__float__", "__bool__"):
                patch.setattr(torch.Tensor, name, guard(getattr(torch.Tensor, name)))
            search_ivfpq_index(index, queries.to("mps"), device=device, backend="torch",
                nprobe=4, topk=k, use_compile=True, torch_pq_scan=scan,
                torch_query_batch_size=4, torch_candidate_chunk_size=1024,
                torch_memory_budget_mb=8, lut_dtype="fp32", persist_layout=False)
        assert transfers == ([4, 1] if scan == "bucketed" else [])
