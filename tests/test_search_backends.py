from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from tensorvs import (IVFFlatBuildResult, IVFPQBuildResult, load_ivfflat_index,
                      load_ivfpq_index, save_ivfflat_index, save_ivfpq_index,
                      search_ivfflat_index, search_ivfpq_index, get_default_device)
from tensorvs.index_ivfpq import _make_pq_rotation_matrix
from tensorvs._torch_search import search_device


@pytest.fixture
def indexes(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORVS_LAYOUT_CACHE_DIR", str(tmp_path / "cache"))
    gen = torch.Generator().manual_seed(103)
    x = torch.randn(257, 7, generator=gen)
    centers = torch.randn(6, 7, generator=gen)
    labels = torch.randint(0, 5, (257,), generator=gen)
    labels[:130] = 0  # Long list, short lists, and an empty list.
    q = torch.randn(11, 7, generator=gen)
    flat_path = save_ivfflat_index(IVFFlatBuildResult(x, labels, centers, torch.device("cpu")), tmp_path / "flat.bin")
    rotation = _make_pq_rotation_matrix(7, 9, device=torch.device("cpu"), seed=4)
    books = torch.randn(3, 16, 3, generator=gen)
    codes = torch.randint(0, 16, (257, 3), generator=gen, dtype=torch.uint8)
    pq_path = save_ivfpq_index(IVFPQBuildResult(x, labels, centers, books, codes,
                              torch.device("cpu"), 3, 4, pq_rotation_matrix=rotation), tmp_path / "pq.bin")
    # Independent decoded-vector oracle for residual-PQ ADC.
    residuals = torch.stack([books[m, codes[:, m].long()] for m in range(3)], 1).flatten(1)
    decoded = centers[labels] @ rotation.T + residuals
    return (load_ivfflat_index(flat_path), load_ivfpq_index(pq_path), x, decoded,
            q, labels, centers, rotation)


def reference(q, vectors, labels, centers, nprobe, k, rotation=None):
    probes = ((q[:, None] - centers).square().sum(-1)).topk(nprobe, largest=False).indices
    allowed = (labels[None, :, None] == probes[:, None, :]).any(-1)
    if rotation is not None:
        q = q @ rotation.T
    distances = (q[:, None] - vectors).square().sum(-1).masked_fill(~allowed, float("inf"))
    values, ids = distances.topk(min(k, vectors.shape[0]), largest=False)
    ids = ids.masked_fill(~torch.isfinite(values), -1)
    if k > vectors.shape[0]:
        values = torch.cat((values, torch.full((q.shape[0], k - vectors.shape[0]), float("inf"))), 1)
        ids = torch.cat((ids, torch.full((q.shape[0], k - vectors.shape[0]), -1, dtype=torch.int64)), 1)
    return values, ids


@pytest.mark.parametrize("scan", ["grouped", "dense"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fixed_flat_scan_matches_selected_candidate_oracle(indexes, scan, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    index, _, vectors, _, queries, labels, centers, _ = indexes
    queries = queries.repeat(6, 1)[:64]
    for nprobe in (1, 3, 6):
        profile = {}
        values, ids = search_ivfflat_index(index, queries, backend="torch", device=device,
                                         nprobe=nprobe, topk=9, torch_flat_scan=scan,
                                         profile=profile)
        expected, expected_ids = reference(queries, vectors, labels, centers, nprobe, 9)
        torch.testing.assert_close(values.cpu(), expected, atol=3e-5, rtol=2e-5)
        torch.testing.assert_close(ids.cpu(), expected_ids)
        assert ("dense_vector_distance_topk" in profile) == (scan == "dense")


@pytest.mark.parametrize("scan", ["invalid", "metal"])
def test_invalid_flat_scan_rejected(indexes, scan):
    with pytest.raises(ValueError, match="flat_scan"):
        search_ivfflat_index(indexes[0], indexes[4], backend="torch", device="cpu",
                             torch_flat_scan=scan)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:0"])
@pytest.mark.parametrize("k,budget", [(10, .0625), (100, 8), (300, .25)])
def test_tensor_native_flat_ragged_chunks_match_oracle(tmp_path, monkeypatch, device, k, budget):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    def reject_custom_shader(*args, **kwargs):
        raise AssertionError("TensorNative search must use standard tensor operators")
    monkeypatch.setattr(torch.mps, "compile_shader", reject_custom_shader, raising=False)
    generator = torch.Generator().manual_seed(907)
    vectors = torch.randn(871, 31, generator=generator)
    centers = torch.randn(6, 31, generator=generator)
    labels = torch.randint(1, 5, (871,), generator=generator)
    labels[:730] = 0  # A long list split into multiple candidate blocks.
    queries = torch.randn(9, 62, generator=generator)[:, ::2]
    queries[-1] = centers[5]  # The nearest list is empty at nprobe=1.
    path = save_ivfflat_index(IVFFlatBuildResult(vectors, labels, centers, torch.device("cpu")),
                             tmp_path / "native.bin")
    index = load_ivfflat_index(path)
    for probe in (1, 3, 6):
        values, ids = search_ivfflat_index(index, queries, device=device, backend="torch",
            topk=k, nprobe=probe, torch_query_batch_size=4, torch_candidate_chunk_size=128,
            torch_memory_budget_mb=budget, torch_flat_scan="grouped", persist_layout=False)
        expected, expected_ids = reference(queries, vectors, labels, centers, probe, k)
        torch.testing.assert_close(values.cpu(), expected, atol=2e-4, rtol=2e-5)
        # Norm/addmm and direct subtraction can order near-ties differently.
        # Verify the selected set and every returned ID's independent distance.
        torch.testing.assert_close(ids.cpu().sort(1).values, expected_ids.sort(1).values)
        actual = (queries[:, None] - vectors[ids.cpu().clamp_min(0)]).square().sum(-1)
        valid = ids.cpu() >= 0
        torch.testing.assert_close(values.cpu()[valid], actual[valid], atol=2e-4, rtol=2e-5)
        assert torch.equal(ids.cpu() < 0, ~torch.isfinite(values.cpu()))


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:0"])
@pytest.mark.parametrize("lut_dtype", ["fp32", "fp16"])
@pytest.mark.parametrize("scan", ["auto", "gpu", "bucketed"])
@pytest.mark.parametrize("k,budget", [(5, .0625), (100, 8), (300, .25)])
def test_tensor_native_pq_ragged_chunks_match_decoded_oracle(indexes, monkeypatch,
                                                           device, lut_dtype, scan, k, budget):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    def reject_custom_shader(*args, **kwargs):
        raise AssertionError("TensorNative search must use standard tensor operators")
    monkeypatch.setattr(torch.mps, "compile_shader", reject_custom_shader, raising=False)
    _, index, _, decoded, queries, labels, centers, rotation = indexes
    queries = queries[:9].clone()
    queries[-1] = centers[5]  # Empty nearest list at nprobe=1.
    tolerance = .05 if lut_dtype == "fp16" else 2e-4
    for probe in (1, 3, 6):
        values, ids = search_ivfpq_index(index, queries, device=device, backend="torch",
            topk=k, nprobe=probe, lut_dtype=lut_dtype, torch_query_batch_size=4,
            torch_candidate_chunk_size=16, torch_memory_budget_mb=budget, persist_layout=False,
            torch_pq_scan=scan)
        expected, _ = reference(queries, decoded, labels, centers, probe, k, rotation)
        torch.testing.assert_close(values.cpu(), expected, atol=tolerance, rtol=2e-5)
        valid = ids.cpu() >= 0
        actual = ((queries @ rotation.T)[:, None] - decoded[ids.cpu().clamp_min(0)]).square().sum(-1)
        torch.testing.assert_close(values.cpu()[valid], actual[valid], atol=tolerance, rtol=2e-5)
        assert torch.equal(valid, torch.isfinite(values.cpu()))
        for row in ids.cpu():
            selected = row[row >= 0]
            assert selected.unique().numel() == selected.numel()


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("lut_dtype", ["fp32", "fp16"])
@pytest.mark.parametrize("scan", ["auto", "gpu", "bucketed"])
def test_tensor_native_pq_many_subspaces_match_decoded_oracle(tmp_path, device, lut_dtype, scan):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    gen = torch.Generator().manual_seed(992)
    vectors = torch.randn(211, 19, generator=gen)
    centers = torch.randn(3, 19, generator=gen)
    labels = torch.randint(0, 2, (211,), generator=gen)
    labels[:170] = 0
    books = torch.randn(19, 16, 1, generator=gen)
    codes = torch.randint(0, 16, (211, 19), generator=gen, dtype=torch.uint8)
    decoded = centers[labels] + torch.stack([books[m, codes[:, m].long(), 0] for m in range(19)], 1)
    build = IVFPQBuildResult(vectors, labels, centers, books, codes, torch.device("cpu"), 19, 4)
    index = load_ivfpq_index(save_ivfpq_index(build, tmp_path / "wide_pq.bin"))
    queries = torch.randn(7, 19, generator=gen)
    queries[-1] = centers[2]
    tolerance = .08 if lut_dtype == "fp16" else 2e-4
    for probe, k in ((1, 7), (3, 240)):
        values, ids = search_ivfpq_index(index, queries, device=device, backend="torch",
            topk=k, nprobe=probe, lut_dtype=lut_dtype, torch_query_batch_size=4,
            torch_candidate_chunk_size=32, torch_memory_budget_mb=.25, persist_layout=False,
            torch_pq_scan=scan)
        expected, _ = reference(queries, decoded, labels, centers, probe, k)
        torch.testing.assert_close(values.cpu(), expected, atol=tolerance, rtol=2e-5)
        valid = ids.cpu() >= 0
        actual = (queries[:, None] - decoded[ids.cpu().clamp_min(0)]).square().sum(-1)
        torch.testing.assert_close(values.cpu()[valid], actual[valid], atol=tolerance, rtol=2e-5)
        assert torch.equal(valid, torch.isfinite(values.cpu()))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
@pytest.mark.parametrize("use_compile", [False, True])
def test_gpu_pq_schedule_has_no_query_readback(indexes, monkeypatch, use_compile):
    import tensorvs._torch_search as native
    _, index, _, decoded, q, labels, centers, rotation = indexes
    q = q.clone()
    q[-1] = centers[5]  # Includes an empty nearest list.
    expected = [reference(q, decoded, labels, centers, p, 15, rotation)[0] for p in (1, 3, 6)]
    index.reside_on_device("mps", backend="torch", persist_layout=False)
    gpu_q = q.to("mps")
    options = dict(device="mps", backend="torch", topk=15, torch_pq_scan="gpu",
        torch_query_batch_size=4, torch_candidate_chunk_size=16,
        torch_memory_budget_mb=.0625, persist_layout=False, use_compile=use_compile)
    if use_compile:
        # Framework tracing/code generation may use host metadata. Guard the
        # warmed execution itself against query-dependent tensor readbacks.
        with torch.inference_mode():
            for probe in (1, 3, 6):
                search_ivfpq_index(index, gpu_q, nprobe=probe, **options)
    def reject_schedule(*args, **kwargs):
        raise AssertionError("GPU scanning must not construct host query tasks")
    def guard(original):
        def checked(tensor, *args, **kwargs):
            if tensor.device.type == "mps":
                raise AssertionError("GPU scanning must not read a query tensor on the host")
            return original(tensor, *args, **kwargs)
        return checked
    actual = []
    with monkeypatch.context() as patch:
        patch.setattr(native, "_schedule", reject_schedule)
        patch.setattr(native, "_task_tensors", reject_schedule)
        if not use_compile:
            patch.setattr(torch.mps, "compile_shader", reject_schedule, raising=False)
        for name in ("cpu", "item", "tolist", "numpy", "__int__", "__float__", "__bool__"):
            patch.setattr(torch.Tensor, name, guard(getattr(torch.Tensor, name)))
        original_to = torch.Tensor.to
        def checked_to(tensor, *args, **kwargs):
            target = kwargs.get("device", args[0] if args else None)
            if isinstance(target, torch.Tensor):
                target = target.device
            if tensor.device.type == "mps" and isinstance(target, (str, torch.device)) and torch.device(target).type == "cpu":
                raise AssertionError("GPU scanning must not transfer query data to the host")
            return original_to(tensor, *args, **kwargs)
        patch.setattr(torch.Tensor, "to", checked_to)
        with torch.inference_mode():
            for probe in (1, 3, 6):
                actual.append(search_ivfpq_index(index, gpu_q, nprobe=probe, **options))
    for (values, ids), oracle in zip(actual, expected):
        assert values.device.type == ids.device.type == "mps"
        torch.testing.assert_close(values.cpu(), oracle, atol=2e-4, rtol=2e-5)
        assert torch.equal(ids.cpu() < 0, ~torch.isfinite(values.cpu()))


def test_invalid_pq_scan_rejected(indexes):
    with pytest.raises(ValueError, match="pq_scan"):
        search_ivfpq_index(indexes[1], indexes[4], device="cpu", backend="torch", torch_pq_scan="invalid")


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("mode", ["row", "blocked"])
def test_experimental_compact_pq_matches_decoded_oracle(indexes, monkeypatch, device, mode):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    import tensorvs._torch_search as native
    from bench._pq_compact_candidate import compact_search
    monkeypatch.setattr(native, "_packed_pq_search", lambda *a, **kw: compact_search(*a, **kw, mode=mode))
    _, index, _, decoded, queries, labels, centers, rotation = indexes
    queries = queries.clone()
    queries[-1] = centers[5]
    for probe, k in ((1, 15), (3, 15), (6, 300)):
        values, ids = search_ivfpq_index(index, queries, device=device, backend="torch",
            nprobe=probe, topk=k, torch_pq_scan="gpu", torch_query_batch_size=4,
            torch_candidate_chunk_size=32, torch_memory_budget_mb=2, lut_dtype="fp32", persist_layout=False)
        expected, _ = reference(queries, decoded, labels, centers, probe, k, rotation)
        torch.testing.assert_close(values.cpu(), expected, atol=2e-4, rtol=2e-5)
        valid = ids.cpu() >= 0
        actual = ((queries @ rotation.T)[:, None] - decoded[ids.cpu().clamp_min(0)]).square().sum(-1)
        torch.testing.assert_close(values.cpu()[valid], actual[valid], atol=2e-4, rtol=2e-5)
        assert torch.equal(valid, torch.isfinite(values.cpu()))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
@pytest.mark.parametrize("use_compile", [False, True])
def test_bucketed_pq_downloads_only_counts_once_per_batch(indexes, monkeypatch, use_compile):
    _, index, _, decoded, q, labels, centers, rotation = indexes
    index.reside_on_device("mps", backend="torch", persist_layout=False)
    q_gpu = q.to("mps")
    options = dict(device="mps", backend="torch", nprobe=3, topk=9, torch_pq_scan="bucketed",
        torch_query_batch_size=4, use_compile=use_compile, persist_layout=False)
    with torch.inference_mode():
        search_ivfpq_index(index, q_gpu, **options)  # prepare caches / compiler
    transfers = []
    original_cpu = torch.Tensor.cpu
    def checked_cpu(tensor, *args, **kwargs):
        if tensor.device.type == "mps":
            assert tensor.dtype == torch.int64 and tensor.ndim == 1
            transfers.append(tensor.shape[0])
        return original_cpu(tensor, *args, **kwargs)
    def guard(original):
        def checked(tensor, *args, **kwargs):
            assert tensor.device.type != "mps", "only the explicit count-vector transfer is allowed"
            return original(tensor, *args, **kwargs)
        return checked
    with monkeypatch.context() as patch, torch.inference_mode():
        patch.setattr(torch.Tensor, "cpu", checked_cpu)
        for name in ("item", "tolist", "numpy", "__int__", "__float__", "__bool__"):
            patch.setattr(torch.Tensor, name, guard(getattr(torch.Tensor, name)))
        values, ids = search_ivfpq_index(index, q_gpu, **options)
    assert transfers == [4, 4, 3]
    expected, _ = reference(q, decoded, labels, centers, 3, 9, rotation)
    torch.testing.assert_close(values.cpu(), expected, atol=2e-4, rtol=2e-5)
    assert values.device.type == ids.device.type == "mps"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_compiled_gpu_pq_schedule_matches_decoded_oracle(indexes):
    _, index, _, decoded, q, labels, centers, rotation = indexes
    # MPS Inductor support is runtime-dependent; an explicitly requested
    # compiled search must succeed rather than silently change its backend.
    with torch.inference_mode():
        values, ids = search_ivfpq_index(index, q, device="mps", backend="torch",
            nprobe=3, topk=15, torch_pq_scan="gpu", use_compile=True,
            torch_query_batch_size=4, torch_candidate_chunk_size=32,
            torch_memory_budget_mb=.25, persist_layout=False)
    expected, _ = reference(q, decoded, labels, centers, 3, 15, rotation)
    torch.testing.assert_close(values.cpu(), expected, atol=2e-4, rtol=2e-5)
    actual = ((q @ rotation.T)[:, None] - decoded[ids.cpu().clamp_min(0)]).square().sum(-1)
    valid = ids.cpu() >= 0
    torch.testing.assert_close(values.cpu()[valid], actual[valid], atol=2e-4, rtol=2e-5)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("counts", [(64,) * 6, (1, 3, 7, 11, 0, 0),
                                     (2049, 33, 65, 0, 0, 0), (2049, 1, 3, 0, 0, 0)])
@pytest.mark.parametrize("budget", [.015625, 2])
def test_adaptive_pq_distributions_match_decoded_oracle(tmp_path, monkeypatch, device, counts, budget):
    """Exercise adaptive tensor expressions independently of compiler support."""
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    import tensorvs._torch_search as native
    monkeypatch.setattr(native, "_compute", lambda fn, use_compile, specialization=(): fn)
    gen = torch.Generator().manual_seed(1209)
    labels = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    vectors = torch.randn(len(labels), 12, generator=gen)
    centers = torch.randn(len(counts), 12, generator=gen)
    books = torch.randn(3, 16, 4, generator=gen)
    codes = torch.randint(0, 16, (len(labels), 3), generator=gen, dtype=torch.uint8)
    decoded = centers[labels] + torch.stack([books[m, codes[:, m].long()] for m in range(3)], 1).flatten(1)
    build = IVFPQBuildResult(vectors, labels, centers, books, codes, torch.device("cpu"), 3, 4)
    index = load_ivfpq_index(save_ivfpq_index(build, tmp_path / "adaptive.bin"))
    queries = torch.randn(7, 12, generator=gen)
    queries[-1] = centers[-1]
    for probe, k in ((1, 7), (3, 7), (6, 7), (6, len(labels) + 9)):
        values, ids = search_ivfpq_index(index, queries, device=device, backend="torch",
            nprobe=probe, topk=k, torch_pq_scan="gpu", use_compile=True,
            torch_query_batch_size=4, torch_candidate_chunk_size=64,
            torch_memory_budget_mb=max(budget, 384 * k / (1 << 20)),
            lut_dtype="fp32", persist_layout=False)
        expected, _ = reference(queries, decoded, labels, centers, probe, k)
        torch.testing.assert_close(values.cpu(), expected, atol=2e-4, rtol=2e-5)
        valid = ids.cpu() >= 0
        actual = (queries[:, None] - decoded[ids.cpu().clamp_min(0)]).square().sum(-1)
        torch.testing.assert_close(values.cpu()[valid], actual[valid], atol=2e-4, rtol=2e-5)
        assert torch.equal(valid, torch.isfinite(values.cpu()))
        for row in ids.cpu():
            assert row[row >= 0].unique().numel() == (row >= 0).sum()


def test_adaptive_pq_plan_respects_budget_and_uses_index_statistics():
    from tensorvs._torch_search import _pq_scan_plan
    uniform = SimpleNamespace(pq_scan_upper_bounds=(0, 1000, 2000, 3000), pq_anchor_grain=512)
    short = SimpleNamespace(pq_scan_upper_bounds=(0, 1000, 1002, 1003), pq_anchor_grain=0)
    large = _pq_scan_plan(uniform, 64, 3, 10, 16 << 20, 1024, True, 64)
    assert large["mapping"] == "anchors" and large["anchor_grain"] == 512
    assert large["selection"] == "regions"
    assert large["estimated_workspace_bytes"] <= 16 << 20
    generic = _pq_scan_plan(short, 64, 3, 10, 16 << 20, 1024, True, 64)
    assert generic["mapping"] == "binary" and generic["selection"] == "global"
    single = _pq_scan_plan(short, 64, 1, 10, 16 << 20, 1024, True, 64)
    assert single["selection"] == "global" and single["anchor_grain"] == 0
    tiny = _pq_scan_plan(uniform, 64, 3, 10, 64 << 10, 64, True, 4)
    assert tiny["selection"] == "stream"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
@pytest.mark.parametrize("counts", [(64, 65, 67, 70, 0, 0), (1, 3, 7, 11, 0, 0)])
@pytest.mark.parametrize("scan", ["gpu", "bucketed"])
def test_compiled_adaptive_pq_empty_lists_and_mapping(tmp_path, counts, scan):
    gen = torch.Generator().manual_seed(313)
    labels = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    vectors = torch.randn(len(labels), 8, generator=gen)
    centers = torch.randn(len(counts), 8, generator=gen)
    books = torch.randn(2, 16, 4, generator=gen)
    codes = torch.randint(0, 16, (len(labels), 2), generator=gen, dtype=torch.uint8)
    decoded = centers[labels] + torch.stack([books[m, codes[:, m].long()] for m in range(2)], 1).flatten(1)
    build = IVFPQBuildResult(vectors, labels, centers, books, codes, torch.device("cpu"), 2, 4)
    index = load_ivfpq_index(save_ivfpq_index(build, tmp_path / "compiled_adaptive.bin"))
    queries = torch.randn(5, 8, generator=gen)
    queries[-1] = centers[-1]
    with torch.inference_mode():
        for probe in (1, 3, 6):
            values, ids = search_ivfpq_index(index, queries, device="mps", backend="torch",
                nprobe=probe, topk=9, torch_pq_scan=scan, use_compile=True,
                torch_query_batch_size=4, torch_candidate_chunk_size=32,
                torch_memory_budget_mb=2, lut_dtype="fp32", persist_layout=False)
            expected, _ = reference(queries, decoded, labels, centers, probe, 9)
            torch.testing.assert_close(values.cpu(), expected, atol=2e-4, rtol=2e-5)
            valid = ids.cpu() >= 0
            assert torch.equal(valid, torch.isfinite(values.cpu()))
            actual = (queries[:, None] - decoded[ids.cpu().clamp_min(0)]).square().sum(-1)
            torch.testing.assert_close(values.cpu()[valid], actual[valid], atol=2e-4, rtol=2e-5)


@pytest.mark.parametrize("cuda,mps,expected", [
    (True, True, "cuda:0"), (True, False, "cuda:0"),
    (False, True, "mps"), (False, False, "cpu"),
])
def test_default_device_priority_and_overrides(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert get_default_device() == torch.device(expected)
    q = torch.zeros(1, 7)
    assert search_device(q, None) == torch.device(expected)
    assert search_device(q, "cpu") == torch.device("cpu")
    assert search_device(q, "cuda:2") == torch.device("cuda:2")
    assert search_device(q, "mps") == torch.device("mps")
    # Check device precedence without allocating unavailable accelerators.
    for device in ("cuda:2", "mps:0"):
        resident_queries = SimpleNamespace(device=torch.device(device))
        assert search_device(resident_queries, None) == torch.device(device)
        assert search_device(resident_queries, "cpu") == torch.device("cpu")


@pytest.mark.parametrize("kind", ["flat", "pq"])
@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_default_mps_routes_search_to_torch(indexes, monkeypatch, kind, backend):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    module = importlib.import_module(f"tensorvs.index_ivf{kind}")
    index = indexes[0 if kind == "flat" else 1]
    search = search_ivfflat_index if kind == "flat" else search_ivfpq_index
    result = object()

    def capture(index_arg, queries, **kwargs):
        assert index_arg is index
        assert kwargs["device"] == torch.device("mps")
        assert kwargs["kind"] == kind
        return result

    monkeypatch.setattr(module, "search_torch", capture)
    assert search(index, indexes[4], backend=backend) is result
    with pytest.raises(RuntimeError, match="Triton"):
        search(index, indexes[4], backend="triton")


@pytest.mark.parametrize("kind", ["flat", "pq"])
@pytest.mark.parametrize("device", ["mps", "cuda:0"])
def test_auto_residency_without_triton_uses_torch(indexes, monkeypatch, kind, device):
    module = importlib.import_module(f"tensorvs.index_ivf{kind}")
    monkeypatch.setattr(module, "_triton", None)
    index = indexes[0 if kind == "flat" else 1]
    calls = []
    monkeypatch.setattr(module, "prepare_torch_layout",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    assert index.reside_on_device(device, backend="auto", persist_layout=False) is index
    assert len(calls) == 1
    assert calls[0][0][0] is index
    assert calls[0][0][1:] == (kind, torch.device(device))
    assert calls[0][1] == {"persist": False}


def test_cpu_auto_search_without_triton(indexes):
    # A fresh interpreter verifies optional imports as well as both searches.
    script = textwrap.dedent("""
        import importlib.abc
        import json
        import sys
        import torch

        for name in list(sys.modules):
            if name == "triton" or name.startswith("triton."):
                del sys.modules[name]

        class NoTriton(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "triton" or fullname.startswith("triton."):
                    raise ModuleNotFoundError("Triton is deliberately unavailable")

        sys.meta_path.insert(0, NoTriton())
        torch.cuda.is_available = lambda: False
        torch.backends.mps.is_available = lambda: False
        from tensorvs import (get_default_device, load_ivfflat_index,
                              load_ivfpq_index, search_ivfflat_index,
                              search_ivfpq_index)
        assert get_default_device() == torch.device("cpu")
        q = torch.load(sys.argv[3], weights_only=True)
        results = []
        for load, search, path in (
            (load_ivfflat_index, search_ivfflat_index, sys.argv[1]),
            (load_ivfpq_index, search_ivfpq_index, sys.argv[2]),
        ):
            index = load(path)
            index.reside_on_device("cpu", backend="auto", persist_layout=False)
            values, ids = search(index, q, topk=5, nprobe=3, persist_layout=False)
            results.append((values.tolist(), ids.tolist()))
        print(json.dumps(results))
    """)
    query_path = indexes[0].index_dir.parent / "queries.pt"
    torch.save(indexes[4], query_path)
    process = subprocess.run(
        [sys.executable, "-c", script, str(indexes[0].index_dir),
         str(indexes[1].index_dir), str(query_path)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    for i, (values, ids) in enumerate(json.loads(process.stdout)):
        search = search_ivfflat_index if i == 0 else search_ivfpq_index
        expected = search(indexes[i], indexes[4], device="cpu", backend="torch",
                          topk=5, nprobe=3, persist_layout=False)
        torch.testing.assert_close(torch.tensor(values), expected[0])
        torch.testing.assert_close(torch.tensor(ids, dtype=torch.int64), expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["flat", "pq"])
def test_auto_cuda_without_triton_matches_torch(indexes, monkeypatch, kind):
    module = importlib.import_module(f"tensorvs.index_ivf{kind}")
    kernels = importlib.import_module(f"tensorvs._ivf{kind}_triton")
    monkeypatch.setattr(kernels, "triton", None)
    monkeypatch.setattr(module, "_triton", None)
    index = indexes[0 if kind == "flat" else 1]
    search = search_ivfflat_index if kind == "flat" else search_ivfpq_index
    index.reside_on_device("cuda:0", backend="auto")
    assert index.layout_cache.torch_layout.lists.data.device == torch.device("cuda:0")
    assert index.layout_cache.triton_fingerprint is None
    actual = search(index, indexes[4], device="cuda:0", backend="auto", nprobe=3)
    expected = search(index, indexes[4], device="cuda:0", backend="torch", nprobe=3)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
@pytest.mark.parametrize("kind", ["flat", "pq"])
def test_mps_search_matches_independent_distance_oracle(indexes, kind):
    flat, pq, x, decoded, q, labels, centers, rotation = indexes
    index, search, vectors = (flat, search_ivfflat_index, x) if kind == "flat" else (pq, search_ivfpq_index, decoded)
    expected, expected_ids = reference(q, vectors, labels, centers, 3, 5,
                                       rotation if kind == "pq" else None)
    index.reside_on_device("mps", backend="auto")
    values, ids = search(index, q, device="mps", nprobe=3, topk=5,
                         torch_query_batch_size=4, torch_candidate_chunk_size=16)
    assert values.device.type == ids.device.type == "mps"
    values, ids = values.cpu(), ids.cpu()
    torch.testing.assert_close(values, expected, atol=2e-4, rtol=2e-4)
    if kind == "flat":
        torch.testing.assert_close(ids, expected_ids)
    else:
        actual = ((q @ rotation.T)[:, None] - decoded[ids]).square().sum(-1)
        torch.testing.assert_close(actual, values, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("kind", ["flat", "pq"])
@pytest.mark.parametrize("nprobe,k", [(1, 10), (3, 7), (6, 300)])
def test_torch_matches_independent_distance_oracle(indexes, kind, nprobe, k):
    flat, pq, x, decoded, q, labels, centers, rotation = indexes
    index, search, vectors = (flat, search_ivfflat_index, x) if kind == "flat" else (pq, search_ivfpq_index, decoded)
    values, ids = search(index, q, topk=k, nprobe=nprobe, device="cpu", backend="torch",
                         torch_query_batch_size=4, torch_candidate_chunk_size=16, torch_memory_budget_mb=2)
    expected, expected_ids = reference(q, vectors, labels, centers, nprobe, k, rotation if kind == "pq" else None)
    torch.testing.assert_close(values, expected, atol=3e-5, rtol=2e-5)
    # Duplicate PQ codes can produce equal distances, so validate returned
    # distances/eligibility independently rather than prescribing tie order.
    if kind == "flat":
        torch.testing.assert_close(ids, expected_ids)
    else:
        valid = ids >= 0
        actual = ((q @ rotation.T)[:, None] - decoded[ids.clamp_min(0)]).square().sum(-1)
        torch.testing.assert_close(actual[valid], values[valid], atol=3e-5, rtol=2e-5)
    assert torch.equal(ids < 0, ~torch.isfinite(values))


@pytest.mark.parametrize("kind", ["flat", "pq"])
def test_auto_cpu_empty_queries_and_validation(indexes, kind):
    index = indexes[0 if kind == "flat" else 1]
    search = search_ivfflat_index if kind == "flat" else search_ivfpq_index
    q = indexes[4]
    result = search(index, q[:0], device="cpu", backend="auto")
    assert result[0].shape == (0, 10)
    expected = search(index, q, device="cpu", backend="torch")
    actual = search(index, q, device="cpu", backend="auto")
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="backend"):
        search(index, q, device="cpu", backend="unknown")
    with pytest.raises(RuntimeError, match="Triton"):
        search(index, q, device="cpu", backend="triton")
    with pytest.raises(ValueError, match="positive"):
        search(index, q, device="cpu", backend="torch", torch_candidate_chunk_size=0)


def test_cache_reload_reuses_conversion_and_invalidates_changed_source(indexes, monkeypatch):
    from tensorvs import _layout_cache
    flat = indexes[0]
    search_ivfflat_index(flat, indexes[4], device="cpu", backend="torch")
    logical = flat.logical_lists
    assert logical.data.shape == (257, 7)
    assert logical.counts.sum() == 257
    reloaded = load_ivfflat_index(flat.index_dir)
    def fail(*args, **kwargs):
        raise AssertionError("cache hit must not decode the archive again")
    original = _layout_cache.logical_lists
    monkeypatch.setattr(_layout_cache, "logical_lists", fail)
    search_ivfflat_index(reloaded, indexes[4], device="cpu", backend="torch")
    torch.testing.assert_close(reloaded.logical_lists.data, logical.data)
    # A source update changes the cache key, even at the same archive path.
    flat.index_dir.touch()
    monkeypatch.setattr(_layout_cache, "logical_lists", original)
    newer = load_ivfflat_index(flat.index_dir)
    search_ivfflat_index(newer, indexes[4], device="cpu", backend="torch")
    torch.testing.assert_close(newer.logical_lists.data, logical.data)


def test_pq_refinement_is_exact_over_its_candidates(indexes):
    flat, pq, x, decoded, q, labels, centers, rotation = indexes
    _, candidates = search_ivfpq_index(pq, q, topk=30, nprobe=3, device="cpu", backend="torch")
    expected_values, order = (q[:, None] - x[candidates.clamp_min(0)]).square().sum(-1).masked_fill(candidates < 0, float("inf")).topk(10, largest=False)
    values, ids = search_ivfpq_index(pq, q, topk=10, nprobe=3, device="cpu", backend="torch", refine_vectors=x, refine_factor=3)
    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(ids, candidates.gather(1, order))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["flat", "pq"])
def test_gpu_torch_preserves_triton_results_and_profile(indexes, kind):
    index = indexes[0 if kind == "flat" else 1]
    search = search_ivfflat_index if kind == "flat" else search_ivfpq_index
    q = indexes[4].cuda()
    profile = {}
    torch_values, torch_ids = search(index, q, topk=10, nprobe=3, backend="torch", profile=profile,
                                    torch_candidate_chunk_size=32, torch_query_batch_size=4)
    triton_values, triton_ids = search(index, q, topk=10, nprobe=3, backend="triton")
    torch.testing.assert_close(torch_values, triton_values, atol=2e-4, rtol=2e-4)
    if kind == "flat":
        torch.testing.assert_close(torch_ids, triton_ids)
    torch.testing.assert_close(search(index, q, topk=10, nprobe=3, backend="auto"), (triton_values, triton_ids))
    assert profile["centroid_distance"] > 0
    assert profile["topk_selection"] > 0


@pytest.mark.parametrize("kind", ["flat", "pq"])
def test_compiled_blocks_match_eager(indexes, kind):
    index = indexes[0 if kind == "flat" else 1]
    search = search_ivfflat_index if kind == "flat" else search_ivfpq_index
    kwargs = dict(device="cpu", backend="torch", topk=5, nprobe=3,
                  torch_candidate_chunk_size=32, torch_query_batch_size=4)
    expected = search(index, indexes[4], **kwargs)
    actual = search(index, indexes[4], use_compile=True, **kwargs)
    torch.testing.assert_close(actual[0], expected[0], atol=3e-5, rtol=3e-5)
    if kind == "flat":
        torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_device_change_preserves_data_and_refreshes_layout(indexes):
    pq = indexes[1]
    search_ivfpq_index(pq, indexes[4], device="cuda:0", backend="torch")
    first = pq.layout_cache.torch_layout
    values, ids = search_ivfpq_index(pq, indexes[4], device="cuda:1", backend="torch")
    second = pq.layout_cache.torch_layout
    assert first.fingerprint != second.fingerprint
    assert second.lists.data.device == torch.device("cuda:1")
    expected = search_ivfpq_index(pq, indexes[4], device="cpu", backend="torch")
    torch.testing.assert_close(values.cpu(), expected[0], atol=2e-4, rtol=2e-4)
    pq.reside_on_device("cuda:0")
    pq.reside_on_device("cuda:1")
    assert pq.resident_device == torch.device("cuda:1")
    assert pq.layout_cache.triton_fingerprint["device"] == "cuda:1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["flat", "pq"])
def test_compiled_gpu_blocks_match_eager(indexes, kind):
    index = indexes[0 if kind == "flat" else 1]
    search = search_ivfflat_index if kind == "flat" else search_ivfpq_index
    kwargs = dict(device="cuda:0", backend="torch", topk=5, nprobe=3,
                  torch_candidate_chunk_size=32, torch_query_batch_size=4)
    expected = search(index, indexes[4], **kwargs)
    actual = search(index, indexes[4], use_compile=True, **kwargs)
    torch.testing.assert_close(actual[0], expected[0], atol=2e-4, rtol=2e-4)
    if kind == "flat":
        torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["flat", "pq"])
@pytest.mark.parametrize("compiled", [False, True])
def test_large_gpu_batches_use_streamed_scans_and_match_oracle(indexes, kind, compiled, monkeypatch):
    flat, pq, x, decoded, q, labels, centers, rotation = indexes
    index, search, vectors = (flat, search_ivfflat_index, x) if kind == "flat" else (pq, search_ivfpq_index, decoded)
    q = q.repeat(6, 1)[:64]
    if kind == "pq":
        # The existing LUT policy allows TF32; force IEEE here so the
        # decoded-vector oracle isolates the scan/merge rather than cuBLAS
        # choosing different LUT precision for different matrix shapes.
        from tensorvs import index_ivfpq
        original = index_ivfpq._build_global_query_luts_bmm
        def ieee_luts(*args, **kwargs):
            kwargs["use_tf32"] = False
            return original(*args, **kwargs)
        monkeypatch.setattr(index_ivfpq, "_build_global_query_luts_bmm", ieee_luts)
    profile = {}
    # K exceeds both the entire index and the number of eligible candidates.
    # This also exercises empty lists and final blocks shorter than K.
    values, ids = search(index, q, device="cuda:0", backend="torch", nprobe=3, topk=300,
                         use_compile=compiled, profile=profile)
    expected, expected_ids = reference(q, vectors, labels, centers, 3, 300,
                                       rotation if kind == "pq" else None)
    torch.testing.assert_close(values.cpu(), expected, atol=3e-5, rtol=2e-5)
    if kind == "flat":
        torch.testing.assert_close(ids.cpu(), expected_ids)
    else:
        valid = ids.cpu() >= 0
        actual = ((q @ rotation.T)[:, None] - decoded[ids.cpu().clamp_min(0)]).square().sum(-1)
        torch.testing.assert_close(actual[valid], values.cpu()[valid], atol=3e-5, rtol=2e-5)
    assert torch.equal(ids < 0, ~torch.isfinite(values))
    stage = "dense_vector_distance_topk" if kind == "flat" else "packed_lut_distance_topk"
    assert profile[stage] > 0
    # Switching to a small-budget grouped schedule must preserve results.
    small = search(index, q, device="cuda:0", backend="torch", nprobe=3, topk=7,
                   torch_memory_budget_mb=1)
    torch.testing.assert_close(small[0], values[:, :7], atol=3e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["flat", "pq"])
def test_streamed_scans_with_only_empty_probes(indexes, kind):
    index = indexes[0 if kind == "flat" else 1]
    search = search_ivfflat_index if kind == "flat" else search_ivfpq_index
    q = indexes[6][5:6].repeat(64, 1)  # The nearest list has no vectors.
    values, ids = search(index, q, device="cuda:0", backend="torch", nprobe=1, topk=7)
    assert torch.isinf(values).all()
    assert (ids == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind,budget", [("flat", 6), ("pq", 2)])
def test_streamed_large_k_uses_bounded_online_merge(indexes, kind, budget):
    index = indexes[0 if kind == "flat" else 1]
    search = search_ivfflat_index if kind == "flat" else search_ivfpq_index
    q = indexes[4].repeat(6, 1)[:64]
    kwargs = dict(device="cuda:0", backend="torch", nprobe=3, topk=300)
    expected = search(index, q, torch_candidate_chunk_size=32, **kwargs)
    profile = {}
    actual = search(index, q, use_compile=True, torch_memory_budget_mb=budget,
                    profile=profile, **kwargs)
    torch.testing.assert_close(actual[0], expected[0], atol=3e-5, rtol=2e-5)
    assert torch.equal(actual[1] < 0, ~torch.isfinite(actual[0]))
    stage = "dense_vector_distance_topk" if kind == "flat" else "packed_lut_distance_topk"
    assert profile[stage] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compiled", [False, True])
def test_packed_query_scan_preserves_fp16_luts_and_refinement(indexes, compiled):
    pq, q, x = indexes[1], indexes[4].repeat(6, 1)[:64], indexes[2]
    kwargs = dict(device="cuda:0", backend="torch", nprobe=3, topk=7, lut_dtype="fp16")
    # Keep the LUT matrix shape identical, changing only the scan schedule.
    expected = search_ivfpq_index(pq, q, torch_candidate_chunk_size=32, **kwargs)
    actual = search_ivfpq_index(pq, q, use_compile=compiled, **kwargs)
    torch.testing.assert_close(actual[0], expected[0], atol=3e-5, rtol=2e-5)
    candidates = search_ivfpq_index(pq, q, device="cuda:0", backend="torch", nprobe=3,
                                     topk=21, lut_dtype="fp16", use_compile=compiled)[1].cpu()
    # Equal PQ distances may select different IDs at the cutoff. Validate
    # exact refinement against this path's own approximate candidates.
    exact = (q[:, None] - x[candidates.clamp_min(0)]).square().sum(-1)
    expected, order = exact.masked_fill(candidates < 0, float("inf")).topk(7, largest=False)
    actual = search_ivfpq_index(pq, q, use_compile=compiled,
                                 refine_vectors=x, refine_factor=3, **kwargs)
    torch.testing.assert_close(actual[0].cpu(), expected, atol=3e-5, rtol=2e-5)
    torch.testing.assert_close(actual[1].cpu(), candidates.gather(1, order))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_existing_triton_pq_workspace_and_graph_replay(indexes):
    index = indexes[1]
    kwargs = dict(device="cuda:0", backend="triton", topk=5, nprobe=3)
    first = search_ivfpq_index(index, indexes[4], **kwargs)
    second = search_ivfpq_index(index, indexes[4] + .2, **kwargs)
    graph_first = search_ivfpq_index(index, indexes[4], use_cuda_graph=True,
                                    use_preallocated_workspace=True, **kwargs)
    actual_first = tuple(t.clone() for t in graph_first)
    graph_second = search_ivfpq_index(index, indexes[4] + .2, use_cuda_graph=True,
                                     use_preallocated_workspace=True, **kwargs)
    torch.testing.assert_close(actual_first, first)
    torch.testing.assert_close(graph_second, second)
    assert index.search_workspace.graph is not None


@pytest.mark.parametrize("kind", ["flat", "pq"])
def test_empty_index_returns_missing_ids(indexes, tmp_path, kind):
    if kind == "flat":
        build = IVFFlatBuildResult(torch.empty(0, 7), torch.empty(0, dtype=torch.int64), indexes[6], torch.device("cpu"))
        index = load_ivfflat_index(save_ivfflat_index(build, tmp_path / "empty.bin"))
        search = search_ivfflat_index
    else:
        pq = indexes[1]
        build = IVFPQBuildResult(torch.empty(0, 7), torch.empty(0, dtype=torch.int64), indexes[6],
                                pq.pq_codebooks, torch.empty(0, 3, dtype=torch.uint8), torch.device("cpu"),
                                3, 4, pq_rotation_matrix=indexes[7])
        index = load_ivfpq_index(save_ivfpq_index(build, tmp_path / "empty.bin"))
        search = search_ivfpq_index
    values, ids = search(index, indexes[4], device="cpu", topk=5)
    assert torch.isinf(values).all()
    assert (ids == -1).all()


@pytest.mark.parametrize("kind", ["flat", "pq"])
def test_tiny_budget_uses_online_merge(indexes, kind, monkeypatch):
    from tensorvs import _torch_search
    index = indexes[0 if kind == "flat" else 1]
    search = search_ivfflat_index if kind == "flat" else search_ivfpq_index
    q = indexes[4][:2]
    kwargs = dict(device="cpu", backend="torch", topk=5, nprobe=3,
                  torch_query_batch_size=2, torch_candidate_chunk_size=16)
    expected = search(index, q, **kwargs)
    def fail(*args, **kwargs):
        raise AssertionError("tiny budgets must not allocate the partial-result reservoir")
    monkeypatch.setattr(_torch_search, "_collect", fail)
    actual = search(index, q, torch_memory_budget_mb=.015625, use_compile=True, **kwargs)
    torch.testing.assert_close(actual[0], expected[0], atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("scan", ["auto", "gpu", "bucketed"])
def test_unsigned_16bit_pq_codes_use_portable_backend(tmp_path, monkeypatch, device, scan):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    monkeypatch.setenv("TENSORVS_LAYOUT_CACHE_DIR", str(tmp_path / "cache"))
    gen = torch.Generator().manual_seed(145)
    x = torch.randn(65, 2, generator=gen)
    centers = torch.randn(2, 2, generator=gen)
    labels = torch.randint(0, 2, (65,), generator=gen)
    books = torch.randn(2, 65536, 1, generator=gen)
    unsigned_codes = torch.randint(0, 65536, (65, 2), generator=gen)
    build = IVFPQBuildResult(x, labels, centers, books, unsigned_codes.to(torch.int16),
                            torch.device("cpu"), 2, 16)
    index = load_ivfpq_index(save_ivfpq_index(build, tmp_path / "pq16.bin"))
    q = torch.randn(3, 2, generator=gen)
    decoded = centers[labels] + torch.stack([books[m, unsigned_codes[:, m], 0] for m in range(2)], 1)
    actual = search_ivfpq_index(index, q, device=device, backend="auto", nprobe=2, topk=5, torch_pq_scan=scan)
    expected = reference(q, decoded, labels, centers, 2, 5)
    torch.testing.assert_close(actual[0].cpu(), expected[0], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(actual[1].cpu(), expected[1])
    if device == "cpu" and torch.cuda.is_available():
        repeated = q.repeat(11, 1)
        profile = {}
        actual = search_ivfpq_index(index, repeated, device="cuda:0", backend="torch",
                                    nprobe=2, topk=5, profile=profile)
        expected = reference(repeated, decoded, labels, centers, 2, 5)
        torch.testing.assert_close(actual[0].cpu(), expected[0], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(actual[1].cpu(), expected[1])
        assert profile["packed_lut_distance_topk"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp16_pq_luts_and_packed_device_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORVS_LAYOUT_CACHE_DIR", str(tmp_path / "cache"))
    gen = torch.Generator().manual_seed(88)
    x = torch.randn(70, 25, generator=gen)
    centers = torch.randn(3, 25, generator=gen)
    labels = torch.randint(0, 3, (70,), generator=gen)
    books = torch.randn(25, 16, 1, generator=gen)
    codes = torch.randint(0, 16, (70, 25), generator=gen, dtype=torch.uint8)
    build = IVFPQBuildResult(x, labels, centers, books, codes, torch.device("cpu"), 25, 4)
    path = save_ivfpq_index(build, tmp_path / "packed.bin")
    index = load_ivfpq_index(path)
    q = torch.randn(5, 25, generator=gen)
    kwargs = dict(device="cuda:0", nprobe=3, topk=5, lut_dtype="fp16")
    expected = search_ivfpq_index(index, q, backend="triton", **kwargs)
    assert index.list_codes_interleaved_packed_16b is not None
    expected_packed = index.list_codes_interleaved_packed_16b.cpu()
    actual = search_ivfpq_index(index, q, backend="torch", **kwargs)
    assert index.list_codes_interleaved_packed_16b is None
    torch.testing.assert_close(actual[0], expected[0], atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual[1], expected[1])
    # Reloading reuses packed conversion; upload itself is device-local.
    import tensorvs.index_ivfpq as pq_module
    def fail(*args, **kwargs):
        raise AssertionError("packed cache should be reused on reload")
    monkeypatch.setattr(pq_module, "_pack_interleaved_pq_code_chunks", fail)
    reloaded = load_ivfpq_index(path).reside_on_device("cuda:0")
    torch.testing.assert_close(reloaded.list_codes_interleaved_packed_16b.cpu(), expected_packed)
