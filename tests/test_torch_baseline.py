"""Compare portable baselines against independent decoded-vector oracles."""
import subprocess
import sys
import textwrap

import pytest
import torch

from tensorvs import (IVFFlatBuildResult, IVFPQBuildResult, load_ivfflat_index,
                      load_ivfpq_index, save_ivfflat_index, save_ivfpq_index)
from tensorvs.torch_baseline import TorchIVFFlatBaseline, TorchIVFPQBaseline


@pytest.fixture
def data():
    generator = torch.Generator().manual_seed(72)
    vectors = torch.randn(73, 7, generator=generator)
    centers = torch.randn(5, 7, generator=generator)
    centers[4] = 10  # An empty list that is nearest to the final query.
    labels = torch.randint(0, 4, (73,), generator=generator)
    labels[:45] = 0
    queries = torch.randn(7, 14, generator=generator)[:, ::2]  # Noncontiguous.
    queries[-1] = centers[4]
    ids = torch.arange(73) * 7 + 100
    return vectors, centers, labels, queries, ids, generator


def oracle(queries, vectors, centers, labels, ids, nprobe, k, rotation=None):
    probes = (queries[:, None] - centers).square().sum(-1).topk(min(nprobe, centers.shape[0]),
                                                            largest=False).indices
    allowed = (labels[None, :, None] == probes[:, None, :]).any(-1)
    q = queries if rotation is None else queries @ rotation.T
    distances = (q[:, None] - vectors).square().sum(-1).masked_fill(~allowed, float("inf"))
    values, order = distances.topk(min(k, vectors.shape[0]), largest=False)
    selected = ids[order].masked_fill(~torch.isfinite(values), -1)
    if k > vectors.shape[0]:
        values = torch.cat((values, values.new_full((q.shape[0], k - vectors.shape[0]), float("inf"))), 1)
        selected = torch.cat((selected, selected.new_full((q.shape[0], k - vectors.shape[0]), -1)), 1)
    return values, selected


def assert_pq_ids(values, actual_ids, queries, decoded, centers, labels, ids, nprobe, rotation=None):
    # Identical residual codes in the same list decode to the same vector;
    # torch.topk is allowed to choose different IDs at these distance ties.
    valid = actual_ids >= 0
    matches = actual_ids[:, :, None] == ids[None, None, :]
    assert (matches.any(-1) == valid).all()
    positions = matches.to(torch.int64).argmax(-1)
    q = queries if rotation is None else queries @ rotation.T
    exact = (q[:, None] - decoded[positions]).square().sum(-1)
    torch.testing.assert_close(values[valid], exact[valid], atol=3e-5, rtol=3e-5)
    probes = (queries[:, None] - centers).square().sum(-1).topk(min(nprobe, centers.shape[0]),
                                                            largest=False).indices
    eligible = (labels[positions, None] == probes[:, None, :]).any(-1)
    assert eligible[valid].all()


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "mps"])
@pytest.mark.parametrize("kind", ["flat", "pq", "rotated_pq"])
def test_raw_baselines_match_oracle(data, kind, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    vectors, centers, labels, queries, ids, generator = data
    rotation = None
    if kind == "flat":
        baseline = TorchIVFFlatBaseline(vectors, labels, centers, ids, device=device)
        decoded = vectors
    else:
        dim = 9 if kind == "rotated_pq" else 7
        m, subdim = (3, 3) if dim == 9 else (1, 7)
        if dim == 9:
            rotation = torch.linalg.qr(torch.randn(9, 7, generator=generator), mode="reduced").Q
        books = torch.randn(m, 8, subdim, generator=generator)
        codes = torch.randint(0, 8, (73, m), generator=generator, dtype=torch.uint8)
        baseline = TorchIVFPQBaseline(codes, labels, centers, books, ids,
                                     rotation=rotation, device=device)
        pq_centers = centers if rotation is None else centers @ rotation.T
        residual = torch.stack([books[j, codes[:, j].long()] for j in range(m)], 1).flatten(1)
        decoded = pq_centers[labels] + residual
    for nprobe in (1, 3, 50):
        for k in (5, 90):
            kwargs = dict(candidate_chunk_size=3)
            if kind != "flat":
                kwargs["lut_probe_chunk_size"] = 2
            values, actual_ids = baseline.search(queries, topk=k, nprobe=nprobe, **kwargs)
            expected, expected_ids = oracle(queries, decoded, centers, labels, ids, nprobe, k, rotation)
            torch.testing.assert_close(values.cpu(), expected, atol=3e-5, rtol=3e-5)
            if kind == "flat":
                torch.testing.assert_close(actual_ids.cpu(), expected_ids)
            else:
                assert_pq_ids(values.cpu(), actual_ids.cpu(), queries, decoded, centers,
                              labels, ids, nprobe, rotation)
            assert values.device.type == actual_ids.device.type == torch.device(device).type
    empty_values, empty_ids = baseline.search(queries[:0], topk=9)
    assert empty_values.shape == empty_ids.shape == (0, 9)
    moved = baseline.to("cpu")
    assert moved is not baseline and moved.device.type == "cpu"
    torch.testing.assert_close(moved.search(queries, nprobe=3)[0], baseline.search(queries, nprobe=3)[0].cpu(),
                               atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("kind", ["flat", "pq", "rotated_pq"])
def test_existing_index_adapter_is_separate(data, tmp_path, kind):
    vectors, centers, labels, queries, _, generator = data
    rotation = None
    if kind == "flat":
        build = IVFFlatBuildResult(vectors, labels, centers, torch.device("cpu"))
        path = save_ivfflat_index(build, tmp_path / "flat.bin")
        index = load_ivfflat_index(path)
        baseline_cls = TorchIVFFlatBaseline
        decoded = vectors
    else:
        m, subdim = (3, 3) if kind == "rotated_pq" else (1, 7)
        if kind == "rotated_pq":
            rotation = torch.linalg.qr(torch.randn(9, 7, generator=generator), mode="reduced").Q
        books = torch.randn(m, 8, subdim, generator=generator)
        codes = torch.randint(0, 8, (73, m), generator=generator, dtype=torch.uint8)
        build = IVFPQBuildResult(vectors, labels, centers, books, codes, torch.device("cpu"), m, 3,
                                pq_rotation_matrix=rotation)
        path = save_ivfpq_index(build, tmp_path / "pq.bin")
        index = load_ivfpq_index(path)
        baseline_cls = TorchIVFPQBaseline
        pq_centers = centers if rotation is None else centers @ rotation.T
        residual = torch.stack([books[j, codes[:, j].long()] for j in range(m)], 1).flatten(1)
        decoded = pq_centers[labels] + residual
    index.reside_on_device("cpu", backend="torch", persist_layout=False)
    cache, residency = index.layout_cache.torch_layout, index.resident_device
    baseline = baseline_cls.from_index(index, device="cpu")
    actual = baseline.search(queries, nprobe=3, topk=7, candidate_chunk_size=5)
    expected = oracle(queries, decoded, centers, labels, torch.arange(73), 3, 7, rotation)
    torch.testing.assert_close(actual[0], expected[0], atol=3e-5, rtol=3e-5)
    if kind == "flat":
        torch.testing.assert_close(actual[1], expected[1])
    else:
        assert_pq_ids(actual[0], actual[1], queries, decoded, centers, labels, torch.arange(73), 3, rotation)
    assert index.layout_cache.torch_layout is cache and index.resident_device == residency


def test_signed_16bit_codes_and_fp16_lut():
    centers = torch.zeros(1, 1)
    books = torch.arange(65536, dtype=torch.float32).reshape(1, 65536, 1) / 65536
    codes = torch.tensor([[0], [32767], [-32768], [-1]], dtype=torch.int16)
    baseline = TorchIVFPQBaseline(codes, torch.zeros(4, dtype=torch.int64), centers, books, device="cpu")
    assert baseline.data[:, 0].tolist() == [0, 32767, 32768, 65535]
    query = torch.tensor([[0.7]])
    values, ids = baseline.search(query, topk=4, nprobe=1, lut_dtype="fp16", candidate_chunk_size=2)
    decoded = books[0, baseline.data[:, 0].long()]
    expected, order = (decoded - query).square().sum(-1).half().float().topk(4, largest=False)
    torch.testing.assert_close(values[0], expected)
    torch.testing.assert_close(ids[0], order)


@pytest.mark.parametrize("kind", ["flat", "pq"])
def test_empty_index_and_invalid_search(kind):
    centers = torch.zeros(2, 3)
    labels = torch.empty(0, dtype=torch.int64)
    if kind == "flat":
        baseline = TorchIVFFlatBaseline(torch.empty(0, 3), labels, centers, device="cpu")
    else:
        baseline = TorchIVFPQBaseline(torch.empty(0, 1, dtype=torch.uint8), labels, centers,
                                     torch.zeros(1, 8, 3), device="cpu")
    values, ids = baseline.search(torch.ones(2, 3), topk=4)
    assert torch.isinf(values).all() and (ids == -1).all()
    for kwargs in ({"topk": 0}, {"nprobe": 0}, {"candidate_chunk_size": 0}):
        with pytest.raises(ValueError, match="positive"):
            baseline.search(torch.ones(2, 3), **kwargs)
    with pytest.raises(ValueError, match="shape"):
        baseline.search(torch.ones(2, 4))


def test_baselines_work_without_triton():
    script = textwrap.dedent('''
        import importlib.abc
        import sys
        import torch

        class NoTriton(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "triton" or fullname.startswith("triton."):
                    raise ModuleNotFoundError("Triton unavailable for baseline test")

        for name in list(sys.modules):
            if name == "triton" or name.startswith("triton."):
                del sys.modules[name]
        sys.meta_path.insert(0, NoTriton())
        from tensorvs.torch_baseline import TorchIVFFlatBaseline, TorchIVFPQBaseline
        centers = torch.tensor([[0., 0.], [5., 5.]])
        x = torch.tensor([[1., 1.], [4., 4.]])
        labels = torch.tensor([0, 1])
        flat = TorchIVFFlatBaseline(x, labels, centers, device="cpu")
        values, ids = flat.search(x[:1], nprobe=2, topk=1)
        assert values.item() == 0 and ids.item() == 0
        books = torch.tensor([[[1., 1.], [-1., -1.]]])
        pq = TorchIVFPQBaseline(torch.tensor([[0], [1]], dtype=torch.uint8), labels,
                               centers, books, device="cpu")
        values, ids = pq.search(x[:1], nprobe=2, topk=1)
        assert values.item() == 0 and ids.item() == 0
    ''')
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=30)
