"""ADC layout/address changes against decoded-vector and wide-index oracles."""
from types import SimpleNamespace

import pytest
import torch

from tensorvs import (IVFPQBuildResult, load_ivfpq_index, save_ivfpq_index,
                      search_ivfpq_index)
from tensorvs._layout_cache import prepare_pq_adc_words
from tensorvs._torch_search import _compute, _pq_anchor_mapping, _pq_anchor_scores


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("m", [4, 5, 10, 16])
def test_adc_word_layout_preserves_unsigned_bytes_and_reuses_storage(device, m):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    gen = torch.Generator().manual_seed(608)
    data = torch.randint(0, 256, (97, m), generator=gen, dtype=torch.uint8).to(device)
    data[0] = 255  # Signed int32 words must also decode their high byte.
    layout = SimpleNamespace(lists=SimpleNamespace(data=data), pq_adc_words=None)
    words = prepare_pq_adc_words(layout)
    assert prepare_pq_adc_words(layout) is words
    decoded = torch.stack([(words >> (8 * byte)) & 255 for byte in range(4)], -1)
    torch.testing.assert_close(decoded.flatten(1)[:, :m].to(torch.uint8), data)
    if m % 4 == 0:
        assert words.untyped_storage().data_ptr() == data.untyped_storage().data_ptr()


def test_adc_wide_addresses_do_not_overflow_int32():
    # Zero-stride views exercise valid >2**31 row addresses without allocating
    # a multi-gigabyte index; the scalar oracle uses the same decoded PQ vector.
    rows = 2**31 + 9
    data = torch.tensor([[255, 128, 1, 0]], dtype=torch.uint8).expand(rows, -1)
    corrections = torch.tensor([7.]).expand(rows)
    lut = torch.arange(1024, dtype=torch.float32).reshape(1, 4, 256)
    delta = torch.tensor([[rows - 2]])
    boundary = torch.tensor([[2]])
    bias = torch.tensor([[3.]])
    scores = _pq_anchor_scores(lut, data, corrections, delta, delta, boundary,
                               bias, bias, torch.tensor([2]), torch.arange(3), 32)
    expected = lut[0, torch.arange(4), data[0].long()].sum() + 10
    torch.testing.assert_close(scores[0, :2], expected.expand(2))
    assert torch.isinf(scores[0, 2])


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16])
def test_narrow_adc_handles_list_boundaries_and_16bit_codes(device, dtype):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    gen = torch.Generator().manual_seed(622)
    ksub = 256 if dtype == torch.uint8 else 65536
    codes = torch.randint(0, ksub, (160, 3), generator=gen)
    codes[0] = ksub - 1
    luts = torch.randn(2, 3, ksub, generator=gen)
    corrections = torch.randn(160, generator=gen)
    probes = torch.tensor([[1, 0], [0, 1]], device=device)
    counts = torch.tensor([[96, 64], [64, 96]], device=device)
    prefix = counts.cumsum(1)
    previous = prefix - counts
    bias = torch.tensor([[1.25, -2.5], [3.5, -1.]], device=device)
    layout = SimpleNamespace(lists=SimpleNamespace(offsets=torch.tensor([0, 64], device=device)))
    mapping = _pq_anchor_mapping(layout, probes, prefix, previous, bias, 192, 64)
    fn = _compute(_pq_anchor_scores, device == "mps", ("adc_boundary", device, dtype))
    with torch.inference_mode():
        actual = fn(luts.to(device), codes.to(device=device, dtype=dtype), corrections.to(device),
                    *mapping, torch.arange(192, device=device), 64)
    order = torch.stack((torch.cat((torch.arange(64, 160), torch.arange(64))), torch.arange(160)))
    query = torch.arange(2)[:, None, None]
    subspace = torch.arange(3)[None, None, :]
    expected = luts[query, subspace, codes[order]].sum(-1)
    biases = torch.stack((torch.cat((bias.cpu()[0, 0].expand(96), bias.cpu()[0, 1].expand(64))),
                          torch.cat((bias.cpu()[1, 0].expand(64), bias.cpu()[1, 1].expand(96)))))
    expected = (expected + corrections[order] + biases).clamp_min(0)
    torch.testing.assert_close(actual.cpu()[:, :160], expected, atol=2e-6, rtol=2e-5)
    assert torch.isinf(actual.cpu()[:, 160:]).all()


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("m", [4, 5, 10])
@pytest.mark.parametrize("scan", ["gpu", "bucketed"])
def test_word_adc_search_matches_decoded_oracle(tmp_path, monkeypatch, device, m, scan):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    # CPU expressions exercise the portable algorithm independently of the
    # machine's unrelated CPU compiler toolchain; MPS uses real compilation.
    if device == "cpu":
        import tensorvs._torch_search as native
        import tensorvs._torch_pq_bucketed as bucket
        monkeypatch.setattr(native, "_compute", lambda fn, use_compile, specialization=(): fn)
        monkeypatch.setattr(bucket, "_compute", lambda fn, use_compile, specialization=(): fn)
    gen = torch.Generator().manual_seed(918)
    counts = torch.tensor([97, 65, 129, 0])
    labels = torch.repeat_interleave(torch.arange(4), counts)
    centers = torch.randn(4, m * 2, generator=gen)
    vectors = torch.randn(len(labels), m * 2, generator=gen)
    books = torch.randn(m, 256, 2, generator=gen)
    codes = torch.randint(0, 256, (len(labels), m), generator=gen, dtype=torch.uint8)
    codes[0] = 255
    residuals = torch.stack([books[s, codes[:, s].long()] for s in range(m)], 1).flatten(1)
    decoded = centers[labels] + residuals
    index = load_ivfpq_index(save_ivfpq_index(IVFPQBuildResult(
        vectors, labels, centers, books, codes, torch.device("cpu"), m, 8), tmp_path / "adc.bin"))
    queries = torch.randn(5, m * 2, generator=gen)
    queries[-1] = centers[-1]
    with torch.inference_mode():
        for probe in (1, 4):
            values, ids = search_ivfpq_index(index, queries, device=device, backend="torch",
                topk=9, nprobe=probe, use_compile=True, torch_pq_scan=scan,
                torch_query_batch_size=4, torch_candidate_chunk_size=64,
                torch_memory_budget_mb=2, lut_dtype="fp32", persist_layout=False)
            probes = (queries[:, None] - centers).square().sum(-1).topk(probe, largest=False).indices
            allowed = (labels[None, :, None] == probes[:, None, :]).any(-1)
            oracle = (queries[:, None] - decoded).square().sum(-1).masked_fill(~allowed, float("inf"))
            expected, expected_ids = oracle.topk(9, largest=False)
            expected_ids.masked_fill_(~torch.isfinite(expected), -1)
            torch.testing.assert_close(values.cpu(), expected, atol=3e-4, rtol=3e-5)
            torch.testing.assert_close(ids.cpu().sort(1).values, expected_ids.sort(1).values)
    assert index.layout_cache.torch_layout.pq_adc_words is not None
    if device == "mps":
        transfers = []
        original_cpu = torch.Tensor.cpu
        def checked_cpu(tensor, *args, **kwargs):
            if tensor.device.type == "mps":
                assert scan == "bucketed" and tensor.ndim == 1 and tensor.dtype == torch.int64
                transfers.append(tensor.shape[0])
            return original_cpu(tensor, *args, **kwargs)
        def reject_readback(original):
            def checked(tensor, *args, **kwargs):
                assert tensor.device.type != "mps", "ADC must keep query data on device"
                return original(tensor, *args, **kwargs)
            return checked
        with monkeypatch.context() as patch, torch.inference_mode():
            patch.setattr(torch.Tensor, "cpu", checked_cpu)
            for name in ("item", "tolist", "numpy", "__int__", "__float__", "__bool__"):
                patch.setattr(torch.Tensor, name, reject_readback(getattr(torch.Tensor, name)))
            search_ivfpq_index(index, queries.to("mps"), device=device, backend="torch",
                topk=9, nprobe=4, use_compile=True, torch_pq_scan=scan,
                torch_query_batch_size=4, torch_candidate_chunk_size=64,
                torch_memory_budget_mb=2, lut_dtype="fp32", persist_layout=False)
        assert transfers == ([4, 1] if scan == "bucketed" else [])
