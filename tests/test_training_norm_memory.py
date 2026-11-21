"""Training norms must preserve values while bounding temporary allocations."""
import pytest
import torch
from flash_kmeans.training import _bounded_squared_norms


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dim", [8, 128])
@pytest.mark.parametrize("rows_per_chunk", [1, 17, 10000])
def test_bounded_norms_are_bitwise_identical(device, dim, rows_per_chunk):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    gen = torch.Generator().manual_seed(921)
    data = torch.randn(4097, dim, generator=gen).to(device)
    expected = (data * data).sum(-1)
    actual = _bounded_squared_norms(data, chunk_bytes=rows_per_chunk * dim * 4)
    assert torch.equal(actual, expected)


def test_norm_temporary_is_bounded_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    data = torch.randn(262144, 128, device="cuda:0")
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = _bounded_squared_norms(data, chunk_bytes=4 << 20)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - baseline
    assert result.shape == (len(data),)
    assert extra < 16 << 20  # Full multiplication alone would require 128 MiB.
