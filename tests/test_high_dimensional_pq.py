"""High-dimensional M384/8-bit ADC correctness on generated inputs."""
import pytest
import torch

from tensorvs import IVFPQBuildResult, load_ivfpq_index, save_ivfpq_index, search_ivfpq_index


@pytest.mark.parametrize("lut_dtype", ["fp16", "fp32"])
@pytest.mark.parametrize("lut_chunk", [16, 32])
def test_m384_gather_preserves_index_candidates_and_decoded_distances(tmp_path, lut_dtype, lut_chunk):
    generator = torch.Generator().manual_seed(306)
    vectors = torch.randn(97, 768, generator=generator) * .02
    centers = torch.randn(5, 768, generator=generator) * .02
    labels = torch.randint(0, 4, (97,), generator=generator)
    labels[:45] = 0
    books = torch.randn(384, 256, 2, generator=generator) * .02
    codes = torch.randint(0, 256, (97, 384), dtype=torch.uint8, generator=generator)
    source = save_ivfpq_index(IVFPQBuildResult(vectors, labels, centers, books, codes,
        torch.device("cpu"), 384, 8), tmp_path / "m384.bin")
    before = source.read_bytes()
    index = load_ivfpq_index(source)
    queries = torch.randn(7, 768, generator=generator) * .02
    decoded = centers[labels] + torch.stack([books[m, codes[:, m].long()] for m in range(384)], 1).flatten(1)
    for probes in (1, 3, 5):
        options = dict(device="cpu", backend="torch", nprobe=probes, topk=10,
            torch_pq_scan="grouped", lut_dtype=lut_dtype, persist_layout=False,
            torch_candidate_chunk_size=32)
        reference, reference_ids = search_ivfpq_index(index, queries, **options)
        actual, ids = search_ivfpq_index(index, queries, torch_query_tile_size=16,
            torch_memory_budget_mb=512, torch_pq_lookup="gather", torch_lut_chunk_size=lut_chunk, **options)
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ids.sort(1).values, reference_ids.sort(1).values)
        selected = ((queries[:, None] - centers).square().sum(-1)).topk(probes, largest=False).indices
        allowed = (labels[None, :, None] == selected[:, None, :]).any(-1)
        oracle = (queries[:, None] - decoded).square().sum(-1).masked_fill(~allowed, float("inf"))
        expected = oracle.topk(10, largest=False).values
        torch.testing.assert_close(actual, expected, atol=1e-4 if lut_dtype == "fp16" else 1e-5, rtol=1e-5)
    assert source.read_bytes() == before
    assert index.metadata["m"] == 384 and index.metadata["nbits"] == 8
