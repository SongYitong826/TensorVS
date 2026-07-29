import json

import pytest
import torch

from tensorvs import (
    build_ivfflat_index,
    build_ivfpq_index,
    compute_recall_at_k,
    load_ivfflat_index,
    load_ivfpq_index,
    save_ivfflat_index,
    save_ivfpq_index,
    search_ivfflat_index,
    search_ivfpq_index,
)
from tensorvs.distance import squared_l2_distance
from tensorvs._ivf_interleaved import (
    INTERLEAVED_GROUP_SIZE,
    build_interleaved_flat_lists,
    build_interleaved_pq_lists,
    interleaved_vector_width,
)
from tensorvs._ivfflat_triton import _direct_scan_plan, _list_grouped_gemm_plan
from tensorvs._ivfpq_triton import _direct_scan_plan as _ivfpq_direct_scan_plan
from tensorvs.index_ivfflat import IVFFlatIndex
from tensorvs.index_ivfpq import (
    _build_interleaved_pq_static_corrections,
    _get_interleaved_pq_lists,
    _tensor_topk_pq_lut_reduce_impl,
)


def test_ivfflat_triton_plan_uses_compact_local_topk():
    assert _direct_scan_plan(topk=1, num_queries=128, nprobe=50) == (2, 2, 4)
    assert _direct_scan_plan(topk=10, num_queries=128, nprobe=50) == (16, 2, 32)
    assert _direct_scan_plan(topk=32, num_queries=128, nprobe=50) == (32, 2, 64)
    # A scan tile contains 128 candidates, so larger k retains the wide path.
    assert _direct_scan_plan(topk=33, num_queries=128, nprobe=50) == (128, 2, 256)
    assert _direct_scan_plan(topk=10, num_queries=64, nprobe=50) == (16, 4, 64)


def test_ivfflat_list_grouped_gemm_plan_uses_one_local_topk_per_list():
    # The common SIFT setting emits 16 local candidates for each of 50 lists,
    # then reduces a 1024-wide query-major merge in the second kernel.
    assert _list_grouped_gemm_plan(topk=10, nprobe=50, dim=128) == (16, 1024)
    assert _list_grouped_gemm_plan(topk=32, nprobe=50, dim=128) == (32, 2048)
    # A partial final dimension tile is masked and zero-padded, so dimensions
    # no longer need to be divisible by the 32-wide matrix tile.
    assert _list_grouped_gemm_plan(topk=10, nprobe=50, dim=16) == (16, 1024)
    assert _list_grouped_gemm_plan(topk=10, nprobe=50, dim=130) == (16, 1024)
    assert _list_grouped_gemm_plan(topk=33, nprobe=50, dim=128) is None


def test_ivfpq_triton_plan_uses_direct_query_partitions():
    assert _ivfpq_direct_scan_plan(topk=10, num_queries=64, nprobe=50) == (16, 4, 64)
    assert _ivfpq_direct_scan_plan(topk=10, num_queries=128, nprobe=50) == (16, 2, 32)
    assert _ivfpq_direct_scan_plan(topk=129, num_queries=64, nprobe=50) is None


def test_tensor_pq_lut_layout_matches_subspace_major_lookup():
    generator = torch.Generator().manual_seed(19)
    num_queries, nprobe, max_candidates, m, ksub = 2, 3, 4, 2, 5
    subspace_major_luts = torch.randn(m, num_queries * nprobe, ksub, generator=generator)
    tensor_luts = subspace_major_luts.permute(1, 0, 2).contiguous()
    probe_idx = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 1]], dtype=torch.int64)
    candidate_codes = torch.randint(
        ksub,
        (num_queries, max_candidates, m),
        generator=generator,
        dtype=torch.int64,
    )
    valid_mask = torch.tensor([[True, True, False, True], [True, False, True, True]])

    lut_rows = (
        torch.arange(num_queries, dtype=torch.int64).unsqueeze(1) * nprobe + probe_idx
    ).reshape(-1)
    subspace_idx = torch.arange(m, dtype=torch.int64)
    expected = subspace_major_luts[
        subspace_idx.unsqueeze(0),
        lut_rows.unsqueeze(1),
        candidate_codes.reshape(-1, m),
    ].sum(dim=1).reshape(num_queries, max_candidates)
    expected = expected.masked_fill(~valid_mask, float("inf"))

    actual = _tensor_topk_pq_lut_reduce_impl(tensor_luts, candidate_codes, probe_idx, valid_mask)
    torch.testing.assert_close(actual, expected)

    fp16_luts = tensor_luts.to(torch.float16)
    fp16_expected = subspace_major_luts.to(torch.float16)[
        subspace_idx.unsqueeze(0),
        lut_rows.unsqueeze(1),
        candidate_codes.reshape(-1, m),
    ].sum(dim=1, dtype=torch.float32).reshape(num_queries, max_candidates)
    fp16_expected = fp16_expected.masked_fill(~valid_mask, float("inf"))
    fp16_actual = _tensor_topk_pq_lut_reduce_impl(fp16_luts, candidate_codes, probe_idx, valid_mask)
    assert fp16_actual.dtype == torch.float32
    torch.testing.assert_close(fp16_actual, fp16_expected)


def test_build_ivfflat_index(tmp_path):
    output_dir = tmp_path / "index"
    vectors = torch.randn(128, 16, dtype=torch.float32)

    result = build_ivfflat_index(
        vectors=vectors,
        nlist=4,
        niter=5,
        use_triton=False,
        device="cpu",
        seed=7,
        n_init=2,
    )
    index_dir = save_ivfflat_index(result, output_dir, dataset_path="in-memory")

    assert index_dir == output_dir
    assert (output_dir / "assignments.pt").exists()
    assert (output_dir / "centroids.pt").exists()
    assert (output_dir / "ids.pt").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "list_vectors.pt").exists()
    assert (output_dir / "list_vectors_interleaved.pt").exists()
    assert (output_dir / "list_ids_interleaved.pt").exists()
    assert (output_dir / "list_group_prefix.pt").exists()

    expected_labels = torch.argmin(squared_l2_distance(vectors, result.centroids.cpu()), dim=1)
    torch.testing.assert_close(result.labels.cpu().to(torch.int64), expected_labels)

    repeated = build_ivfflat_index(
        vectors=vectors,
        nlist=4,
        niter=5,
        use_triton=False,
        device="cpu",
        seed=7,
        n_init=2,
    )
    torch.testing.assert_close(result.centroids, repeated.centroids)
    torch.testing.assert_close(result.labels, repeated.labels)
    assert result.coarse_seed == repeated.coarse_seed
    assert result.coarse_inertia == repeated.coarse_inertia

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_path"] == "in-memory"
    assert metadata["output_dir"] == str(output_dir)
    assert metadata["num_vectors"] == 128
    assert metadata["dim"] == 16
    assert metadata["nlist"] == 4
    assert metadata["seed"] == 7
    assert metadata["n_init"] == 2
    assert metadata["coarse_seed"] in (7, 8)
    assert metadata["coarse_inertia"] > 0.0
    assert metadata["interleaved_layout"] == {
        "available": True,
        "group_size": INTERLEAVED_GROUP_SIZE,
        "vector_width": 4,
    }

    index = load_ivfflat_index(output_dir)
    assert index.persisted_list_vectors_interleaved is not None
    assert index.persisted_list_ids_interleaved is not None
    assert index.persisted_list_group_prefix is not None

    single_file_path = tmp_path / "index.bin"
    save_ivfflat_index(result, single_file_path, dataset_path="in-memory")
    single_file_index = load_ivfflat_index(single_file_path)
    assert single_file_index.persisted_list_vectors_interleaved is not None
    assert single_file_index.persisted_list_ids_interleaved is not None
    assert single_file_index.persisted_list_group_prefix is not None

    queries = vectors[:16] + 0.01 * torch.randn(16, 16)
    profile: dict[str, float] = {}
    tensor_distances, tensor_ids = search_ivfflat_index(
        index,
        queries,
        topk=4,
        nprobe=4,
        device="cpu",
        profile=profile,
        selection_mode="tensor_topk",
    )
    assert "sort_by_query_and_distance" in profile

    triton_profile: dict[str, float] = {}
    triton_distances, triton_ids = search_ivfflat_index(
        index,
        queries,
        topk=4,
        nprobe=4,
        device="cpu",
        profile=triton_profile,
        selection_mode="triton_topk",
    )
    assert "sort_by_query_and_distance" in triton_profile
    torch.testing.assert_close(triton_distances, tensor_distances)
    torch.testing.assert_close(triton_ids, tensor_ids)

    for removed_mode in (
        "global_sort",
        "list_gemm_topk",
        "triton_interleaved_direct_topk",
        "triton_interleaved_topk",
    ):
        try:
            search_ivfflat_index(
                index,
                queries,
                topk=4,
                nprobe=4,
                device="cpu",
                selection_mode=removed_mode,
            )
        except ValueError as error:
            assert "Unsupported IVFFlat selection_mode" in str(error)
        else:  # pragma: no cover - guards the removed public API
            raise AssertionError(f"Removed IVFFlat mode was accepted: {removed_mode}")


def test_ivfflat_tensor_topk_handles_empty_lists_and_short_lists(tmp_path):
    index = IVFFlatIndex(
        index_dir=tmp_path,
        metadata={"nlist": 3, "dim": 2},
        vectors=None,
        assignments=torch.tensor([0, 0, 2], dtype=torch.int64),
        centroids=torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]),
        ids=torch.tensor([40, 41, 42], dtype=torch.int64),
        lists=[],
        list_vectors=torch.tensor([[0.0, 0.0], [1.0, 0.0], [20.0, 20.0]]),
        list_vector_norms=torch.tensor([0.0, 1.0, 800.0]),
        list_ids_flat=torch.tensor([40, 41, 42], dtype=torch.int64),
        list_prefix=torch.tensor([0, 2, 2], dtype=torch.int64),
        list_count=torch.tensor([2, 0, 1], dtype=torch.int64),
    )
    queries = torch.tensor([[0.1, 0.0], [11.0, 11.0]])

    tensor_distances, tensor_ids = search_ivfflat_index(
        index,
        queries,
        topk=5,
        nprobe=3,
        device="cpu",
        selection_mode="tensor_topk",
    )

    assert torch.isinf(tensor_distances[:, 3:]).all()
    assert torch.equal(tensor_ids[:, 3:], torch.full((2, 2), -1, dtype=torch.int64))


def test_ivfpq_seed_metadata_and_exact_refinement(tmp_path):
    generator = torch.Generator().manual_seed(123)
    vectors = torch.randn(128, 8, generator=generator)
    queries = vectors[:16] + 0.01 * torch.randn(16, 8, generator=generator)
    output_path = tmp_path / "ivfpq.bin"

    result = build_ivfpq_index(
        vectors=vectors,
        nlist=4,
        niter=5,
        use_triton=False,
        device="cpu",
        m=2,
        nbits=3,
        seed=11,
        n_init=2,
    )
    save_ivfpq_index(result, output_path)
    index = load_ivfpq_index(output_path)

    assert index.metadata["has_pq_static_vector_corrections"]

    assert index.metadata["seed"] == 11
    assert index.metadata["n_init"] == 2
    assert index.metadata["coarse_seed"] in (11, 12)
    assert len(index.metadata["pq_seeds"]) == 2
    assert len(set(index.metadata["pq_seeds"])) == 2

    coarse_expected = torch.argmin(squared_l2_distance(vectors, result.centroids.cpu()), dim=1)
    torch.testing.assert_close(result.labels.cpu().to(torch.int64), coarse_expected)

    profile: dict[str, float] = {}
    approximate_distances, approximate_ids = search_ivfpq_index(
        index,
        queries,
        topk=4,
        nprobe=4,
        device="cpu",
        profile=profile,
        detailed_tensor_topk_profile=True,
    )
    assert "tensor_topk_select" in profile
    for timing_name in (
        "tensor_topk_candidate_map",
        "tensor_topk_code_gather",
        "tensor_topk_lut_reduce",
        "tensor_topk_select_only",
        "tensor_topk_id_lookup",
        "tensor_topk_writeback",
    ):
        assert timing_name in profile

    fp16_lut_profile: dict[str, float] = {}
    fp16_lut_distances, fp16_lut_ids = search_ivfpq_index(
        index,
        queries,
        topk=4,
        nprobe=4,
        device="cpu",
        profile=fp16_lut_profile,
        lut_dtype="fp16",
    )
    assert "LUT_prepare" in fp16_lut_profile
    assert fp16_lut_distances.dtype == torch.float32
    assert fp16_lut_ids.dtype == torch.int64
    torch.testing.assert_close(fp16_lut_distances, approximate_distances, rtol=2e-3, atol=2e-3)

    with pytest.raises(ValueError, match="Unsupported LUT dtype"):
        search_ivfpq_index(
            index,
            queries,
            topk=4,
            nprobe=4,
            device="cpu",
            lut_dtype="fp8",
        )

    triton_distances, triton_ids = search_ivfpq_index(
        index,
        queries,
        topk=4,
        nprobe=4,
        device="cpu",
        selection_mode="triton_topk",
    )
    torch.testing.assert_close(triton_distances, approximate_distances)
    torch.testing.assert_close(triton_ids, approximate_ids)
    for removed_mode in ("global_sort", "per_query_topk", "triton_interleaved_topk"):
        try:
            search_ivfpq_index(
                index,
                queries,
                topk=4,
                nprobe=4,
                device="cpu",
                selection_mode=removed_mode,
            )
        except ValueError as error:
            assert "Unsupported IVFPQ selection_mode" in str(error)
        else:  # pragma: no cover - guards the removed public API
            raise AssertionError(f"Removed IVFPQ mode was accepted: {removed_mode}")
    refined_distances, refined_ids = search_ivfpq_index(
        index,
        queries,
        topk=1,
        nprobe=4,
        device="cpu",
        refine_vectors=vectors,
        refine_factor=4,
    )

    groundtruth = torch.argmin(squared_l2_distance(queries, vectors), dim=1, keepdim=True)
    assert compute_recall_at_k(refined_ids, groundtruth, 1) >= compute_recall_at_k(
        approximate_ids,
        groundtruth,
        1,
    )
    expected_distances = ((queries - vectors[refined_ids[:, 0]]) ** 2).sum(dim=1)
    torch.testing.assert_close(refined_distances[:, 0], expected_distances)


def test_ivfpq_persists_dual_list_layouts(tmp_path):
    generator = torch.Generator().manual_seed(79)
    vectors = torch.randn(128, 8, generator=generator)
    result = build_ivfpq_index(
        vectors=vectors,
        nlist=4,
        niter=5,
        use_triton=False,
        device="cpu",
        m=2,
        nbits=3,
        seed=5,
        n_init=2,
    )

    output_dir = tmp_path / "ivfpq"
    save_ivfpq_index(result, output_dir)
    for name in (
        "pq_codebook_norms.pt",
        "list_codes.pt",
        "list_ids_flat.pt",
        "list_prefix.pt",
        "list_count.pt",
        "list_codes_interleaved_by_subspace.pt",
        "list_ids_interleaved.pt",
        "list_group_prefix.pt",
        "list_pq_static_corrections_interleaved.pt",
    ):
        assert (output_dir / name).exists()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["interleaved_layout"] == {
        "available": True,
        "group_size": INTERLEAVED_GROUP_SIZE,
        "code_layout": "subspace_major",
    }

    index = load_ivfpq_index(output_dir)
    assert index.list_codes is not None
    assert index.persisted_list_codes_interleaved_by_subspace is not None
    assert index.persisted_list_ids_interleaved is not None
    assert index.persisted_list_group_prefix is not None
    assert index.persisted_list_pq_static_corrections_interleaved is not None
    torch.testing.assert_close(
        index.pq_codebook_norms,
        (index.pq_codebooks * index.pq_codebooks).sum(dim=-1),
    )
    expected_layout = build_interleaved_pq_lists(
        index.list_codes,
        index.list_ids_flat,
        index.list_prefix,
        index.list_count,
    )
    torch.testing.assert_close(
        index.persisted_list_codes_interleaved_by_subspace,
        expected_layout.codes_by_subspace,
    )
    torch.testing.assert_close(index.persisted_list_ids_interleaved, expected_layout.ids)
    torch.testing.assert_close(index.persisted_list_group_prefix, expected_layout.group_prefix)
    expected_static_corrections = _build_interleaved_pq_static_corrections(
        index.list_codes,
        index.list_prefix,
        index.list_count,
        index.centroids,
        index.pq_codebooks,
    )
    torch.testing.assert_close(
        index.persisted_list_pq_static_corrections_interleaved,
        expected_static_corrections,
    )

    # Existing indexes did not persist this scalar cache. Loading one remains
    # supported: reconstruct it once from its row-major codes and R table.
    (output_dir / "list_pq_static_corrections_interleaved.pt").unlink()
    legacy_static_index = load_ivfpq_index(output_dir)
    assert legacy_static_index.persisted_list_pq_static_corrections_interleaved is not None
    torch.testing.assert_close(
        legacy_static_index.persisted_list_pq_static_corrections_interleaved,
        expected_static_corrections,
    )

    index.reside_on_device("cpu")
    assert index.list_codes_interleaved_by_subspace is not None
    assert index.list_ids_interleaved is not None
    assert index.list_group_prefix is not None
    assert index.list_pq_static_corrections_interleaved is not None
    torch.testing.assert_close(index.list_codes_interleaved_by_subspace, expected_layout.codes_by_subspace)
    torch.testing.assert_close(index.list_ids_interleaved, expected_layout.ids)
    torch.testing.assert_close(index.list_group_prefix, expected_layout.group_prefix)
    torch.testing.assert_close(
        index.list_pq_static_corrections_interleaved,
        expected_static_corrections,
    )
    interleaved_codes, interleaved_ids, interleaved_prefix = _get_interleaved_pq_lists(
        index,
        index.list_codes,
        index.list_ids_flat,
        index.list_prefix,
        index.list_count,
        torch.device("cpu"),
    )
    torch.testing.assert_close(interleaved_codes, expected_layout.codes_by_subspace)
    torch.testing.assert_close(interleaved_ids, expected_layout.ids)
    torch.testing.assert_close(interleaved_prefix, expected_layout.group_prefix)

    single_file_path = tmp_path / "ivfpq.bin"
    save_ivfpq_index(result, single_file_path)
    single_file_index = load_ivfpq_index(single_file_path)
    assert single_file_index.persisted_list_codes_interleaved_by_subspace is not None
    assert single_file_index.persisted_list_ids_interleaved is not None
    assert single_file_index.persisted_list_group_prefix is not None
    assert single_file_index.persisted_list_pq_static_corrections_interleaved is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("nbits", [3, 4])
def test_ivfpq_triton_profile_splits_direct_scan_and_final_merge(tmp_path, nbits):
    pytest.importorskip("triton")
    generator = torch.Generator().manual_seed(97)
    vectors = torch.randn(128, 8, generator=generator)
    result = build_ivfpq_index(
        vectors=vectors,
        nlist=4,
        niter=5,
        use_triton=False,
        device="cpu",
        m=2,
        nbits=nbits,
        seed=6,
        n_init=2,
    )
    output_path = tmp_path / f"ivfpq-nbits{nbits}.bin"
    save_ivfpq_index(result, output_path)
    index = load_ivfpq_index(output_path)
    index.reside_on_device("cuda:0")

    profile: dict[str, float] = {}
    triton_distances, triton_ids = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="triton_topk",
        profile=profile,
    )
    tensor_distances, _ = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="tensor_topk",
    )
    assert "triton_direct_scan" in profile
    assert "triton_final_merge" in profile
    assert "triton_topk_select" not in profile
    torch.testing.assert_close(triton_distances, tensor_distances)

    fp16_profile: dict[str, float] = {}
    fp16_triton_distances, fp16_triton_ids = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="triton_topk",
        profile=fp16_profile,
        lut_dtype="fp16",
    )
    assert "LUT_prepare" in fp16_profile
    assert "triton_direct_scan" in fp16_profile
    assert fp16_triton_ids.dtype == torch.int64
    torch.testing.assert_close(fp16_triton_distances, triton_distances, rtol=2e-3, atol=2e-3)

    fp32_profile: dict[str, float] = {}
    fp32_distances, fp32_ids = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="triton_topk",
        profile=fp32_profile,
        lut_dtype="fp32",
        lut_strategy="decomposed_vector_correction",
    )
    assert "triton_direct_scan" in fp32_profile
    assert "triton_group_probe_lists" not in fp32_profile
    torch.testing.assert_close(fp32_ids, triton_ids)
    torch.testing.assert_close(fp32_distances, triton_distances, rtol=2e-4, atol=2e-4)


def test_interleaved_layout_preserves_ivf_list_values():
    vectors = torch.arange(7 * 10, dtype=torch.float32).reshape(7, 10)
    codes = torch.arange(7 * 2, dtype=torch.uint8).reshape(7, 2)
    ids = torch.arange(10, 17, dtype=torch.int64)
    list_count = torch.tensor([2, 0, 3, 2], dtype=torch.int64)
    list_prefix = torch.tensor([0, 2, 2, 5], dtype=torch.int64)

    flat_layout = build_interleaved_flat_lists(vectors, ids, list_prefix, list_count)
    pq_layout = build_interleaved_pq_lists(codes, ids, list_prefix, list_count)
    physical_positions = torch.tensor(
        [0, 1, INTERLEAVED_GROUP_SIZE, INTERLEAVED_GROUP_SIZE + 1, INTERLEAVED_GROUP_SIZE + 2,
         2 * INTERLEAVED_GROUP_SIZE, 2 * INTERLEAVED_GROUP_SIZE + 1],
        dtype=torch.int64,
    )
    assert interleaved_vector_width(vectors.shape[1], vectors.dtype) == 1
    assert flat_layout.vector_width == 1
    assert flat_layout.vectors.shape == (3, vectors.shape[1], INTERLEAVED_GROUP_SIZE, 1)
    flat_rows = flat_layout.vectors.permute(0, 2, 1, 3).reshape(-1, vectors.shape[1])
    torch.testing.assert_close(flat_rows[physical_positions], vectors)
    assert interleaved_vector_width(8, vectors.dtype) == 4
    torch.testing.assert_close(flat_layout.ids[physical_positions], ids)
    torch.testing.assert_close(pq_layout.codes_by_subspace[:, physical_positions].transpose(0, 1), codes)
    torch.testing.assert_close(pq_layout.ids[physical_positions], ids)
