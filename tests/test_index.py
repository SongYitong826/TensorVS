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
from tensorvs.index_ivfflat import (
    IVFFlatIndex,
    _coarse_squared_l2_distance_addmm as _ivfflat_coarse_squared_l2_distance_addmm,
)
from tensorvs.index_ivfpq import (
    _coarse_squared_l2_distance_addmm,
    _build_global_query_luts_bmm,
    _build_interleaved_pq_static_corrections,
    _build_pq_static_corrections,
    _get_interleaved_pq_lists,
    _tensor_list_topk_pq_distance_impl,
    _tensor_nopadding_topk_pq_distance_impl,
    _tensor_topk_pq_distance_impl,
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


def test_tensor_pq_global_lut_lookup_reconstructs_residual_adc():
    generator = torch.Generator().manual_seed(19)
    num_queries, nprobe, max_candidates, m, ksub = 2, 3, 4, 2, 5
    global_luts = torch.randn(m, num_queries, ksub, generator=generator)
    candidate_codes = torch.randint(
        ksub,
        (m, num_queries, max_candidates),
        generator=generator,
        dtype=torch.int64,
    )
    list_codes = candidate_codes.permute(1, 2, 0).reshape(-1, m).to(torch.uint8)
    static_corrections = torch.randn(list_codes.shape[0], generator=generator)
    probe_distance_bias = torch.randn(num_queries, nprobe, generator=generator)
    safe_candidate_positions = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        dtype=torch.int64,
    )
    probe_idx = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 1]], dtype=torch.int64)
    valid_mask = torch.tensor([[True, True, False, True], [True, False, True, True]])

    subspace_idx = torch.arange(m, dtype=torch.int64)
    query_idx = torch.arange(num_queries, dtype=torch.int64)[:, None, None]
    expected = global_luts[
        subspace_idx[None, None, :],
        query_idx,
        candidate_codes.permute(1, 2, 0),
    ].sum(dim=2)
    expected += static_corrections[safe_candidate_positions]
    expected += torch.gather(probe_distance_bias, 1, probe_idx)
    expected = expected.masked_fill(~valid_mask, float("inf"))

    actual = _tensor_topk_pq_lut_reduce_impl(
        global_luts.permute(1, 0, 2).contiguous(),
        list_codes[safe_candidate_positions],
        static_corrections,
        probe_distance_bias,
        safe_candidate_positions,
        probe_idx,
        valid_mask,
    )
    torch.testing.assert_close(actual, expected)

    fp16_luts = global_luts.to(torch.float16)
    fp16_expected = fp16_luts[
        subspace_idx[None, None, :],
        query_idx,
        candidate_codes.permute(1, 2, 0),
    ].sum(dim=2, dtype=torch.float32)
    fp16_expected += static_corrections[safe_candidate_positions]
    fp16_expected += torch.gather(probe_distance_bias, 1, probe_idx)
    fp16_expected = fp16_expected.masked_fill(~valid_mask, float("inf"))
    fp16_actual = _tensor_topk_pq_lut_reduce_impl(
        fp16_luts.permute(1, 0, 2).contiguous(),
        list_codes[safe_candidate_positions],
        static_corrections,
        probe_distance_bias,
        safe_candidate_positions,
        probe_idx,
        valid_mask,
    )
    assert fp16_actual.dtype == torch.float32
    torch.testing.assert_close(fp16_actual, fp16_expected)


def test_tensor_pq_global_lut_topk_matches_full_query_merge():
    generator = torch.Generator().manual_seed(31)
    num_queries, nprobe, m, ksub = 2, 3, 2, 7
    global_luts = torch.randn(m, num_queries, ksub, generator=generator)
    list_codes = torch.randint(ksub, (8, m), generator=generator, dtype=torch.uint8)
    list_ids = torch.arange(100, 108, dtype=torch.int64)
    static_corrections = torch.randn(8, generator=generator)
    probe_distance_bias = torch.randn(num_queries, nprobe, generator=generator)
    probe_list_prefix = torch.tensor([[0, 3, 5], [3, 0, 5]], dtype=torch.int64)
    probe_list_count = torch.tensor([[3, 2, 3], [2, 3, 3]], dtype=torch.int64)
    candidate_offsets = torch.arange(8, dtype=torch.int64)
    actual_distances, actual_ids = _tensor_topk_pq_distance_impl(
        global_luts.permute(1, 0, 2).contiguous(),
        list_codes,
        list_ids,
        static_corrections,
        probe_distance_bias,
        probe_list_prefix,
        probe_list_count,
        candidate_offsets,
        topk=3,
    )

    expected_distances = []
    expected_ids = []
    for query_idx in range(num_queries):
        candidate_distances = []
        candidate_ids = []
        for probe_idx in range(nprobe):
            start = int(probe_list_prefix[query_idx, probe_idx])
            count = int(probe_list_count[query_idx, probe_idx])
            positions = torch.arange(start, start + count, dtype=torch.int64)
            codes = list_codes[positions].to(torch.int64)
            distances = global_luts[
                torch.arange(m, dtype=torch.int64)[:, None],
                query_idx,
                codes.transpose(0, 1),
            ].sum(dim=0)
            distances += static_corrections[positions] + probe_distance_bias[query_idx, probe_idx]
            candidate_distances.append(distances)
            candidate_ids.append(list_ids[positions])
        distances = torch.cat(candidate_distances)
        ids = torch.cat(candidate_ids)
        selected_distances, selected_positions = torch.topk(distances, k=3, largest=False)
        expected_distances.append(selected_distances)
        expected_ids.append(ids[selected_positions])

    torch.testing.assert_close(actual_distances, torch.stack(expected_distances))
    torch.testing.assert_close(actual_ids, torch.stack(expected_ids))


def test_tensor_pq_global_lut_handles_empty_and_short_lists():
    global_luts = torch.tensor([[[1.0, 2.0]]])
    list_codes = torch.tensor([[1]], dtype=torch.uint8)
    list_ids = torch.tensor([71], dtype=torch.int64)
    static_corrections = torch.tensor([0.25])
    probe_distance_bias = torch.tensor([[10.0, 5.0, 20.0]])
    probe_list_prefix = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    probe_list_count = torch.tensor([[0, 1, 0]], dtype=torch.int64)

    distances, ids = _tensor_topk_pq_distance_impl(
        global_luts.permute(1, 0, 2).contiguous(),
        list_codes,
        list_ids,
        static_corrections,
        probe_distance_bias,
        probe_list_prefix,
        probe_list_count,
        torch.arange(1, dtype=torch.int64),
        topk=1,
    )

    torch.testing.assert_close(distances, torch.tensor([[7.25]]))
    torch.testing.assert_close(ids, torch.tensor([[71]], dtype=torch.int64))


def test_tensor_list_topk_matches_query_major_without_distance_ties():
    generator = torch.Generator().manual_seed(43)
    num_queries, nprobe, m, subdim, ksub = 2, 3, 3, 2, 5
    queries = torch.randn(num_queries, m * subdim, generator=generator)
    codebooks = torch.randn(m, ksub, subdim, generator=generator)
    codebook_norms = (codebooks * codebooks).sum(dim=-1)
    global_luts = _build_global_query_luts_bmm(
        queries,
        codebooks,
        codebook_norms,
        m=m,
        subdim=subdim,
        use_tf32=False,
    ).permute(1, 0, 2).contiguous()
    list_codes = torch.randint(ksub, (8, m), generator=generator, dtype=torch.uint8)
    list_ids = torch.arange(200, 208, dtype=torch.int64)
    # Random per-vector gamma makes equality at the final top-k boundary
    # overwhelmingly unlikely, so this checks the packed global reduction.
    static_corrections = torch.randn(8, generator=generator)
    probe_distance_bias = torch.randn(num_queries, nprobe, generator=generator)
    probe_cids = torch.tensor([[0, 2, 1], [2, 0, 1]], dtype=torch.int64)
    list_prefix = torch.tensor([0, 3, 5], dtype=torch.int64)
    list_count = torch.tensor([3, 2, 3], dtype=torch.int64)
    probe_list_prefix = list_prefix[probe_cids]
    probe_list_count = list_count[probe_cids]
    expected_distances, expected_ids = _tensor_topk_pq_distance_impl(
        global_luts,
        list_codes,
        list_ids,
        static_corrections,
        probe_distance_bias,
        probe_list_prefix,
        probe_list_count,
        torch.arange(8, dtype=torch.int64),
        topk=3,
    )
    actual_distances, actual_ids = _tensor_list_topk_pq_distance_impl(
        global_luts,
        list_codes,
        list_ids,
        static_corrections,
        probe_cids,
        probe_distance_bias,
        list_prefix,
        list_count,
        topk=3,
    )

    torch.testing.assert_close(actual_distances, expected_distances)
    torch.testing.assert_close(actual_ids, expected_ids)

    nopadding_distances, nopadding_ids = _tensor_nopadding_topk_pq_distance_impl(
        global_luts,
        list_codes,
        list_ids,
        static_corrections,
        probe_cids,
        probe_distance_bias,
        list_prefix,
        list_count,
        topk=3,
    )
    torch.testing.assert_close(nopadding_distances, expected_distances)
    torch.testing.assert_close(nopadding_ids, expected_ids)


def test_query_norm_free_global_lut_preserves_residual_pq_distance():
    generator = torch.Generator().manual_seed(71)
    num_queries, num_lists, m, subdim, ksub = 3, 2, 3, 4, 5
    dim = m * subdim
    queries = torch.randn(num_queries, dim, generator=generator)
    centroids = torch.randn(num_lists, dim, generator=generator)
    codebooks = torch.randn(m, ksub, subdim, generator=generator)
    codebook_norms = (codebooks * codebooks).sum(dim=-1)
    candidate_lists = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    candidate_codes = torch.randint(ksub, (candidate_lists.numel(), m), generator=generator)

    global_luts = _build_global_query_luts_bmm(
        queries,
        codebooks,
        codebook_norms,
        m=m,
        subdim=subdim,
        use_tf32=False,
    )
    expected_global_luts = codebook_norms[:, None, :] - 2.0 * torch.einsum(
        "qmd,mkd->mqk",
        queries.view(num_queries, m, subdim),
        codebooks,
    )
    torch.testing.assert_close(global_luts, expected_global_luts)

    probe_distances = squared_l2_distance(queries, centroids)
    subspace_idx = torch.arange(m, dtype=torch.int64)
    for query_idx in range(num_queries):
        for candidate_idx, list_idx in enumerate(candidate_lists.tolist()):
            codes = candidate_codes[candidate_idx]
            gamma = 2.0 * (
                centroids[list_idx].view(m, subdim) * codebooks[subspace_idx, codes]
            ).sum()
            decomposed_distance = (
                probe_distances[query_idx, list_idx]
                + gamma
                + global_luts[subspace_idx, query_idx, codes].sum()
            )
            reconstructed_residual = codebooks[subspace_idx, codes].reshape(dim)
            expected_distance = (
                (queries[query_idx] - centroids[list_idx] - reconstructed_residual) ** 2
            ).sum()
            torch.testing.assert_close(decomposed_distance, expected_distance)


def test_cached_centroid_norm_addmm_matches_squared_l2_distance():
    generator = torch.Generator().manual_seed(83)
    queries = torch.randn(7, 12, generator=generator)
    centroids = torch.randn(11, 12, generator=generator)
    centroid_norms = (centroids * centroids).sum(dim=-1)

    actual = _coarse_squared_l2_distance_addmm(
        queries,
        centroids,
        centroid_norms,
    )
    expected = squared_l2_distance(queries, centroids)
    torch.testing.assert_close(actual, expected)


def test_ivfflat_cached_centroid_norm_addmm_matches_squared_l2_distance():
    generator = torch.Generator().manual_seed(89)
    queries = torch.randn(7, 12, generator=generator)
    centroids = torch.randn(11, 12, generator=generator)
    centroid_norms = (centroids * centroids).sum(dim=-1)

    actual = _ivfflat_coarse_squared_l2_distance_addmm(
        queries,
        centroids,
        centroid_norms,
    )
    expected = squared_l2_distance(queries, centroids)
    torch.testing.assert_close(actual, expected)


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
    assert (output_dir / "centroid_norms.pt").exists()
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
    torch.testing.assert_close(
        index.centroid_norms,
        (index.centroids * index.centroids).sum(dim=-1),
    )
    assert index.persisted_list_vectors_interleaved is not None
    assert index.persisted_list_ids_interleaved is not None
    assert index.persisted_list_group_prefix is not None

    single_file_path = tmp_path / "index.bin"
    save_ivfflat_index(result, single_file_path, dataset_path="in-memory")
    single_file_index = load_ivfflat_index(single_file_path)
    torch.testing.assert_close(
        single_file_index.centroid_norms,
        (single_file_index.centroids * single_file_index.centroids).sum(dim=-1),
    )
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

    padding_profile: dict[str, float] = {}
    padding_distances, padding_ids = search_ivfflat_index(
        index,
        queries,
        topk=4,
        nprobe=4,
        device="cpu",
        profile=padding_profile,
        selection_mode="tensor_padding_topk",
    )
    assert "tensor_padding_topk_select" in padding_profile
    torch.testing.assert_close(padding_distances, tensor_distances)
    torch.testing.assert_close(padding_ids, tensor_ids)

    list_profile: dict[str, float] = {}
    list_distances, list_ids = search_ivfflat_index(
        index,
        queries,
        topk=4,
        nprobe=4,
        device="cpu",
        profile=list_profile,
        selection_mode="tensor_list_topk",
    )
    assert "tensor_list_topk_select" in list_profile
    torch.testing.assert_close(list_distances, tensor_distances)
    torch.testing.assert_close(list_ids, tensor_ids)

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

    padding_distances, padding_ids = search_ivfflat_index(
        index,
        queries,
        topk=5,
        nprobe=3,
        device="cpu",
        selection_mode="tensor_padding_topk",
    )

    assert torch.isinf(tensor_distances[:, 3:]).all()
    assert torch.equal(tensor_ids[:, 3:], torch.full((2, 2), -1, dtype=torch.int64))
    torch.testing.assert_close(padding_distances, tensor_distances)
    torch.testing.assert_close(padding_ids, tensor_ids)

    list_distances, list_ids = search_ivfflat_index(
        index,
        queries,
        topk=5,
        nprobe=3,
        device="cpu",
        selection_mode="tensor_list_topk",
    )
    torch.testing.assert_close(list_distances, tensor_distances)
    torch.testing.assert_close(list_ids, tensor_ids)


@pytest.mark.parametrize(
    "selection_mode",
    ("tensor_topk", "tensor_padding_topk", "tensor_list_topk", "triton_topk"),
)
def test_ivfflat_modes_select_nearest_nprobe_lists_from_all_centroids(tmp_path, selection_mode):
    centroids = torch.tensor([[0.0, 0.0], [20.0, 0.0], [2.0, 0.0], [40.0, 0.0]])
    index = IVFFlatIndex(
        index_dir=tmp_path,
        metadata={"nlist": 4, "dim": 2},
        vectors=None,
        assignments=torch.arange(4, dtype=torch.int64),
        centroids=centroids,
        centroid_norms=(centroids * centroids).sum(dim=-1),
        ids=torch.tensor([100, 101, 102, 103], dtype=torch.int64),
        lists=[],
        list_vectors=centroids.clone(),
        list_vector_norms=(centroids * centroids).sum(dim=-1),
        list_ids_flat=torch.tensor([100, 101, 102, 103], dtype=torch.int64),
        list_prefix=torch.arange(4, dtype=torch.int64),
        list_count=torch.ones(4, dtype=torch.int64),
    )

    _, ids = search_ivfflat_index(
        index,
        torch.tensor([[1.8, 0.0], [38.8, 0.0]]),
        topk=1,
        nprobe=1,
        device="cpu",
        selection_mode=selection_mode,
    )

    torch.testing.assert_close(ids, torch.tensor([[102], [103]], dtype=torch.int64))


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
    tensor_list_profile: dict[str, float] = {}
    tensor_list_distances, tensor_list_ids = search_ivfpq_index(
        index,
        queries,
        topk=4,
        nprobe=4,
        device="cpu",
        selection_mode="tensor_list_topk",
        profile=tensor_list_profile,
    )
    assert "tensor_list_topk_select" in tensor_list_profile
    torch.testing.assert_close(tensor_list_distances, approximate_distances, rtol=1e-5, atol=1e-5)
    assert torch.all(tensor_list_ids >= 0)
    tensor_list_fp16_distances, tensor_list_fp16_ids = search_ivfpq_index(
        index,
        queries,
        topk=4,
        nprobe=4,
        device="cpu",
        selection_mode="tensor_list_topk",
        lut_dtype="fp16",
    )
    assert tensor_list_fp16_ids.dtype == torch.int64
    torch.testing.assert_close(
        tensor_list_fp16_distances,
        approximate_distances,
        rtol=5e-3,
        atol=5e-3,
    )
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
        "centroid_norms.pt",
        "pq_codebook_norms.pt",
        "list_codes.pt",
        "list_ids_flat.pt",
        "list_prefix.pt",
        "list_count.pt",
        "list_pq_static_corrections.pt",
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
    assert index.persisted_list_pq_static_corrections is not None
    assert index.persisted_list_pq_static_corrections_interleaved is not None
    torch.testing.assert_close(
        index.centroid_norms,
        (index.centroids * index.centroids).sum(dim=-1),
    )
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
    expected_row_major_static_corrections = _build_pq_static_corrections(
        index.list_codes,
        index.list_prefix,
        index.list_count,
        index.centroids,
        index.pq_codebooks,
    )
    torch.testing.assert_close(
        index.persisted_list_pq_static_corrections,
        expected_row_major_static_corrections,
    )

    # Existing indexes did not persist these static caches. Loading one remains
    # supported: reconstruct them once from the original index tensors.
    (output_dir / "list_pq_static_corrections_interleaved.pt").unlink()
    (output_dir / "list_pq_static_corrections.pt").unlink()
    (output_dir / "centroid_norms.pt").unlink()
    legacy_static_index = load_ivfpq_index(output_dir)
    torch.testing.assert_close(
        legacy_static_index.centroid_norms,
        (legacy_static_index.centroids * legacy_static_index.centroids).sum(dim=-1),
    )
    assert legacy_static_index.persisted_list_pq_static_corrections_interleaved is not None
    assert legacy_static_index.persisted_list_pq_static_corrections is not None
    torch.testing.assert_close(
        legacy_static_index.persisted_list_pq_static_corrections,
        expected_row_major_static_corrections,
    )
    torch.testing.assert_close(
        legacy_static_index.persisted_list_pq_static_corrections_interleaved,
        expected_static_corrections,
    )

    index.reside_on_device("cpu")
    assert index.list_codes_interleaved_by_subspace is not None
    assert index.list_ids_interleaved is not None
    assert index.list_group_prefix is not None
    assert index.list_pq_static_corrections is not None
    torch.testing.assert_close(index.list_pq_static_corrections, expected_row_major_static_corrections)
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
    assert single_file_index.persisted_list_pq_static_corrections is not None


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
