import json

import pytest
import torch

import tensorvs._ivfpq_triton as _ivfpq_triton
import tensorvs._ivfsq_triton as _ivfsq_triton
from tensorvs._kmeans import _assigned_inertia, _kmeans_cpu_chunk_rows
from tensorvs import (
    build_ivfflat_index,
    build_ivfpq_index,
    build_ivfsq_index,
    compute_recall_at_k,
    load_ivfflat_index,
    load_ivfpq_index,
    load_ivfsq_index,
    save_ivfflat_index,
    save_ivfpq_index,
    save_ivfsq_index,
    search_ivfflat_index,
    search_ivfpq_index,
    search_ivfsq_index,
)
from tensorvs.index_ivfsq import _pack_sq_codes, _unpack_sq_codes
from tensorvs.distance import squared_l2_distance
from tensorvs._ivf_interleaved import (
    INTERLEAVED_GROUP_SIZE,
    build_interleaved_flat_lists,
    build_interleaved_pq_code_chunks,
    build_interleaved_pq_lists,
    interleaved_vector_width,
)
from tensorvs._ivfflat_triton import _direct_scan_plan, _list_grouped_gemm_plan
from tensorvs._ivfpq_triton import (
    _default_launch_config as _ivfpq_default_launch_config,
    _hierarchical_merge_plan as _ivfpq_hierarchical_merge_plan,
    _list_cta_launch_plan as _ivfpq_list_cta_launch_plan,
    _list_cta_plan as _ivfpq_list_cta_plan,
)
from tensorvs._ivfpq_list_major_triton import (
    _ListMajorLaunchConfig,
    _build_list_major_task_plan,
    _estimated_query_tile as _ivfpq_list_major_estimated_query_tile,
    _layout_list_major_global_lut,
    _list_major_plan as _ivfpq_list_major_plan,
)
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
    _tensor_topk_pq_distance_impl,
    _tensor_topk_pq_lut_reduce_impl,
)


def test_kmeans_memory_chunks_scale_with_vector_width():
    assert _kmeans_cpu_chunk_rows(960) < 1_000_000
    assert _kmeans_cpu_chunk_rows(960) * 960 * torch.float32.itemsize <= 256 << 20


def test_assigned_inertia_chunking_matches_direct_sum():
    generator = torch.Generator().manual_seed(619)
    data = torch.randn(11, 7, generator=generator)
    centroids = torch.randn(3, 7, generator=generator)
    labels = torch.randint(0, 3, (11,), generator=generator)
    expected = ((data - centroids[labels]) ** 2).sum(dtype=torch.float64).item()

    actual = _assigned_inertia(
        data,
        labels,
        centroids,
        device=torch.device("cpu"),
        chunk_size=3,
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_ivfflat_triton_plan_uses_compact_local_topk():
    assert _direct_scan_plan(topk=1, num_queries=128, nprobe=50) == (2, 2, 4)
    assert _direct_scan_plan(topk=10, num_queries=128, nprobe=50) == (16, 2, 32)
    assert _direct_scan_plan(topk=32, num_queries=128, nprobe=50) == (32, 2, 64)
    # A scan tile contains 128 candidates, so larger k retains the wide path.
    assert _direct_scan_plan(topk=33, num_queries=128, nprobe=50) == (128, 2, 256)
    assert _direct_scan_plan(topk=10, num_queries=64, nprobe=50) == (16, 4, 64)
    # The direct scanner partitions wide probe sets across GRID_X programs;
    # its final merge width does not grow with the full nprobe value.
    assert _direct_scan_plan(topk=10, num_queries=64, nprobe=256) == (16, 4, 64)


def test_ivfflat_list_grouped_gemm_plan_uses_one_local_topk_per_list():
    # The common SIFT setting emits 16 local candidates for each of 50 lists,
    # then reduces a 1024-wide query-major merge in the second kernel.
    assert _list_grouped_gemm_plan(topk=10, nprobe=50, dim=128) == (16, 1024)
    assert _list_grouped_gemm_plan(topk=32, nprobe=50, dim=128) == (32, 2048)
    # A partial final dimension tile is masked and zero-padded, so dimensions
    # no longer need to be divisible by the 32-wide matrix tile.
    assert _list_grouped_gemm_plan(topk=10, nprobe=50, dim=16) == (16, 1024)
    assert _list_grouped_gemm_plan(topk=10, nprobe=50, dim=130) == (16, 1024)
    # High-dimensional GIST-style vectors use the runtime-loop direct scan;
    # the grouped GEMM specialization would exceed the GPU shared-memory cap.
    assert _list_grouped_gemm_plan(topk=10, nprobe=50, dim=960) is None
    assert _list_grouped_gemm_plan(topk=33, nprobe=50, dim=128) is None


def test_ivfpq_triton_plan_emits_one_list_row_per_probe():
    assert _ivfpq_list_cta_plan(topk=10, nprobe=50) == (16, 1024)
    assert _ivfpq_list_cta_plan(topk=32, nprobe=50) == (32, 2048)
    assert _ivfpq_list_cta_plan(topk=10, nprobe=129) is None


def test_ivfpq_triton_uses_generic_fallback_scan_plan():
    fallback = _ivfpq_default_launch_config()
    assert fallback.scan_block_candidates == 256
    assert fallback.num_warps == 4
    assert fallback.probe_lists_per_cta == 1

    # Without a persistent measured result the fallback keeps one local row
    # per probe, preserving list-level load balance for any M.
    assert _ivfpq_list_cta_launch_plan(topk=10, nprobe=50, m=25) == (
        16,
        50,
        fallback,
        1024,
    )
    assert _ivfpq_list_cta_launch_plan(topk=10, nprobe=51, m=25) == (
        16,
        51,
        fallback,
        1024,
    )


def test_ivfpq_triton_hierarchical_merge_supports_wide_probe_sets():
    fallback = _ivfpq_default_launch_config()
    assert _ivfpq_list_cta_launch_plan(topk=10, nprobe=256, m=16) == (
        16,
        256,
        fallback,
        4096,
    )
    assert _ivfpq_hierarchical_merge_plan(
        cta_probe_groups=256,
        local_k=16,
    ) == (((256, 2),), 2, 32)
    assert _ivfpq_hierarchical_merge_plan(
        cta_probe_groups=1024,
        local_k=16,
    ) == (((1024, 8),), 8, 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ivfpq_triton_hierarchical_merge_matches_reference():
    pytest.importorskip("triton")
    torch.manual_seed(123)
    device = torch.device("cuda:0")
    num_queries, m, ksub, topk = 2, 2, 4, 10

    for nprobe in (256, 1024):
        padded_vectors = nprobe * INTERLEAVED_GROUP_SIZE
        global_luts = torch.rand((m, num_queries, ksub), device=device)
        codes = torch.randint(
            0,
            ksub,
            (m, padded_vectors),
            device=device,
            dtype=torch.uint8,
        )
        list_ids = torch.arange(padded_vectors, device=device, dtype=torch.int64)
        valid_positions = torch.arange(nprobe, device=device, dtype=torch.int64)
        valid_positions = valid_positions * INTERLEAVED_GROUP_SIZE
        corrections = torch.zeros(padded_vectors, device=device)
        corrections[valid_positions] = valid_positions.float() * 1e-4
        probe_prefix = torch.arange(nprobe, device=device, dtype=torch.int64).repeat(
            num_queries, 1
        )
        probe_count = torch.ones(
            (num_queries, nprobe),
            device=device,
            dtype=torch.int64,
        )
        probe_bias = torch.zeros((num_queries, nprobe), device=device)
        distances = torch.empty((num_queries, topk), device=device)
        result_ids = torch.empty(
            (num_queries, topk),
            device=device,
            dtype=torch.int64,
        )

        _ivfpq_triton.triton_topk(
            global_query_luts=global_luts,
            static_vector_corrections=corrections,
            probe_distance_bias=probe_bias,
            list_codes_interleaved=codes,
            list_ids_interleaved=list_ids,
            probe_group_prefix=probe_prefix,
            probe_list_count=probe_count,
            topk=topk,
            output_distances=distances,
            output_ids=result_ids,
            list_codes_packed_16b=None,
        )

        reference_distances = torch.zeros(
            (num_queries, nprobe),
            device=device,
        )
        for subspace in range(m):
            reference_distances += global_luts[
                subspace,
                :,
                codes[subspace, valid_positions].long(),
            ]
        reference_distances += corrections[valid_positions]
        expected_distances, expected_indices = torch.topk(
            reference_distances,
            topk,
            dim=1,
            largest=False,
        )
        expected_ids = valid_positions[expected_indices]
        torch.testing.assert_close(
            distances,
            expected_distances,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(result_ids, expected_ids)


def test_ivfpq_list_major_plan_adapts_query_tile_to_local_key_footprint():
    assert _ivfpq_list_major_plan(topk=10, nprobe=50) == (16, 1024)
    assert _ivfpq_list_major_plan(topk=33, nprobe=50) is None

    # The default resource estimate uses a wider tile for the small local
    # reduction and shrinks it once the key matrix itself becomes wider.
    assert _ivfpq_list_major_estimated_query_tile(
        m=32,
        local_k=16,
        lut_dtype=torch.float32,
    ) == 8
    assert _ivfpq_list_major_estimated_query_tile(
        m=32,
        local_k=32,
        lut_dtype=torch.float32,
    ) == 4
    assert _ListMajorLaunchConfig(query_tile=8, num_warps=4).query_tile == 8


def test_ivfpq_list_major_task_plan_uses_candidate_tiles():
    probe_cids = torch.tensor(
        [[0, 1], [0, 1], [0, 1], [0, 1]],
        dtype=torch.int64,
    )
    list_count = torch.tensor([129, 0], dtype=torch.int64)
    plan = _build_list_major_task_plan(
        probe_cids=probe_cids,
        list_count=list_count,
        query_tile=4,
        candidate_groups=4,
    )

    # Five 32-vector groups are now consumed by one list/query task, which
    # loops over two candidate tiles while retaining the local top-k.
    assert plan["max_vector_chunks"] == 1
    assert plan["num_tasks"] == 2


def test_ivfpq_list_major_lut_keeps_query_major_rows_for_vector_reorder():
    query_major_lut = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    list_major_lut = _layout_list_major_global_lut(query_major_lut)

    assert list_major_lut.shape == (2, 3, 4)
    torch.testing.assert_close(list_major_lut[1, :, 2], query_major_lut[1, :, 2])
    assert list_major_lut.is_contiguous()


def test_ivfsq_triton_plan_bounds_local_and_chunk_topk():
    assert _ivfsq_triton._task_plan(topk=10, nprobe=50, max_list_group_count=40) == (
        16,
        5,
        128,
    )
    assert _ivfsq_triton._task_plan(topk=33, nprobe=50, max_list_group_count=40) is None
    assert _ivfsq_triton._task_plan(topk=10, nprobe=129, max_list_group_count=40) is None


def test_ivfpq_list_cta_plan_supports_two_probe_lists_per_cta(monkeypatch):
    config = _ivfpq_triton._MAdaptiveLaunchConfig(
        scan_block_candidates=128,
        num_warps=4,
        probe_lists_per_cta=2,
    )
    monkeypatch.setattr(_ivfpq_triton, "_default_launch_config", lambda: config)
    # The final odd probe is part of a masked two-list CTA group, so only 26
    # group-local rows need to be merged for 51 probes.
    assert _ivfpq_list_cta_launch_plan(topk=10, nprobe=51, m=25) == (
        16,
        26,
        config,
        512,
    )


def test_ivfpq_triton_autotune_cache_persists_and_reloads(tmp_path, monkeypatch):
    cache_path = tmp_path / "ivfpq-triton-autotune.json"
    monkeypatch.setenv("TENSORVS_TRITON_AUTOTUNE_CACHE", str(cache_path))
    monkeypatch.setattr(_ivfpq_triton, "_PERSISTED_AUTOTUNE_CONFIGS", None)
    _ivfpq_triton._AUTOTUNE_CONFIG_CACHE.clear()
    cache_key = ("test GPU", (8, 9), 32, 256, 50, 10, 128, True, 1165, 3021, 4848)
    expected = _ivfpq_triton._MAdaptiveLaunchConfig(512, 4, 1)

    _ivfpq_triton._persist_autotune_config(cache_key, expected)
    assert json.loads(cache_path.read_text())["version"] == 1

    _ivfpq_triton._AUTOTUNE_CONFIG_CACHE.clear()
    monkeypatch.setattr(_ivfpq_triton, "_PERSISTED_AUTOTUNE_CONFIGS", None)
    assert _ivfpq_triton._cached_autotune_config(cache_key) == expected


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
    list_major_global_luts = _build_global_query_luts_bmm(
        queries,
        codebooks,
        codebook_norms,
        m=m,
        subdim=subdim,
        use_tf32=False,
        layout="list_major",
    )
    assert list_major_global_luts.shape == (m, num_queries, ksub)
    torch.testing.assert_close(list_major_global_luts, expected_global_luts)

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
        "tensor_nopadding_topk",
        "tensor_list_topk",
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

@pytest.mark.parametrize(
    "selection_mode",
    ("tensor_topk", "tensor_padding_topk", "triton_topk"),
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
    list_major_distances, list_major_ids = search_ivfpq_index(
        index,
        queries,
        topk=4,
        nprobe=4,
        device="cpu",
        selection_mode="triton_list_major",
    )
    # CPU has no Triton kernel, so the public mode intentionally preserves
    # exact tensor_topk semantics through the normal fallback.
    torch.testing.assert_close(list_major_distances, approximate_distances)
    torch.testing.assert_close(list_major_ids, approximate_ids)
    for removed_mode in (
        "list_major",
        "triton_list_major_vector_lut",
        "triton_list_topk",
        "triton_list_cta_topk",
    ):
        with pytest.raises(ValueError, match="Unsupported IVFPQ selection_mode"):
            search_ivfpq_index(
                index,
                queries,
                topk=4,
                nprobe=4,
                device="cpu",
                selection_mode=removed_mode,
            )
    for removed_mode in (
        "tensor_list_topk",
        "global_sort",
        "per_query_topk",
        "triton_interleaved_topk",
    ):
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


def test_ivfpq_nondivisible_m_uses_orthogonal_pq_working_space(tmp_path):
    """Match cuVS's padded-and-rotated PQ semantics for ``dim % m != 0``."""
    generator = torch.Generator().manual_seed(211)
    vectors = torch.randn(64, 20, generator=generator)
    queries = vectors[:5] + 0.01 * torch.randn(5, 20, generator=generator)

    result = build_ivfpq_index(
        vectors=vectors,
        nlist=4,
        niter=1,
        use_triton=False,
        device="cpu",
        m=3,
        nbits=2,
        seed=17,
        n_init=1,
    )
    assert result.pq_rotation_matrix is not None
    assert result.pq_rotation_matrix.shape == (21, 20)
    torch.testing.assert_close(
        result.pq_rotation_matrix.transpose(0, 1) @ result.pq_rotation_matrix,
        torch.eye(20),
        rtol=1e-4,
        atol=1e-4,
    )

    output_dir = tmp_path / "ivfpq_nondivisible"
    save_ivfpq_index(result, output_dir)
    assert (output_dir / "pq_rotation_matrix.pt").exists()
    assert (output_dir / "pq_centroids.pt").exists()

    index = load_ivfpq_index(output_dir)
    assert index.metadata["dim"] == 20
    assert index.metadata["subdim"] == 7
    assert index.metadata["pq_working_dim"] == 21
    assert index.metadata["pq_rotation"] == "random_orthogonal"
    assert index.pq_rotation_matrix is not None
    assert index.pq_centroids is not None
    assert index.pq_centroids.shape == (4, 21)

    actual_distances, actual_ids = search_ivfpq_index(
        index,
        queries,
        topk=3,
        nprobe=4,
        device="cpu",
        selection_mode="tensor_topk",
    )

    assert index.list_codes is not None
    assert index.list_ids_flat is not None
    rotated_queries = queries @ index.pq_rotation_matrix.transpose(0, 1)
    subspace_indices = torch.arange(3, dtype=torch.int64).unsqueeze(0)
    decoded_residuals = index.pq_codebooks[
        subspace_indices,
        index.list_codes.to(torch.int64),
    ].reshape(index.list_codes.shape[0], 21)
    candidate_lists = index.assignments[index.list_ids_flat].to(torch.int64)
    reconstructed_vectors = index.pq_centroids[candidate_lists] + decoded_residuals
    full_distances = squared_l2_distance(rotated_queries, reconstructed_vectors)
    expected_distances, expected_positions = torch.topk(full_distances, k=3, largest=False)

    torch.testing.assert_close(actual_distances, expected_distances, rtol=1e-4, atol=1e-4)
    # The decomposed scanner and the explicit reconstruction may resolve a
    # near-equal final boundary in a different order. Verify every returned
    # ID maps back to its own reconstructed distance instead of imposing an
    # accidental tie-ordering contract.
    id_to_position = torch.empty(index.list_ids_flat.numel(), dtype=torch.int64)
    id_to_position[index.list_ids_flat] = torch.arange(index.list_ids_flat.numel())
    actual_positions = id_to_position[actual_ids]
    torch.testing.assert_close(
        actual_distances,
        torch.gather(full_distances, 1, actual_positions),
        rtol=1e-4,
        atol=1e-4,
    )


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
def test_ivfpq_triton_profile_splits_cta_scan_and_final_merge(tmp_path, nbits):
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
    assert "triton_scan" in profile
    assert "triton_final_merge" in profile
    assert "triton_topk_select" not in profile
    torch.testing.assert_close(triton_distances, tensor_distances)

    list_major_profile: dict[str, float] = {}
    list_major_distances, list_major_ids = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="triton_list_major",
        profile=list_major_profile,
    )
    assert "triton_scan" in list_major_profile
    assert "triton_final_merge" in list_major_profile
    torch.testing.assert_close(list_major_ids, triton_ids)
    torch.testing.assert_close(list_major_distances, triton_distances)

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
    assert "triton_scan" in fp16_profile
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
    assert "triton_scan" in fp32_profile
    torch.testing.assert_close(fp32_ids, triton_ids)
    torch.testing.assert_close(fp32_distances, triton_distances, rtol=2e-4, atol=2e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [25, 32])
def test_ivfpq_triton_packed_chunked_matches_tensor_topk(tmp_path, m):
    pytest.importorskip("triton")
    generator = torch.Generator().manual_seed(509)
    vectors = torch.randn(128, m, generator=generator)
    result = build_ivfpq_index(
        vectors=vectors,
        nlist=4,
        niter=2,
        use_triton=False,
        device="cpu",
        m=m,
        nbits=3,
        seed=11,
        n_init=1,
    )
    output_path = tmp_path / f"ivfpq-m{m}-packed.bin"
    save_ivfpq_index(result, output_path)
    index = load_ivfpq_index(output_path)
    index.reside_on_device("cuda:0")

    profile: dict[str, float] = {}
    packed_distances, packed_ids = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="triton_topk",
        profile=profile,
    )
    reference_distances, reference_ids = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="tensor_topk",
    )
    assert "triton_packed_prepare" in profile
    assert index.list_codes_interleaved_packed_16b is not None
    assert index.list_codes_interleaved_packed_16b.shape[1:] == (2, 32, 2)
    unpacked_distances, unpacked_ids = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="triton_topk",
        use_packed_16b_codes=False,
    )
    torch.testing.assert_close(packed_ids, reference_ids)
    torch.testing.assert_close(packed_distances, reference_distances, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(unpacked_ids, reference_ids)
    torch.testing.assert_close(unpacked_distances, reference_distances, rtol=2e-4, atol=2e-4)

    list_major_distances, list_major_ids = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="triton_list_major",
    )
    torch.testing.assert_close(list_major_ids, reference_ids)
    torch.testing.assert_close(list_major_distances, reference_distances, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ivfpq_triton_chunked_m96_matches_tensor_topk(tmp_path):
    pytest.importorskip("triton")
    generator = torch.Generator().manual_seed(733)
    vectors = torch.randn(128, 96, generator=generator)
    result = build_ivfpq_index(
        vectors=vectors,
        nlist=4,
        niter=1,
        use_triton=False,
        device="cpu",
        m=96,
        nbits=2,
        seed=13,
        n_init=1,
    )
    output_path = tmp_path / "ivfpq-m96-chunked.bin"
    save_ivfpq_index(result, output_path)
    index = load_ivfpq_index(output_path)
    index.reside_on_device("cuda:0")

    profile: dict[str, float] = {}
    chunked_distances, chunked_ids = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="triton_topk",
        profile=profile,
    )
    reference_distances, reference_ids = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="tensor_topk",
    )

    assert "triton_packed_prepare" in profile
    assert "triton_scan" in profile
    assert index.list_codes_interleaved_packed_16b is not None
    assert index.list_codes_interleaved_packed_16b.shape[1:] == (6, 32, 2)
    torch.testing.assert_close(chunked_ids, reference_ids)
    torch.testing.assert_close(
        chunked_distances,
        reference_distances,
        rtol=5e-4,
        atol=5e-4,
    )

    list_major_distances, list_major_ids = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="triton_list_major",
    )
    torch.testing.assert_close(list_major_ids, chunked_ids)
    torch.testing.assert_close(
        list_major_distances,
        chunked_distances,
        rtol=5e-4,
        atol=5e-4,
    )




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

    wide_codes = torch.arange(7 * 25, dtype=torch.uint8).reshape(7, 25)
    packed_codes = build_interleaved_pq_code_chunks(wide_codes, list_prefix, list_count)
    assert packed_codes.shape == (3, 2, INTERLEAVED_GROUP_SIZE, 2)
    packed_rows = packed_codes.view(torch.uint8).permute(0, 2, 1, 3).reshape(-1, 32)
    torch.testing.assert_close(packed_rows[physical_positions, :25], wide_codes)
    assert torch.count_nonzero(packed_rows[:, 25:]) == 0


def test_ivfsq_nibble_pack_roundtrip_odd_dimension():
    codes = torch.tensor(
        [[0, 1, 2, 3, 4], [15, 14, 13, 12, 11]],
        dtype=torch.uint8,
    )
    packed = _pack_sq_codes(codes)
    assert packed.dtype == torch.uint8
    assert packed.shape == (2, 3)
    torch.testing.assert_close(_unpack_sq_codes(packed, dim=5), codes)


@pytest.mark.parametrize("bits", [8, 4])
def test_ivfsq_build_save_load_search_and_refine(tmp_path, bits):
    generator = torch.Generator().manual_seed(321)
    vectors = torch.randn(96, 7, generator=generator)
    queries = vectors[:8] + 0.01 * torch.randn(8, 7, generator=generator)
    result = build_ivfsq_index(
        vectors=vectors,
        nlist=4,
        niter=4,
        use_triton=False,
        device="cpu",
        bits=bits,
        seed=17,
        n_init=1,
    )

    assert result.bits == bits
    assert result.sq_codes.dtype == torch.uint8
    assert result.sq_codes.shape == (96, 7 if bits == 8 else 4)
    output_path = tmp_path / f"ivfsq_{bits}"
    save_ivfsq_index(result, output_path, dataset_path="in-memory")
    index = load_ivfsq_index(output_path)
    assert index.metadata["sq_bits"] == bits
    assert index.metadata["sq_mode"] == "residual_per_list_affine"
    assert index.list_codes is not None
    assert index.list_codes.shape == result.sq_codes.shape

    profile: dict[str, float] = {}
    distances, ids = search_ivfsq_index(
        index,
        queries,
        topk=4,
        nprobe=4,
        device="cpu",
        profile=profile,
    )
    assert distances.shape == (8, 4)
    assert ids.shape == (8, 4)
    assert distances.dtype == torch.float32
    assert ids.dtype == torch.int64
    assert torch.all(ids >= 0)
    assert torch.all(ids < vectors.shape[0])
    assert "sq_distance_topk" in profile

    refined_distances, refined_ids = search_ivfsq_index(
        index,
        queries,
        topk=3,
        nprobe=4,
        device="cpu",
        refine_vectors=vectors,
        refine_factor=2,
    )
    expected = ((queries[:, None, :] - vectors[refined_ids]) ** 2).sum(dim=2)
    torch.testing.assert_close(refined_distances, expected, rtol=1e-5, atol=1e-5)

    single_file = tmp_path / f"ivfsq_{bits}.bin"
    save_ivfsq_index(result, single_file)
    single_file_index = load_ivfsq_index(single_file)
    torch.testing.assert_close(single_file_index.sq_min, index.sq_min)
    torch.testing.assert_close(single_file_index.sq_scale, index.sq_scale)


def test_ivfsq_rejects_unsupported_bits_and_selection_mode():
    vectors = torch.randn(16, 4)
    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        build_ivfsq_index(vectors, 2, 2, False, "cpu", bits=3)
