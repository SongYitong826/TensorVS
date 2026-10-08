import json
import math

import pytest
import torch

import tensorvs._ivfflat_triton as _ivfflat_triton
import tensorvs._ivfpq_triton as _ivfpq_triton
from flash_kmeans.training import _assigned_inertia, _kmeans_cpu_chunk_rows
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
    build_interleaved_pq_code_chunks,
    build_interleaved_pq_lists,
    interleaved_vector_width,
)
from tensorvs._ivfflat_triton import (
    _direct_scan_plan,
    _list_grouped_gemm_plan,
    _list_grouped_large_k_task_plan,
    _list_grouped_task_plan,
    triton_topk as _ivfflat_triton_topk,
)
from tensorvs._ivfpq_triton import (
    _default_launch_config as _ivfpq_default_launch_config,
    _hierarchical_merge_plan as _ivfpq_hierarchical_merge_plan,
    _list_cta_launch_plan as _ivfpq_list_cta_launch_plan,
    _list_cta_plan as _ivfpq_list_cta_plan,
    chunk_probe_metadata as _chunk_probe_metadata,
)
from tensorvs.index_ivfflat import (
    IVFFlatIndex,
    _coarse_squared_l2_distance_addmm as _ivfflat_coarse_squared_l2_distance_addmm,
)
from tensorvs.index_ivfpq import (
    _coarse_squared_l2_distance_addmm,
    _build_global_query_luts_bmm,
    _build_interleaved_pq_static_corrections,
    _get_interleaved_pq_lists,
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
    assert _list_grouped_gemm_plan(topk=40, nprobe=16, dim=128) == (64, 1024)
    # A partial final dimension tile is masked and zero-padded, so dimensions
    # no longer need to be divisible by the 32-wide matrix tile.
    assert _list_grouped_gemm_plan(topk=10, nprobe=50, dim=16) == (16, 1024)
    assert _list_grouped_gemm_plan(topk=10, nprobe=50, dim=130) == (16, 1024)
    # The one-program-per-list specialization remains bounded to avoid
    # excessive static specialization for high-dimensional vectors.
    assert _list_grouped_gemm_plan(topk=10, nprobe=50, dim=960) is None
    assert _list_grouped_gemm_plan(topk=33, nprobe=50, dim=128) is None
    # A wide final merge must not disable the adaptive grouped scan: it can
    # reduce probe results in multiple bounded merge stages instead.
    assert _list_grouped_gemm_plan(topk=10, nprobe=129, dim=200) is None
    assert _list_grouped_task_plan(
        topk=10,
        nprobe=129,
        dim=200,
        max_list_group_count=152,
    ) == (16, 19, 512)
    assert _list_grouped_task_plan(
        topk=40,
        nprobe=100,
        dim=256,
        max_list_group_count=80,
    ) == (64, 10, 1024)
    assert _list_grouped_task_plan(
        topk=100,
        nprobe=100,
        dim=384,
        max_list_group_count=781,
    ) == (128, 98, 2048)
    assert _list_grouped_large_k_task_plan(
        topk=10_000,
        nprobe=100,
        dim=1536,
        max_list_group_count=781,
    ) == (128, 196)
    # All high-dimensional inputs use the runtime-loop chunked GEMM path
    # instead of direct scalar scanning; the task plan has no 960-D special
    # case or high-dimensional dimension limit.
    for high_dim in (257, 960, 1536, 4096):
        assert _list_grouped_task_plan(
            topk=10,
            nprobe=50,
            dim=high_dim,
            max_list_group_count=168,
        ) == (16, 21, 512)


def test_ivfpq_triton_plan_emits_one_list_row_per_probe():
    assert _ivfpq_list_cta_plan(topk=10, nprobe=50) == (16, 1024)
    assert _ivfpq_list_cta_plan(topk=32, nprobe=50) == (32, 2048)
    assert _ivfpq_list_cta_plan(topk=10, nprobe=129) is None


def test_ivfpq_chunk_probe_metadata_splits_oversized_lists():
    prefixes = torch.tensor([[10, 100], [20, 30]], dtype=torch.int64)
    counts = torch.tensor([[5000, 100], [100, 4096]], dtype=torch.int64)
    biases = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    chunk_prefixes, chunk_counts, chunk_biases, chunk_groups, max_chunks = (
        _chunk_probe_metadata(
            prefixes,
            counts,
            biases,
            max_list_count=5000,
        )
    )

    assert chunk_groups == 128
    assert max_chunks == 2
    assert chunk_counts.shape == (2, 4)
    torch.testing.assert_close(
        chunk_counts,
        torch.tensor(
            [[4096, 904, 100, 0], [100, 0, 4096, 0]],
            dtype=torch.int64,
        ),
    )
    torch.testing.assert_close(
        chunk_prefixes,
        torch.tensor(
            [[10, 138, 100, 228], [20, 148, 30, 158]],
            dtype=torch.int64,
        ),
    )
    torch.testing.assert_close(
        chunk_biases,
        torch.tensor([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]),
    )


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

        # The production path leaves probe metadata unexpanded.  Each scan
        # CTA resolves the selected centroid and list offsets itself.
        probe_cids = torch.arange(nprobe, device=device, dtype=torch.int64).repeat(
            num_queries, 1
        )
        list_group_prefix = torch.arange(nprobe, device=device, dtype=torch.int64)
        list_count = torch.ones(nprobe, device=device, dtype=torch.int64)
        _ivfpq_triton.triton_topk(
            global_query_luts=global_luts,
            static_vector_corrections=corrections,
            probe_distance_bias=probe_bias,
            list_codes_interleaved=codes,
            list_ids_interleaved=list_ids,
            probe_group_prefix=None,
            probe_list_count=None,
            probe_cids=probe_cids,
            list_group_prefix=list_group_prefix,
            list_count=list_count,
            topk=topk,
            output_distances=distances,
            output_ids=result_ids,
            list_codes_packed_16b=None,
        )
        torch.testing.assert_close(
            distances,
            expected_distances,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(result_ids, expected_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ivfpq_fused_metadata_matches_expanded_oversized_list():
    pytest.importorskip("triton")
    torch.manual_seed(701)
    device = torch.device("cuda:0")
    m, ksub, topk = 2, 8, 10
    list_count = torch.tensor([5000, 100], device=device, dtype=torch.int64)
    list_group_prefix = torch.tensor([0, 157], device=device, dtype=torch.int64)
    padded_vectors = (157 + 4) * INTERLEAVED_GROUP_SIZE
    global_luts = torch.rand((m, 1, ksub), device=device)
    codes = torch.randint(0, ksub, (m, padded_vectors), device=device, dtype=torch.uint8)
    ids = torch.arange(padded_vectors, device=device, dtype=torch.int64)
    corrections = torch.rand(padded_vectors, device=device)
    probe_cids = torch.tensor([[0, 1]], device=device, dtype=torch.int64)
    probe_bias = torch.tensor([[0.25, 0.5]], device=device)
    expanded = _chunk_probe_metadata(
        list_group_prefix[probe_cids],
        list_count[probe_cids],
        probe_bias,
        max_list_count=5000,
    )
    expanded_prefix, expanded_count, expanded_bias, chunk_groups, max_chunks = expanded

    legacy_distances = torch.empty((1, topk), device=device)
    legacy_ids = torch.empty((1, topk), device=device, dtype=torch.int64)
    _ivfpq_triton.triton_topk(
        global_query_luts=global_luts,
        static_vector_corrections=corrections,
        probe_distance_bias=expanded_bias,
        list_codes_interleaved=codes,
        list_ids_interleaved=ids,
        probe_group_prefix=expanded_prefix,
        probe_list_count=expanded_count,
        topk=topk,
        output_distances=legacy_distances,
        output_ids=legacy_ids,
        list_codes_packed_16b=None,
    )

    fused_distances = torch.empty_like(legacy_distances)
    fused_ids = torch.empty_like(legacy_ids)
    _ivfpq_triton.triton_topk(
        global_query_luts=global_luts,
        static_vector_corrections=corrections,
        probe_distance_bias=probe_bias,
        list_codes_interleaved=codes,
        list_ids_interleaved=ids,
        probe_group_prefix=None,
        probe_list_count=None,
        probe_cids=probe_cids,
        list_group_prefix=list_group_prefix,
        list_count=list_count,
        scan_chunk_groups=chunk_groups,
        scan_max_chunks=max_chunks,
        topk=topk,
        output_distances=fused_distances,
        output_ids=fused_ids,
        list_codes_packed_16b=None,
    )
    torch.testing.assert_close(fused_distances, legacy_distances)
    torch.testing.assert_close(fused_ids, legacy_ids)


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
    assert json.loads(cache_path.read_text())["version"] == _ivfpq_triton._AUTOTUNE_CACHE_FORMAT_VERSION

    _ivfpq_triton._AUTOTUNE_CONFIG_CACHE.clear()
    monkeypatch.setattr(_ivfpq_triton, "_PERSISTED_AUTOTUNE_CONFIGS", None)
    assert _ivfpq_triton._cached_autotune_config(cache_key) == expected


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
    assert not (output_dir / "list_vectors.pt").exists()
    assert not (output_dir / "list_ids_flat.pt").exists()
    assert not (output_dir / "list_prefix.pt").exists()
    assert (output_dir / "list_count.pt").exists()
    assert (output_dir / "list_vectors_interleaved.pt").exists()
    assert (output_dir / "list_ids_interleaved.pt").exists()
    assert (output_dir / "list_group_prefix.pt").exists()
    assert (output_dir / "list_vector_norms_interleaved.pt").exists()

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
    assert index.persisted_list_vector_norms_interleaved is not None

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
    assert single_file_index.persisted_list_vector_norms_interleaved is not None

    for removed_mode in (
        "tensor_topk",
        "tensor_padding_topk",
        "tensor_nopadding_topk",
        "tensor_list_topk",
        "global_sort",
        "list_gemm_topk",
        "triton_interleaved_direct_topk",
        "triton_interleaved_topk",
    ):
        with pytest.raises(ValueError, match="Unsupported IVFFlat selection_mode"):
            search_ivfflat_index(
                index,
                vectors[:16],
                topk=4,
                nprobe=4,
                device="cpu",
                selection_mode=removed_mode,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ivfflat_triton_topk_selects_nearest_nprobe_lists_from_all_centroids(tmp_path):
    pytest.importorskip("triton")
    centroids = torch.tensor([[0.0, 0.0], [20.0, 0.0], [2.0, 0.0], [40.0, 0.0]])
    list_count = torch.ones(4, dtype=torch.int64)
    list_prefix = torch.arange(4, dtype=torch.int64)
    list_ids = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
    layout = build_interleaved_flat_lists(centroids, list_ids, list_prefix, list_count)
    interleaved_norms = layout.vectors.square().sum(dim=(1, 3)).reshape(-1)
    index = IVFFlatIndex(
        index_dir=tmp_path,
        metadata={"nlist": 4, "dim": 2},
        vectors=None,
        assignments=torch.arange(4, dtype=torch.int64),
        centroids=centroids,
        centroid_norms=(centroids * centroids).sum(dim=-1),
        ids=list_ids,
        lists=[],
        list_count=list_count,
        list_vectors_interleaved=layout.vectors,
        list_ids_interleaved=layout.ids,
        list_group_prefix=layout.group_prefix,
        list_vector_norms_interleaved=interleaved_norms,
    )

    _, ids = search_ivfflat_index(
        index,
        torch.tensor([[1.8, 0.0], [38.8, 0.0]]),
        topk=1,
        nprobe=1,
        device="cuda:0",
        selection_mode="triton_topk",
    )

    torch.testing.assert_close(ids.cpu(), torch.tensor([[102], [103]], dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dim", [960, 961])
def test_ivfflat_high_dim_task_gemm_matches_reference(dim):
    pytest.importorskip("triton")
    generator = torch.Generator().manual_seed(2718)
    nlist = 4
    vectors = torch.randn(nlist * 64, dim, generator=generator)
    ids = torch.arange(vectors.shape[0], dtype=torch.int64)
    list_count = torch.full((nlist,), 64, dtype=torch.int64)
    list_prefix = torch.arange(0, vectors.shape[0], 64, dtype=torch.int64)
    layout = build_interleaved_flat_lists(vectors, ids, list_prefix, list_count)
    device = torch.device("cuda:0")
    list_vectors = layout.vectors.to(device)
    list_ids = layout.ids.to(device)
    list_group_prefix = layout.group_prefix.to(device)
    list_count_cuda = list_count.to(device)
    list_vector_norms = list_vectors.square().sum(dim=(1, 3)).reshape(-1)
    queries = vectors[:8].to(device)
    probe_cids = torch.arange(nlist, device=device, dtype=torch.int64).repeat(8, 1)

    actual_distances, actual_ids = _ivfflat_triton_topk(
        queries=queries,
        list_vectors_interleaved=list_vectors,
        list_ids_interleaved=list_ids,
        list_group_prefix=list_group_prefix,
        list_count=list_count_cuda,
        probe_cids=probe_cids,
        topk=10,
        list_vector_norms_interleaved=list_vector_norms,
        max_list_group_count=2,
    )
    reference_distances, reference_ids = torch.topk(
        torch.cdist(queries, vectors.to(device), p=2).square(),
        k=10,
        dim=1,
        largest=False,
    )

    torch.testing.assert_close(actual_ids, reference_ids)
    torch.testing.assert_close(actual_distances, reference_distances, rtol=2e-4, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("topk", [40, 100, 200, 10_000])
def test_ivfflat_task_gemm_supports_large_local_k(topk):
    pytest.importorskip("triton")
    generator = torch.Generator().manual_seed(31415)
    nlist = 4
    vectors = torch.randn(nlist * 64, 64, generator=generator)
    ids = torch.arange(vectors.shape[0], dtype=torch.int64)
    list_count = torch.full((nlist,), 64, dtype=torch.int64)
    list_prefix = torch.arange(0, vectors.shape[0], 64, dtype=torch.int64)
    layout = build_interleaved_flat_lists(vectors, ids, list_prefix, list_count)
    device = torch.device("cuda:0")
    list_vectors = layout.vectors.to(device)
    list_ids = layout.ids.to(device)
    list_group_prefix = layout.group_prefix.to(device)
    list_count_cuda = list_count.to(device)
    list_vector_norms = list_vectors.square().sum(dim=(1, 3)).reshape(-1)
    queries = vectors[:8].to(device)
    probe_cids = torch.arange(nlist, device=device, dtype=torch.int64).repeat(8, 1)

    original_direct = _ivfflat_triton._run_direct_interleaved_topk

    def forbidden_direct(*args, **kwargs):
        raise AssertionError("large-k task GEMM selected direct scan")

    _ivfflat_triton._run_direct_interleaved_topk = forbidden_direct
    try:
        actual_distances, actual_ids = _ivfflat_triton_topk(
            queries=queries,
            list_vectors_interleaved=list_vectors,
            list_ids_interleaved=list_ids,
            list_group_prefix=list_group_prefix,
            list_count=list_count_cuda,
            probe_cids=probe_cids,
            topk=topk,
            list_vector_norms_interleaved=list_vector_norms,
            max_list_group_count=2,
        )
    finally:
        _ivfflat_triton._run_direct_interleaved_topk = original_direct

    reference_k = min(topk, vectors.shape[0])
    reference_distances, reference_ids = torch.topk(
        torch.cdist(queries, vectors.to(device), p=2).square(),
        k=reference_k,
        dim=1,
        largest=False,
    )
    torch.testing.assert_close(actual_ids[:, :reference_k], reference_ids)
    torch.testing.assert_close(
        actual_distances[:, :reference_k],
        reference_distances,
        rtol=2e-4,
        atol=2e-3,
    )
    assert torch.equal(actual_ids[:, reference_k:], torch.full_like(actual_ids[:, reference_k:], -1))
    assert torch.isinf(actual_distances[:, reference_k:]).all()


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
    assert index.metadata["coarse_kmeans_method"] == "balanced"
    assert index.metadata["pq_kmeans_method"] == "balanced"
    assert index.metadata["coarse_seed"] in (11, 12)
    assert len(index.metadata["pq_seeds"]) == 2
    assert len(set(index.metadata["pq_seeds"])) == 2

    coarse_labels = result.labels.cpu().to(torch.int64)
    assert coarse_labels.shape == (vectors.shape[0],)
    assert int(coarse_labels.min()) >= 0
    assert int(coarse_labels.max()) < result.centroids.shape[0]

    with pytest.raises(ValueError, match="Unsupported LUT dtype"):
        search_ivfpq_index(
            index,
            queries,
            topk=4,
            nprobe=4,
            device="cpu",
            lut_dtype="fp8",
        )

    for removed_mode in (
        "tensor_topk",
        "triton_list_major",
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
    with pytest.raises(RuntimeError, match="Triton IVF-PQ backend"):
        search_ivfpq_index(
            index,
            queries,
            topk=4,
            nprobe=4,
            device="cpu",
            selection_mode="triton_topk",
            backend="triton",
        )


def test_ivfpq_separate_coarse_and_pq_training_controls(tmp_path):
    generator = torch.Generator().manual_seed(321)
    vectors = torch.randn(256, 8, generator=generator)
    output_path = tmp_path / "ivfpq-separate-training.bin"

    result = build_ivfpq_index(
        vectors=vectors,
        nlist=4,
        niter=1,
        use_triton=False,
        device="cpu",
        m=2,
        nbits=3,
        seed=19,
        n_init=1,
        coarse_niter=2,
        pq_niter=3,
        coarse_n_init=1,
        pq_n_init=2,
        coarse_max_points_per_centroid=32,
        pq_max_points_per_centroid=48,
        coarse_init_method="kmeans++",
        pq_init_method="kmeans++",
    )
    assert result.coarse_niter == 2
    assert result.pq_niter == 3
    assert result.coarse_n_init == 1
    assert result.pq_n_init == 2
    assert result.coarse_max_points_per_centroid == 32
    assert result.pq_max_points_per_centroid == 48

    save_ivfpq_index(result, output_path)
    index = load_ivfpq_index(output_path)
    assert index.metadata["coarse_niter"] == 2
    assert index.metadata["pq_niter"] == 3
    assert index.metadata["coarse_n_init"] == 1
    assert index.metadata["pq_n_init"] == 2
    assert index.metadata["coarse_max_points_per_centroid"] == 32
    assert index.metadata["pq_max_points_per_centroid"] == 48


def test_ivfpq_keeps_pq_sampling_independent_from_coarse():
    generator = torch.Generator().manual_seed(322)
    vectors = torch.randn(128, 8, generator=generator)

    result = build_ivfpq_index(
        vectors=vectors,
        nlist=4,
        niter=2,
        use_triton=False,
        device="cpu",
        m=2,
        nbits=2,
        seed=23,
        n_init=1,
        coarse_trainset_fraction=0.5,
        pq_max_points_per_centroid=4,
    )

    # The coarse quantizer uses a 64-row sample, but PQ keeps its own
    # per-subspace sampling path instead of reusing a shared prefix.
    assert not result.pq_reuses_coarse_trainset
    assert result.pq_training_rows is None
    assert result.coarse_trainset_fraction == 0.5
    assert result.pq_max_points_per_centroid == 4


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

    assert index.persisted_list_codes_interleaved_by_subspace is not None
    assert index.persisted_list_codes_interleaved_by_subspace.shape[0] == 3
    assert index.persisted_list_ids_interleaved is not None
    assert index.persisted_list_group_prefix is not None


def test_ivfpq_persists_only_triton_list_layout(tmp_path):
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
        "list_count.pt",
        "list_codes_interleaved_by_subspace.pt",
        "list_ids_interleaved.pt",
        "list_group_prefix.pt",
        "list_pq_static_corrections_interleaved.pt",
    ):
        assert (output_dir / name).exists()
    for name in (
        "list_codes.pt",
        "list_ids_flat.pt",
        "list_prefix.pt",
        "list_pq_static_corrections.pt",
    ):
        assert not (output_dir / name).exists()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["interleaved_layout"] == {
        "available": True,
        "group_size": INTERLEAVED_GROUP_SIZE,
        "code_layout": "subspace_major",
    }
    index = load_ivfpq_index(output_dir)
    assert index.persisted_list_codes_interleaved_by_subspace is not None
    assert index.persisted_list_ids_interleaved is not None
    assert index.persisted_list_group_prefix is not None
    assert index.persisted_list_pq_static_corrections_interleaved is not None
    torch.testing.assert_close(
        index.centroid_norms,
        (index.centroids * index.centroids).sum(dim=-1),
    )
    torch.testing.assert_close(
        index.pq_codebook_norms,
        (index.pq_codebooks * index.pq_codebooks).sum(dim=-1),
    )
    sort_perm = torch.argsort(result.labels.cpu().to(torch.int64), stable=True)
    expected_list_codes = result.pq_codes.cpu()[sort_perm]
    expected_list_ids = torch.arange(vectors.shape[0], dtype=torch.int64)[sort_perm]
    expected_list_count = torch.bincount(
        result.labels.cpu().to(torch.int64), minlength=4
    ).to(torch.int64)
    expected_list_prefix = torch.cumsum(expected_list_count, dim=0) - expected_list_count
    expected_layout = build_interleaved_pq_lists(
        expected_list_codes,
        expected_list_ids,
        expected_list_prefix,
        expected_list_count,
    )
    torch.testing.assert_close(
        index.persisted_list_codes_interleaved_by_subspace,
        expected_layout.codes_by_subspace,
    )
    torch.testing.assert_close(index.persisted_list_ids_interleaved, expected_layout.ids)
    torch.testing.assert_close(index.persisted_list_group_prefix, expected_layout.group_prefix)
    expected_static_corrections = _build_interleaved_pq_static_corrections(
        expected_list_codes,
        expected_list_prefix,
        expected_list_count,
        index.centroids,
        index.pq_codebooks,
    )
    torch.testing.assert_close(
        index.persisted_list_pq_static_corrections_interleaved,
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
    assert "triton_scan" in profile
    assert "triton_final_merge" in profile
    assert "triton_topk_select" not in profile

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
def test_ivfpq_cuda_graph_replays_new_queries_with_workspace(tmp_path):
    pytest.importorskip("triton")
    generator = torch.Generator().manual_seed(311)
    vectors = torch.randn(128, 8, generator=generator)
    result = build_ivfpq_index(
        vectors=vectors,
        nlist=4,
        niter=2,
        use_triton=False,
        device="cpu",
        m=2,
        nbits=3,
        seed=19,
        n_init=1,
    )
    output_path = tmp_path / "ivfpq-cuda-graph.bin"
    save_ivfpq_index(result, output_path)
    index = load_ivfpq_index(output_path)
    index.reside_on_device("cuda:0")

    first_queries = vectors[:16]
    second_queries = vectors[16:32]
    expected_first = search_ivfpq_index(
        index, first_queries, topk=4, nprobe=4, device="cuda:0"
    )
    expected_second = search_ivfpq_index(
        index, second_queries, topk=4, nprobe=4, device="cuda:0"
    )

    graph_first = search_ivfpq_index(
        index,
        first_queries,
        topk=4,
        nprobe=4,
        device="cuda:0",
        use_preallocated_workspace=True,
        use_cuda_graph=True,
    )
    actual_first = tuple(tensor.clone() for tensor in graph_first)
    graph_second = search_ivfpq_index(
        index,
        second_queries,
        topk=4,
        nprobe=4,
        device="cuda:0",
        use_preallocated_workspace=True,
        use_cuda_graph=True,
    )
    actual_second = tuple(tensor.clone() for tensor in graph_second)

    assert index.search_workspace is not None
    assert index.search_workspace.graph is not None
    torch.testing.assert_close(actual_first[0], expected_first[0], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_first[1], expected_first[1])
    torch.testing.assert_close(actual_second[0], expected_second[0], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_second[1], expected_second[1])

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [17, 25, 31, 32, 33, 47, 48])
def test_ivfpq_triton_packed_chunked_matches_unpacked(tmp_path, m):
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
    assert index.list_codes_interleaved_by_subspace is None
    assert index.list_codes_interleaved_packed_16b is not None
    packed_data_ptr = index.list_codes_interleaved_packed_16b.data_ptr()
    index.reside_on_device("cuda:0")
    assert index.list_codes_interleaved_packed_16b.data_ptr() == packed_data_ptr

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
    assert "triton_packed_prepare" in profile
    assert index.list_codes_interleaved_packed_16b is not None
    assert index.list_codes_interleaved_packed_16b.shape[1:] == (
        math.ceil(m / 16),
        32,
        2,
    )
    unpacked_distances, unpacked_ids = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="triton_topk",
        use_packed_16b_codes=False,
    )
    torch.testing.assert_close(unpacked_ids, packed_ids)
    torch.testing.assert_close(unpacked_distances, packed_distances, rtol=2e-4, atol=2e-4)

    unpacked_graph_request = search_ivfpq_index(
        index,
        vectors[:16],
        topk=4,
        nprobe=4,
        device="cuda:0",
        selection_mode="triton_topk",
        use_packed_16b_codes=False,
        use_preallocated_workspace=True,
        use_cuda_graph=True,
    )
    assert index.search_workspace is not None
    assert index.search_workspace.graph is None
    torch.testing.assert_close(unpacked_graph_request[1], packed_ids)
    torch.testing.assert_close(
        unpacked_graph_request[0], packed_distances, rtol=2e-4, atol=2e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("m", "num_chunks"), [(96, 6), (192, 12)])
def test_ivfpq_triton_chunked_wide_m(tmp_path, m, num_chunks):
    pytest.importorskip("triton")
    generator = torch.Generator().manual_seed(733)
    vectors = torch.randn(128, m, generator=generator)
    result = build_ivfpq_index(
        vectors=vectors,
        nlist=4,
        niter=1,
        use_triton=False,
        device="cpu",
        m=m,
        nbits=2,
        seed=13,
        n_init=1,
    )
    output_path = tmp_path / f"ivfpq-m{m}-chunked.bin"
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
    assert "triton_packed_prepare" in profile
    assert "triton_scan" in profile
    assert index.list_codes_interleaved_packed_16b is not None
    assert index.list_codes_interleaved_packed_16b.shape[1:] == (num_chunks, 32, 2)
    assert chunked_ids.shape == (16, 4)
    assert chunked_distances.shape == (16, 4)





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
