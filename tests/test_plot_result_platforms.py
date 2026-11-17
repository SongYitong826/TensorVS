"""Keep AMD result renames compatible without mixing GPU platforms in figures."""

import pytest

from plot import ivf_resource_table, ivfflat_cuda_kernel, ivfpq_cuda_kernel
from plot import plot_amd_ivf_compare as amd


@pytest.mark.parametrize("q_batch", [None, 128])
def test_ascend_flat_identifies_cpu_fallback_and_uses_shared_recall_range(tmp_path, monkeypatch, q_batch):
    from matplotlib.figure import Figure
    from plot import plot_ascend_ivf_compare as ascend

    def inspect_figure(figure, *_args, **_kwargs):
        assert len(figure.axes) == (6 if q_batch is None else 2)
        for axis in figure.axes:
            assert axis.get_xlim() == pytest.approx((0.85, 1.005))
            assert {line.get_label() for line in axis.lines} == {
                ascend.CPU_METHOD, ascend.TENSOR_METHOD,
            }
            for line in axis.lines:
                expected_color = "#00609d" if line.get_label() == ascend.TENSOR_METHOD else "#2ca02c"
                assert line.get_color() == expected_color
                assert line.get_linestyle() == "-"
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            ascend.CPU_METHOD, ascend.TENSOR_METHOD,
        ]

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    monkeypatch.setattr(ascend, "PAPER_FIGURE_DIR", tmp_path / "no-paper-dir")
    pdf_path, _ = ascend.plot(ascend.ROOT / "result", tmp_path, "ivfflat", q_batch=q_batch)
    if q_batch is not None:
        assert pdf_path.name == "IVFFLAT_Ascend_QBatch128_K10.pdf"


@pytest.mark.parametrize("module,kind", [(ivfflat_cuda_kernel, "IVFFLAT"), (ivfpq_cuda_kernel, "IVFPQ")])
def test_cuda_sift_column_uses_complete_sift100m_results(module, kind):
    assert module.DATASETS[0].key == "sift100m"
    assert module.DATASETS[0].label == "SIFT"
    found = module.discover_result_files({method: (module.PROJECT_ROOT / "result",) for method in module.METHODS})
    if kind == "IVFPQ":
        found, _ = found
    for method in module.METHODS:
        source = found[method]["sift100m"]
        assert "MultiGPU3" in source.path.name
        rows = module.read_result_rows(source)
        assert len(rows) == 42
        assert {row["q_batch"] for row in rows} == {32, 128, 512}


@pytest.mark.parametrize("kind", ["ivfflat", "ivfpq"])
@pytest.mark.parametrize("dataset,label,pq_config", amd.DATASETS)
@pytest.mark.parametrize("method", amd.BASE_METHODS + (amd.TRITON_METHOD,))
def test_checked_in_amd_results_are_readable(kind, dataset, label, pq_config, method):
    path = amd.result_path(amd.ROOT / "result", method, kind, dataset, pq_config)
    assert path.name.startswith("AMD_")
    curves = amd.read_curves(path)
    assert set(curves) == set(amd.QBATCHES)
    assert all(curves.values())


@pytest.mark.parametrize("kind", ["ivfflat", "ivfpq"])
def test_amd_prefers_renamed_results_and_accepts_legacy_files(tmp_path, kind):
    suffix = "_M32_nbits8_train512" if kind == "ivfpq" else ""
    legacy = tmp_path / f"TensorVS_Tensor_{kind.upper()}_glove1.2m_nlist1024{suffix}.csv"
    legacy.touch()
    args = (tmp_path, "TensorVS (Tensor)", kind, "glove1.2m", "M32_nbits8_train512")
    assert amd.result_path(*args) == legacy
    renamed = tmp_path / f"AMD_{legacy.name}"
    renamed.touch()
    assert amd.result_path(*args) == renamed


@pytest.mark.parametrize("dataset,kind,config", [
    ("glove1.2m", "ivfflat", "M32_nbits8_train512"),
    ("hunyuan", "ivfflat", "M384_nbits8_train256"),
    ("glove1.2m", "ivfpq", "M32_nbits8_train512"),
])
def test_faiss_rocm_results_use_the_measured_configuration(dataset, kind, config):
    path = amd.result_path(amd.ROOT / "result", amd.FAISS_METHOD, kind, dataset, config)
    assert path.name.startswith("AMD_FaissROCm_")
    if kind == "ivfpq":
        assert "M32_nbits8_train256_pad224" in path.name
    assert all(amd.read_curves(path).values())


@pytest.mark.parametrize("kind", ["ivfflat", "ivfpq"])
def test_faiss_rocm_appears_only_in_supported_panels(tmp_path, monkeypatch, kind):
    from matplotlib.figure import Figure

    snapshots = []

    def inspect_figure(figure, *_args, **_kwargs):
        snapshots.append(figure)
        for index, axis in enumerate(figure.axes):
            labels = {line.get_label() for line in axis.lines}
            assert (amd.FAISS_METHOD in labels) == (kind == "ivfflat" or index < 3)
            assert len(labels) == (3 if kind == "ivfpq" and index >= 3 else 4)
        assert [label.get_text() for label in figure.legends[0].get_texts()] == list(amd.METHODS)

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    monkeypatch.setattr(amd, "PAPER_FIGURE_DIR", tmp_path / "no-paper-dir")
    amd.plot(amd.ROOT / "result", tmp_path, kind)
    assert len(snapshots) == 2


@pytest.mark.parametrize("kind,module", [
    ("IVFFLAT", ivfflat_cuda_kernel), ("IVFPQ", ivfpq_cuda_kernel),
])
def test_cuda_discovery_does_not_select_amd_results(tmp_path, kind, module):
    suffix = "_M32_nbits8_train512" if kind == "IVFPQ" else ""
    stem = f"{kind}_glove1.2m_nlist1024{suffix}.csv"
    for prefix in ("AMD_TensorVS_Tensor_", "AMD_TensorVS_Triton_",
                   "AMD_FaissROCm_", "AMD_hipVS_", "TensorVS_Tensor_"):
        (tmp_path / f"{prefix}{stem}").touch()
    result_dirs = {method: (tmp_path,) for method in module.METHODS}
    found = module.discover_result_files(result_dirs)
    if kind == "IVFPQ":
        found, _ = found
    assert all(not results for results in found.values())
    cuda = tmp_path / stem
    cuda.touch()
    found = module.discover_result_files(result_dirs)
    if kind == "IVFPQ":
        found, _ = found
    assert found["tensorvs"]["glove1.2m"].path == cuda


@pytest.mark.parametrize("dataset", ["glove1.2m", "hunyuan"])
@pytest.mark.parametrize("kind", ["IVF-Flat", "IVF-PQ"])
def test_cuda_resource_table_reads_cuda_memory_only(tmp_path, monkeypatch, dataset, kind):
    calls = []
    def load(dataset, method, family, nlist, **kwargs):
        path = tmp_path / f"CUDA_{method}_{family.upper()}_{dataset}_nlist{nlist}_build.csv"
        calls.append(path)
        return path, dict(index_size_bytes=1024**3, build_seconds=1.0,
                          legacy_table_statistics={"peak_gpu_memory_gib": 0.5},
                          peak_build_gpu_memory_mib=9999)
    monkeypatch.setattr(ivf_resource_table, "load_build", load)
    ivf_resource_table.collect_records(ivf_resource_table.Dataset(dataset, dataset), kind)
    assert len(calls) == 3
    assert all(path.name.startswith("CUDA_") and path.name.endswith("_build.csv") for path in calls)
