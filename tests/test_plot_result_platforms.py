"""Keep AMD result renames compatible without mixing GPU platforms in figures."""

import pytest

from plot import ivf_resource_table, ivfflat_cuda_kernel, ivfpq_cuda_kernel
from plot import plot_amd_ivf_compare as amd


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
    index = tmp_path / "index.bin"
    index.write_bytes(b"index")
    monkeypatch.setattr(ivf_resource_table, "index_path", lambda *_: index)
    monkeypatch.setattr(ivf_resource_table, "matching_build_time", lambda *_: 1.0)
    memory_paths = []

    def memory(path):
        memory_paths.append(path)
        return {1: 0.5}

    monkeypatch.setattr(ivf_resource_table, "peak_memory_by_nprobe_gib", memory)
    ivf_resource_table.collect_records(ivf_resource_table.Dataset(dataset, dataset), kind)
    assert memory_paths
    prefix = "IVFFLAT_" if kind == "IVF-Flat" else "IVFPQ_"
    assert all(path.name.startswith(prefix) for path in memory_paths)
