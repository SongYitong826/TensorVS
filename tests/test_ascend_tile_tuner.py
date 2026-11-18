"""Exercise portable tuning and sampled/full comparisons without vendor hardware."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorvs import IVFFlatBuildResult, IVFPQBuildResult, save_ivfflat_index, save_ivfpq_index


@pytest.fixture(autouse=True)
def project_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("reference_limit", [0, 3])
@pytest.mark.parametrize("kind", ["ivfflat", "ivfpq"])
def test_tuner_keeps_index_and_search_settings_fixed(tmp_path, monkeypatch, reference_limit, kind):
    tuner = importlib.import_module("bench.tune_tensor_tiles")
    generator = torch.Generator().manual_seed(709)
    vectors, centroids = torch.randn(97, 8, generator=generator), torch.randn(4, 8, generator=generator)
    labels = torch.randint(0, 4, (97,), generator=generator)
    books = torch.randn(4, 16, 2, generator=generator)
    codes = torch.randint(0, 16, (97, 4), dtype=torch.uint8, generator=generator)
    if kind == "ivfpq":
        source = save_ivfpq_index(IVFPQBuildResult(vectors, labels, centroids, books, codes,
            torch.device("cpu"), 4, 4), tmp_path / "index.bin")
        decoded = centroids[labels] + torch.stack([books[m, codes[:, m].long()] for m in range(4)], 1).flatten(1)
    else:
        source = save_ivfflat_index(IVFFlatBuildResult(vectors, labels, centroids,
            torch.device("cpu")), tmp_path / "index.bin")
        decoded = vectors
    original_bytes = source.read_bytes()
    queries = torch.randn(16, 8, generator=generator)
    truth = (queries[:, None] - decoded).square().sum(-1).topk(10, largest=False).indices
    query_path, truth_path = tmp_path / "query.npy", tmp_path / "truth.npy"
    np.save(query_path, queries.numpy())
    np.save(truth_path, truth.numpy())
    search_name = f"search_{kind}_index"
    search = getattr(tuner, search_name)
    seen_options = []

    def capture_options(index, batch, **options):
        seen_options.append(options)
        return search(index, batch, **options)

    monkeypatch.setattr(tuner, search_name, capture_options)
    # Equal measured latencies force the test to detect accidentally selecting
    # the smaller sampled baseline as the best full-batch configuration.
    clock = iter(range(10000))
    monkeypatch.setattr(tuner.time, "perf_counter", lambda: next(clock))
    output = tmp_path / "tuning.json"
    monkeypatch.setattr("sys.argv", ["tuner", kind, "--platform", "cpu", "--q-batch", "8", "--query-offset", "4",
        "--index", str(source), "--queries", str(query_path), "--groundtruth", str(truth_path),
        "--nprobe", "4", "--topk", "3", "--query-tiles", "4", "--candidate-chunks", "32",
        "--task-caps", "32", "--memory-budgets", "2", "--pq-lookup", "gather", "--lut-dtype", "fp32",
        "--repetitions", "1", "--reference-query-limit", str(reference_limit), "--skip-stage-profile",
        "--output", str(output)])
    tuner.main()
    report = json.loads(output.read_text())
    baseline, trial = report["results"]
    assert baseline["queries_measured"] == (reference_limit or 8)
    assert trial["queries_measured"] == 8 and report["best"]["queries_measured"] == 8
    assert trial["baseline_id_overlap"] == 1
    assert baseline["validation_recall"] == trial["validation_recall"]
    assert report["query_offset"] == 4 and source.read_bytes() == original_bytes
    assert report["platform"] == "cpu" and report["execution_device"] == "cpu"
    assert all(options["nprobe"] == 4 and options["topk"] == 3
        and options["backend"] == "torch" and not options["persist_layout"]
        and not options["use_compile"] for options in seen_options)
    if kind == "ivfpq":
        assert all(options["lut_dtype"] == "fp32" for options in seen_options)
    if reference_limit:
        assert report["speedup_over_default"] is None and not report["speedup_comparable"]
    else:
        assert report["speedup_over_default"] == 1 and report["speedup_comparable"]


@pytest.mark.parametrize("platform, device, plugin", [
    ("ascend", "npu", "torch_npu"), ("mlu", "mlu", "torch_mlu"),
    ("kunlun", "cuda", "torch_xmlir"),
])
def test_vendor_runtime_registers_plugin_and_synchronizes_selected_device(monkeypatch, platform, device, plugin):
    tuner = importlib.import_module("bench.tune_tensor_tiles")
    selected, synced, imported = [], [], []
    matmul = SimpleNamespace(allow_tf32=True, allow_hf32=True)
    api = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
        set_device=selected.append, synchronize=lambda: synced.append(True),
        get_device_name=lambda dev: "Vendor accelerator", matmul=matmul)
    monkeypatch.setattr(torch, device, api, raising=False)
    monkeypatch.setattr(torch, "device", lambda name: SimpleNamespace(type=name.split(":")[0], index=int(name.split(":")[1])))
    monkeypatch.setattr(tuner.importlib, "import_module", lambda name: imported.append(name) or SimpleNamespace(__version__="test"))
    runtime = tuner.configure_runtime(platform, f"{device}:1")
    runtime.synchronize()
    assert imported == [plugin] and selected == [runtime.device] and synced == [True]
    assert runtime.name == "Vendor accelerator" and runtime.modules == {plugin: "test"}
    assert not matmul.allow_tf32 and not matmul.allow_hf32


def test_kunlun_supports_explicit_vendor_registration_and_rejects_unavailable_device(monkeypatch):
    tuner = importlib.import_module("bench.tune_tensor_tiles")
    imported = []
    monkeypatch.setattr(tuner.importlib, "import_module", lambda name: imported.append(name) or SimpleNamespace())
    monkeypatch.setattr(torch, "xpu", SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="unavailable"):
        tuner.configure_runtime("kunlun", "xpu:0", ["custom_kunlun_plugin"])
    assert imported == ["custom_kunlun_plugin"]


def test_groundtruth_preserves_large_integer_ids(tmp_path):
    tuner = importlib.import_module("bench.tune_tensor_tiles")
    ids = np.array([[2**24 + 1, 2**24 + 3]], dtype=np.int64)
    path = tmp_path / "truth.npy"
    np.save(path, ids)
    assert tuner.load_groundtruth(path).tolist() == ids.tolist()
    np.save(path, ids.astype(np.float32))
    with pytest.raises(ValueError, match="int32/int64"):
        tuner.load_groundtruth(path)


def test_legacy_ascend_entrypoint_delegates_to_portable_tuner():
    from bench import tune_ascend_tensor_tiles, tune_tensor_tiles
    assert tune_ascend_tensor_tiles.main is tune_tensor_tiles.main
