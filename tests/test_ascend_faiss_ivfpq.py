"""CPU-only checks of padding, build accounting/cache, and Faiss API dispatch."""
from types import SimpleNamespace
from pathlib import Path

import faiss
import numpy as np
import pytest
import torch

from bench.ascend_faiss_ivfpq_bench import search_batch
from bench._ascend_process_memory import process_hbm_mib
from bench._ascend_faiss_build import pad_vectors, build_or_load, create_residual_ip_index, exact_ip_ground_truth
from bench.ascend_faiss_ivfpq_bench import save_build_summary
import json
import os
import subprocess


def test_zero_padding_preserves_inner_product():
    vectors = np.random.default_rng(0).normal(size=(4, 200)).astype("float32")
    padded = pad_vectors(vectors)
    assert padded.shape == (4, 256)
    np.testing.assert_array_equal(padded[:, :200], vectors)
    assert not padded[:, 200:].any()
    np.testing.assert_allclose(vectors @ vectors.T, padded @ padded.T, rtol=1e-6, atol=1e-5)


def test_release_uses_explicit_residual_ip_constructor():
    calls = []
    sentinel = object()
    def create(*args):
        calls.append(args)
        return sentinel
    old = SimpleNamespace(NpuIndexIVFPQ=create, METRIC_INNER_PRODUCT=0)
    assert create_residual_ip_index(old, object(), object()) is sentinel
    assert len(calls) == 1 and calls[0][-3:-1] == (0, True)


def test_ip_truth_is_exhaustive_unnormalized_and_cache_tracks_queries(tmp_path, monkeypatch):
    # L2 nearest neighbor is row 1; maximum raw inner product is row 0.
    base = np.array([[10., .1], [1., .2], [.3, 3.]], dtype="float32")
    query = np.array([[1., 0.], [0., 2.]], dtype="float32")
    source, cache = tmp_path / "base.fvecs", tmp_path / "truth.npz"
    source.write_bytes(b"signature")
    calls = []
    def load(_):
        calls.append(1)
        return torch.from_numpy(base)
    monkeypatch.setattr("tensorvs.load_dataset", load)
    found, info = exact_ip_ground_truth(faiss, source, query, 2, cache)
    np.testing.assert_array_equal(found, np.argsort(-(query @ base.T), axis=1)[:, :2])
    assert not info["cached"]
    again, info = exact_ip_ground_truth(faiss, source, query, 2, cache)
    np.testing.assert_array_equal(again, found)
    assert info["cached"] and len(calls) == 1
    changed, info = exact_ip_ground_truth(faiss, source, -query, 2, cache)
    assert not info["cached"] and len(calls) == 2
    assert changed[0, 0] != found[0, 0]


@pytest.mark.parametrize("method,dataset", [(m, d) for m in ("ivfflat", "ivfpq")
                         for d in ("glove1.2m", "hunyuan")])
def test_tensor_launcher_runs_exactly_one_requested_method(tmp_path, method, dataset):
    conda = tmp_path / "conda"
    conda.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    conda.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "bench/run_ascend_tensor_bench.sh"),
                             method, dataset, "--resume"], check=True, capture_output=True,
                            text=True, env=dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}"))
    argv = result.stdout.splitlines()
    assert argv.count("--kind") == 1 and argv[argv.index("--kind") + 1] == method
    assert argv[argv.index("--dataset") + 1] == dataset
    assert argv[-1] == "--resume"


def test_faiss_launcher_explicit_method_and_dataset(tmp_path):
    conda = tmp_path / "conda"
    conda.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    conda.chmod(0o755)
    (tmp_path / "env.sh").write_text("")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "bench/run_ascend_faiss_ivfpq_bench.sh"),
                             "ivfpq", "glove1.2m", "--resume"], check=True, capture_output=True,
                            text=True, env=dict(os.environ, FAISS_ASCEND_RUNTIME=str(tmp_path),
                                                PATH=f"{tmp_path}:{os.environ['PATH']}"))
    argv = result.stdout.splitlines()
    assert argv.count(str(root / "bench/ascend_faiss_ivfpq_bench.py")) == 1
    assert argv[-1] == "--resume"


def test_build_measurements_and_resume_retain_original_timing(tmp_path, monkeypatch):
    from bench import _ascend_faiss_build as builder
    clock = [0.0]
    calls = []
    def advance(seconds):
        clock[0] += seconds
    class Index:
        def __init__(self, *_args):
            self.ntotal = 0
            self.is_trained = False
            self.pq = SimpleNamespace(cp=SimpleNamespace())
            self.metric_type, self.by_residual = 0, True
        def train(self, x):
            calls.append("train")
            assert x.shape == (1024, 256)
            advance(10)
            self.is_trained = True
        def add(self, x):
            calls.append("add")
            advance(2)
            self.ntotal += len(x)
        def copyTo(self, cpu):
            advance(7)
            cpu.ntotal = self.ntotal
        def getListLength(self, i):
            return self.ntotal // 1024 + (i < self.ntotal % 1024)
    def create(*a):
        index = Index()
        if len(a) == 3:  # cached CPU -> NPU; never retrain
            index.ntotal = a[1].ntotal
        return index
    def write(cpu, path):
        advance(3)
        Path(path).write_text(json.dumps(dict(ntotal=cpu.ntotal)))
    def read(path):
        cpu = Index()
        cpu.ntotal = json.loads(Path(path).read_text())["ntotal"]
        return cpu
    native = SimpleNamespace(__version__="test", __file__=__file__, METRIC_INNER_PRODUCT=0,
        NpuIndexIVFPQ=create, IndexFlatIP=lambda d: object(), IndexIVFPQ=Index,
        write_index=write, read_index=read)
    args = SimpleNamespace(index_dir=tmp_path / "index", rebuild_index=False, seed=1234,
        coarse_niter=25, pq_niter=50, train_size=0, add_batch_size=512,
        lut_dtype="fp16", npu_id=0)
    source = tmp_path / "base.fvecs"
    source.write_bytes(b"raw vectors signature")
    monkeypatch.setattr("tensorvs.load_dataset", lambda p: torch.zeros(1024, 200))
    monkeypatch.setattr(builder.time, "perf_counter", lambda: clock[0])
    resources = SimpleNamespace(getResources=lambda: SimpleNamespace(syncDefaultStream=lambda d: None))
    _, _, counts, build, loaded = build_or_load(native, resources, object(), args, source)
    assert not loaded and counts.sum() == 1024
    assert build["train_seconds"] == 10
    assert build["add_seconds"] == 4
    assert build["build_time_seconds"] == 14
    assert build["serialize_seconds"] == 10  # serialization excluded from build
    assert build["index_size_bytes"] == Path(build["index_path"]).stat().st_size
    _, _, _, cached_build, loaded = build_or_load(native, resources, object(), args, source)
    assert loaded and cached_build == build and calls == ["train", "add", "add"]
    summary = tmp_path / "build.csv"
    save_build_summary(summary, build)
    assert "build_time_seconds" in summary.read_text().splitlines()[0]
    Path(build["index_path"]).write_text("corrupted")
    with pytest.raises(ValueError, match="Cached index"):
        build_or_load(native, resources, object(), args, source)


def test_hbm_counts_current_process_not_other_users_or_device_total():
    output = "| 5 0 | 123 | python | 100 |\n| 5 0 | 456 | python | 3000 |"
    assert process_hbm_mib(output, 123) == (5, 0, 100)
    with pytest.raises(RuntimeError, match="PID 999"):
        process_hbm_mib(output, 999)


def test_search_pipeline_exposes_assignments_actually_scanned():
    rng = np.random.default_rng(2)
    quantizer = faiss.IndexFlatIP(32)
    centroids = rng.normal(size=(64, 32)).astype("float32")
    faiss.normalize_L2(centroids)
    quantizer.add(centroids)
    cpu = faiss.IndexIVFPQ(quantizer, 32, 64, 32, 8, faiss.METRIC_INNER_PRODUCT)
    cpu.is_trained = True
    cpu.by_residual = True
    faiss.copy_array_to_vector(np.zeros(32 * 256, dtype="float32"), cpu.pq.centroids)
    ci, cc = np.array([33], dtype="int64"), np.zeros(32, dtype="uint8")
    cpu.invlists.add_entries(2, 1, faiss.swig_ptr(ci), faiss.swig_ptr(cc))
    cpu.ntotal, cpu.nprobe = 1, 1
    batch = quantizer.reconstruct_n(2, 1)
    synced = []
    resources = SimpleNamespace(getResources=lambda: SimpleNamespace(syncDefaultStream=synced.append))
    _, found, assignments = search_batch(faiss, cpu, quantizer, resources, 0, batch, None, 1)
    np.testing.assert_array_equal(assignments, [[2]])
    np.testing.assert_array_equal(found, [[33]])
    assert synced == [0]


def test_disjoint_probe_scans_merge_ip_topk_and_restore_nprobe():
    rng = np.random.default_rng(12)
    centroids = rng.normal(size=(128, 32)).astype("float32")
    faiss.normalize_L2(centroids)
    quantizer = faiss.IndexFlatIP(32)
    quantizer.add(centroids)
    cpu = faiss.IndexIVFPQ(quantizer, 32, 128, 32, 8, faiss.METRIC_INNER_PRODUCT)
    cpu.is_trained = cpu.by_residual = True
    faiss.copy_array_to_vector(np.zeros(32 * 256, dtype="float32"), cpu.pq.centroids)
    code = np.zeros(32, dtype="uint8")
    for list_no in range(128):
        ids = np.array([list_no], dtype="int64")
        cpu.invlists.add_entries(list_no, 1, faiss.swig_ptr(ids), faiss.swig_ptr(code))
    cpu.ntotal, cpu.nprobe = 128, 100
    query = rng.normal(size=(4, 32)).astype("float32")
    expected_d, expected_i = cpu.search(query, 10)
    syncs = []
    resources = SimpleNamespace(getResources=lambda: SimpleNamespace(syncDefaultStream=syncs.append))
    actual_d, actual_i, lists = search_batch(faiss, cpu, quantizer, resources, 0, query, None, 10)
    np.testing.assert_array_equal(actual_i, expected_i)
    np.testing.assert_allclose(actual_d, expected_d, atol=1e-5)
    assert cpu.nprobe == 100 and lists.shape == (4, 100) and syncs == [0, 0]
