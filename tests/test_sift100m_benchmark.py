"""SIFT100M configuration and binary input compatibility, without GPU work."""
import importlib
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from bench._benchmark_utils import load_benchmark_vectors, load_benchmark_groundtruth


def write_matrix(path, values):
    with path.open("wb") as output:
        np.array(values.shape, dtype="<u4").tofile(output)
        values.tofile(output)


def test_binary_vectors_and_groundtruth(tmp_path):
    vectors = np.array([[0, 128, 255], [17, 2, 9]], dtype="u1")
    ids = np.array([[0, 99_999_999], [42, 123]], dtype="<u4")
    write_matrix(tmp_path / "base.bin", vectors)
    write_matrix(tmp_path / "groundtruth.bin", ids)
    actual = load_benchmark_vectors(tmp_path / "base.bin")
    truth = load_benchmark_groundtruth(tmp_path / "groundtruth.bin")
    assert actual.dtype == torch.float32 and truth.dtype == torch.int64
    torch.testing.assert_close(actual, torch.tensor(vectors.astype(np.float32)))
    torch.testing.assert_close(truth, torch.tensor(ids.astype(np.int64)))
    assert (tmp_path / "base.bin").read_bytes()[8:] == vectors.tobytes()


@pytest.mark.parametrize("payload", [b"", b"1234", np.array([2, 3], dtype="<u4").tobytes(), np.array([0, 3], dtype="<u4").tobytes()])
def test_malformed_binary_rejected(tmp_path, payload):
    path = tmp_path / "base.bin"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        load_benchmark_vectors(path)


def test_sift100m_runners_reuse_sift10m_defaults(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "bench"))
    flat = importlib.import_module("run_ivfflat_figure")
    pq = importlib.import_module("run_ivfpq_figure")
    folder = tmp_path / "sift100M"
    folder.mkdir()
    for name in ("base.bin", "query.bin", "groundtruth.bin"):
        (folder / name).touch()
    args = SimpleNamespace(dataset="SIFT100M", dataset_dir=None, nlist=None,
        nprobes=None, q_batches=[32, 128, 512], topk=10, nq=1024, warmup=2,
        gpu_id=0, pq_dim=None, pq_bits=8)
    for runner in (flat, pq):
        monkeypatch.setattr(runner, "DATA_ROOT", tmp_path)
        monkeypatch.setattr(runner, "select_gpu", lambda requested: 0)
        config = runner.make_config(args)
        assert config.dataset == "sift100m" and config.dataset_dir == folder
        assert config.base_path == folder / "base.bin"
        assert config.query_path == folder / "query.bin"
        assert config.groundtruth_path == folder / "groundtruth.bin"
        assert config.nlist == 1024 and config.q_batches == (32, 128, 512)
        assert config.nprobes == pq.benchmark.DATASET_PROFILES["sift10m"]["nprobes"]
    assert pq.benchmark.DATASET_PROFILES["sift100m"] == pq.benchmark.DATASET_PROFILES["sift10m"]
    assert config.pq_dim == 16 and config.pq_bits == 8
