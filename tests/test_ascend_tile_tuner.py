"""Exercise sampled/full diagnostic comparisons without an NPU."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorvs import IVFPQBuildResult, save_ivfpq_index


@pytest.mark.parametrize("reference_limit", [0, 3])
def test_tuner_keeps_holdout_and_sampled_baseline_comparisons_honest(tmp_path, monkeypatch, reference_limit):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "bench"))
    tuner = importlib.import_module("tune_ascend_tensor_tiles")
    generator = torch.Generator().manual_seed(709)
    vectors, centroids = torch.randn(97, 8, generator=generator), torch.randn(4, 8, generator=generator)
    labels = torch.randint(0, 4, (97,), generator=generator)
    books = torch.randn(4, 16, 2, generator=generator)
    codes = torch.randint(0, 16, (97, 4), dtype=torch.uint8, generator=generator)
    source = save_ivfpq_index(IVFPQBuildResult(vectors, labels, centroids, books, codes,
        torch.device("cpu"), 4, 4), tmp_path / "index.bin")
    original_bytes = source.read_bytes()
    queries = torch.randn(16, 8, generator=generator)
    decoded = centroids[labels] + torch.stack([books[m, codes[:, m].long()] for m in range(4)], 1).flatten(1)
    truth = (queries[:, None] - decoded).square().sum(-1).topk(10, largest=False).indices
    monkeypatch.setattr(tuner, "index_path_for", lambda *args: source)
    monkeypatch.setattr(tuner, "load_dataset", lambda path: queries)
    monkeypatch.setattr(tuner, "load_ivecs", lambda path: truth)
    monkeypatch.setattr(tuner, "configure_npu", lambda device: torch.device("cpu"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda *args: None,
        get_device_name=lambda device: "CPU test"), raising=False)
    monkeypatch.setattr(tuner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="No running processes"))
    # Equal measured latencies force the test to detect accidentally selecting
    # the smaller sampled baseline as the best full-batch configuration.
    clock = iter(range(10000))
    monkeypatch.setattr(tuner.time, "perf_counter", lambda: next(clock))
    output = tmp_path / "tuning.json"
    monkeypatch.setattr("sys.argv", ["tuner", "ivfpq", "--q-batch", "8", "--query-offset", "4",
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
    if reference_limit:
        assert report["speedup_over_default"] is None and not report["speedup_comparable"]
    else:
        assert report["speedup_over_default"] == 1 and report["speedup_comparable"]
