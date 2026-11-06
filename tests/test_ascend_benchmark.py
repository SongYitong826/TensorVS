"""Memory accounting must use the requested process, never device-total HBM."""
import pytest
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess

import torch

from bench.ascend_tensor_bench import process_hbm_mib
from bench import ascend_tensor_bench
from bench.amd_tensor_bench import output_name


def test_process_hbm_uses_current_pid():
    output = """
| 5     910B4               | OK            | 93.2        37                0 / 0              |
| 0                        | 0000:41:00.0  | 0          0 / 0             2936 / 32768        |
| 5       0                 | 50917         | python                   | 110                     |
| 5       0                 | 50918         | python                   | 1024                    |
"""
    assert process_hbm_mib(output, 50917) == (5, 0, 110)


@pytest.mark.parametrize("output", ["No running processes found", "| 5 0 | 7 | python | 64 |"])
def test_missing_process_memory_is_an_error(output):
    with pytest.raises(RuntimeError, match="PID 50917"):
        process_hbm_mib(output, 50917)


@pytest.mark.parametrize("platform", ["AMD", "Ascend"])
@pytest.mark.parametrize("kind", ["ivfflat", "ivfpq"])
def test_result_filename_identifies_platform(platform, kind):
    index = SimpleNamespace(centroids=torch.empty(1024, 7), metadata={"m": 384, "nbits": 8})
    name = output_name("hunyuan", kind, index, "tensor", platform=platform)
    expected = ("TensorVS_Tensor_IVFFLAT_hunyuan_nlist1024" if kind == "ivfflat" else
                "TensorVS_Tensor_IVFPQ_hunyuan_nlist1024_M384_nbits8_train256")
    assert name == f"{platform}_{expected}.csv"


def test_default_output_directory_is_shared_result(monkeypatch):
    captured = []
    monkeypatch.setattr("sys.argv", ["ascend_tensor_bench.py", "--dataset", "hunyuan", "--kind", "ivfflat"])
    monkeypatch.setattr(ascend_tensor_bench, "benchmark", captured.append)
    ascend_tensor_bench.main()
    assert captured[0].output_dir == ascend_tensor_bench.PROJECT_ROOT / "result"


def test_grouped_execution_tiles_are_exposed_on_cli(monkeypatch):
    captured = []
    monkeypatch.setattr("sys.argv", ["ascend_tensor_bench.py", "--dataset", "glove1.2m", "--kind", "ivfpq",
        "--query-tile-size", "32", "--candidate-chunk-size", "2048", "--max-tasks-per-batch", "64"])
    monkeypatch.setattr(ascend_tensor_bench, "benchmark", captured.append)
    ascend_tensor_bench.main()
    assert captured[0].query_tile_size == 32 and captured[0].max_tasks_per_batch == 64
    assert captured[0].candidate_chunk_size == 2048


def test_npu_stage_profiling_uses_device_events(monkeypatch):
    from tensorvs._torch_search import _Timer
    synchronized, recorded = [], []
    class Event:
        def __init__(self, enable_timing):
            assert enable_timing
        def record(self, stream):
            recorded.append(stream)
        def elapsed_time(self, end):
            return 3.5  # Device events report milliseconds.
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=Event,
        current_stream=lambda device: "npu-stream", synchronize=synchronized.append), raising=False)
    profile, device = {}, SimpleNamespace(type="npu")
    timer = _Timer(device, profile)
    timer.end("scan", timer.begin())
    timer.finish()
    assert profile["scan"] == .0035 and recorded == ["npu-stream", "npu-stream"]
    assert synchronized == [device]


def test_hunyuan_preset_changes_only_execution_parameters(tmp_path):
    conda = tmp_path / "conda"
    conda.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    conda.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "bench/run_ascend_tensor_bench.sh"),
        "ivfpq", "hunyuan", "--resume"], capture_output=True, text=True, check=True,
        env=dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}"))
    argv = result.stdout.splitlines()
    assert argv[argv.index("--pq-lookup") + 1] == "gather"
    assert argv[argv.index("--memory-budget-mb") + 1] == "256"
    assert argv[argv.index("--query-tile-size") + 1] == "64"
    assert argv[argv.index("--lut-chunk-size") + 1] == "32"
    assert not set(("--nlist", "--M", "--nbits", "--nprobes", "--topk", "--lut-dtype")) & set(argv)
    assert argv[-1] == "--resume"
