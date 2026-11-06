"""CPU-only checks for the Ascend-host Faiss IVF-Flat baseline."""
import csv
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from bench import ascend_faiss_cpu_ivfflat_bench as bench


def write_vectors(path, values, integer=False):
    records = np.empty((len(values), values.shape[1] + 1), dtype="int32")
    records[:, 0] = values.shape[1]
    records[:, 1:] = values if integer else np.asarray(values, dtype="float32").view("int32")
    records.tofile(path)


def test_native_cpu_build_metrics_search_accounting_and_resume(tmp_path, capfd):
    faiss.omp_set_num_threads(1)
    folder = tmp_path / "data/glove1.2m"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(17)
    base = rng.normal(size=(1024, 8)).astype("float32")
    queries = rng.normal(size=(13, 8)).astype("float32")
    exact = faiss.IndexFlatL2(8)
    exact.add(base)
    _, truth = exact.search(queries, 10)
    for suffix, array, integer in (("base.fvecs", base, False), ("query.fvecs", queries, False),
                                    ("groundtruth.ivecs", truth, True)):
        write_vectors(folder / ("glove1.2m_" + suffix), array, integer)
    args = bench.parse_args(["--dataset", "glove1.2m", "--data-root", str(tmp_path / "data"),
        "--output-dir", str(tmp_path / "result"), "--index-dir", str(tmp_path / "indexes"),
        "--nlist", "8", "--coarse-niter", "2", "--cpu-threads", "1", "--nq", "13",
        "--topk", "5", "--q-batches", "4", "32", "--nprobes", "1", "8", "--warmup", "0",
        "--repetitions", "1", "--add-batch-size", "256"])
    bench.run(args)
    progress = capfd.readouterr().out
    assert "[1/4] Loading" in progress and "[2/4] Training complete" in progress
    assert "Iteration 0" in progress and "Iteration 1" in progress
    assert "[3/4] Added 1,024/1,024 (100.0%)" in progress and "ETA~" in progress
    assert "[4/4] Saved" in progress
    stem = "Ascend_Faiss_IVFFLAT_glove1.2m_nlist8"
    csv_path, json_path = args.output_dir / (stem + ".csv"), args.output_dir / (stem + ".json")
    with csv_path.open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(bench.FIELDS)
        rows = list(reader)
    assert len(rows) == 4
    for row in rows:
        assert float(row["peak_gpu_memory"]) == 0
        if int(row["nprobes"]) == 8:
            assert float(row["recall"]) == 1
            assert float(row["visited_vectors_per_query"]) == len(base)
    meta = json.loads(json_path.read_text())
    assert meta["device"] == "cpu" and meta["backend"] == "faiss_cpu" and meta["status"] == "complete"
    assert all(r["peak_rss_memory_mib"] > 0 for r in meta["results"])
    assert meta["results"][-1]["visit_statistics"]["visited_vectors_total"] == len(base) * len(queries)
    build = meta["build"]
    assert build["build_time_seconds"] >= build["train_seconds"] + build["add_seconds"]
    assert build["index_size_bytes"] == Path(build["index_path"]).stat().st_size
    assert (args.output_dir / "Build" / (stem + "_build.csv")).exists()
    assert (args.output_dir / "Build" / (stem + "_build.json")).exists()
    index_mtime = Path(build["index_path"]).stat().st_mtime_ns
    # Resume a partial search without retraining or changing original build times.
    bench.write_csv(csv_path, rows[:-1])
    meta["results"] = meta["results"][:-1]
    bench.write_json(json_path, meta)
    args.resume = True
    bench.run(args)
    assert "Iteration" not in capfd.readouterr().out
    resumed = json.loads(json_path.read_text())
    assert resumed["status"] == "complete" and resumed["completed_rows"] == 4
    assert resumed["build"] == build and resumed["index_loaded_from_cache"]
    assert Path(build["index_path"]).stat().st_mtime_ns == index_mtime
    assert resumed["results"][:-1] == meta["results"]
    # Progress-only updates must reuse old native indexes and preserve their
    # builder provenance, output-resume configuration and original build time.
    manifest_path = Path(build["index_path"]).with_suffix(".build.json")
    manifest = json.loads(manifest_path.read_text())
    previous_builder = next(iter(bench.PROGRESS_COMPATIBLE_BUILDERS))
    manifest["signature"]["builder_sha256"] = previous_builder
    resumed["config"]["build_signature"]["builder_sha256"] = previous_builder
    bench.write_json(manifest_path, manifest)
    bench.write_json(json_path, resumed)
    bench.run(args)
    compatible = json.loads(json_path.read_text())
    assert compatible["build"] == build
    assert compatible["config"]["build_signature"]["builder_sha256"] == previous_builder
    assert Path(build["index_path"]).stat().st_mtime_ns == index_mtime
    unknown = dict(manifest, signature=dict(manifest["signature"], builder_sha256="0" * 64))
    bench.write_json(manifest_path, unknown)
    with pytest.raises(ValueError, match="Cached CPU index/settings differ"):
        bench.run(args)
    bench.write_json(manifest_path, manifest)
    args.resume = False
    args.append = True
    args.q_batches = [4, 16]
    args.nprobes = [1, 4]
    bench.run(args)
    appended = json.loads(json_path.read_text())
    assert appended["status"] == "complete" and appended["completed_rows"] == 7
    assert appended["results"][:4] == compatible["results"]
    with csv_path.open() as stream:
        assert len(list(csv.DictReader(stream))) == 7
    bench.run(args)
    assert json.loads(json_path.read_text())["completed_rows"] == 7
    with pytest.raises(ValueError, match="Cached CPU index/settings differ"):
        args.cpu_threads = 2
        bench.run(args)
    faiss.omp_set_num_threads(1)


def test_visit_accounting_matches_native_thread_slices_and_excludes_padding():
    sizes = []
    class Quantizer:
        def search(self, x, nprobe):
            sizes.append(len(x))
            # Simulate coarse assignments changing at the BLAS threshold.
            ids = np.full((len(x), nprobe), int(len(x) >= 20), dtype=np.int64)
            return None, ids
    api = SimpleNamespace(omp_get_max_threads=lambda: 24)
    index = SimpleNamespace(quantizer=Quantizer(), nprobe=1, parallel_mode=0)
    batch = np.zeros((32, 8), dtype=np.float32)
    visits = bench.account_visits(api, index, np.array([3, 7]), batch, 13, 96)
    assert sizes == [32 * (i + 1) // 24 - 32 * i // 24 for i in range(24)]
    assert visits.tolist() == [3] * 13
    sizes.clear()
    index.parallel_mode = 3
    visits = bench.account_visits(api, index, np.array([3, 7]), batch, 13, 224)
    assert sizes == [32] and visits.tolist() == [7] * 13


def test_build_timing_excludes_preparation_and_serialization(tmp_path, monkeypatch):
    clock, calls = [0.], []
    def advance(seconds):
        clock[0] += seconds
    class IndexIVFFlat:
        def __init__(self, quantizer, dim, nlist, metric):
            advance(1)
            self.d, self.nlist, self.metric_type = dim, nlist, metric
            self.ntotal, self.is_trained, self.cp = 0, False, SimpleNamespace()
            self.invlists = SimpleNamespace(list_size=lambda i: self.ntotal // nlist)
        def train(self, x):
            calls.append("train")
            assert self.cp.verbose
            advance(10)
            self.is_trained = True
        def add(self, x):
            calls.append("add")
            assert not self.cp.verbose
            advance(2)
            self.ntotal += len(x)
    def write(index, path):
        advance(3)
        Path(path).write_text("serialized index")
    def read_vectors(path):
        advance(5)
        return np.zeros((1024, 8), dtype="float32")
    native = SimpleNamespace(IndexIVFFlat=IndexIVFFlat, IndexFlatL2=lambda d: None,
                             METRIC_L2=1, __version__="test", write_index=write)
    args = bench.parse_args(["--dataset", "glove1.2m", "--index-dir", str(tmp_path / "index"),
                             "--nlist", "8", "--cpu-threads", "1", "--add-batch-size", "512"])
    source = tmp_path / "raw.fvecs"
    source.write_bytes(b"signature")
    monkeypatch.setattr(bench, "read_vectors", read_vectors)
    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock[0])
    _, counts, build, loaded, _ = bench.build_or_load(native, args, source, "index", {})
    assert not loaded and counts.sum() == 1024 and calls == ["train", "add", "add"]
    assert build["build_time_seconds"] == 15
    assert build["train_seconds"] == 10 and build["add_seconds"] == 4
    assert build["preparation_seconds"] == 5 and build["serialize_seconds"] == 3


def test_schema_and_probe_grids_match_tensor_benchmarks():
    from bench.amd_tensor_bench import FIELDS, NPROBES
    assert bench.FIELDS == FIELDS and bench.NPROBES == NPROBES


@pytest.mark.parametrize("dataset", ["glove1.2m", "hunyuan"])
def test_launcher_selects_one_cpu_method_and_dataset(tmp_path, dataset):
    (tmp_path / "env.sh").write_text("")
    conda = tmp_path / "conda"
    conda.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    conda.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "bench/run_ascend_faiss_cpu_ivfflat_bench.sh"),
        "ivfflat", dataset, "--cpu-threads", "4", "--resume"], check=True, capture_output=True, text=True,
        env=dict(os.environ, FAISS_ASCEND_RUNTIME=str(tmp_path), PATH=f"{tmp_path}:{os.environ['PATH']}"))
    argv = result.stdout.splitlines()
    assert argv[argv.index("--dataset") + 1] == dataset and argv.count("--dataset") == 1
    assert "--require-ascend-package" in argv and argv[-1] == "--resume"


def test_bad_native_visit_count_is_rejected():
    quantizer = SimpleNamespace(search=lambda q, k: (None, np.zeros((len(q), k), dtype="int64")))
    index = SimpleNamespace(quantizer=quantizer, nprobe=1, parallel_mode=3)
    with pytest.raises(RuntimeError, match="ndis"):
        bench.account_visits(None, index, np.array([10]), np.zeros((4, 8), dtype="float32"), 3, 999)
