"""Apple result migration must retain measurements and their actual devices."""
import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from bench import _mps_results as io
from bench.organize_mps_results import normalize
from plot import mps_kernel_performance, qbatch_comparison, ivf_resource_table


@pytest.fixture
def comparison(tmp_path):
    index = tmp_path / "TenVS_IVFPQResidual_nlist1024_m32_nbits8_train256.bin"
    rows = [dict(dataset="glove1.2m", index_type="ivfpq", method=method,
                 execution_device=device, q_batch="128", K="10", nprobes="50",
                 qps="123.456", recall="0.40", nlist="1024", M="32", nbits="8",
                 index_path=str(index), original_extra_field="retained", sample7_seconds="1.25")
            for method, device in (("TensorVS", "mps"), ("Faiss Metal", "cpu"))]
    source = tmp_path / "source.csv"
    with source.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return source, rows


def test_split_preserves_all_fields_and_does_not_label_cpu_as_gpu(comparison, tmp_path):
    source, rows = comparison
    outputs = io.split_comparison(source, tmp_path / "result")
    recovered = [row for path in outputs for row in io.read_rows(path)]
    assert io.rows_sha256(recovered) == io.rows_sha256(rows)
    assert {p.name for p in outputs} == {
        "MPS_TensorVS_Tensor_IVFPQ_glove1.2m_nlist1024_M32_nbits8_train256.csv",
        "MPS_FaissMetal_IVFPQ_glove1.2m_nlist1024_M32_nbits8_train256.csv"}
    for path in outputs:
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert metadata["status"] == "complete" and metadata["completed_rows"] == 1
        assert metadata["csv_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline="") as file:
            assert tuple(next(csv.reader(file))) == io.QUERY_FIELDS
        assert metadata["records"] and "original_extra_field" in metadata["records"][0]
        if "FaissMetal" in path.name:
            assert metadata["config"]["execution_device"] == "cpu"
    assert source.exists()
    with pytest.raises(FileExistsError):
        io.split_comparison(source, tmp_path / "result")


def test_historical_runs_and_cpu_memory_are_not_read_as_current_curves(comparison, tmp_path):
    source, rows = comparison
    destination = tmp_path / "result"
    archives = io.split_comparison(source, destination, archive=True)
    memory = io.split_comparison(source, destination, cpu_memory=True)
    assert all(path.parent.name == "Archive" and "QBatch" not in path.name for path in archives)
    assert all(path.parent.name == "Build" for path in memory)
    with pytest.raises(FileNotFoundError):
        io.comparison_rows(destination, "glove1.2m", "ivfpq")
    io.split_comparison(source, destination)
    assert io.comparison_rows(destination, "glove1.2m", "ivfpq") == rows
    assert io.comparison_rows(destination, "glove1.2m", "ivfpq", archive=True) == rows


def test_csv_json_mismatch_is_rejected(comparison, tmp_path):
    source, _ = comparison
    path = io.split_comparison(source, tmp_path / "result")[0]
    path.write_text(path.read_text().replace("123.456", "999.999"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        io.read_rows(path)


def test_normalize_keeps_overlapping_runs_and_incomplete_results_separate(comparison, tmp_path):
    source, original = comparison
    root = tmp_path / "result"
    current = io.split_comparison(source, root)
    for path in current:
        historical = path.with_name(path.stem + "_QBatch128_256_512.csv")
        io.write_result(historical, io.read_rows(path))
    for path in io.split_comparison(source, root, cpu_memory=True):
        path.rename(root / path.name)
        path.with_suffix(".json").rename(root / path.with_suffix(".json").name)
    partial = root / "MPS_FaissMetal_IVFPQ_hunyuan_nlist1024_M384_nbits8_train256.csv"
    incomplete = [dict(original[1], dataset="hunyuan")]
    io.write_result(partial, incomplete, complete=False)
    normalize(root)
    assert io.comparison_rows(root, "glove1.2m", "ivfpq") == original
    assert io.comparison_rows(root, "glove1.2m", "ivfpq", archive=True) == original
    assert len(list((root / "Build").glob("*_cpu_memory.csv"))) == 2
    assert not list(root.rglob("*QBatch*"))
    archived_partial = root / "Archive" / "incomplete" / partial.name
    assert io.read_rows(archived_partial) == incomplete
    assert json.loads(archived_partial.with_suffix(".json").read_text())["status"] == "in_progress"
    assert not partial.exists()
    with pytest.raises(FileNotFoundError):
        io.comparison_rows(root, "hunyuan", "ivfpq")


def test_source_build_keeps_original_cuda_device(tmp_path):
    index = tmp_path / "TenVS_IVFPQResidual_nlist1024_m32_nbits8_train256.bin"
    metadata = dict(device="cuda:6", build_time_seconds=8.81, nlist=1024,
                    num_vectors=99, dim=200, m=32, nbits=8)
    torch.save({"metadata": metadata, "tensor": torch.empty(1)}, index)
    output = io.save_source_index(index, "glove1.2m", "ivfpq", tmp_path / "result")
    assert output.parent.name == "Build"
    record = json.loads(output.with_suffix(".json").read_text())
    assert record["build_device"] == "cuda:6" and record["build_seconds"] == 8.81
    assert record["status"] == "source_index_metadata"
    assert record["source_index_metadata"] == metadata


def test_interrupted_results_are_not_marked_complete(comparison, tmp_path):
    _, rows = comparison
    path = tmp_path / "run.csv"
    io.write_result(path, rows[:1], complete=False)
    assert json.loads(path.with_suffix(".json").read_text())["status"] == "in_progress"
    io.write_result(path, rows, complete=True)
    assert json.loads(path.with_suffix(".json").read_text())["completed_rows"] == 2


@pytest.mark.parametrize("dataset", ["glove1.2m", "hunyuan"])
@pytest.mark.parametrize("kind", ["ivfflat", "ivfpq"])
def test_current_mps_figure_and_memory_read_migrated_files(dataset, kind):
    curves, rows = mps_kernel_performance.read_curves(io.RESULT_DIR, dataset, kind)
    assert curves and {int(row["q_batch"]) for row in rows} == {32, 128, 512}
    memory = ivf_resource_table.collect_mps_memory(
        ivf_resource_table.Dataset(dataset, dataset), "IVF-Flat" if kind == "ivfflat" else "IVF-PQ")
    assert "TensorVS" in memory
    assert ("Faiss GPU" in memory) == (kind == "ivfflat")


@pytest.mark.parametrize("dataset", ["glove1.2m", "sift1m"])
@pytest.mark.parametrize("kind", ["ivfflat", "ivfpq"])
def test_historical_figure_reads_migrated_files(dataset, kind):
    rows, curves = qbatch_comparison.read_results(io.RESULT_DIR, dataset, kind)
    assert curves and {int(row["q_batch"]) for row in rows} == {128, 256, 512}


@pytest.mark.parametrize("device,metal_memory,valid", [
    ("cpu", "0", True), ("metal", "0", False), ("cpu", "-1", False),
])
def test_cpu_curve_allows_zero_metal_memory_only(monkeypatch, device, metal_memory, valid):
    rows = [dict(dataset="hunyuan", index_type="ivfpq", method=method,
                 execution_device=device if method == "Faiss Metal" else "mps",
                 q_batch=str(batch), K="10", nprobes="350", qps="25", recall="0.49",
                 peak_gpu_memory=metal_memory if method == "Faiss Metal" else "100",
                 peak_rss_memory="500")
            for method in mps_kernel_performance.METHODS
            for batch in mps_kernel_performance.QBATCHES]
    monkeypatch.setattr(mps_kernel_performance, "comparison_rows", lambda *args: rows)
    if valid:
        curves, _ = mps_kernel_performance.read_curves(Path("unused"), "hunyuan", "ivfpq")
        assert all(("Faiss Metal", batch) in curves for batch in (32, 128, 512))
    else:
        with pytest.raises(ValueError, match="peak-memory"):
            mps_kernel_performance.read_curves(Path("unused"), "hunyuan", "ivfpq")


def test_all_migrated_rows_match_original_hashes():
    manifest = json.loads((io.RESULT_DIR / "MPS_migration_manifest.json").read_text())
    for source in manifest["sources"]:
        rows = [row for path in source["outputs"] for row in io.read_rows(io.RESULT_DIR / path)]
        assert len(rows) == source["rows"]
        assert io.rows_sha256(rows) == source["rows_sha256"]
    assert manifest["source_rows"] == 739
    assert len(manifest["build_records"]) == 6
