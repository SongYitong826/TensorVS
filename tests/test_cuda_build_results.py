"""Keep CUDA build statistics precise, portable, and separate from query peaks."""
import csv
import json
from pathlib import Path

import pytest
import torch

from bench import _cuda_build_results as io
from bench.organize_cuda_build_results import match_statistics
from plot import ivf_resource_table


@pytest.mark.parametrize("unit", [2**20, 10**6])
def test_legacy_size_units_are_checked_against_actual_bytes(unit):
    size = 123456789
    row = {"Index Type": "IVF PQ", "Method Name": "cuVS",
           "Index Size (MB)": str(size / unit), "Build Time (s)": "12.3456789"}
    value, candidates = match_statistics([row], "ivfpq", "cuvs", size)
    assert value == 12.3456789
    assert candidates[0]["recorded_size_unit"] == ("MiB" if unit == 2**20 else "MB")
    other = dict(row, **{"Build Time (s)": "99.0"})
    value, candidates = match_statistics([row, other], "ivfpq", "cuvs", size)
    assert value is None and len(candidates) == 2


def test_all_single_device_build_pairs_use_the_existing_cuda_schema():
    reference = io.BUILD_DIR / "CUDA_FaissGPU_MultiGPU3_IVFFLAT_sift100m_nlist1024_build.csv"
    with reference.open(newline="") as file:
        assert tuple(next(csv.reader(file))) == io.BUILD_FIELDS
    paths = [p for p in io.BUILD_DIR.glob("CUDA_*_build.json") if "MultiGPU" not in p.name]
    assert len(paths) == 59
    assert len({json.loads(p.read_text())["dataset"] for p in paths}) == 8
    legacy_count = 0
    for path in paths:
        metadata = json.loads(path.read_text())
        token = "TensorVS_Triton" if metadata["method"] == "TensorVS" else "FaissGPU" if metadata["method"] == "faiss" else "cuVS"
        _, loaded = io.load_build(metadata["dataset"], token, metadata["index_type"], metadata["nlist"],
                                  m=metadata["M"], nbits=metadata["nbits"], index_name=Path(metadata["index_path"]).name)
        assert loaded["index_size_mib"] == loaded["index_size_bytes"] / 2**20
        assert loaded["peak_build_gpu_memory_mib"] is None
        assert loaded["save_seconds"] is None
        if loaded["source_index_metadata"]:
            assert loaded["build_seconds"] == loaded["source_index_metadata"]["build_time_seconds"]
        legacy_count += "legacy_table_statistics" in loaded
    assert legacy_count == 36


def test_future_tensorvs_builds_keep_exact_metadata_and_query_statistics(tmp_path):
    index = tmp_path / "TenVS_IVFPQResidual_nlist1024_m25_nbits8_train256.bin"
    metadata = dict(nlist=1024, m=25, nbits=8, num_vectors=100, dim=200,
                    device="cuda:6", pq_max_points_per_centroid=512, build_time_seconds=3.123456789)
    torch.save({"metadata": metadata}, index)
    query = Path("CUDA_TensorVS_Triton_IVFPQ_glove1.2m_nlist1024_M25_nbits8_train512.csv")
    path = io.save_tensorvs_build(index, "glove1.2m", "ivfpq", query_path=query, build_dir=tmp_path / "Build")
    assert path.name == query.stem + "_build.csv"
    record = json.loads(path.with_suffix(".json").read_text())
    record["legacy_table_statistics"] = {"peak_gpu_memory_gib": 0.61}
    path.with_suffix(".json").write_text(json.dumps(record))
    io.save_tensorvs_build(index, "glove1.2m", "ivfpq", query_path=query, build_dir=tmp_path / "Build")
    _, restored = io.load_build("glove1.2m", "TensorVS_Triton", "ivfpq", 1024, m=25, nbits=8, build_dir=tmp_path / "Build")
    assert restored["build_seconds"] == 3.123456789
    assert restored["build_device"] == "cuda:6"
    assert restored["index_size_bytes"] == index.stat().st_size
    assert restored["legacy_table_statistics"]["peak_gpu_memory_gib"] == 0.61
    assert restored["peak_build_gpu_memory_mib"] is None
    path.write_text(path.read_text().replace("3.123456789", "4.56789"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        io.load_build("glove1.2m", "TensorVS_Triton", "ivfpq", 1024, m=25, nbits=8, build_dir=tmp_path / "Build")


def test_table_uses_query_peak_not_build_peak():
    for dataset in ivf_resource_table.DATASETS:
        for kind in ("IVF-Flat", "IVF-PQ"):
            records = ivf_resource_table.collect_records(dataset, kind)
            for record in records.values():
                assert record.source_path.parent == io.BUILD_DIR
                metadata = json.loads(record.source_path.with_suffix(".json").read_text())
                assert record.peak_memory_gib == metadata["legacy_table_statistics"]["peak_gpu_memory_gib"]
