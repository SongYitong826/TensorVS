"""Validate portable Torch SIFT results against preserved benchmark tables."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[1]
results = root / "results"
original_path = results / "original_torch_archive/torch_mps_IVFFLAT_sift_nlist1024.csv"
if not original_path.exists():
    original_path = results / "torch_mps_IVFFLAT_sift_nlist1024.csv"
paths = [original_path] + [results / name for name in (
    "torch_mps_IVFFLAT_sift_nlist1024_tensor_native.csv",
    "faiss_metal_IVFFLAT_sift_nlist1024.csv")]
fields = ["nprobes", "q_batch", "K", "qps", "recall", "visited_vectors_per_query", "peak_gpu_memory"]
tables = []
for path in paths:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == fields
        tables.append(list(reader))
original, native, faiss = tables
key = lambda row: tuple(int(row[name]) for name in ("nprobes", "q_batch", "K"))
assert len(original) == len(native) == len(faiss) == 84
assert [key(row) for row in original] == [key(row) for row in native] == [key(row) for row in faiss]
assert len({key(row) for row in native}) == 84
metadata = json.loads(paths[1].with_suffix(".json").read_text())
old_metadata = json.loads(paths[0].with_suffix(".json").read_text())
faiss_metadata = json.loads(paths[2].with_suffix(".json").read_text())
assert metadata["status"] == "complete" and metadata["cpu_operator_fallback"] is False
assert metadata["backend"] == "torch" and metadata["use_compile"] is False
for field in ("query_count", "warmup_iterations", "repetitions"):
    assert metadata[field] == old_metadata[field]
assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == faiss_metadata["reference_sha256"]
source = Path(metadata["index"])
index_stat = source.stat()
assert (index_stat.st_size, index_stat.st_mtime_ns) == (old_metadata["index_size_bytes"], old_metadata["index_mtime_ns"])
assert all(math.isfinite(float(value)) for row in native for value in row.values())
assert all(float(row["qps"]) > 0 and float(row["peak_gpu_memory"]) > 0 and 0 <= float(row["recall"]) <= 1 for row in native)
recall_delta = [abs(float(a["recall"]) - float(b["recall"])) for a, b in zip(original, native)]
assert max(recall_delta) < 0.002
assert all(float(a["visited_vectors_per_query"]) == float(b["visited_vectors_per_query"]) for a, b in zip(original, native))
assert not (root / "src/tensorvs/_ivfflat_mps.py").exists()
assert not (root / "src/tensorvs/_layout_cache.py").read_text().count("metal_vectors")
for path in (root / "src/tensorvs/_torch_search.py", root / "src/tensorvs/index_ivfflat.py"):
    text = path.read_text()
    assert "compile_shader" not in text and "_ivfflat_mps" not in text
gains = [float(new["qps"]) / float(old["qps"]) for old, new in zip(original, native)]
test_summary = (results / "tensor_native_tests.log").read_text().strip().splitlines()[-1]
assert "failed" not in test_summary and "passed" in test_summary
report = dict(
    validation="passed", method="TensorVS TensorNative / eager PyTorch", backend="torch", device="MPS / Apple M5",
    handwritten_gpu_kernels=False, use_compile=False, cpu_operator_fallback=False,
    configurations=84, query_count=metadata["query_count"], q_batch=128,
    warmup=metadata["warmup_iterations"], repetitions=metadata["repetitions"],
    configurations_match=True, visited_vectors_match=True,
    reference_unchanged=True, reference_csv=str(original_path), index_unchanged=True,
    max_recall_difference=max(recall_delta),
    median_speedup=statistics.median(gains), minimum_speedup=min(gains), maximum_speedup=max(gains),
    faster_configurations=sum(gain > 1 for gain in gains),
    benchmark_wall_seconds=metadata["wall_seconds"],
    implementation="Flattened index_select row reads; compact eager query groups; cached batch query norms; selected-ID reads after block Top-K; bounded Torch scatter/Top-K merges",
    resident_layout="Existing contiguous logical lists; no extra device-specific vector copy",
    numerical_policy="FP32 distances and coarse selection, preserving norm/addmm formula; near-tie ordering may vary with matrix shapes",
    portability="The scan is the same standard-operator implementation on CPU, MPS and supported Torch accelerator backends. CPU/MPS were tested here; CUDA-specific tests require unavailable hardware.",
    comparison_limit=faiss_metadata["timing_comparison_limit"],
    test_result=test_summary,
    cpu_compilation_limit="Four CPU torch.compile tests are excluded because the existing host compiler SDK configuration fails even standalone torch.compile(x + 1); the measured method uses use_compile=False.",
    timing_variability_note="The full-table CSV retains the original warmup and repetition protocol. For K=10/nprobes=10, the three sweeps vary substantially; a separate same-protocol rerun is retained below without replacing the full-table observation.",
    source_sha256=metadata.get("source_sha256") or metadata.get("tensor_native_validation", {}).get("source_sha256") or
                  {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (root / "src/tensorvs/_torch_search.py", root / "src/tensorvs/index_ivfflat.py")},
    artifact_sha256={str(path.relative_to(results)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    examples=[dict(K=int(new["K"]), nprobes=int(new["nprobes"]),
                   original_qps=float(old["qps"]), tensor_native_qps=float(new["qps"]),
                   faiss_qps=float(other["qps"]), speedup=float(new["qps"]) / float(old["qps"]),
                   original_recall=float(old["recall"]), tensor_native_recall=float(new["recall"]))
              for old, new, other in zip(original, native, faiss)
              if int(new["K"]) == 10 and int(new["nprobes"]) in (5, 10, 50, 150)],
)
timing_check = results / "tensor_native_timing_check/torch_mps_IVFFLAT_sift_nlist1024_tensor_native.json"
if timing_check.exists():
    check = json.loads(timing_check.read_text())
    assert check["status"] == "complete"
    report["timing_recheck"] = dict(metadata=str(timing_check), results=check["results"])
(results / "tensor_native_validation.json").write_text(json.dumps(report, indent=2) + "\n")
metadata["tensor_native_validation"] = report
temporary = paths[1].with_suffix(".json.tmp")
temporary.write_text(json.dumps(metadata, indent=2) + "\n")
temporary.replace(paths[1].with_suffix(".json"))
print(json.dumps(report, indent=2))
