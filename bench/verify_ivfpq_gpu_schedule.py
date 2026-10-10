"""Validate retained latest GPU-scheduled IVF-PQ results without old tables."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NEW = RESULTS / "ivfpq_gpu_schedule"
SUFFIX = "IVFPQ_sift_nlist1024_M16_nbits8_train256"
FIELDS = ["nprobes", "q_batch", "K", "qps", "recall", "visited_vectors_per_query", "peak_gpu_memory"]
KEYS = ("index", "index_size_bytes", "index_mtime_ns", "query_count", "q_batch", "topks", "nprobes",
        "warmup_iterations", "warmup_scope", "repetitions")


def read(path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == FIELDS
        rows = list(reader)
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["status"] == "complete" and len(rows) == 14
    assert all(math.isfinite(float(v)) for row in rows for v in row.values())
    assert all(float(r["qps"]) > 0 and float(r["peak_gpu_memory"]) > 0 and 0 <= float(r["recall"]) <= 1 for r in rows)
    assert len(meta["results"]) == len(rows)
    for row, measured in zip(rows, meta["results"]):
        assert all(float(row[key]) == float(measured[key]) for key in FIELDS)
    return rows, meta


prior_path = NEW / "validation_summary.json"
prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
tests = (NEW / "tests_tiled.log").read_text().strip().splitlines()[-1]
assert re.search(r"\b\d+ passed\b", tests) and "failed" not in tests
report = dict(status="passed", K=10, query_count=1024, cpu_operator_fallback=False,
    handwritten_gpu_kernels=False,
    use_compile=True, python_submits_operations=True, tests=tests, batches=[])
if "historical_validation_before_cleanup" in prior:
    report["historical_validation_before_cleanup"] = prior["historical_validation_before_cleanup"]
report["coverage"] = ["decoded-vector oracles on CPU and MPS", "short, empty, uniform and skewed lists",
    "large K and small workspaces", "unsigned 16-bit PQ codes", "eager and compiled query readback guards",
    "word ADC with divisible and odd subspace counts", "int64 address fallback beyond 2**31 rows",
    "region-min Top-K: clustered/spread candidates, ties, odd tails and all-invalid rows",
    "region-min full-search decoded oracles and device readback guards on non-SIFT indexes",
    "bucket planning before LUT generation; no full-table reorder; original query identity and tail batches",
    "FP16/FP32 conversion and OPQ rotation before ADC on CPU and real MPS compilation"]
adc_tests = NEW / "adc_unsigned_boundary_tests.log"
if adc_tests.exists():
    extra = adc_tests.read_text().strip().splitlines()[-1]
    assert "4 passed" in extra and "failed" not in extra
    report["additional_adc_boundary_tests"] = extra
lut_order_comparison = NEW / "lut_order_comparison.json"
if lut_order_comparison.exists():
    comparison = json.loads(lut_order_comparison.read_text())
    assert comparison["status"] == "complete"
    assert len(comparison["records"]) == 6
    assert comparison["index_and_faiss_reference_unchanged"] is True
    assert len(comparison["full_curve_validation"]) == 2
    assert all(r["bit_exact_distances"] and r["only_boundary_tie_swaps"]
               and r["neighbor_ids_match"] and r["recall"]["previous"] == r["recall"]["current"] for r in comparison["records"])
    assert comparison["faiss_threads"] == 4
    for source, digest in comparison["source_sha256"].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
    report["lut_order_comparison"] = dict(path=str(lut_order_comparison),
        paired_configurations=len(comparison["records"]), bit_exact_distances=True,
        neighbor_ids_match=True, recall_unchanged=True, faiss_threads=4,
        minimum_speedup=min(r["speedup"] for r in comparison["records"]),
        maximum_speedup=max(r["speedup"] for r in comparison["records"]))
report["cpu_compilation_limit"] = "Four existing CPU torch.compile tests were deselected due to the host SDK toolchain; CPU tensor expressions and real MPS compilation were tested. CUDA hardware is unavailable."
guard_log = NEW / "count_readback_guard.log"
if guard_log.exists():
    guard = guard_log.read_text().strip().splitlines()[-1]
    assert "2 passed" in guard and "failed" not in guard
    report["additional_count_readback_guards"] = guard
for batch in (256, 512):
    path = NEW / f"qbatch{batch}" / f"torch_mps_{SUFFIX}.csv"
    rows, meta = read(path)
    cpu_path = RESULTS / f"qbatch{batch}" / f"faiss_metal_cpu_{SUFFIX}.csv"
    cpu_rows, cpu = read(cpu_path)
    assert meta["backend"] == "torch" and meta["device"] == "mps" and meta["use_compile"] is True
    scan = meta["search_options"]["torch_pq_scan"]
    assert scan in {"gpu", "bucketed"}
    uses_host_lengths = scan == "bucketed"
    assert meta["query_dependent_cpu_scheduling"] is uses_host_lengths and meta["query_tensor_host_readback"] is uses_host_lengths
    assert meta["cpu_operator_fallback"] is False
    if lut_order_comparison.exists():
        validated = next(r for r in comparison["full_curve_validation"] if r["q_batch"] == batch)
        assert validated["points"] == 14 and validated["recall_unchanged"] and validated["visits_unchanged"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == validated["csv_sha256"]
        assert meta["adc_access"]["pq_codes_per_word"] == 4
        assert meta["adc_access"]["packed_code_storage_aliases_logical_codes"] is True
        assert meta["adc_access"]["fp32_subspace_accumulation_order_unchanged"] is True
        assert meta["topk_selection"]["exact_distances"] is True
        assert meta["topk_selection"]["handwritten_gpu_kernels"] is False
        assert meta["lut_preparation"]["full_lut_reordering"] is False
        assert meta["lut_preparation"]["fp16_layout_and_cast_in_one_copy"] is True
        assert hashlib.sha256(Path(meta["index"]).read_bytes()).hexdigest() == comparison["source_index_sha256"]
    assert cpu["backend"] == "faiss_metal_cpu" and cpu["query_execution_device"] == "cpu"
    assert cpu["gpu_query_execution"] is False
    assert cpu["index_conversion"]["source"] == dict(path=meta["index"],
        size_bytes=meta["index_size_bytes"], mtime_ns=meta["index_mtime_ns"])
    provenance = cpu["reference_provenance"]
    assert provenance["verified_before_cleanup"] is True and provenance["csv_sha256"] == cpu["reference_sha256"]
    assert all(provenance["configuration"][key] == meta[key] for key in KEYS)
    source = Path(meta["index"])
    assert (source.stat().st_size, source.stat().st_mtime_ns) == (meta["index_size_bytes"], meta["index_mtime_ns"])
    for source, digest in meta["source_sha256"].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
    assert [int(r["nprobes"]) for r in rows] == [int(r["nprobes"]) for r in cpu_rows] == meta["nprobes"]
    assert all(int(r["q_batch"]) == batch and int(r["K"]) == 10 for r in rows + cpu_rows)
    report["batches"].append(dict(q_batch=batch, csv=str(path),
        torch_pq_scan=scan, query_dependent_cpu_scheduling=uses_host_lengths,
        recall_and_visits_unchanged_from_previous_curve=True if lut_order_comparison.exists() else None,
        query_tensor_host_readback=uses_host_lengths,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), faiss_csv=str(cpu_path),
        faiss_sha256=hashlib.sha256(cpu_path.read_bytes()).hexdigest(), current_csv_matches_recorded_measurements=True))
report["comparison_limit"] = "Faiss Metal IVF-PQ queries execute on CPU with FP32 LUT; Torch queries execute on MPS with FP16 LUT and FP32 accumulation. Historical tables were deleted at user request; correctness comparisons were validated before cleanup."
prior_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
