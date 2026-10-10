"""Verify matching IVF-PQ configurations and summarize the second optimization."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NEW = RESULTS / "ivfpq_torch_round2"
STEM = "IVFPQ_sift_nlist1024_M16_nbits8_train256"
FIELDS = ["nprobes", "q_batch", "K", "qps", "recall", "visited_vectors_per_query", "peak_gpu_memory"]
PROBES = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]


def read(path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == FIELDS
        rows = list(reader)
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["status"] == "complete"
    assert len(rows) == 14
    assert [int(row["nprobes"]) for row in rows] == PROBES
    assert all(math.isfinite(float(v)) for row in rows for v in row.values())
    assert all(float(r["qps"]) > 0 and 0 <= float(r["recall"]) <= 1 for r in rows)
    return rows, meta


report = dict(status="passed",K=10,query_count=1024,handwritten_gpu_kernels=False,
              cpu_operator_fallback=False,precision="FP16 LUT, FP32 accumulation",batches=[])
for batch in (256, 512):
    name = f"torch_mps_{STEM}.csv"
    old, old_meta = read(RESULTS / f"qbatch{batch}" / name)
    previous, previous_meta = read(RESULTS / "ivfpq_torch_optimization" / f"qbatch{batch}" / name)
    cpu, cpu_meta = read(RESULTS / f"qbatch{batch}" / f"faiss_metal_cpu_{STEM}.csv")
    entry = dict(q_batch=batch,methods=[])
    for compiled in (False, True):
        folder = NEW / "compiled" if compiled else NEW
        path = folder / f"qbatch{batch}" / name
        rows, meta = read(path)
        assert meta["backend"] == "torch" and meta["device"] == "mps"
        assert meta["use_compile"] is compiled and meta["cpu_operator_fallback"] is False
        assert meta["search_options"]["lut_dtype"] == "fp16"
        for key in ("index", "index_size_bytes", "index_mtime_ns", "query_count", "q_batch",
                    "topks", "nprobes", "warmup_iterations", "warmup_scope", "repetitions"):
            assert old_meta[key] == meta[key] == previous_meta[key]
        source = Path(meta["index"])
        assert (source.stat().st_size,source.stat().st_mtime_ns) == (meta["index_size_bytes"],meta["index_mtime_ns"])
        for source, digest in meta["source_sha256"].items():
            assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
        assert all(int(r["q_batch"]) == batch and int(r["K"]) == 10 for r in rows)
        assert all(a["visited_vectors_per_query"] == b["visited_vectors_per_query"] for a,b in zip(old,rows))
        differences = [abs(float(a["recall"])-float(b["recall"])) for a,b in zip(old,rows)]
        assert max(differences) <= .001
        original_gains = [float(a["qps"])/float(b["qps"]) for a,b in zip(rows,old)]
        previous_gains = [float(a["qps"])/float(b["qps"]) for a,b in zip(rows,previous)]
        summary = dict(mode="compiled" if compiled else "eager",csv=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            max_recall_difference=max(differences),visited_vectors_identical=True,
            median_speedup_vs_original=statistics.median(original_gains),
            median_speedup_vs_previous=statistics.median(previous_gains),
            median_cpu_over_torch_qps=statistics.median(float(a["qps"])/float(b["qps"]) for a,b in zip(cpu,rows)),
            examples=[dict(nprobes=int(r["nprobes"]),recall=float(r["recall"]),
                           previous_qps=float(p["qps"]),new_qps=float(r["qps"]),faiss_cpu_qps=float(c["qps"]))
                      for r,p,c in zip(rows,previous,cpu) if int(r["nprobes"]) in (5,50,150)])
        entry["methods"].append(summary)
    report["batches"].append(entry)

report["paired_validation"] = json.loads((NEW / "paired_comparison.json").read_text())
assert all(r["neighbor_set_overlap"] == 1 and r["recalls"]["baseline"] == r["recalls"]["current"]
           for r in report["paired_validation"]["records"])
report["portability"] = "Standard Torch operators; CPU and MPS tested. CUDA unavailable on this machine. Compilation is optional and backend-dependent."
report["timing_limit"] = "Separate benchmark runs may differ thermally. Alternating paired eager measurements are included; compilation excluded by warmup."
(NEW / "validation_summary.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
