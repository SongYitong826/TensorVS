"""Warm, paired SIFT comparisons against a saved TensorNative implementation.

Run sequentially with other GPU benchmarks. This tests six representative
configurations; the normal MPS benchmark writes the complete CSV matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import types

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
from tensorvs import compute_recall_at_k, load_dataset, load_ivecs, load_ivfflat_index
from tensorvs._torch_search import search_torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/Users/yitongsong/Dataset/sift"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_source = args.baseline_source.read_text()
    baseline = types.ModuleType("tensorvs._baseline_experiment")
    baseline.__package__ = "tensorvs"
    exec(compile(baseline_source, str(args.baseline_source), "exec"), baseline.__dict__)
    index = load_ivfflat_index(args.dataset / "GPUIndex/TenVS_IVFFlat_nlist1024.bin")
    index.reside_on_device("mps", backend="torch", persist_layout=False)
    queries = load_dataset(args.dataset / "sift_query.fvecs")[:1024].to("mps")
    truth = load_ivecs(args.dataset / "sift_groundtruth.ivecs")[:1024]
    batches = list(queries.split(128))
    current_source = Path(__file__).resolve().parents[1] / "src/tensorvs/_torch_search.py"
    report = dict(status="running", device="mps", backend="torch", cpu_operator_fallback=False,
                  query_count=1024, q_batch=128, warmup_full_sweeps=2, repetitions=3,
                  measurement="alternating before/after order; synchronized per-batch wall time",
                  baseline_source=str(args.baseline_source.resolve()),
                  baseline_sha256=hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
                  current_sha256=hashlib.sha256(current_source.read_bytes()).hexdigest(), records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for k, probe in ((10, 5), (10, 50), (10, 150), (100, 5), (100, 50), (100, 150)):
            def sweep(search):
                elapsed, values, ids = 0.0, [], []
                for batch in batches:
                    torch.mps.synchronize()
                    start = time.perf_counter()
                    v, i = search(index, batch, kind="flat", topk=k, nprobe=probe,
                                  device="mps", query_batch_size=128, candidate_chunk_size=2048,
                                  memory_budget_mb=128, use_compile=False, persist_layout=False,
                                  flat_scan="grouped")
                    torch.mps.synchronize()
                    elapsed += time.perf_counter() - start
                    values.append(v)
                    ids.append(i)
                return elapsed, torch.cat(values).cpu(), torch.cat(ids).cpu()
            functions = {"before": baseline.search_torch, "after": search_torch}
            for _ in range(2):
                for fn in functions.values():
                    sweep(fn)
            samples = {name: [] for name in functions}
            results = {}
            for repetition in range(3):
                order = ("before", "after") if repetition % 2 == 0 else ("after", "before")
                for name in order:
                    elapsed, values, ids = sweep(functions[name])
                    samples[name].append(elapsed)
                    results[name] = (values, ids)
            before, after = results["before"], results["after"]
            assert torch.isfinite(after[0]).all() and (after[1] >= 0).all()
            assert all(len(set(row)) == k for row in after[1].tolist())
            before_recall = float(compute_recall_at_k(before[1], truth, k))
            after_recall = float(compute_recall_at_k(after[1], truth, k))
            assert abs(after_recall - before_recall) < .002
            torch.testing.assert_close(after[0], before[0], atol=.125, rtol=2e-5)
            before_qps = 1024 / statistics.median(samples["before"])
            after_qps = 1024 / statistics.median(samples["after"])
            row = dict(K=k, nprobes=probe, before_qps=before_qps, after_qps=after_qps,
                       speedup=after_qps / before_qps, before_recall=before_recall,
                       after_recall=after_recall, samples_seconds=samples,
                       max_distance_difference=float((after[0] - before[0]).abs().max()))
            report["records"].append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "passed"
    report["median_speedup"] = statistics.median(r["speedup"] for r in report["records"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
