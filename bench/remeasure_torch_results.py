"""Remeasure the existing QBatch=128 matrix and regenerate comparison figures.

Run: conda run -n tensorvs python -m bench.remeasure_torch_results
Each GPU runs one benchmark process at a time. Existing results and figures
are archived before publishing the complete, validated replacement matrix.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from .torch_search_bench import FIELDS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-gpu", default="2")
    parser.add_argument("--large-gpu", default="1")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--staging-dir", type=Path,
                        help="Use a fresh directory for a new experiment; an existing directory resumes it")
    args = parser.parse_args()
    if args.small_gpu == args.large_gpu:
        parser.error("small-gpu and large-gpu must differ to avoid concurrent measurements on one GPU")
    root = Path(__file__).resolve().parents[1]
    results = root / "result"
    staging = args.staging_dir or results / "remeasured_qbatch128"
    staging = staging.resolve()
    staging.mkdir(parents=True, exist_ok=True)
    jobs = []
    for old_path in sorted(results.glob("torch_*.csv")):
        old_metadata = json.loads(old_path.with_suffix(".json").read_text())
        reference = Path(old_metadata["reference"])
        jobs.append(dict(reference=reference, output=old_path, dataset=old_metadata["dataset"],
                         algorithm=old_metadata["algorithm"], reference_sha256=digest(reference)))
    if len(jobs) != 12:
        raise ValueError(f"Expected 12 existing Torch result files, got {len(jobs)}")
    small_jobs = [j for j in jobs if j["algorithm"] == "IVFPQ" or j["dataset"] in
                  {"glove1.2m", "deep1m", "sift1m"}]
    large_jobs = [j for j in jobs if j not in small_jobs]

    def worker(gpu, pending, retest=False):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        for job in pending:
            log = staging / (job["output"].stem + ".log")
            command = [sys.executable, "-u", str(root / "bench" / "torch_search_bench.py"),
                       "--reference", str(job["reference"]), "--device", "cuda:0",
                       "--output-dir", str(staging), "--flat-scan", "grouped",
                       "--warmup", "2", "--warmup-seconds", "0.5",
                       "--repetitions", str(args.repetitions), "--resume"]
            if retest:
                command.append("--retest-unstable")
            phase = "retesting unstable cases in" if retest else "starting"
            print(f"GPU {gpu}: {phase} {job['output'].name}; log={log}", flush=True)
            with log.open("a") as file:
                process = subprocess.run(command, cwd=root, env=env, stdout=file, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f"Benchmark failed, see {log}")
            print(f"GPU {gpu}: completed {job['output'].name}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, args.small_gpu, small_jobs),
                   pool.submit(worker, args.large_gpu, large_jobs)]
        for future in futures:
            future.result()

    # Keep every raw sample when extending unstable measurements; choose no
    # points based on throughput, recall, or the desired curve shape.
    retest_small = [j for j in jobs if json.loads((staging / j["output"].with_suffix(".json").name)
                                                .read_text())["cuda_visible_devices"] == args.small_gpu]
    retest_large = [j for j in jobs if j not in retest_small]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, args.small_gpu, retest_small, True),
                   pool.submit(worker, args.large_gpu, retest_large, True)]
        for future in futures:
            future.result()

    reports = []
    for job in jobs:
        current = staging / job["output"].name
        with current.open() as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == list(FIELDS)
            rows = list(reader)
        with job["reference"].open() as file:
            reference = [row for row in csv.DictReader(file) if int(row["q_batch"]) == 128]
        keys = lambda values: [(int(x["K"]), int(x["nprobes"]), int(x["q_batch"])) for x in values]
        assert keys(rows) == keys(reference) and len(set(keys(rows))) == len(rows)
        assert digest(job["reference"]) == job["reference_sha256"]
        assert all(all(math.isfinite(float(row[field])) for field in FIELDS) for row in rows)
        assert all(float(row["qps"]) > 0 and 0 <= float(row["recall"]) <= 1 for row in rows)
        recall_error = max(abs(float(x["recall"]) - float(y["recall"])) for x, y in zip(rows, reference))
        assert recall_error <= 2e-4
        assert all(math.isclose(float(x["visited_vectors_per_query"]), float(y["visited_vectors_per_query"]),
                                rel_tol=1e-5, abs_tol=1e-5) for x, y in zip(rows, reference))
        metadata = json.loads(current.with_suffix(".json").read_text())
        assert metadata["status"] == "complete" and len(metadata["results"]) == len(rows)
        assert metadata["repetitions"] == args.repetitions and metadata["q_batch"] == 128
        assert metadata["flat_scan"] == ("grouped" if job["algorithm"] == "IVFFLAT" else None)
        for row, measured in zip(rows, metadata["results"]):
            assert len(measured["samples_seconds"]) >= args.repetitions
            expected = metadata["query_count"] / statistics.median(measured["samples_seconds"])
            assert math.isclose(float(row["qps"]), expected, rel_tol=1e-12)
        reports.append(dict(file=str(job["output"]), rows=len(rows), reference=str(job["reference"]),
                            max_recall_difference=recall_error, parameter_order_matches=True,
                            reference_unchanged=True, physical_gpu=metadata["cuda_visible_devices"],
                            flat_scan=metadata["flat_scan"], median_repetitions=args.repetitions,
                            extended_measurements=sum(len(x["samples_seconds"]) > args.repetitions
                                                      for x in metadata["results"]),
                            sample_spread_over_20_percent=sum(x["relative_sample_spread"] > .2
                                                             for x in metadata["results"])))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = results / "archive" / ("torch_before_remeasurement_" + stamp)
    backup.mkdir(parents=True)
    figure_dir = root / "plot" / "Figures"
    figure_backup = figure_dir / "archive" / backup.name
    figure_backup.mkdir(parents=True)
    for job in jobs:
        for path in (job["output"], job["output"].with_suffix(".json")):
            shutil.copy2(path, backup / path.name)
    validation = results / "torch_search_validation.json"
    if validation.exists():
        shutil.copy2(validation, backup / validation.name)
    for path in figure_dir.glob("*Triton_vs_Torch*QBatch128*"):
        if path.is_file():
            shutil.copy2(path, figure_backup / path.name)
    for job in jobs:
        current = staging / job["output"].name
        shutil.copy2(current, job["output"])
        metadata = json.loads(current.with_suffix(".json").read_text())
        metadata["measurement_output"] = metadata["output"]
        metadata["output"] = str(job["output"])
        job["output"].with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = dict(status="passed", backend="torch", use_compile=False, q_batch=128,
                  files=len(jobs), rows=sum(x["rows"] for x in reports),
                  repetitions=args.repetitions, flat_scan="grouped", original_csvs_unchanged=True,
                  previous_torch_results=str(backup), previous_figures=str(figure_backup), results=reports)
    previous_validation = json.loads((backup / validation.name).read_text()) if (backup / validation.name).exists() else {}
    history = previous_validation.get("result_history", [])
    if previous_validation.get("previous_torch_results") and previous_validation["previous_torch_results"] not in history:
        history.append(previous_validation["previous_torch_results"])
    report["result_history"] = history + [str(backup)]
    validation.write_text(json.dumps(report, indent=2) + "\n")
    subprocess.run([sys.executable, str(root / "plot" / "search_backends_qps_recall.py")],
                   cwd=root, check=True)
    print(f"Validated and published {report['rows']} rows in {len(jobs)} files; figures regenerated.", flush=True)


if __name__ == "__main__":
    main()
