"""Replace the QBatch=128 Triton/Torch matrix and its plots, without archives.

Run: conda run -n tensorvs python -m bench.remeasure_search_backends --gpus 1 2
Both backends for a dataset run sequentially on the same GPU. A fresh staging
directory prevents old measurements from being reused. --resume only resumes
the current source version. Publication requires the complete validated matrix.
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

from .torch_search_bench import FIELDS, source_hashes


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"Unexpected CSV header: {path}")
        return list(reader)


def row_keys(rows):
    return [(int(row["K"]), int(row["nprobes"]), int(row["q_batch"])) for row in rows]


def check_gpus(gpus):
    """Avoid publishing timings taken alongside another compute workload."""
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"
    ], text=True)
    devices = {row[0].strip(): (row[1].strip(), int(row[2]))
               for row in csv.reader(query.splitlines())}
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits"
    ], text=True)
    busy = {row[0].strip() for row in csv.reader(processes.splitlines()) if row}
    for gpu in gpus:
        if gpu not in devices:
            raise RuntimeError(f"GPU {gpu} does not exist")
        uuid, free_mb = devices[gpu]
        if uuid in busy or free_mb < 12000:
            raise RuntimeError(f"GPU {gpu}: {free_mb} MiB free, other compute process={uuid in busy}; "
                               "a free GPU with at least 12 GB is required")


def validate(staging, jobs, repetitions):
    reports = []
    sources = source_hashes()
    for job in jobs:
        paired = {}
        for backend, name in (("triton", job["reference"].name), ("torch", job["torch_name"])):
            path = staging / name
            rows = read_rows(path)
            metadata = json.loads(path.with_suffix(".json").read_text())
            if row_keys(rows) != job["keys"] or len(set(row_keys(rows))) != len(rows):
                raise ValueError(f"Parameter matrix differs: {path}")
            if (metadata["status"] != "complete" or metadata["backend"] != backend
                    or metadata["source_sha256"] != sources or metadata["repetitions"] != repetitions
                    or metadata["q_batch"] != 128 or metadata["use_compile"]
                    or len(metadata["results"]) != len(rows)):
                raise ValueError(f"Incomplete or stale metadata: {path}")
            for row, measured in zip(rows, metadata["results"]):
                samples = measured["samples_seconds"]
                if (not all(math.isfinite(float(row[field])) for field in FIELDS)
                        or float(row["qps"]) <= 0 or not 0 <= float(row["recall"]) <= 1
                        or len(samples) < repetitions or any(not math.isfinite(t) or t <= 0 for t in samples)
                        or (measured["K"], measured["nprobe"]) != (int(row["K"]), int(row["nprobes"]))
                        or not math.isclose(float(row["qps"]), metadata["query_count"] / statistics.median(samples),
                                            rel_tol=1e-12)):
                    raise ValueError(f"Invalid measured row: {path}: {row}")
            paired[backend] = rows, metadata
        triton, tm = paired["triton"]
        torch, pm = paired["torch"]
        for field in ("index", "index_size_bytes", "index_mtime_ns", "query_count", "gpu",
                      "cuda_visible_devices", "torch", "cuda", "lut_dtype", "timing",
                      "torch_num_threads", "torch_num_interop_threads"):
            if tm[field] != pm[field]:
                raise ValueError(f"Unmatched comparison setting {field}: {job['reference']}")
        error = max(abs(float(t["recall"]) - float(p["recall"])) for t, p in zip(triton, torch))
        if error > 2e-4:
            raise ValueError(f"Backend recall difference exceeds tolerance: {error}: {job['reference']}")
        if not all(math.isclose(float(t["visited_vectors_per_query"]),
                                float(p["visited_vectors_per_query"]), rel_tol=1e-5, abs_tol=1e-5)
                   for t, p in zip(triton, torch)):
            raise ValueError(f"Backend candidate counts differ: {job['reference']}")
        reports.append(dict(dataset=tm["dataset"], algorithm=tm["algorithm"], rows=len(triton),
                            max_recall_difference=error, physical_gpu=tm["cuda_visible_devices"],
                            sample_spread_over_20_percent={backend: sum(
                                m["relative_sample_spread"] > .2 for m in paired[backend][1]["results"])
                                for backend in paired}))
    return reports


def remove_previous_copies(results, figures):
    """Delete only the previous copies belonging to this comparison."""
    for directory in (results / "archive", figures / "archive"):
        if directory.exists():
            for old in directory.glob("torch_before_remeasurement_*"):
                if old.is_dir():
                    shutil.rmtree(old)
            if not any(directory.iterdir()):
                directory.rmdir()
    for name in ("remeasured_qbatch128", "remeasured_qbatch128_refined"):
        old = results / name
        if old.is_dir():
            shutil.rmtree(old)
    (results / "torch_search_validation.json").unlink(missing_ok=True)


def torch_speedup(staging, jobs):
    """Keep aggregate speed ratios, without retaining previous result rows."""
    groups, all_ratios = [], []
    for job in jobs:
        old = [row for row in read_rows(ROOT / "result" / job["torch_name"])
               if int(row["q_batch"]) == 128]
        current = read_rows(staging / job["torch_name"])
        if row_keys(old) != row_keys(current):
            raise ValueError(f"Previous Torch parameter matrix differs: {job['torch_name']}")
        ratios = [float(new["qps"]) / float(previous["qps"]) for previous, new in zip(old, current)]
        metadata = json.loads((staging / job["torch_name"]).with_suffix(".json").read_text())
        old_metadata = json.loads((ROOT / "result" / job["torch_name"]).with_suffix(".json").read_text())
        differences = {field: dict(previous=old_metadata.get(field), current=metadata.get(field))
                       for field in ("index", "index_size_bytes", "index_mtime_ns", "query_count",
                                     "q_batch", "flat_scan", "lut_dtype", "torch", "gpu", "repetitions")
                       if old_metadata.get(field) != metadata.get(field)}
        groups.append(dict(dataset=metadata["dataset"], algorithm=metadata["algorithm"], cases=len(ratios),
                           median_qps_speedup=statistics.median(ratios),
                           geometric_mean_qps_speedup=statistics.geometric_mean(ratios),
                           minimum_qps_speedup=min(ratios), maximum_qps_speedup=max(ratios),
                           cases_faster_by_over_5_percent=sum(r > 1.05 for r in ratios),
                           cases_slower_by_over_5_percent=sum(r < 1 / 1.05 for r in ratios),
                           previous_torch_version=old_metadata["torch"],
                           previous_gpu=old_metadata["gpu"], current_gpu=metadata["gpu"],
                           measurement_setting_differences=differences))
        all_ratios.extend(ratios)
    return dict(comparison="Current Torch QPS divided by the previous published Torch QPS at identical parameters",
                limitation="Historical measurements; CPU load and GPU clocks may differ. Ratios alone do not isolate code changes.",
                cases=len(all_ratios), median_qps_speedup=statistics.median(all_ratios),
                geometric_mean_qps_speedup=statistics.geometric_mean(all_ratios), results=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["1", "2"], help="Free physical GPU indices")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=Path("/storage/vector_data"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 2 or len(set(args.gpus)) != len(args.gpus):
        parser.error("Use at least two repetitions and distinct GPU indices")
    results, figures = ROOT / "result", ROOT / "plot" / "Figures"
    staging = results / ".search_backends_current"
    jobs = []
    for path in sorted(results.glob("torch_*.csv")):
        reference = results / path.name.removeprefix("torch_").replace("IVFFlat_", "IVFFLAT_", 1)
        rows = [row for row in read_rows(reference) if int(row["q_batch"]) == 128]
        jobs.append(dict(reference=reference, torch_name=path.name, keys=row_keys(rows),
                         sha256=hashlib.sha256(reference.read_bytes()).hexdigest()))
    if len(jobs) != 12 or any(not job["keys"] for job in jobs):
        raise ValueError("Expected 12 existing paired result files at QBatch=128")
    print(f"Matrix: {len(jobs)} pairs, {sum(len(j['keys']) for j in jobs)} cases per backend", flush=True)
    check_gpus(args.gpus)
    if staging.exists() and not args.resume:
        raise FileExistsError(f"Current run already exists: {staging}; use --resume or remove it")
    staging.mkdir(parents=True, exist_ok=True)

    def worker(gpu, pending):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        for job in pending:
            check_gpus([gpu])
            for backend in ("triton", "torch"):
                name = job["reference"].stem if backend == "triton" else Path(job["torch_name"]).stem
                command = [sys.executable, "-u", str(ROOT / "bench" / "torch_search_bench.py"),
                           "--backend", backend, "--reference", str(job["reference"]),
                           "--device", "cuda:0", "--data-root", str(args.data_root),
                           "--output-dir", str(staging), "--flat-scan", "grouped",
                           "--repetitions", str(args.repetitions), "--resume"]
                log = staging / f"{name}.log"
                print(f"GPU {gpu}: {backend} {name}; log={log}", flush=True)
                with log.open("a") as file:
                    process = subprocess.run(command, cwd=ROOT, env=env, stdout=file, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f"Measurement failed: {log}")
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu, jobs[i::len(args.gpus)]) for i, gpu in enumerate(args.gpus)]
        for future in futures:
            future.result()
    reports = validate(staging, jobs, args.repetitions)
    speedup = torch_speedup(staging, jobs)
    # Validate rendering before replacing any published data.
    subprocess.run([sys.executable, str(ROOT / "plot" / "search_backends_qps_recall.py"),
                    "--result-dir", str(staging), "--output-dir", str(staging / "figures")],
                   cwd=ROOT, check=True)
    for job in jobs:
        if hashlib.sha256(job["reference"].read_bytes()).hexdigest() != job["sha256"]:
            raise RuntimeError(f"Reference changed during measurements: {job['reference']}")
    for job in jobs:
        for name in (job["reference"].name, job["torch_name"]):
            current, target = staging / name, results / name
            shutil.copy2(current, target)
            metadata = json.loads(current.with_suffix(".json").read_text())
            metadata["output"] = str(target)
            metadata["reference_sha256"] = hashlib.sha256((results / job["reference"].name).read_bytes()).hexdigest()
            metadata["reference_role"] = "paired current Triton measurements; parameter matrix replayed from prior run"
            target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "plot" / "search_backends_qps_recall.py")], cwd=ROOT, check=True)
    remove_previous_copies(results, figures)
    report = dict(status="passed", measured_at_utc=datetime.now(timezone.utc).isoformat(),
                  backends=["triton", "torch"], q_batch=128, files=2 * len(jobs),
                  rows=2 * sum(r["rows"] for r in reports), repetitions=args.repetitions,
                  source_sha256=source_hashes(), results=reports, previous_results_retained=False)
    (results / "search_backends_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (results / "torch_update_speedup.json").write_text(json.dumps(speedup, indent=2) + "\n")
    shutil.rmtree(staging)
    print(f"Published {report['rows']} fresh rows; regenerated figures; removed previous copies.", flush=True)


if __name__ == "__main__":
    main()
