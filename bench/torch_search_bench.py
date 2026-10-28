"""Replay a reference CSV with repeated, warmed Torch or Triton searches.

Flat defaults to fixed list-grouped scanning. Measurements retain every
sample and report median elapsed time; optional retesting extends noisy cases.

Example:
    conda run -n tensorvs python bench/torch_search_bench.py \
        --reference result/IVFFLAT_glove1.2m_nlist1024.csv --device cuda:1
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

import torch

from tensorvs import (
    compute_recall_at_k, load_dataset, load_ivecs, load_ivfflat_index,
    load_ivfpq_index, search_ivfflat_index, search_ivfpq_index,
)

try:
    from ._benchmark_utils import _make_padded_batch, probe_visit_statistics, process_gpu_memory_bytes
except ImportError:
    from _benchmark_utils import _make_padded_batch, probe_visit_statistics, process_gpu_memory_bytes


FIELDS = ("nprobes", "q_batch", "K", "qps", "recall",
          "visited_vectors_per_query", "peak_gpu_memory")


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src" / "tensorvs").glob("*.py")) + [Path(__file__).resolve()]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def measure_repeated(*, batches, query_count, groundtruth, device, search_one,
                     topk, nprobe, visits, warmup, warmup_seconds, repetitions):
    """Warm every query batch, then report a median of synchronized sweeps."""
    def sweep(retain=False):
        elapsed, saved = 0.0, []
        for batch, valid in batches:
            started = time.perf_counter()
            values, ids = search_one(batch, nprobe)
            torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            if retain:
                # Triton workspace/graph outputs are overwritten by the next
                # search. Copy each batch outside timing before reusing them.
                saved.append(ids[:valid].cpu())
            del values, ids
        return elapsed, saved

    rounds, started = 0, time.perf_counter()
    while rounds < warmup or time.perf_counter() - started < warmup_seconds:
        sweep()
        rounds += 1
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    baseline_process = process_gpu_memory_bytes(device)
    samples, peak_process = [], baseline_process
    for iteration in range(repetitions):
        elapsed, saved = sweep(retain=iteration == repetitions - 1)
        samples.append(elapsed)
        peak_process = max(peak_process, process_gpu_memory_bytes(device))
    recall = float(compute_recall_at_k(torch.cat(saved).cpu(), groundtruth, topk))
    elapsed = statistics.median(samples)
    qps_samples = [query_count / value for value in samples]
    return dict(nprobe=nprobe, qps=query_count / elapsed, qp=query_count / elapsed,
                recall=recall, search_seconds=elapsed, samples_seconds=samples,
                minimum_qps=min(qps_samples), maximum_qps=max(qps_samples),
                relative_sample_spread=(max(samples) - min(samples)) / elapsed,
                warmup_full_rounds=rounds,
                baseline_gpu_memory_allocated_mb=baseline_allocated / (1 << 20),
                baseline_gpu_memory_reserved_mb=baseline_reserved / (1 << 20),
                baseline_gpu_memory_process_mb=baseline_process / (1 << 20),
                peak_gpu_memory_allocated_mb=torch.cuda.max_memory_allocated(device) / (1 << 20),
                peak_gpu_memory_reserved_mb=torch.cuda.max_memory_reserved(device) / (1 << 20),
                peak_gpu_memory_driver_mb=peak_process / (1 << 20),
                peak_gpu_memory_process_mb=peak_process / (1 << 20), **visits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/storage/vector_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("result"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--q-batch", type=int, default=128)
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--flat-scan", choices=("auto", "grouped", "dense"), default="grouped")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retest-unstable", action="store_true",
                        help="Add another measurement set to completed cases with >20%% sample spread")
    args = parser.parse_args()
    if (min(args.q_batch, args.nq, args.warmup, args.repetitions) <= 0
            or not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0):
        parser.error("q-batch, nq, warmup, repetitions must be positive; warmup-seconds must be nonnegative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("This comparison requires a CUDA device")

    reference_hash = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    with args.reference.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"Unexpected reference header: {reader.fieldnames}")
        # Preserve the reference order and use only the requested batch size.
        reference_rows = {
            (int(row["K"]), int(row["nprobes"])): row
            for row in reader if int(row["q_batch"]) == args.q_batch
        }
    if not reference_rows:
        raise ValueError(f"No q_batch={args.q_batch} rows in {args.reference}")
    match = re.fullmatch(
        r"(IVFFLAT|IVFPQ)_(.+)_nlist(\d+)(?:_M(\d+)_nbits(\d+)_train(\d+))?\.csv",
        args.reference.name,
    )
    if match is None:
        raise ValueError(f"Unrecognized reference filename: {args.reference.name}")
    algorithm, dataset, nlist, m, nbits, _ = match.groups()
    root = args.data_root / dataset
    index_name = (f"TenVS_IVFFlat_nlist{nlist}.bin" if algorithm == "IVFFLAT" else
                  f"TenVS_IVFPQResidual_nlist{nlist}_m{m}_nbits{nbits}_train256.bin")
    index_path = root / "GPUIndex" / index_name
    if not index_path.exists():
        raise FileNotFoundError(f"Existing index required: {index_path}")
    name = ("torch_" + args.reference.name.replace("IVFFLAT_", "IVFFlat_", 1)
            if args.backend == "torch" else args.reference.name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / name
    json_path = csv_path.with_suffix(".json")
    if csv_path.resolve() == args.reference.resolve():
        raise ValueError("Measure into a staging directory before replacing the reference CSV")
    sources = source_hashes()
    # Serialize writers and make resume safe across queues on different GPUs.
    lock_file = csv_path.with_suffix(".lock").open("a")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    completed = {}
    retiming_previous = {}
    if csv_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume to continue existing results: {csv_path}")
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != list(FIELDS):
                raise ValueError(f"Unexpected output header in {csv_path}")
            for row in reader:
                key = (int(row["K"]), int(row["nprobes"]))
                if key not in reference_rows or key in completed or int(row["q_batch"]) != args.q_batch:
                    raise ValueError(f"Unexpected or duplicate output row: {row}")
                if not all(math.isfinite(float(row[field])) for field in FIELDS):
                    raise ValueError(f"Incomplete output row: {row}")
                completed[key] = row
        previous = json.loads(json_path.read_text())
        needs_measurement = (args.retest_unstable or len(completed) != len(reference_rows)
                             or previous["status"] != "complete")
        if (previous.get("backend") != args.backend or previous.get("source_sha256") != sources
                or previous["reference_sha256"] != reference_hash
                or previous["use_compile"] or previous["nq_limit"] != args.nq
                or previous.get("repetitions") != args.repetitions
                or previous.get("warmup_seconds") != args.warmup_seconds
                or previous.get("warmup_iterations") != args.warmup
                or previous.get("flat_scan") != (args.flat_scan if algorithm == "IVFFLAT" and args.backend == "torch" else None)
                or (needs_measurement and (previous.get("device") != str(device)
                    or previous.get("cuda_visible_devices") != os.environ.get("CUDA_VISIBLE_DEVICES")))):
            raise ValueError("Resume metadata does not match the requested experiment")
        if args.retest_unstable:
            retiming_previous = {(r["K"], r["nprobe"]): r for r in previous["results"]
                                if r["relative_sample_spread"] > .2
                                and not r.get("additional_measurement_sets", 0)}
            for key in retiming_previous:
                completed.pop(key, None)
        if len(completed) == len(reference_rows) and previous["status"] == "complete":
            print(f"Already complete: {csv_path}", flush=True)
            return

    load, search = ((load_ivfflat_index, search_ivfflat_index) if algorithm == "IVFFLAT"
                    else (load_ivfpq_index, search_ivfpq_index))
    started = time.perf_counter()
    print(f"Loading {index_path} ({len(reference_rows)} configurations)", flush=True)
    index = load(index_path, load_lists=True)
    if int(index.centroids.shape[0]) != int(nlist):
        raise ValueError("Index nlist does not match the reference")
    if algorithm == "IVFPQ" and (int(index.metadata["m"]) != int(m)
                                 or int(index.metadata["nbits"]) != int(nbits)):
        raise ValueError("Index PQ configuration does not match the reference")
    queries = load_dataset(root / f"{dataset}_query.fvecs")
    groundtruth = load_ivecs(root / f"{dataset}_groundtruth.ivecs")
    if queries.shape[0] != groundtruth.shape[0]:
        raise ValueError("Query and ground-truth counts differ")
    nq = min(args.nq, queries.shape[0])
    queries, groundtruth = queries[:nq], groundtruth[:nq]
    if groundtruth.shape[1] < max(k for k, _ in reference_rows):
        raise ValueError("Ground truth does not contain all requested K values")
    data_bytes = (int(index.centroids.shape[1]) * 4 if algorithm == "IVFFLAT" else
                  int(m) * index.persisted_list_codes_interleaved_by_subspace.element_size())
    with torch.inference_mode():
        visits = {
            probe: probe_visit_statistics(index, queries, probe, device, data_bytes)
            for probe in dict.fromkeys(probe for _, probe in reference_rows)
        }
        # Matching list counts also checks that the same index/query subset is used.
        for (_, probe), row in reference_rows.items():
            if not math.isclose(float(row["visited_vectors_per_query"]),
                                visits[probe]["visited_vectors_per_query"], rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f"Reference list visits differ for {dataset}, nprobe={probe}: "
                                 f"reference={row['visited_vectors_per_query']}, "
                                 f"current={visits[probe]['visited_vectors_per_query']}")
        index.reside_on_device(device, backend=args.backend)
        batches = [_make_padded_batch(queries, start, args.q_batch, device)
                   for start in range(0, nq, args.q_batch)]
        torch.cuda.synchronize(device)
        metadata = {
            "status": "running", "backend": args.backend, "use_compile": False,
            "source_sha256": sources, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "reference": str(args.reference.resolve()), "reference_sha256": reference_hash,
            "output": str(csv_path.resolve()), "index": str(index_path),
            "index_size_bytes": index_path.stat().st_size,
            "index_mtime_ns": index_path.stat().st_mtime_ns, "index_metadata": index.metadata,
            "dataset": dataset, "algorithm": algorithm, "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(device), "torch": str(torch.__version__),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "process_id": os.getpid(),
            "cuda": torch.version.cuda, "query_count": nq, "nq_limit": args.nq,
            "q_batch": args.q_batch, "warmup_iterations": args.warmup,
            "warmup_seconds": args.warmup_seconds, "repetitions": args.repetitions,
            "flat_scan": args.flat_scan if algorithm == "IVFFLAT" and args.backend == "torch" else None,
            "lut_dtype": "fp16" if algorithm == "IVFPQ" else None,
            "refine_factor": 1, "torch_candidate_chunk_size": 2048,
            "torch_memory_budget_mb": 128,
            "timing": "median of repeated full-query sweeps; sum of synchronized per-batch wall clock; "
                      "every batch warmed; resident queries; tail padded to q_batch; real query count for QPS",
            "memory_metric": "NVML process resident MiB sampled after warmup and each full sweep",
            "expected_rows": len(reference_rows), "completed_rows": len(completed),
            "results": [],
        }
        measured_results = {(r["K"], r["nprobe"]): r for r in previous["results"]
                            if (r["K"], r["nprobe"]) in completed} if completed else {}
        metadata["results"] = [measured_results[key] for key in reference_rows if key in measured_results]
        json_path.write_text(json.dumps(metadata, indent=2) + "\n")
        kwargs = dict(device=device, backend=args.backend, use_compile=False,
                      torch_query_batch_size=args.q_batch)
        if algorithm == "IVFPQ":
            kwargs["lut_dtype"] = "fp16"
            if args.backend == "triton":
                kwargs.update(use_packed_16b_codes=True, use_preallocated_workspace=True,
                              use_cuda_graph=True)
        elif args.backend == "torch":
            kwargs["torch_flat_scan"] = args.flat_scan
        metadata["search_options"] = {key: str(value) if isinstance(value, torch.device) else value
                                      for key, value in kwargs.items()}
        metadata["triton_autotune"] = args.backend == "triton" and algorithm == "IVFPQ"
        tuned = set()
        def search_one(batch, current_probe, current_k):
            options = dict(kwargs)
            if args.backend == "triton" and algorithm == "IVFPQ":
                anchors = (5, 50, 150, 300, 500, 1000)
                anchor = min(anchors, key=lambda value: abs(value - current_probe))
                key = current_k, anchor
                options.update(triton_autotune=key not in tuned,
                               triton_autotune_cache_nprobe=anchor)
                output = search(index, batch, topk=current_k, nprobe=current_probe, **options)
                tuned.add(key)
                return output
            return search(index, batch, topk=current_k, nprobe=current_probe, **options)
        verification_profile = {}
        search(index, batches[0][0], topk=min(k for k, _ in reference_rows),
               nprobe=max(probe for _, probe in reference_rows), profile=verification_profile, **kwargs)
        if args.backend == "torch" and algorithm == "IVFFLAT" and args.flat_scan == "grouped":
            if "dense_vector_distance_topk" in verification_profile:
                raise RuntimeError("Grouped benchmark unexpectedly selected dense scanning")
        metadata["verified_scan_profile"] = verification_profile
        def save_csv():
            temporary = csv_path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(completed[key] for key in reference_rows if key in completed)
            temporary.replace(csv_path)

        for (k, probe), reference_row in reference_rows.items():
            if (k, probe) in completed:
                continue
            result = measure_repeated(
                batches=batches, query_count=nq, groundtruth=groundtruth, nprobe=probe, topk=k,
                warmup=args.warmup, warmup_seconds=args.warmup_seconds,
                repetitions=args.repetitions, device=device,
                search_one=lambda batch, current_probe, current_k=k: search_one(
                    batch, current_probe, current_k),
                visits=visits[probe],
            )
            if (k, probe) in retiming_previous:
                old = retiming_previous[k, probe]
                samples = old["samples_seconds"] + result["samples_seconds"]
                elapsed = statistics.median(samples)
                result.update(samples_seconds=samples, search_seconds=elapsed,
                              qps=nq / elapsed, qp=nq / elapsed,
                              minimum_qps=nq / max(samples), maximum_qps=nq / min(samples),
                              relative_sample_spread=(max(samples) - min(samples)) / elapsed,
                              additional_measurement_sets=old.get("additional_measurement_sets", 0) + 1,
                              previous_measurement=old)
                for field in ("peak_gpu_memory_driver_mb", "peak_gpu_memory_process_mb",
                              "peak_gpu_memory_allocated_mb", "peak_gpu_memory_reserved_mb"):
                    result[field] = max(old[field], result[field])
            row = dict(nprobes=probe, q_batch=args.q_batch, K=k, qps=result["qps"],
                       recall=result["recall"],
                       visited_vectors_per_query=result["visited_vectors_per_query"],
                       peak_gpu_memory=result["peak_gpu_memory_driver_mb"])
            completed[k, probe] = row
            save_csv()
            result.update(K=k, reference_qps=float(reference_row["qps"]),
                          reference_recall=float(reference_row["recall"]))
            measured_results[k, probe] = result
            metadata["results"] = [measured_results[key] for key in reference_rows if key in measured_results]
            metadata["completed_rows"] = len(completed)
            json_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"{args.backend} {dataset} {algorithm} {len(completed)}/{len(reference_rows)} "
                  f"K={k} nprobe={probe} QPS={result['qps']:.2f} "
                  f"recall={result['recall']:.6f} "
                  f"memory={row['peak_gpu_memory']:.1f} MiB", flush=True)
    if hashlib.sha256(args.reference.read_bytes()).hexdigest() != reference_hash:
        raise RuntimeError("The reference CSV changed during the run")
    if source_hashes() != sources:
        raise RuntimeError("The implementation changed during the run")
    metadata.update(status="complete", wall_seconds=time.perf_counter() - started)
    json_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(completed)} rows: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
