"""Benchmark existing SIFT IVF indexes with Torch on Apple MPS.

Run with ``conda run -n tensorvs python bench/mps_search_bench.py``.
CSV memory is a sampled Metal-driver allocation peak in MiB, including caches.
Memory sampling runs separately from the synchronized QPS measurements.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

# Must be set before importing Torch: unsupported GPU operators should fail.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch

from tensorvs import (
    compute_recall_at_k, load_dataset, load_ivecs, load_ivfflat_index,
    load_ivfpq_index, search_ivfflat_index, search_ivfpq_index,
)

try:
    from ._benchmark_utils import probe_visit_statistics
except ImportError:
    from _benchmark_utils import probe_visit_statistics


FIELDS = ("nprobes", "q_batch", "K", "qps", "recall",
          "visited_vectors_per_query", "peak_gpu_memory")
SIFT_NPROBES = (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150)


class MemorySampler:
    """Sample process Metal allocations; MPS has no public peak counter."""

    def __init__(self):
        self.peak = 0
        self.samples = 0
        self.error = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        memory = torch.mps.driver_allocated_memory()
        with self.lock:
            self.peak = max(self.peak, memory)
            self.samples += 1

    def run(self):
        try:
            while not self.stop.wait(0.01):
                self.sample()
        except Exception as error:
            self.error = error

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()
        if self.error is not None:
            raise self.error


def save_csv(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def benchmark(args, kind, queries, groundtruth):
    flat = kind == "ivfflat"
    index_name = ("TenVS_IVFFlat_nlist1024.bin" if flat else
                  "TenVS_IVFPQResidual_nlist1024_m16_nbits8_train256.bin")
    index_path = args.dataset / "GPUIndex" / index_name
    stem = ("torch_mps_IVFFLAT_sift_nlist1024" if flat else
            "torch_mps_IVFPQ_sift_nlist1024_M16_nbits8_train256")
    if args.result_suffix:
        stem += "_" + args.result_suffix
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = csv_path.with_suffix(".json")
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(f"Output already exists: {csv_path}")
    load, search = ((load_ivfflat_index, search_ivfflat_index) if flat else
                    (load_ivfpq_index, search_ivfpq_index))
    q_batch = args.flat_batch if flat else args.pq_batch
    topks = args.flat_topks if flat else args.pq_topks
    if groundtruth.shape[1] < max(topks):
        raise ValueError("Insufficient ground-truth neighbors for requested K")
    print(f"Loading {kind}: {index_path}", flush=True)
    started = time.perf_counter()
    index = load(index_path, load_lists=True)
    if index.centroids.shape != (1024, queries.shape[1]):
        raise ValueError("Index does not match SIFT query dimensions/nlist")
    if not flat and (index.metadata["m"] != 16 or index.metadata["nbits"] != 8):
        raise ValueError("Unexpected PQ configuration")
    batches = [queries[start:start + q_batch].to("mps")
               for start in range(0, len(queries), q_batch)]
    kwargs = dict(device="mps", backend="torch", use_compile=args.use_compile,
                  persist_layout=False, torch_query_batch_size=q_batch,
                  torch_candidate_chunk_size=args.candidate_chunk_size, torch_memory_budget_mb=128)
    if flat:
        kwargs["torch_flat_scan"] = args.flat_scan
    else:
        kwargs["lut_dtype"] = "fp16"
        kwargs["torch_pq_scan"] = args.pq_scan
    index.reside_on_device("mps", backend="torch", persist_layout=False)
    torch.mps.synchronize()
    metadata = dict(
        status="running", algorithm=kind, backend="torch", device="mps",
        use_compile=args.use_compile, cpu_operator_fallback=False, torch=str(torch.__version__),
        implementation="TensorNative: standard PyTorch tensor operations, no custom GPU kernels",
        compiler="PyTorch Inductor" if args.use_compile else None,
        compilation_timing="excluded by full-query warmup sweeps" if args.use_compile else None,
        source_sha256={str(path.relative_to(Path(__file__).resolve().parents[1])):
                       hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__).resolve().parents[1] / "src/tensorvs/_torch_search.py",
                                    Path(__file__).resolve().parents[1] / "src/tensorvs/_layout_cache.py",
                                    Path(__file__).resolve().parents[1] / f"src/tensorvs/index_{'ivfflat' if flat else 'ivfpq'}.py",
                                    *(() if flat else (Path(__file__).resolve().parents[1] / "src/tensorvs/_torch_pq_bucketed.py",)))},
        dataset=str(args.dataset.resolve()), index=str(index_path.resolve()),
        index_size_bytes=index_path.stat().st_size,
        index_mtime_ns=index_path.stat().st_mtime_ns, index_metadata=index.metadata,
        query_count=len(queries), query_subset="first NQ queries in file order",
        q_batch=q_batch, topks=topks, nprobes=args.nprobes,
        warmup_iterations=args.warmup, warmup_scope=args.warmup_scope, repetitions=args.repetitions,
        search_options=kwargs, expected_rows=len(topks) * len(args.nprobes),
        timing="median of repeated full-query sweeps; sum of synchronized per-batch "
               "wall time; index and queries resident on MPS; excludes recall and memory sampling",
        recall="mean set overlap with ground-truth top-K divided by K",
        memory_unit="MiB", memory_metric="sampled process Metal-driver allocation peak "
               "including resident index, query batches, allocator caches and MPS framework allocations; "
               "10 ms background sampling and synchronized batch boundaries in a separate sweep; "
               "not an exact hardware peak",
        results=[],
    )
    if not flat and args.pq_scan in {"gpu", "bucketed"}:
        metadata.update(query_dependent_cpu_scheduling=args.pq_scan == "bucketed",query_tensor_host_readback=args.pq_scan == "bucketed",
                        python_submits_tensor_operations=True,
                        gpu_scan_bounds={str(p):index.layout_cache.torch_layout.pq_scan_upper_bounds[p]
                                         for p in args.nprobes},
                        gpu_scan_query_tiling="Metadata-adaptive fixed power-of-two query tiles; compiled scans choose whole-span or streaming Top-K, and anchor or binary positioning; coarse/LUT batch remains q_batch",
                        gpu_scan_adaptation=dict(minimum_positive_list_count=min(
                            c for c in index.layout_cache.torch_layout.host_counts if c > 0),
                            anchor_grain=index.layout_cache.torch_layout.pq_anchor_grain,
                            has_empty_lists=index.layout_cache.torch_layout.pq_has_empty_lists,
                            uses_dataset_name=False,
                            strategy_inputs=["list length statistics", "query batch shape", "nprobe",
                                             "K", "memory budget", "candidate chunk limit", "use_compile"],
                            static_upper_bound_padding_remains=True))
        if args.pq_scan == "bucketed":
            metadata["gpu_scan_query_tiling"] = "CPU sorts actual candidate counts once per API batch; GPU scans length buckets under the workspace budget"
            metadata["gpu_scan_adaptation"].update(static_upper_bound_padding_remains=False,
                local_bucket_padding_remains=True, cpu_transfer="one int64 candidate count per query; uploads query permutation only")

    def save_metadata():
        if not flat:
            metadata["lut_preparation"] = dict(
                order="plan length buckets, reorder query matrix, rotate and build LUT" if args.pq_scan == "bucketed" else "build LUT in original query order",
                full_lut_reordering=False,
                fp16_layout_and_cast_in_one_copy=True,
                generation_batch="one LUT build per API query batch, not per scan bucket",
                precision_unchanged=True, query_permutation_device="mps" if args.pq_scan == "bucketed" else None)
            metadata["topk_selection"] = dict(
                method="exact region-minimum pruning for long whole-span scans and small K; global/blocked/streaming fallback",
                target_minima=2048, region_width_range=[32, 1024],
                eligibility="candidate span >= 2048 and K * region_width * 4 < span",
                span_alignment="region width for pruned scans; no 8192-score padding",
                exact_distances=True, equal_distance_boundary_ids_may_differ=True,
                handwritten_gpu_kernels=False)
            layout = index.layout_cache.torch_layout
            words = layout.pq_adc_words
            metadata["adc_access"] = dict(
                index_address_width="int32 when index rows + ordinal span <= 2**31-1; otherwise int64",
                applies_to="compiled anchor ADC; binary and eager scans retain previous access",
                pq_codes_per_word=4 if words is not None else 1,
                packed_code_storage_aliases_logical_codes=(words.untyped_storage().data_ptr()
                    == layout.lists.data.untyped_storage().data_ptr()) if words is not None else None,
                fp32_subspace_accumulation_order_unchanged=True,
                handwritten_gpu_kernels=False)
        json_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    rows = []
    save_csv(csv_path, rows)
    save_metadata()
    try:
        with torch.inference_mode():
            for topk in topks:
                for nprobe in args.nprobes:
                    print(f"Starting {kind} K={topk} nprobe={nprobe}", flush=True)
                    visits = probe_visit_statistics(index, queries, nprobe, torch.device("mps"),
                                                    4 * queries.shape[1] if flat else 16)
                    for _ in range(args.warmup):
                        warmup_batches = batches if args.warmup_scope == "all-batches" else batches[:1]
                        for batch in warmup_batches:
                            values, ids = search(index, batch, topk=topk, nprobe=nprobe, **kwargs)
                            if values.device.type != "mps" or ids.device.type != "mps":
                                raise RuntimeError("Search outputs are not on GPU")
                            torch.mps.synchronize()
                            del values, ids

                    def sweep(retain=False, sampler=None):
                        elapsed, saved = 0.0, []
                        for batch in batches:
                            torch.mps.synchronize()
                            begin = time.perf_counter()
                            values, ids = search(index, batch, topk=topk, nprobe=nprobe, **kwargs)
                            torch.mps.synchronize()
                            elapsed += time.perf_counter() - begin
                            if sampler is not None:
                                sampler.sample()
                            if retain:
                                if not torch.isfinite(values).all().item():
                                    raise RuntimeError("Nonfinite search distances")
                                saved.append(ids.cpu())
                            del values, ids
                        return elapsed, saved

                    samples = []
                    for repetition in range(args.repetitions):
                        elapsed, saved = sweep(retain=repetition == args.repetitions - 1)
                        samples.append(elapsed)
                    result_ids = torch.cat(saved)
                    recall = float(compute_recall_at_k(result_ids, groundtruth, topk))
                    if result_ids.shape != (len(queries), topk) or (result_ids < 0).any():
                        raise RuntimeError("Missing query results or invalid neighbor IDs")
                    if not all(len(set(row)) == topk for row in result_ids.tolist()):
                        raise RuntimeError("Duplicate neighbor IDs")
                    del result_ids, saved
                    # Release cached allocations between memory sweeps so earlier
                    # configurations do not inflate this configuration's metric.
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                    with MemorySampler() as sampler:
                        sweep(sampler=sampler)
                    row = dict(nprobes=nprobe, q_batch=q_batch, K=topk,
                               qps=len(queries) / statistics.median(samples), recall=recall,
                               visited_vectors_per_query=visits["visited_vectors_per_query"],
                               peak_gpu_memory=sampler.peak / (1 << 20))
                    if not all(math.isfinite(float(value)) for value in row.values()):
                        raise RuntimeError("Non-finite benchmark result")
                    rows.append(row)
                    save_csv(csv_path, rows)
                    metadata["results"].append(dict(row, samples_seconds=samples,
                                                   memory_samples=sampler.samples))
                    save_metadata()
                    print(f"{kind} {len(rows)}/{metadata['expected_rows']} "
                          f"K={topk} nprobe={nprobe} QPS={row['qps']:.2f} "
                          f"recall={recall:.6f} memory={row['peak_gpu_memory']:.1f} MiB", flush=True)
        metadata.update(status="complete", wall_seconds=time.perf_counter() - started)
        save_metadata()
    except BaseException as error:
        metadata.update(status="failed", error=repr(error), wall_seconds=time.perf_counter() - started)
        save_metadata()
        raise
    finally:
        index.release_device_residency()
        del index, batches
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
    print(f"Saved {csv_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/Users/yitongsong/Dataset/sift"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--kind", choices=("both", "ivfflat", "ivfpq"), default="both")
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--nprobes", type=int, nargs="+", default=SIFT_NPROBES)
    parser.add_argument("--flat-topks", type=int, nargs="+", default=(20, 40, 60, 80, 100))
    parser.add_argument("--pq-topks", type=int, nargs="+", default=(10,))
    parser.add_argument("--flat-batch", type=int, default=128)
    parser.add_argument("--flat-scan", choices=("auto", "grouped", "dense"), default="auto")
    parser.add_argument("--result-suffix", default="")
    parser.add_argument("--use-compile", action="store_true",
                        help="Compile the existing PyTorch expressions with torch.compile; no handwritten GPU kernels")
    parser.add_argument("--candidate-chunk-size", type=int, default=2048)
    parser.add_argument("--pq-batch", type=int, default=16)
    parser.add_argument("--pq-scan", choices=("auto", "grouped", "gpu", "bucketed"), default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warmup-scope", choices=("first-batch", "all-batches"), default="first-batch")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.result_suffix and not all(c.isalnum() or c in "_-" for c in args.result_suffix):
        parser.error("Result suffix may contain letters, digits, underscores and hyphens")
    if not torch.backends.mps.is_available():
        parser.error("An available Apple MPS GPU is required")
    if min(args.nq, args.flat_batch, args.pq_batch, args.warmup, args.repetitions, args.candidate_chunk_size,
           *args.flat_topks, *args.pq_topks, *args.nprobes) <= 0 or max(args.nprobes) > 1024:
        parser.error("Sizes/counts must be positive; nprobes must be <= 1024")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.kind == "both":
        # Separate processes avoid carrying one algorithm's Metal/MPSGraph
        # caches into the other's process-memory measurements.
        for kind in ("ivfflat", "ivfpq"):
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                            *sys.argv[1:], "--kind", kind], check=True)
        return
    queries = load_dataset(args.dataset / "sift_query.fvecs")
    groundtruth = load_ivecs(args.dataset / "sift_groundtruth.ivecs")
    if len(queries) != len(groundtruth) or queries.shape[1] != 128:
        raise ValueError("Invalid SIFT queries/ground truth")
    queries, groundtruth = queries[:args.nq], groundtruth[:args.nq]
    for kind in (("ivfflat", "ivfpq") if args.kind == "both" else (args.kind,)):
        benchmark(args, kind, queries, groundtruth)


if __name__ == "__main__":
    main()
