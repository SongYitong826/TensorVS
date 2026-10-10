"""Paired pre/post LUT ordering and contiguous-cast benchmark.

Pass --baseline-dir containing trusted pre-change source snapshots named
_torch_search.py_tensorvs and _torch_pq_bucketed.py_tensorvs. Optional Faiss
uses a separate process to avoid incompatible OpenMP runtimes.
"""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def faiss_worker():
    # Torch and this local Faiss build link different OpenMP runtimes. Keep
    # them in separate processes; IPC is outside the reported search timing.
    parser = argparse.ArgumentParser()
    parser.add_argument("--faiss-worker", action="store_true")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--nq", type=int, required=True)
    args = parser.parse_args()
    faiss_root = Path(__file__).resolve().parents[2] / "faiss-metal"
    sys.path.insert(0, str(faiss_root / "local_bench"))
    import benchmark_sift as common
    import faiss
    import numpy as np
    faiss.omp_set_num_threads(4)
    cpu = faiss.read_index(str(faiss_root / "local_bench/indexes/sift_ivfpq_from_tensorvs.faiss"))
    resources = faiss.StandardGpuResources()
    wrapper = faiss.index_cpu_to_gpu(resources, 0, cpu)
    queries = common.read_vectors(args.dataset / "sift_query.fvecs", np.float32)[:args.nq]
    for line in sys.stdin:
        request = json.loads(line)
        params = faiss.SearchParametersIVF(); params.nprobe = request["nprobe"]
        elapsed, saved_values, saved_ids = 0., [], []
        for start in range(0, len(queries), request["q_batch"]):
            batch = queries[start:start + request["q_batch"]]
            begin = time.perf_counter()
            values, ids = wrapper.search(batch, 10, params=params)
            elapsed += time.perf_counter() - begin
            if request["retain"]:
                saved_values.append(values); saved_ids.append(ids)
        result = dict(elapsed=elapsed)
        if request["retain"]:
            result.update(values=np.concatenate(saved_values).tolist(), ids=np.concatenate(saved_ids).tolist())
        print(json.dumps(result), flush=True)


if __name__ == "__main__" and "--faiss-worker" in sys.argv:
    faiss_worker()
    raise SystemExit

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
from tensorvs import (compute_recall_at_k, load_dataset, load_ivecs,
                      load_ivfpq_index, search_ivfpq_index)
import tensorvs._torch_pq_bucketed as bucket
import tensorvs._torch_search as native
import tensorvs.index_ivfpq as pq


def load_baseline(directory):
    """Read a trusted pre-change source snapshot, preserving its original plan."""
    import types
    modules = []
    for name in ("_torch_search", "_torch_pq_bucketed"):
        path = directory / (name + ".py_tensorvs")
        module = types.ModuleType("tensorvs.topk_baseline" + name)
        module.__package__ = "tensorvs"
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
        modules.append(module)
    return modules[0].search_torch, modules[1].search_bucketed_pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/Users/yitongsong/Dataset/sift"))
    parser.add_argument("--q-batches", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--nprobes", nargs="+", type=int, default=[5, 50, 150])
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--faiss", action="store_true")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/tensorvs_lut_order_paired_comparison.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.dataset / "GPUIndex/TenVS_IVFPQResidual_nlist1024_m16_nbits8_train256.bin"
    index = load_ivfpq_index(source)
    index.reside_on_device("mps", backend="torch", persist_layout=False)
    host_queries = load_dataset(args.dataset / "sift_query.fvecs")[:args.nq]
    queries = host_queries.to("mps")
    truth = load_ivecs(args.dataset / "sift_groundtruth.ivecs")[:args.nq]
    original_search = bucket.search_bucketed_pq
    original_torch = pq.search_torch
    baseline_torch, baseline_search = load_baseline(args.baseline_dir)
    modes = ["previous", "current"]
    worker = None
    if args.faiss:
        worker = subprocess.Popen(["conda", "run", "--no-capture-output", "-n", "tensorvs",
            "python", str(Path(__file__).resolve()), "--faiss-worker", "--dataset", str(args.dataset),
            "--nq", str(args.nq)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        atexit.register(worker.terminate)
        modes.append("faiss_metal_cpu")
    records = []
    with torch.inference_mode():
        for qb in args.q_batches:
            for probe in args.nprobes:
                options = dict(topk=10, nprobe=probe, device="mps", backend="torch",
                    use_compile=True, torch_pq_scan="bucketed", lut_dtype="fp16",
                    persist_layout=False, torch_query_batch_size=qb,
                    torch_candidate_chunk_size=2048, torch_memory_budget_mb=128)
                def sweep(mode, retain=False):
                    bucket.search_bucketed_pq = baseline_search if mode == "previous" else original_search
                    pq.search_torch = baseline_torch if mode == "previous" else original_torch
                    elapsed, saved = 0., []
                    try:
                        if mode == "faiss_metal_cpu":
                            worker.stdin.write(json.dumps(dict(q_batch=qb, nprobe=probe, retain=retain)) + "\n")
                            worker.stdin.flush()
                            while True:
                                line = worker.stdout.readline()
                                if not line:
                                    raise RuntimeError("Faiss benchmark worker exited")
                                if line.startswith('{"elapsed":'):
                                    reply = json.loads(line); break
                            elapsed = reply["elapsed"]
                            if retain:
                                saved.append((torch.tensor(reply["values"], dtype=torch.float32),
                                              torch.tensor(reply["ids"], dtype=torch.int64)))
                        else:
                            for batch in queries.split(qb):
                                torch.mps.synchronize(); start = time.perf_counter()
                                values, ids = search_ivfpq_index(index, batch, **options)
                                torch.mps.synchronize(); elapsed += time.perf_counter() - start
                                if retain: saved.append((values.cpu(), ids.cpu()))
                    finally:
                        bucket.search_bucketed_pq = original_search
                        pq.search_torch = original_torch
                    result = (torch.cat([v for v, i in saved]), torch.cat([i for v, i in saved])) if retain else None
                    return elapsed, result
                for mode in modes:
                    print(f"warmup {qb=} {probe=} {mode=}", flush=True)
                    for _ in range(2): sweep(mode)
                samples = {mode: [] for mode in modes}; outputs = {}
                for rep in range(args.repetitions):
                    for mode in modes if rep % 2 == 0 else modes[::-1]:
                        elapsed, result = sweep(mode, rep == args.repetitions - 1)
                        samples[mode].append(elapsed)
                        if result is not None: outputs[mode] = result
                torch.testing.assert_close(outputs["current"][0], outputs["previous"][0], atol=0, rtol=0)
                torch.testing.assert_close(outputs["current"][1], outputs["previous"][1], atol=0, rtol=0)
                swaps = 0
                for row in range(args.nq):
                    old = {int(i): float(v) for i, v in zip(outputs["previous"][1][row], outputs["previous"][0][row])}
                    new = {int(i): float(v) for i, v in zip(outputs["current"][1][row], outputs["current"][0][row])}
                    boundary = float(outputs["current"][0][row, -1])
                    for i in old.keys() - new.keys():
                        assert old[i] == boundary; swaps += 1
                    for i in new.keys() - old.keys():
                        assert new[i] == boundary
                assert float(compute_recall_at_k(outputs["previous"][1], truth, 10)) == float(compute_recall_at_k(outputs["current"][1], truth, 10))
                record = dict(q_batch=qb, nprobes=probe, query_count=args.nq,
                    qps={mode: args.nq / statistics.median(v) for mode, v in samples.items()},
                    speedup=statistics.median(samples["previous"]) / statistics.median(samples["current"]),
                    samples_seconds=samples, bit_exact_distances=True, only_boundary_tie_swaps=True, tie_swaps=swaps,
                    neighbor_ids_match=torch.equal(outputs["current"][1], outputs["previous"][1]),
                    recall={mode: float(compute_recall_at_k(result[1], truth, 10)) for mode, result in outputs.items()})
                records.append(record); print(json.dumps(record), flush=True)
                metadata = dict(status="running", records=records,
                    changes=["plan buckets before LUT; reorder query matrix instead of LUT",
                             "combine FP16 cast and contiguous layout conversion in one tensor copy"],
                    baseline_source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (args.baseline_dir / "_torch_search.py_tensorvs", args.baseline_dir / "_torch_pq_bucketed.py_tensorvs")},
                    source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (Path(native.__file__), Path(bucket.__file__), root / "src/tensorvs/_layout_cache.py")},
                    source_index_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    cpu_operator_fallback=False, handwritten_gpu_kernels=False,
                    protocol="2 full-query warmup sweeps; alternating timed sweep order; median of repetitions; GPU batch boundaries synchronized; compile/index/query preparation excluded",
                    faiss_threads=4 if args.faiss else None,
                    precision="Torch FP16 LUT / FP32 accumulation; Faiss FP32 LUT")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    metadata["status"] = "complete"
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    if worker is not None:
        worker.stdin.close()
        worker.wait(timeout=10)
        atexit.unregister(worker.terminate)


if __name__ == "__main__":
    main()
