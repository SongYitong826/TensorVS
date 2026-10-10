"""Paired comparison of current padded scan and experimental compact ADC."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
from tensorvs import load_dataset, load_ivecs, load_ivfpq_index, compute_recall_at_k
import tensorvs.index_ivfpq as pq
import tensorvs._torch_search as native
from _pq_compact_candidate import compact_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--q-batches", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--nprobes", nargs="+", type=int, default=[5, 50, 150])
    parser.add_argument("--modes", nargs="+", default=["padded", "row", "blocked", "bucketed"])
    parser.add_argument("--output", type=Path, default=Path("/tmp/tensorvs_compact_compare.json"))
    args = parser.parse_args()
    dataset = Path("/Users/yitongsong/Dataset/sift")
    index = load_ivfpq_index(dataset / "GPUIndex/TenVS_IVFPQResidual_nlist1024_m16_nbits8_train256.bin")
    index.reside_on_device("mps", backend="torch", persist_layout=False)
    queries = load_dataset(dataset / "sift_query.fvecs")[:args.nq].to("mps")
    truth = load_ivecs(dataset / "sift_groundtruth.ivecs")[:args.nq]
    original = pq.search_torch
    spec = importlib.util.spec_from_file_location("tensorvs.compact_experiment", Path(native.__file__))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    records = []
    try:
        with torch.inference_mode():
            for batch_size in args.q_batches:
                for probe in args.nprobes:
                    kwargs = dict(device="mps", backend="torch", use_compile=True, torch_pq_scan="gpu",
                        lut_dtype="fp16", persist_layout=False, topk=10, nprobe=probe,
                        torch_query_batch_size=batch_size, torch_memory_budget_mb=128)
                    def sweep(mode, retain=False):
                        module._packed_pq_search = lambda *a, **kw: compact_search(*a, **kw, mode=mode)
                        pq.search_torch = original if mode == "padded" else module.search_torch
                        if mode == "bucketed":
                            pq.search_torch = original
                        elapsed, saved = 0., []
                        for batch in queries.split(batch_size):
                            torch.mps.synchronize(); start = time.perf_counter()
                            options = dict(kwargs, torch_pq_scan="bucketed") if mode == "bucketed" else kwargs
                            values, ids = pq.search_ivfpq_index(index, batch, **options)
                            torch.mps.synchronize(); elapsed += time.perf_counter() - start
                            if retain:
                                saved.append((values.cpu(), ids.cpu()))
                        return elapsed, (torch.cat([s[0] for s in saved]), torch.cat([s[1] for s in saved])) if retain else None
                    for mode in args.modes:
                        print(f"warmup {batch_size=} {probe=} {mode=}", flush=True)
                        for _ in range(2): sweep(mode)
                    samples = {mode: [] for mode in args.modes}; outputs = {}
                    for rep in range(3):
                        for mode in args.modes if rep % 2 == 0 else args.modes[::-1]:
                            elapsed, result = sweep(mode, rep == 2)
                            samples[mode].append(elapsed)
                            if result is not None: outputs[mode] = result
                    validation = {}
                    for mode, (values, ids) in outputs.items():
                        torch.testing.assert_close(values, outputs["padded"][0], atol=.0625, rtol=2e-5)
                        assert all(row.unique().numel() == 10 for row in ids)
                        overlap = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ids, outputs["padded"][1])) / (10 * len(ids))
                        assert overlap == 1.
                        validation[mode] = dict(recall=float(compute_recall_at_k(ids, truth, 10)), neighbor_overlap=overlap)
                    record = dict(q_batch=batch_size, nprobes=probe, samples_seconds=samples,
                        qps={mode: args.nq / statistics.median(v) for mode, v in samples.items()}, validation=validation)
                    records.append(record); print(json.dumps(record), flush=True)
                    args.output.write_text(json.dumps(dict(status="running", records=records), indent=2) + "\n")
    finally:
        pq.search_torch = original
    args.output.write_text(json.dumps(dict(status="complete", records=records), indent=2) + "\n")


if __name__ == "__main__": main()
