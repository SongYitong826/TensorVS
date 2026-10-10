"""Validate and time metadata-adaptive PQ on non-SIFT synthetic indexes.

Baseline snapshot and outputs deliberately live in /tmp; retained result
tables continue to contain only the latest SIFT comparison requested earlier.
"""
import importlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
from tensorvs import IVFPQBuildResult, load_ivfpq_index, save_ivfpq_index
import tensorvs.index_ivfpq as pq
from mps_ivfpq_gpu_opt_candidates import loader


def main():
    baseline = loader("adaptive_distribution_baseline")
    adaptive = importlib.import_module("tensorvs._torch_search")
    records = []
    cases = [("uniform", [256] * 128, 32, 8, 64, (8, 32)),
             ("short_and_skewed", [16384, 1, 3] + [96] * 61, 24, 6, 64, (8, 32)),
             ("empty_and_long", [1024] * 32 + [0] * 32, 48, 12, 64, (8, 32)),
             ("different_dimension", [512] * 128, 80, 10, 128, (8, 32))]
    original = pq.search_torch
    output = Path("/tmp/tensorvs_ivfpq_adaptive_distributions.json")
    try:
        with tempfile.TemporaryDirectory(prefix="tensorvs_adaptive_") as temporary, torch.inference_mode():
            for name, counts, dim, m, batch_size, probes in cases:
                print(f"building {name}", flush=True)
                gen = torch.Generator().manual_seed(819)
                labels = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
                vectors = torch.randn(len(labels), dim, generator=gen)
                centers = torch.randn(len(counts), dim, generator=gen)
                books = torch.randn(m, 256, dim // m, generator=gen)
                codes = torch.randint(0, 256, (len(labels), m), generator=gen, dtype=torch.uint8)
                index = load_ivfpq_index(save_ivfpq_index(IVFPQBuildResult(vectors, labels, centers,
                    books, codes, torch.device("cpu"), m, 8), Path(temporary) / (name + ".bin")))
                index.reside_on_device("mps", backend="torch", persist_layout=False)
                queries = torch.randn(batch_size, dim, generator=gen).to("mps")
                for probe in probes:
                    kwargs = dict(device="mps", backend="torch", nprobe=probe, topk=10,
                        torch_pq_scan="gpu", use_compile=True, lut_dtype="fp32",
                        torch_query_batch_size=batch_size, persist_layout=False)
                    def run(module):
                        pq.search_torch = module.search_torch
                        torch.mps.synchronize()
                        start = time.perf_counter()
                        values, ids = pq.search_ivfpq_index(index, queries, **kwargs)
                        torch.mps.synchronize()
                        return time.perf_counter() - start, values, ids
                    for module in (baseline, adaptive):
                        for _ in range(10):
                            run(module)
                    samples = {"baseline": [], "adaptive": []}
                    outputs = {}
                    for repetition in range(7):
                        order = [("baseline", baseline), ("adaptive", adaptive)]
                        if repetition % 2:
                            order.reverse()
                        for variant, module in order:
                            elapsed = 0.
                            for _ in range(16):
                                duration, values, ids = run(module)
                                elapsed += duration
                            samples[variant].append(elapsed / 16)
                            outputs[variant] = values.cpu(), ids.cpu()
                    torch.testing.assert_close(outputs["baseline"][0], outputs["adaptive"][0], atol=2e-4, rtol=2e-5)
                    assert torch.equal(outputs["baseline"][1].sort(1).values, outputs["adaptive"][1].sort(1).values)
                    record = dict(case=name, dimension=dim, subspaces=m, counts=counts,
                        q_batch=batch_size, nprobes=probe, K=10, samples_seconds=samples,
                        speedup=statistics.median(samples["baseline"]) / statistics.median(samples["adaptive"]),
                        selected_sets_match=True)
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    output.write_text(json.dumps(dict(status="running", records=records), indent=2) + "\n")
                index.release_device_residency()
    finally:
        pq.search_torch = original
    output.write_text(json.dumps(dict(status="complete", records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
