"""Synchronized MPS stage diagnostics on the existing SIFT IVF-Flat index."""
import json
import argparse
import os
from pathlib import Path
import statistics
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
from tensorvs import load_dataset, load_ivfflat_index, search_ivfflat_index
from tensorvs._torch_search import _schedule
from tensorvs.index_ivfflat import _coarse_centroid_scores_addmm

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scan", choices=("grouped", "dense"), default="grouped")
args = parser.parse_args()
dataset = Path("/Users/yitongsong/Dataset/sift")
index = load_ivfflat_index(dataset / "GPUIndex/TenVS_IVFFlat_nlist1024.bin")
index.reside_on_device("mps", backend="torch", persist_layout=False)
q = load_dataset(dataset / "sift_query.fvecs")[:1024].to("mps")
batches = list(q.split(128))
layout = index.layout_cache.torch_layout
records = []
kwargs = dict(device="mps", backend="torch", use_compile=False, persist_layout=False,
              torch_query_batch_size=128, torch_candidate_chunk_size=2048,
              torch_memory_budget_mb=128, torch_flat_scan=args.scan)
with torch.inference_mode():
    for k, probe in ((10, 5), (10, 50), (10, 150), (100, 5), (100, 50), (100, 150)):
        for _ in range(2):
            search_ivfflat_index(index, batches[0], topk=k, nprobe=probe, **kwargs)
        torch.mps.synchronize()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            for batch in batches:
                search_ivfflat_index(index, batch, topk=k, nprobe=probe, **kwargs)
                torch.mps.synchronize()
            times.append(time.perf_counter() - start)
        profile = {}
        search_ivfflat_index(index, batches[0], topk=k, nprobe=probe, profile=profile, **kwargs)
        probes = _coarse_centroid_scores_addmm(batches[0], layout.tensors["centroids"], layout.tensors["centroid_norms"]).topk(probe, largest=False).indices
        schedule = _schedule(probes, layout.host_counts, layout.host_offsets, 32, 2048,
                             minimum_query_width=1)
        valid_pairs = sum(len(task[0]) * task[2] for tasks in schedule.values() for task in tasks)
        padded_pairs = sum(qw * cw * len(tasks) for (qw, cw), tasks in schedule.items())
        record = dict(K=k, nprobes=probe, median_qps=1024 / statistics.median(times),
                      samples_seconds=times, first_batch_profile_seconds=profile,
                      tasks=sum(map(len, schedule.values())), shape_buckets=len(schedule),
                      valid_distance_pairs=valid_pairs, padded_distance_pairs=padded_pairs,
                      padding_ratio=padded_pairs / valid_pairs,
                      index_max_list_size=max(layout.host_counts))
        records.append(record)
        print(json.dumps(record), flush=True)
output = root / f"results/mps_ivfflat_diagnosis_{args.scan}.json"
output.write_text(json.dumps(dict(torch=str(torch.__version__), device="mps", cpu_fallback=False,
    profiling="GPU synchronized at stage boundaries; profiling adds synchronization and is separate from QPS",
    records=records), indent=2) + "\n")
