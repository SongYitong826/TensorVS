"""Synchronized stage diagnostics and padding analysis for eager MPS IVF-PQ."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
from tensorvs import load_dataset, load_ivfpq_index, search_ivfpq_index
import tensorvs._torch_search as native
from tensorvs.index_ivfpq import _coarse_centroid_scores_addmm

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-batches", type=int, nargs="+", default=[256,512])
    parser.add_argument("--nprobes", type=int, nargs="+", default=[5,50,150])
    parser.add_argument("--minimum-query-width", type=int, default=1)
    parser.add_argument("--query-tile", type=int, default=8)
    parser.add_argument("--candidate-width", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=ROOT/"results/ivfpq_torch_optimization/current_diagnosis.json")
    args = parser.parse_args()
    dataset = Path("/Users/yitongsong/Dataset/sift")
    index = load_ivfpq_index(dataset/"GPUIndex/TenVS_IVFPQResidual_nlist1024_m16_nbits8_train256.bin")
    index.reside_on_device("mps", backend="torch", persist_layout=False)
    q = load_dataset(dataset/"sift_query.fvecs")[:1024].to("mps")
    layout = index.layout_cache.torch_layout
    records = []
    with torch.inference_mode():
        for batch_size in args.q_batches:
            batches = list(q.split(batch_size))
            options = dict(device="mps",backend="torch",lut_dtype="fp16",use_compile=False,
                           persist_layout=False,torch_query_batch_size=batch_size,
                           torch_candidate_chunk_size=2048,torch_memory_budget_mb=128)
            for probe in args.nprobes:
                for _ in range(2):
                    for batch in batches:
                        search_ivfpq_index(index,batch,topk=10,nprobe=probe,**options)
                torch.mps.synchronize()
                times = []
                for _ in range(3):
                    elapsed = 0.
                    for batch in batches:
                        start = time.perf_counter()
                        search_ivfpq_index(index,batch,topk=10,nprobe=probe,**options)
                        torch.mps.synchronize()
                        elapsed += time.perf_counter()-start
                    times.append(elapsed)
                profile = {}
                search_ivfpq_index(index,batches[0],topk=10,nprobe=probe,profile=profile,**options)
                probes = _coarse_centroid_scores_addmm(batches[0],layout.tensors["centroids"],
                    layout.tensors["centroid_norms"]).topk(probe,largest=False,sorted=False).indices
                schedule = native._schedule(probes,layout.host_counts,layout.host_offsets,args.query_tile,args.candidate_width,
                                            minimum_query_width=args.minimum_query_width)
                valid = sum(len(task[0])*task[2] for tasks in schedule.values() for task in tasks)
                padded = sum(qw*cw*len(tasks) for (qw,cw),tasks in schedule.items())
                record = dict(q_batch=batch_size,nprobes=probe,qps=1024/statistics.median(times),
                    samples_seconds=times,first_batch_profile_seconds=profile,
                    valid_distance_pairs=valid,padded_distance_pairs=padded,padding_ratio=padded/valid,
                    task_count=sum(map(len,schedule.values())),shape_buckets=len(schedule),
                    expanded_lut_values=padded*16,
                    selected_pq_code_bytes=valid*16,expanded_int32_lut_index_bytes=padded*16*4,
                    query_tile=args.query_tile,candidate_width=args.candidate_width,
                    estimate_limit="Padding and address bytes cover all first-batch tasks, not simultaneous allocations.")
                records.append(record)
                print(json.dumps(record),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(device="mps",cpu_fallback=False,K=10,lut_dtype="fp16",
        source_sha256=hashlib.sha256((ROOT/"src/tensorvs/_torch_search.py").read_bytes()).hexdigest(),
        profile_limit="Stage synchronization adds overhead; profiles are separate from ordinary full-query QPS timing. lut_lookup_accumulation includes block-local Top-K; final Top-K merging is timed separately.",
        records=records),indent=2)+"\n")


if __name__ == "__main__":
    main()
