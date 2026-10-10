"""Alternating, fully warmed eager MPS IVF-PQ comparisons against a source snapshot."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
from tensorvs import compute_recall_at_k,load_dataset,load_ivecs,load_ivfpq_index
import tensorvs.index_ivfpq as pq
import tensorvs._torch_search as current

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=ROOT/"results/ivfpq_torch_optimization/baseline_torch_search.py")
    parser.add_argument("--output",type=Path,default=ROOT/"results/ivfpq_torch_optimization/paired_comparison.json")
    parser.add_argument("--repetitions",type=int,default=3)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location("tensorvs._torch_search_baseline",args.baseline)
    baseline=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    dataset=Path("/Users/yitongsong/Dataset/sift")
    index=load_ivfpq_index(dataset/"GPUIndex/TenVS_IVFPQResidual_nlist1024_m16_nbits8_train256.bin")
    index.reside_on_device("mps",backend="torch",persist_layout=False)
    queries=load_dataset(dataset/"sift_query.fvecs")[:1024].to("mps")
    truth=load_ivecs(dataset/"sift_groundtruth.ivecs")[:1024]
    original=pq.search_torch
    records=[]
    try:
        with torch.inference_mode():
            for batch_size,probe in ((256,5),(256,50),(256,150),(512,5),(512,150)):
                batches=list(queries.split(batch_size))
                options=dict(device="mps",backend="torch",use_compile=False,persist_layout=False,
                    lut_dtype="fp16",torch_query_batch_size=batch_size,torch_candidate_chunk_size=2048,
                    torch_memory_budget_mb=128,topk=10,nprobe=probe)
                def sweep(function,retain=False):
                    pq.search_torch=function
                    elapsed,all_values,all_ids=0.,[],[]
                    for batch in batches:
                        torch.mps.synchronize()
                        begin=time.perf_counter()
                        values,ids=pq.search_ivfpq_index(index,batch,**options)
                        torch.mps.synchronize()
                        elapsed+=time.perf_counter()-begin
                        if retain:
                            all_values.append(values.cpu());all_ids.append(ids.cpu())
                    return elapsed,(torch.cat(all_values),torch.cat(all_ids)) if retain else None
                for function in (baseline.search_torch,current.search_torch):
                    for _ in range(2):sweep(function)
                samples={"baseline":[],"current":[]}
                results={}
                for repetition in range(args.repetitions):
                    order=(("baseline",baseline.search_torch),("current",current.search_torch))
                    if repetition%2:order=order[::-1]
                    for label,function in order:
                        elapsed,result=sweep(function,retain=repetition==args.repetitions-1)
                        samples[label].append(elapsed)
                        if result is not None:results[label]=result
                before,after=results["baseline"],results["current"]
                assert torch.isfinite(after[0]).all() and ((after[1]>=0)&(after[1]<1000000)).all()
                assert all(len(set(row.tolist()))==10 for row in after[1])
                torch.testing.assert_close(after[0],before[0],atol=.0625,rtol=2e-5)
                recalls={label:float(compute_recall_at_k(result[1],truth,10)) for label,result in results.items()}
                assert abs(recalls["baseline"]-recalls["current"])<=.001
                record=dict(q_batch=batch_size,nprobes=probe,samples_seconds=samples,
                    qps={label:1024/statistics.median(values) for label,values in samples.items()},
                    speedup=statistics.median(samples["baseline"])/statistics.median(samples["current"]),
                    recalls=recalls,max_distance_difference=float((before[0]-after[0]).abs().max()),
                    neighbor_set_overlap=float(torch.tensor([len(set(a.tolist())&set(b.tolist()))/10
                        for a,b in zip(before[1],after[1])]).mean()))
                records.append(record)
                print(json.dumps(record),flush=True)
    finally:pq.search_torch=original
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(device="mps",cpu_fallback=False,K=10,query_count=1024,
        full_warmup_sweeps=2,repetitions=args.repetitions,alternating_order=True,
        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        current_sha256=hashlib.sha256((ROOT/"src/tensorvs/_torch_search.py").read_bytes()).hexdigest(),
        records=records),indent=2)+"\n")


if __name__=="__main__":main()
