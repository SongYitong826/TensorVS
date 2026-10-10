"""Alternating grouped/GPU-only TensorNative PQ schedule comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
from tensorvs import load_dataset, load_ivecs, load_ivfpq_index, search_ivfpq_index, compute_recall_at_k
from tensorvs.index_ivfpq import _coarse_centroid_scores_addmm

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-compiled", action="store_true")
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--q-batches", type=int, nargs="+", default=[256,512])
    parser.add_argument("--nprobes", type=int, nargs="+", default=[5,50,150])
    parser.add_argument("--output", type=Path, default=ROOT/"results/ivfpq_gpu_schedule/paired_comparison.json")
    args = parser.parse_args()
    dataset = Path("/Users/yitongsong/Dataset/sift")
    source = dataset/"GPUIndex/TenVS_IVFPQResidual_nlist1024_m16_nbits8_train256.bin"
    index = load_ivfpq_index(source)
    index.reside_on_device("mps", backend="torch", persist_layout=False)
    queries = load_dataset(dataset/"sift_query.fvecs")[:args.nq].to("mps")
    truth = load_ivecs(dataset/"sift_groundtruth.ivecs")[:args.nq]
    variants = {"grouped_eager":("grouped",False),"gpu_eager":("gpu",False)}
    if args.include_compiled:
        variants.update(grouped_compiled=("grouped",True),gpu_compiled=("gpu",True))
    report = dict(status="running",device="mps",K=10,query_count=args.nq,
        cpu_operator_fallback=False,handwritten_kernels=False,full_warmup_sweeps=2,repetitions=3,
        gpu_schedule="GPU prefix sums and searchsorted; static index-only scan bound; no query readback",
        python_launch_loop=True,index=str(source),index_size_bytes=source.stat().st_size,
        index_mtime_ns=source.stat().st_mtime_ns,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ("src/tensorvs/_torch_search.py","src/tensorvs/_layout_cache.py","src/tensorvs/index_ivfpq.py")},records=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    save()
    with torch.inference_mode():
        for batch_size in args.q_batches:
            batches = list(queries.split(batch_size))
            for probe in args.nprobes:
                kwargs = dict(device="mps",backend="torch",persist_layout=False,lut_dtype="fp16",
                    torch_query_batch_size=batch_size,torch_candidate_chunk_size=2048,
                    torch_memory_budget_mb=128,topk=10,nprobe=probe)
                def sweep(name,retain=False):
                    scan,compiled=variants[name]
                    elapsed,values_out,ids_out=0.,[],[]
                    for batch in batches:
                        torch.mps.synchronize(); start=time.perf_counter()
                        values,ids=search_ivfpq_index(index,batch,torch_pq_scan=scan,use_compile=compiled,**kwargs)
                        torch.mps.synchronize(); elapsed+=time.perf_counter()-start
                        if retain:
                            values_out.append(values.cpu());ids_out.append(ids.cpu())
                    return elapsed,(torch.cat(values_out),torch.cat(ids_out)) if retain else None
                for name in variants:
                    print(f"warmup batch={batch_size} probe={probe} {name}",flush=True)
                    for _ in range(2):sweep(name)
                samples={name:[] for name in variants}; outputs={}
                for repetition in range(3):
                    names=list(variants)
                    if repetition%2:names.reverse()
                    for name in names:
                        elapsed,result=sweep(name,retain=repetition==2)
                        samples[name].append(elapsed)
                        if result is not None:outputs[name]=result
                baseline=outputs["grouped_eager"]
                reference_recall=float(compute_recall_at_k(baseline[1],truth,10))
                validation={}
                for name,(values,ids) in outputs.items():
                    torch.testing.assert_close(values,baseline[0],atol=.0625,rtol=2e-5)
                    assert torch.isfinite(values).all() and (ids>=0).all()
                    assert all(len(set(row.tolist()))==10 for row in ids)
                    recall=float(compute_recall_at_k(ids,truth,10))
                    assert abs(recall-reference_recall)<=.001
                    validation[name]=dict(recall=recall,
                        max_distance_difference=float((values-baseline[0]).abs().max()),
                        neighbor_set_overlap=sum(len(set(a.tolist())&set(b.tolist())) for a,b in zip(ids,baseline[1]))/(10*len(ids)))
                # Untimed reporting only: these host reads are outside search.
                layout=index.layout_cache.torch_layout
                selected=_coarse_centroid_scores_addmm(queries,layout.tensors["centroids"],
                    layout.tensors["centroid_norms"]).topk(probe,largest=False,sorted=False).indices
                counts=layout.lists.counts[selected].sum(1).cpu()
                qps={name:len(queries)/statistics.median(s) for name,s in samples.items()}
                record=dict(q_batch=batch_size,nprobes=probe,samples_seconds=samples,qps=qps,
                    speedup_gpu_vs_grouped_eager=qps["gpu_eager"]/qps["grouped_eager"],validation=validation,
                    fixed_scan_bound=layout.pq_scan_upper_bounds[probe],
                    mean_actual_candidates=float(counts.float().mean()),max_actual_candidates=int(counts.max()))
                if args.include_compiled:
                    record["speedup_gpu_vs_grouped_compiled"]=qps["gpu_compiled"]/qps["grouped_compiled"]
                report["records"].append(record);save();print(json.dumps(record),flush=True)
    report["status"]="complete";save()


if __name__=="__main__":main()
