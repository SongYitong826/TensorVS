"""Measure tensor-only eager PQ scan candidates on the original SIFT index."""
from __future__ import annotations

import hashlib
import argparse
import inspect
import json
import os
from pathlib import Path
import statistics
import time
import types

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="0"
import torch
from tensorvs import load_dataset,load_ivecs,load_ivfpq_index,compute_recall_at_k
import tensorvs.index_ivfpq as pq
from mps_ivfpq_round2_probe import candidate_major,finish,embedding_bag

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/"results/ivfpq_torch_round2/baseline_torch_search.py"


def deferred_finish(distances,ids,corrections,bias,query_ids,list_ids,starts,counts,positions,offsets,k):
    distances.add_(corrections.index_select(0,positions.flatten()).reshape_as(positions)[:,None,:])
    distances.masked_fill_(~(offsets[None,None,:]<counts[:,None,None]),float("inf"))
    values,order=distances.topk(k,dim=-1,largest=False,sorted=False)
    bias_positions=query_ids*bias.shape[1]+list_ids[:,None]
    values.add_(bias.flatten().index_select(0,bias_positions.flatten()).reshape_as(query_ids)[:,:,None]).clamp_min_(0.)
    selected=ids.index_select(0,(starts[:,None,None]+order).clamp_max(ids.shape[0]-1).flatten()).reshape_as(order)
    return values,selected.masked_fill(~torch.isfinite(values),-1)


def packed_candidate(luts,data,ids,corrections,bias,probes,offsets,prefix,previous,ordinals,k,subspace_chunk):
    ranks=torch.searchsorted(prefix,ordinals[None,:].expand(prefix.shape[0],-1).contiguous(),right=True).clamp_max(probes.shape[1]-1)
    positions=(offsets[probes.gather(1,ranks)]+ordinals[None,:]-previous.gather(1,ranks)).clamp_max(data.shape[0]-1)
    dtype=torch.int32 if luts.numel()<=2147483647 else torch.int64
    codes=data.index_select(0,positions.flatten()).reshape(*positions.shape,data.shape[1]).to(dtype)
    if data.dtype==torch.int16:codes=codes&65535
    query_offsets=torch.arange(luts.shape[0],device=luts.device,dtype=dtype)[:,None,None]*(luts.shape[1]*luts.shape[2])
    distances=None
    for start in range(0,data.shape[1],16):
        end=min(start+16,data.shape[1])
        subspace=torch.arange(start,end,device=luts.device,dtype=dtype)*luts.shape[2]
        indices=query_offsets+subspace[None,None,:]+codes[:,:,start:end]
        values=luts.flatten().index_select(0,indices.flatten()).reshape_as(indices)
        summed=values.sum(-1,dtype=torch.float32)
        if distances is None:distances=summed
        else:distances.add_(summed)
        del indices,values,summed
    distances.add_(corrections.index_select(0,positions.flatten()).reshape_as(positions)).add_(bias.gather(1,ranks)).clamp_min_(0.)
    distances.masked_fill_(ordinals[None,:]>=prefix[:,-1:],float("inf"))
    values,order=distances.topk(k,largest=False,sorted=False)
    selected=ids.index_select(0,positions.gather(1,order).flatten()).reshape_as(order)
    return values,selected.masked_fill(~torch.isfinite(values),-1)


def module(name,packed=False):
    source=SNAPSHOT.read_text()
    if packed:
        source=source.replace('kind == "pq" and device.type == "cuda"','kind == "pq" and device.type in ("cuda", "mps")')
        source=source.replace('(32 if use_compile else 100)',
            '(32 if use_compile else (32 + min(m,16) * (4 + lut.element_size() + 4)))')
    result=types.ModuleType("tensorvs."+name)
    result.__package__="tensorvs"
    exec(compile(source,str(SNAPSHOT),"exec"),result.__dict__)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants",nargs="+",default=["current","candidate_major","deferred_bias","packed"])
    parser.add_argument("--output",type=Path,default=ROOT/"results/ivfpq_torch_round2/candidate_comparison.json")
    args=parser.parse_args()
    variants={"current":module("round2_current"),"candidate_major":module("round2_candidate"),
              "deferred_bias":module("round2_deferred"),"packed":module("round2_packed",packed=True)}
    variants["candidate_major"]._pq_block_eager=candidate_major
    variants["deferred_bias"]._pq_block_eager=types.FunctionType(candidate_major.__code__,
        {**candidate_major.__globals__,"finish":deferred_finish},candidate_major.__name__)
    variants["packed"]._pq_packed_block=packed_candidate
    for tile in (8,32):
        source=SNAPSHOT.read_text().replace("tile = min(32, query_batch_size)",f"tile = min({tile}, query_batch_size)")
        source=source.replace('                bias = (scores + batch.square()',
            '                lut = lut.float()\n                bias = (scores + batch.square()')
        source=source.replace('lane_bytes = 32 + (address_bytes + lut.element_size() + 4) * min(m, 16)',
            'lane_bytes = 32 + 2 * address_bytes * min(m,16)')
        name=f"embedding_tile{tile}"
        implementation=types.ModuleType("tensorvs.round2_"+name);implementation.__package__="tensorvs"
        exec(compile(source,str(SNAPSHOT),"exec"),implementation.__dict__)
        implementation._pq_block_eager=types.FunctionType(embedding_bag.__code__,
            {**embedding_bag.__globals__,"finish":deferred_finish},embedding_bag.__name__)
        variants[name]=implementation
    for tile,chunk in ((8,1024),(16,1024),(64,1024),(8,512),(16,512),(16,256)):
        source=SNAPSHOT.read_text().replace("tile = min(32, query_batch_size)",f"tile = min({tile}, query_batch_size)")
        source=source.replace("preferred = min(cchunk, 1024)",f"preferred = min(cchunk, {chunk})")
        name=f"tile{tile}_chunk{chunk}"
        implementation=types.ModuleType("tensorvs.round2_"+name);implementation.__package__="tensorvs"
        exec(compile(source,str(SNAPSHOT),"exec"),implementation.__dict__)
        implementation._pq_block_eager=variants["deferred_bias"]._pq_block_eager
        variants[name]=implementation
    variants={name:variants[name] for name in args.variants}
    assert "current" in variants
    dataset=Path("/Users/yitongsong/Dataset/sift")
    index=load_ivfpq_index(dataset/"GPUIndex/TenVS_IVFPQResidual_nlist1024_m16_nbits8_train256.bin")
    index.reside_on_device("mps",backend="torch",persist_layout=False)
    queries=load_dataset(dataset/"sift_query.fvecs")[:512].to("mps")
    truth=load_ivecs(dataset/"sift_groundtruth.ivecs")[:512]
    original=pq.search_torch;records=[]
    try:
        with torch.inference_mode():
            for batch_size,probe in ((256,5),(256,50),(256,150),(512,5),(512,150)):
                batches=list(queries.split(batch_size))
                kwargs=dict(device="mps",backend="torch",use_compile=False,persist_layout=False,lut_dtype="fp16",
                    torch_query_batch_size=batch_size,torch_candidate_chunk_size=2048,torch_memory_budget_mb=128,
                    topk=10,nprobe=probe)
                def sweep(function,retain=False):
                    pq.search_torch=function
                    elapsed=0.;saved=[]
                    for batch in batches:
                        torch.mps.synchronize();begin=time.perf_counter()
                        values,ids=pq.search_ivfpq_index(index,batch,**kwargs)
                        torch.mps.synchronize();elapsed+=time.perf_counter()-begin
                        if retain:saved.append((values.cpu(),ids.cpu()))
                    return elapsed,(torch.cat([s[0] for s in saved]),torch.cat([s[1] for s in saved])) if retain else None
                for implementation in variants.values():
                    for _ in range(2):sweep(implementation.search_torch)
                samples={name:[] for name in variants};results={}
                for repetition in range(3):
                    names=list(variants)
                    if repetition%2:names.reverse()
                    for name in names:
                        elapsed,result=sweep(variants[name].search_torch,retain=repetition==2)
                        samples[name].append(elapsed)
                        if result is not None:results[name]=result
                reference=results["current"]
                validation={}
                for name,(values,ids) in results.items():
                    torch.testing.assert_close(values,reference[0],atol=.0625,rtol=2e-5)
                    assert torch.isfinite(values).all() and ((ids>=0)&(ids<1000000)).all()
                    assert all(len(set(row.tolist()))==10 for row in ids)
                    recall=float(compute_recall_at_k(ids,truth,10))
                    assert abs(recall-float(compute_recall_at_k(reference[1],truth,10)))<=.001
                    validation[name]=dict(recall=recall,max_distance_difference=float((values-reference[0]).abs().max()),
                        neighbor_set_overlap=sum(len(set(a.tolist())&set(b.tolist()))/10 for a,b in zip(ids,reference[1]))/len(ids))
                record=dict(q_batch=batch_size,nprobes=probe,query_count=512,samples_seconds=samples,
                    qps={name:512/statistics.median(s) for name,s in samples.items()},validation=validation)
                records.append(record);print(json.dumps(record),flush=True)
    finally:pq.search_torch=original
    args.output.write_text(json.dumps(dict(
        device="mps",cpu_fallback=False,baseline_sha256=hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),records=records),indent=2)+"\n")


if __name__=="__main__":main()
