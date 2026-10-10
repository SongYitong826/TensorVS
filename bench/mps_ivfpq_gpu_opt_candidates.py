"""Compare portable, metadata-adaptive GPU PQ scan/selection candidates."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="0"
import torch
from tensorvs import load_dataset,load_ivecs,load_ivfpq_index,compute_recall_at_k
import tensorvs.index_ivfpq as pq

SNAPSHOT=Path("/tmp/tensorvs_ivfpq_before_padding_optimization.py")


def mapped_positions(delta0,delta1,boundaries,total,ordinals,grain):
    bins=ordinals//grain
    positions=ordinals[None,:]+torch.where(ordinals[None,:]>=boundaries[:,bins],delta1[:,bins],delta0[:,bins])
    return torch.where(ordinals[None,:]<total[:,None],positions,0)


def mapped_scores(luts,data,corrections,delta0,delta1,boundaries,bias0,bias1,total,ordinals,grain):
    bins=ordinals//grain
    next_list=ordinals[None,:]>=boundaries[:,bins]
    positions=ordinals[None,:]+torch.where(next_list,delta1[:,bins],delta0[:,bins])
    valid=ordinals[None,:]<total[:,None]
    positions=torch.where(valid,positions,0)
    distances=torch.zeros_like(positions,dtype=torch.float32)
    for subspace in range(data.shape[1]):
        codes=data[positions,subspace:subspace+1].to(torch.int64)
        if data.dtype==torch.int16:codes=codes&65535
        values=luts[:,subspace:subspace+1].gather(2,codes.transpose(1,2))
        distances=distances+values.to(torch.float32).sum(1)
    distances=(distances+corrections[positions]+torch.where(next_list,bias1[:,bins],bias0[:,bins])).clamp_min_(0.)
    return distances.masked_fill(~valid,float("inf"))


def selected_ids(ids,delta0,delta1,boundaries,total,ordinals,grain,values):
    ordinals=torch.where(torch.isfinite(values),ordinals,0)
    bins=ordinals//grain
    positions=ordinals+torch.where(ordinals>=boundaries.gather(1,bins),delta1.gather(1,bins),delta0.gather(1,bins))
    positions=torch.where(torch.isfinite(values),positions,0)
    return ids[positions].masked_fill(~torch.isfinite(values),-1)


def loader(name):
    spec=importlib.util.spec_from_file_location("tensorvs."+name,SNAPSHOT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def make_search(module,mode,block_width=2048):
    def search(batch,layout,lut,bias,probes,best,best_ids,budget,chunk,timer,use_compile,query_tile=None):
        positive=[c for c in layout.host_counts if c>0]
        grain=1<<min(10,min(positive).bit_length()-1)
        if grain<32 or not use_compile:
            return module._original_packed(batch,layout,lut,bias,probes,best,best_ids,budget,chunk,timer,use_compile,query_tile)
        counts=layout.lists.counts[probes]
        if 0 in layout.host_counts:
            order=counts.argsort(1,descending=True)
            probes=probes.gather(1,order);counts=counts.gather(1,order)
        prefix=counts.cumsum(1)
        previous=torch.cat((torch.zeros_like(prefix[:,:1]),prefix[:,:-1]),1)
        selected_bias=bias.gather(1,probes)
        deltas=layout.lists.offsets[probes]-previous
        upper=layout.pq_scan_upper_bounds[probes.shape[1]]
        size=upper if mode in ("global", "exact") else math.ceil(upper/block_width)*block_width if mode=="blocked" else math.ceil(upper/chunk)*chunk
        starts=torch.arange(math.ceil(size/grain),device=batch.device)*grain
        ranks=torch.searchsorted(prefix,starts[None,:].expand(prefix.shape[0],-1).contiguous(),right=True).clamp_max(probes.shape[1]-1)
        following=(ranks+1).clamp_max(probes.shape[1]-1)
        mapping=(deltas.gather(1,ranks),deltas.gather(1,following),prefix.gather(1,ranks),
                 selected_bias.gather(1,ranks),selected_bias.gather(1,following),prefix[:,-1])
        if mode=="stream":
            width=query_tile
        else:
            partial=0 if mode=="global" else 48*best.shape[1]*math.ceil(size/block_width)
            available=budget*7//8-8*size
            capacity=min(batch.shape[0],max(1,available//(8*size+partial)))
            width=1<<(capacity.bit_length()-1)
        for qs in range(0,batch.shape[0],width):
            qe=min(qs+width,batch.shape[0]);tables=lut[qs:qe];plan=tuple(m[qs:qe] for m in mapping)
            key=(mode,qe-qs,size,chunk,block_width,grain,layout.lists.data.dtype,lut.dtype,best.shape[1])
            scores=module._compute(mapped_scores,True,key)
            select=module._compute(selected_ids,True,key)
            if mode=="stream":
                ordinals=torch.arange(chunk,device=batch.device)
                def scan(start):
                    values=scores(tables,layout.lists.data,layout.lists.auxiliary,*plan,ordinals+start,grain)
                    values,order=values.topk(min(best.shape[1],chunk),largest=False,sorted=False)
                    ids=select(layout.lists.ids,*plan[:3],plan[-1],order+start,grain,values)
                    return values,ids
                values,ids=module._chunked_topk(scan,size,chunk,best[qs:qe],best_ids[qs:qe],budget,timer,
                                              "candidate_scan",True,key)
            else:
                ordinals=torch.arange(size,device=batch.device)
                distances=scores(tables,layout.lists.data,layout.lists.auxiliary,*plan,ordinals,grain)
                if mode=="global":
                    values,order=distances.topk(min(best.shape[1],size),largest=False,sorted=False)
                else:
                    head=size//block_width*block_width
                    local_values,local_order=distances[:,:head].reshape(qe-qs,-1,block_width).topk(min(best.shape[1],block_width),largest=False,sorted=False)
                    offsets=torch.arange(head//block_width,device=batch.device)[None,:,None]*block_width
                    local_values=local_values.flatten(1);local_order=(local_order+offsets).flatten(1)
                    if head<size:
                        tail_values,tail_order=distances[:,head:].topk(min(best.shape[1],size-head),largest=False,sorted=False)
                        local_values=torch.cat((local_values,tail_values),1)
                        local_order=torch.cat((local_order,tail_order+head),1)
                    values,which=local_values.topk(min(best.shape[1],size),largest=False,sorted=False)
                    order=local_order.gather(1,which)
                ids=select(layout.lists.ids,*plan[:3],plan[-1],order,grain,values)
                if values.shape[1]<best.shape[1]:
                    values,ids=module._merge_rows(best[qs:qe],best_ids[qs:qe],values,ids)
            best[qs:qe],best_ids[qs:qe]=values,ids
        return best,best_ids
    return search


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants",nargs="+",default=["baseline","anchors","global","blocked2048","blocked8192"])
    parser.add_argument("--nq",type=int,default=512)
    parser.add_argument("--q-batches",type=int,nargs="+",default=[256])
    parser.add_argument("--nprobes",type=int,nargs="+",default=[5,50,150])
    parser.add_argument("--output",type=Path,default=Path("/tmp/tensorvs_ivfpq_gpu_opt_candidates.json"))
    args=parser.parse_args()
    variants={}
    for name in args.variants:
        if name=="adaptive":
            module=importlib.import_module("tensorvs._torch_search")
            variants[name]=module
            continue
        module=loader("opt_"+name)
        module._original_packed=module._packed_pq_search
        if name!="baseline":
            mode="stream" if name=="anchors" else "exact" if name.endswith("exact") else "blocked" if name.startswith("blocked") else "global"
            block_width=int(name[7:].removesuffix("exact")) if name.startswith("blocked") else 2048
            module._packed_pq_search=make_search(module,mode,block_width)
        variants[name]=module
    dataset=Path("/Users/yitongsong/Dataset/sift")
    index=load_ivfpq_index(dataset/"GPUIndex/TenVS_IVFPQResidual_nlist1024_m16_nbits8_train256.bin")
    index.reside_on_device("mps",backend="torch",persist_layout=False)
    queries=load_dataset(dataset/"sift_query.fvecs")[:args.nq].to("mps")
    truth=load_ivecs(dataset/"sift_groundtruth.ivecs")[:args.nq]
    original=pq.search_torch;records=[]
    try:
        with torch.inference_mode():
            for batch_size in args.q_batches:
                batches=list(queries.split(batch_size))
                for probe in args.nprobes:
                    kwargs=dict(device="mps",backend="torch",use_compile=True,torch_pq_scan="gpu",lut_dtype="fp16",persist_layout=False,
                        topk=10,nprobe=probe,torch_query_batch_size=batch_size,torch_memory_budget_mb=128,torch_candidate_chunk_size=2048)
                    def sweep(name,retain=False):
                        pq.search_torch=variants[name].search_torch
                        saved=[];elapsed=0.
                        for batch in batches:
                            torch.mps.synchronize();begin=time.perf_counter()
                            values,ids=pq.search_ivfpq_index(index,batch,**kwargs)
                            torch.mps.synchronize();elapsed+=time.perf_counter()-begin
                            if retain:saved.append((values.cpu(),ids.cpu()))
                        return elapsed,(torch.cat([s[0] for s in saved]),torch.cat([s[1] for s in saved])) if retain else None
                    for name in variants:
                        print(f"warmup {batch_size=} {probe=} {name}",flush=True)
                        for _ in range(2):sweep(name)
                    samples={name:[] for name in variants};outputs={}
                    for rep in range(3):
                        names=list(variants)
                        if rep%2:names.reverse()
                        for name in names:
                            elapsed,result=sweep(name,rep==2);samples[name].append(elapsed)
                            if result is not None:outputs[name]=result
                    base=outputs["baseline"];validation={}
                    for name,(values,ids) in outputs.items():
                        print(f"validating {name}: first={values[0].tolist()}",flush=True)
                        torch.testing.assert_close(values,base[0],atol=.0625,rtol=2e-5)
                        assert torch.isfinite(values).all() and (ids>=0).all()
                        recall=float(compute_recall_at_k(ids,truth,10))
                        assert abs(recall-float(compute_recall_at_k(base[1],truth,10)))<=.001
                        assert all(len(set(row.tolist()))==10 for row in ids)
                        validation[name]=dict(recall=recall,neighbor_overlap=sum(len(set(a.tolist())&set(b.tolist())) for a,b in zip(ids,base[1]))/(10*len(ids)))
                    record=dict(q_batch=batch_size,nprobes=probe,samples_seconds=samples,
                        qps={name:args.nq/statistics.median(s) for name,s in samples.items()},validation=validation)
                    records.append(record);print(json.dumps(record),flush=True)
                    args.output.write_text(json.dumps(dict(status="running",records=records),indent=2)+"\n")
    finally:pq.search_torch=original
    args.output.write_text(json.dumps(dict(status="complete",baseline_sha256=hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),records=records),indent=2)+"\n")


if __name__=="__main__":main()
