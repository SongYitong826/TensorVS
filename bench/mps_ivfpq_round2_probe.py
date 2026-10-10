"""Compare portable tensor-only PQ lookup layouts before changing search."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="0"
import torch
import tensorvs._torch_search as native

ROOT=Path(__file__).resolve().parents[1]


def finish(distances,ids,corrections,probe_bias,query_ids,list_ids,starts,counts,
           positions,offsets,k):
    auxiliary=corrections.index_select(0,positions.flatten()).reshape_as(positions)
    bias_positions=query_ids*probe_bias.shape[1]+list_ids[:,None]
    biases=probe_bias.flatten().index_select(0,bias_positions.flatten()).reshape_as(query_ids)
    distances.add_(auxiliary[:,None,:]).add_(biases[:,:,None]).clamp_min_(0.)
    valid=offsets[None,:]<counts[:,None]
    distances.masked_fill_(~valid[:,None,:],float("inf"))
    values,order=distances.topk(k,dim=-1,largest=False,sorted=False)
    selected_positions=(starts[:,None,None]+order).clamp_max(ids.shape[0]-1)
    selected=ids.index_select(0,selected_positions.flatten()).reshape_as(order)
    return values,selected.masked_fill(~torch.isfinite(values),-1)


def candidate_major(luts,data,ids,corrections,bias,query_ids,list_ids,starts,counts,valid,offsets,k,chunk):
    positions=(starts[:,None]+offsets[None,:]).clamp_max(data.shape[0]-1)
    codes=data.index_select(0,positions.flatten()).reshape(*positions.shape,data.shape[1]).to(torch.int32)
    if data.dtype==torch.int16:codes=codes&65535
    query_offsets=query_ids.to(torch.int32)[:,:,None,None]*(luts.shape[1]*luts.shape[2])
    distances=None
    for start in range(0,data.shape[1],chunk):
        end=min(start+chunk,data.shape[1])
        subspace=torch.arange(start,end,device=luts.device,dtype=torch.int32)*luts.shape[2]
        indices=query_offsets+subspace[None,None,None,:]+codes[:,None,:,start:end]
        values=luts.flatten().index_select(0,indices.flatten()).reshape_as(indices)
        summed=values.sum(-1,dtype=torch.float32)
        if distances is None:distances=summed
        else:distances.add_(summed)
        del indices,values,summed
    distances.masked_fill_(~valid[:,:,None],float("inf"))
    return finish(distances,ids,corrections,bias,query_ids,list_ids,starts,counts,positions,offsets,k)


def flat_gather(luts,data,ids,corrections,bias,query_ids,list_ids,starts,counts,valid,offsets,k,chunk):
    positions=(starts[:,None]+offsets[None,:]).clamp_max(data.shape[0]-1)
    codes=data.index_select(0,positions.flatten()).reshape(*positions.shape,data.shape[1]).long()
    if data.dtype==torch.int16:codes=codes&65535
    task_tables=luts.index_select(0,query_ids.flatten()).reshape(*query_ids.shape,-1)
    subspace=torch.arange(data.shape[1],device=luts.device)*luts.shape[2]
    indices=(codes+subspace[None,None,:]).flatten(1)[:,None,:].expand(-1,query_ids.shape[1],-1)
    values=task_tables.gather(2,indices).reshape(*query_ids.shape,positions.shape[1],data.shape[1])
    distances=values.sum(-1,dtype=torch.float32)
    distances.masked_fill_(~valid[:,:,None],float("inf"))
    return finish(distances,ids,corrections,bias,query_ids,list_ids,starts,counts,positions,offsets,k)


def channel_gather(luts,data,ids,corrections,bias,query_ids,list_ids,starts,counts,valid,offsets,k,chunk):
    positions=(starts[:,None]+offsets[None,:]).clamp_max(data.shape[0]-1)
    codes=data.index_select(0,positions.flatten()).reshape(*positions.shape,data.shape[1]).long()
    if data.dtype==torch.int16:codes=codes&65535
    tables=luts.index_select(0,query_ids.flatten()).reshape(*query_ids.shape,-1).transpose(1,2).contiguous()
    subspace=torch.arange(data.shape[1],device=luts.device)*luts.shape[2]
    indices=(codes+subspace[None,None,:]).flatten(1)[:,:,None].expand(-1,-1,query_ids.shape[1])
    values=tables.gather(1,indices).reshape(positions.shape[0],positions.shape[1],data.shape[1],query_ids.shape[1])
    distances=values.sum(2,dtype=torch.float32).transpose(1,2).contiguous()
    distances.masked_fill_(~valid[:,:,None],float("inf"))
    return finish(distances,ids,corrections,bias,query_ids,list_ids,starts,counts,positions,offsets,k)


def dense_onehot(luts,data,ids,corrections,bias,query_ids,list_ids,starts,counts,valid,offsets,k,chunk):
    positions=(starts[:,None]+offsets[None,:]).clamp_max(data.shape[0]-1)
    codes=data.index_select(0,positions.flatten()).reshape(*positions.shape,data.shape[1]).long()
    if data.dtype==torch.int16:codes=codes&65535
    tables=luts.index_select(0,query_ids.flatten()).reshape(*query_ids.shape,-1).float()
    subspace=torch.arange(data.shape[1],device=luts.device)*luts.shape[2]
    indices=codes+subspace[None,None,:]
    indicators=torch.zeros((*positions.shape,luts.shape[1]*luts.shape[2]),device=luts.device)
    indicators.scatter_(2,indices,1.)
    distances=torch.bmm(tables,indicators.transpose(1,2))
    distances.masked_fill_(~valid[:,:,None],float("inf"))
    return finish(distances,ids,corrections,bias,query_ids,list_ids,starts,counts,positions,offsets,k)


def embedding_bag(luts,data,ids,corrections,bias,query_ids,list_ids,starts,counts,valid,offsets,k,chunk):
    positions=(starts[:,None]+offsets[None,:]).clamp_max(data.shape[0]-1)
    dtype=torch.int32 if luts.numel()<=2147483647 else torch.int64
    codes=data.index_select(0,positions.flatten()).reshape(*positions.shape,data.shape[1]).to(dtype)
    if data.dtype==torch.int16:codes=codes&65535
    query_offsets=query_ids.to(dtype)[:,:,None,None]*(luts.shape[1]*luts.shape[2])
    weights=luts.float().reshape(-1,1)
    distances=None
    for start in range(0,data.shape[1],chunk):
        end=min(start+chunk,data.shape[1])
        subspace=torch.arange(start,end,device=luts.device,dtype=dtype)*luts.shape[2]
        indices=query_offsets+subspace[None,None,None,:]+codes[:,None,:,start:end]
        summed=torch.nn.functional.embedding_bag(indices.reshape(-1,end-start),weights,mode="sum").reshape(
            *query_ids.shape,positions.shape[1])
        if distances is None:distances=summed
        else:distances.add_(summed)
        del indices,summed
    distances.masked_fill_(~valid[:,:,None],float("inf"))
    return finish(distances,ids,corrections,bias,query_ids,list_ids,starts,counts,positions,offsets,k)


VARIANTS={"current":native._pq_block_eager,"candidate_major":candidate_major,
          "flat_gather":flat_gather,"channel_gather":channel_gather,"dense_onehot":dense_onehot,
          "embedding_bag":embedding_bag}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants",nargs="+",default=list(VARIANTS))
    parser.add_argument("--output",type=Path,default=ROOT/"results/ivfpq_torch_round2/block_layouts.json")
    args=parser.parse_args()
    generator=torch.Generator().manual_seed(614)
    records=[]
    with torch.inference_mode():
        for width in (1,8,32):
            tasks,candidates,queries,m,ksub=4,512,512,16,256
            luts=torch.randn(queries,m,ksub,generator=generator).to("mps",dtype=torch.float16)
            data=torch.randint(0,ksub,(tasks*candidates,m),generator=generator,dtype=torch.uint8).to("mps")
            ids=torch.arange(tasks*candidates,device="mps")
            corrections=torch.randn(tasks*candidates,generator=generator).to("mps")
            bias=(20+torch.randn(queries,tasks,generator=generator)).to("mps")
            query_ids=torch.randint(0,queries,(tasks,width),generator=generator).to("mps")
            lists=torch.arange(tasks,device="mps")
            counts=torch.full((tasks,),candidates,device="mps",dtype=torch.int64)
            valid=torch.ones((tasks,width),device="mps",dtype=torch.bool)
            offsets=torch.arange(candidates,device="mps")
            arguments=(luts,data,ids,corrections,bias,query_ids,lists,lists*candidates,counts,valid,offsets,10,16)
            expected=native._pq_block_eager(*arguments)
            for name in args.variants:
                function=VARIANTS[name]
                for _ in range(12):actual=function(*arguments)
                torch.mps.synchronize()
                samples=[]
                for _ in range(21):
                    begin=time.perf_counter();actual=function(*arguments);torch.mps.synchronize()
                    samples.append(time.perf_counter()-begin)
                # Each block uses the same FP16 LUT and FP32 accumulation.
                torch.testing.assert_close(actual[0].sort(-1).values,expected[0].sort(-1).values,atol=1e-4,rtol=1e-5)
                record=dict(query_width=width,variant=name,median_seconds=statistics.median(samples),samples_seconds=samples)
                records.append(record);print(json.dumps(record),flush=True)
    args.output.write_text(json.dumps(dict(device="mps",cpu_fallback=False,records=records),indent=2)+"\n")


if __name__=="__main__":main()
