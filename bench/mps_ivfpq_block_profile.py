"""Separate operator groups using the saved baseline eager PQ source snapshot."""
import json
import importlib.util
import os
from pathlib import Path
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="0"
import torch
from tensorvs import load_dataset,load_ivfpq_index,search_ivfpq_index
import tensorvs.index_ivfpq as pq_module

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("tensorvs._torch_search_profile_baseline",
    ROOT/"results/ivfpq_torch_optimization/baseline_torch_search.py")
baseline=importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
profile={}


def phase(name,fn):
    torch.mps.synchronize()
    begin=time.perf_counter()
    result=fn()
    torch.mps.synchronize()
    profile[name]=profile.get(name,0.)+time.perf_counter()-begin
    return result


def block(luts,data,ids,corrections,probe_bias,query_ids,list_ids,
          starts,counts,valid_queries,candidate_offsets,k,subspace_chunk):
    def prepare():
        positions=starts[:,None]+candidate_offsets[None,:]
        return candidate_offsets[None,:]<counts[:,None],positions.clamp_max(data.shape[0]-1)
    valid,safe=phase("candidate_positions",prepare)
    codes=phase("pq_code_read_and_int64_conversion",lambda:data[safe].to(torch.int64))
    if data.dtype==torch.int16:codes=codes&65535
    tables=phase("query_lut_read",lambda:luts[query_ids])
    distances=torch.zeros((query_ids.shape[0],query_ids.shape[1],candidate_offsets.shape[0]),
                          device=luts.device,dtype=torch.float32)
    for subspace in range(0,data.shape[1],subspace_chunk):
        indices=codes[:,:,subspace:subspace+subspace_chunk].transpose(1,2)[:,None,:,:].expand(-1,query_ids.shape[1],-1,-1)
        values=phase("lut_gather",lambda:tables[:,:,subspace:subspace+subspace_chunk,:].gather(3,indices))
        distances=phase("fp32_conversion_sum_accumulation",lambda:distances+values.to(torch.float32).sum(2))
    def correct():
        result=(distances+corrections[safe][:,None,:]+probe_bias[query_ids,list_ids[:,None]][:,:,None]).clamp_min_(0.)
        result.masked_fill_(~(valid[:,None,:]&valid_queries[:,:,None]),float("inf"))
        return result
    distances=phase("corrections_bias_mask",correct)
    values,order=phase("block_topk",lambda:distances.topk(k,dim=-1,largest=False,sorted=False))
    selected=phase("candidate_id_read",lambda:ids[safe].unsqueeze(1).expand(-1,query_ids.shape[1],-1).gather(2,order))
    return values,selected.masked_fill(~torch.isfinite(values),-1)


dataset=Path("/Users/yitongsong/Dataset/sift")
index=load_ivfpq_index(dataset/"GPUIndex/TenVS_IVFPQResidual_nlist1024_m16_nbits8_train256.bin")
index.reside_on_device("mps",backend="torch",persist_layout=False)
q=load_dataset(dataset/"sift_query.fvecs")[:256].to("mps")
original=baseline._pq_block
original_search=pq_module.search_torch
pq_module.search_torch=baseline.search_torch
records=[]
try:
    with torch.inference_mode():
        for probe in (5,50):
            kwargs=dict(device="mps",backend="torch",use_compile=False,persist_layout=False,
                        lut_dtype="fp16",torch_query_batch_size=256,nprobe=probe,topk=10)
            for _ in range(2):search_ivfpq_index(index,q,**kwargs)
            baseline._pq_block=block
            profile={}
            search_ivfpq_index(index,q,**kwargs)
            baseline._pq_block=original
            records.append(dict(q_batch=256,nprobes=probe,seconds=profile))
            print(json.dumps(records[-1]),flush=True)
finally:
    baseline._pq_block=original
    pq_module.search_torch=original_search
(ROOT/"results/ivfpq_torch_optimization/baseline_block_profile.json").write_text(json.dumps(dict(
    profile_limit="Each operator group synchronizes GPU; these times include diagnostic synchronization overhead.",records=records),indent=2)+"\n")
