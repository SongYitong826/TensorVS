"""Compare standard tensor ADC address widths and subspace tile sizes on MPS."""
import inspect
import json
import os
from pathlib import Path
import statistics
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="0"
import torch
import tensorvs._torch_search as native

ROOT=Path(__file__).resolve().parents[1]
source=inspect.getsource(native._pq_block_eager)
source64=source.replace("index_dtype = torch.int32 if luts.numel() <= 2147483647 else torch.int64",
                        "index_dtype = torch.int64")
namespace=dict(native.__dict__)
exec(source64,namespace)
fn64=namespace["_pq_block_eager"]
fn32=native._pq_block_eager
generator=torch.Generator().manual_seed(611)
records=[]
with torch.inference_mode():
    for width in (1,8,32):
        tasks,candidates,queries,m,ksub=16,1024,512,16,256
        luts=torch.randn(queries,m,ksub,generator=generator,dtype=torch.float32).to("mps",dtype=torch.float16)
        data=torch.randint(0,ksub,(tasks*candidates,m),generator=generator,dtype=torch.uint8).to("mps")
        ids=torch.arange(tasks*candidates,device="mps")
        corrections=torch.randn(tasks*candidates,generator=generator).to("mps")
        bias=torch.randn(queries,tasks,generator=generator).to("mps")
        query_ids=torch.randint(0,queries,(tasks,width),generator=generator).to("mps")
        lists=torch.arange(tasks,device="mps")
        starts=lists*candidates
        counts=torch.full((tasks,),candidates,device="mps",dtype=torch.int64)
        valid=torch.ones((tasks,width),device="mps",dtype=torch.bool)
        offsets=torch.arange(candidates,device="mps")
        args=(luts,data,ids,corrections,bias,query_ids,lists,starts,counts,valid,offsets,10)
        expected=native._pq_block_eager(*args,8)
        for label,fn in (("int64",fn64),("int32",fn32)):
            for chunk in (4,8,16):
                for _ in range(3):actual=fn(*args,chunk)
                torch.mps.synchronize()
                samples=[]
                for _ in range(9):
                    begin=time.perf_counter()
                    actual=fn(*args,chunk)
                    torch.mps.synchronize()
                    samples.append(time.perf_counter()-begin)
                torch.testing.assert_close(actual[0],expected[0],atol=1e-5,rtol=1e-5)
                record=dict(query_width=width,index_dtype=label,subspace_chunk=chunk,
                    median_seconds=statistics.median(samples),samples_seconds=samples)
                records.append(record)
                print(json.dumps(record),flush=True)
(ROOT/"results/ivfpq_torch_optimization/block_variants.json").write_text(json.dumps(dict(
    device="mps",cpu_fallback=False,operations="standard PyTorch tensor operators only",records=records),indent=2)+"\n")
