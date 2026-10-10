"""Check existing torch.compile tensor expressions on the actual MPS PQ index."""
import json
import os
from pathlib import Path
import statistics
import time
import traceback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="0"
import torch
from tensorvs import load_dataset,load_ivfpq_index,search_ivfpq_index

ROOT=Path(__file__).resolve().parents[1]
dataset=Path("/Users/yitongsong/Dataset/sift")
report=dict(device="mps",custom_kernel_source=False,cpu_fallback=False,records=[])
try:
    index=load_ivfpq_index(dataset/"GPUIndex/TenVS_IVFPQResidual_nlist1024_m16_nbits8_train256.bin")
    index.reside_on_device("mps",backend="torch",persist_layout=False)
    q=load_dataset(dataset/"sift_query.fvecs")[:512].to("mps")
    with torch.inference_mode():
        for probe in (5,50,150):
            kwargs=dict(device="mps",backend="torch",persist_layout=False,lut_dtype="fp16",topk=10,
                nprobe=probe,torch_query_batch_size=256,torch_candidate_chunk_size=2048,torch_memory_budget_mb=128)
            before=search_ivfpq_index(index,q,**kwargs)
            times={}
            for compiled in (False,True):
                for _ in range(2):
                    actual=search_ivfpq_index(index,q,use_compile=compiled,**kwargs)
                torch.mps.synchronize()
                samples=[]
                for _ in range(3):
                    begin=time.perf_counter()
                    actual=search_ivfpq_index(index,q,use_compile=compiled,**kwargs)
                    torch.mps.synchronize();samples.append(time.perf_counter()-begin)
                torch.testing.assert_close(actual[0],before[0],atol=.0625,rtol=2e-5)
                times["compiled" if compiled else "eager"]=samples
            record=dict(nprobes=probe,samples_seconds=times,qps={name:512/statistics.median(s) for name,s in times.items()})
            report["records"].append(record);print(json.dumps(record),flush=True)
    report["status"]="complete"
except Exception as error:
    report.update(status="failed",error=repr(error));traceback.print_exc()
(ROOT/"results/ivfpq_torch_round2/compiled_search_check.json").write_text(json.dumps(report,indent=2)+"\n")
