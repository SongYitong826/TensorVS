# TensorVS Triton experiments on AMD MI300X

The Triton benchmark uses the same existing TensorVS indexes, query vectors,
ground truth, query batches, nprobe grids, and seven CSV metrics as the Tensor
benchmark. It runs `backend="triton"` and saves results alongside the Tensor
results under `/root/TensorVS/result/`. No index rebuild is required.

Run each dataset and index kind in a separate process:

```bash
cd /root/TensorVS
PY=/root/tensorvs-venv/bin/python
$PY -u bench/amd_triton_bench.py --dataset glove1.2m --kind ivfflat
$PY -u bench/amd_triton_bench.py --dataset glove1.2m --kind ivfpq
$PY -u bench/amd_triton_bench.py --dataset hunyuan --kind ivfflat
$PY -u bench/amd_triton_bench.py --dataset hunyuan --kind ivfpq
```

The defaults are K=10, NQ=1000 for these datasets, QBatch=32/128/512,
two warmups, three synchronized timed repetitions, and the same nprobe
grids as the Tensor results. Output names begin with
`TensorVS_Triton_IVFFLAT_` or `TensorVS_Triton_IVFPQ_`. The pre-existing
Tensor route results begin with `TensorVS_Tensor_`. Each CSV has a JSON
sidecar and exactly these columns:

```
nprobes,q_batch,K,qps,recall,visited_vectors_per_query,peak_gpu_memory
```

GPU memory is sampled for the current process using AMD-SMI in a separate
search sweep, as in the Tensor benchmark. The script writes each completed
row immediately. Add `--resume` to the same command after an interruption;
use `--overwrite` only to replace an existing result.

Small smoke tests completed for both IVF kinds on GloVe and Hunyuan, and
the Hunyuan M384 IVF-PQ and IVF-Flat paths compiled at QBatch=512,
nprobe=1000. Tensor and Triton recall matched on the tested query subsets.
The full sweep has not been run by this setup step.
