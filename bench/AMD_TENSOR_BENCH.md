# AMD ROCm TensorVS experiment

Use the existing indexes in `/root/vector_data`. Each command runs one dataset
and index type in a fresh process so the GPU memory measurements are isolated.
TensorVS always uses `backend="torch"` and `use_compile=False` in this script.

```bash
cd /root/TensorVS
PY=/root/tensorvs-venv/bin/python
$PY -u bench/amd_tensor_bench.py --dataset glove1.2m --kind ivfflat
$PY -u bench/amd_tensor_bench.py --dataset glove1.2m --kind ivfpq
$PY -u bench/amd_tensor_bench.py --dataset hunyuan --kind ivfflat
$PY -u bench/amd_tensor_bench.py --dataset hunyuan --kind ivfpq
```

The defaults follow the CUDA figure sweep: `K=10`, query batches `32 128 512`,
up to 1024 queries (1000 in these files), two warmup sweeps, three timed
repetitions, and the dataset's predefined nprobe values. For a short check,
append `--nq 8 --q-batches 4 --nprobes 50 --warmup 0 --repetitions 1`
(use nprobe 350 for Hunyuan) and specify a separate `--output-dir`.

Results are written to `result/` as `AMD_TensorVS_Tensor_IVFFLAT_<dataset>_nlist1024.csv` or
`AMD_TensorVS_Tensor_IVFPQ_<dataset>_nlist1024_M<m>_nbits8_train<label>.csv`, with a JSON file
of the same stem. The CSV has exactly these columns, in this order:

`nprobes,q_batch,K,qps,recall,visited_vectors_per_query,peak_gpu_memory`

`qps` is the number of real queries divided by the median synchronized
full-query search time. `recall` is strict average recall@K against the
corresponding ground-truth file. `visited_vectors_per_query` uses the same
coarse-list accounting helper as the CUDA benchmark. `peak_gpu_memory` is
the sampled current-process VRAM peak in MiB reported by `amd-smi` during a
separate search sweep. It includes the resident index and query batches.
The JSON includes individual time samples, memory sample count, and PyTorch
allocator peaks. AMD-SMI's sampled process VRAM and the older MPS driver
allocation metric have different measurement APIs, so compare their absolute
values with that distinction in mind.

The GloVe IVF-PQ result filename uses the old benchmark's `train512` label;
the existing source index filename ends in `train256`. The JSON records the
exact source index path and metadata.

The script refuses to replace an existing result by default. Use `--resume`
after an interruption with the same options, or `--overwrite` to rerun a
result from the beginning.
