# SIFT100M on three NVIDIA GPUs

Run from `/home/songyitong/TensorVS`, using Conda environment `tensorvs`:

```bash
bash bench/run_sift100m_multigpu_bench.sh --kind ivfflat --phase all --gpus 0 1 2
bash bench/run_sift100m_multigpu_bench.sh --kind ivfpq --phase all --gpus 0 1 2
```

Run these methods sequentially. `--phase build` only constructs the indexes;
`--phase query` reuses the saved indexes and original build measurements.
`--resume` verifies sources, data fingerprints, build parameters, index files,
and result configuration before reusing indexes or skipping completed rows.
Existing files are never silently overwritten with new measurements. Use a
new `--index-dir` for changed build parameters and `--output-dir` for a new
query run. Interrupted `.partial.bin` files are never treated as complete.

## Dataset and index semantics

Default input: `/storage/vector_data/sift100M/{base,query,groundtruth}.bin`.
Each file starts with two little-endian uint32 values `(rows, columns)`.
Base/query payloads are uint8, ground-truth IDs are uint32. The expected base
shape is 100,000,000 x 128. Only each worker's contiguous third is converted
to FP32. Queries use the first min(1024, actual query count), currently 1000.

Each GPU independently trains and searches one contiguous shard, with
`nlist=1024` **per shard**, totaling 3072 IVF lists. PQ codebooks are also
independently trained. `nprobe` applies to each shard. Local IDs are converted
to original global row IDs. The coordinator merges the three local Top-K
sets by squared-L2 distance into one global Top-K. This is a sharded index,
not a shared-codebook global-nlist=1024 index; comparisons must retain this
distinction.

Defaults match the existing SIFT10M figure scripts: K=10,
QBatch=32/128/512, nprobe=5/10/15/20/30/40/50/60/70/80/90/100/120/150,
two warmup sweeps, three timing sweeps. Flat uses 25 Lloyd iterations, one
restart, and at most 256 training points per centroid. PQ uses M=16, nbits=8,
25 balanced hierarchical coarse iterations, 50% of each shard for coarse
training, 50 balanced PQ iterations, two PQ restarts and at most 512 training
points per PQ centroid. Seed=1234; LUT=FP16. Default search backend is Triton;
`--backend torch` selects the portable Tensor implementation. CPU thread
count defaults to four per worker. Explicit flags can override the defaults.

The existing PQ figure filenames use the legacy `train256` label despite
actually requesting 512 points per centroid. This new runner uses the actual
`train512` value in its filenames and JSON.

The optional `cpu_residual_chunk_size=65536` bounds the host centroid gather
without changing residual values or training parameters. GPU training runs
concurrently on all three workers. CPU sorting/interleaving/serialization runs
one shard at a time to avoid simultaneous full-size host copies.

## Files and metrics

Indexes are stored under:

```text
/storage/vector_data/sift100M/GPUIndex/TensorVS_MultiGPU3/
  CUDA_TensorVS_MultiGPU3_IVFFLAT_sift100m_nlist1024_build/
    shard0of3.bin, shard1of3.bin, shard2of3.bin
    shard0of3.json, shard1of3.json, shard2of3.json, build.json
  CUDA_TensorVS_MultiGPU3_IVFPQ_sift100m_nlist1024_M16_nbits8_train512_build/
    shard0of3.bin, shard1of3.bin, shard2of3.bin
    shard0of3.json, shard1of3.json, shard2of3.json, build.json
```

Query CSV/JSON pairs:

```text
result/CUDA_TensorVS_Triton_MultiGPU3_IVFFLAT_sift100m_nlist1024.{csv,json}
result/CUDA_TensorVS_Triton_MultiGPU3_IVFPQ_sift100m_nlist1024_M16_nbits8_train512.{csv,json}
```

CSV has the same seven columns as AMD results:
`nprobes,q_batch,K,qps,recall,visited_vectors_per_query,peak_gpu_memory`.
QPS is the median end-to-end full-query throughput, including coordinator IPC,
GPU computation, local results copied to CPU, and CPU global Top-K merge.
Query upload, recall computation, visit accounting and the separate memory
sweep are excluded. Padded final batches are searched but only valid queries
count toward QPS/recall. Visited vectors are summed across all shards.

`peak_gpu_memory` is **MiB**, the maximum sampled simultaneous sum of the
three worker processes' NVML memory in a separate query sweep. Other users'
processes are excluded. JSON additionally retains per-GPU sampled peaks,
Torch allocated/reserved peaks, raw timing samples, each shard's visit
statistics, source hashes, data/index fingerprints and actual devices.
Sampled peaks need not equal the instantaneous hardware maximum.

Build CSV/JSON pairs:

```text
result/Build/CUDA_TensorVS_MultiGPU3_IVFFLAT_sift100m_nlist1024_build.{csv,json}
result/Build/CUDA_TensorVS_MultiGPU3_IVFPQ_sift100m_nlist1024_M16_nbits8_train512_build.{csv,json}
```

Build CSV follows the AMD build schema:
`dataset,method,nlist,n_vectors,dim,build_seconds,save_seconds,index_size_bytes,index_size_mib,peak_build_gpu_memory_mib`.
`build_seconds` is parallel wall time for training plus assignment/encoding;
input conversion and serialization are excluded. `save_seconds` measures the
wall time for the three serialized sorting/interleaving/write operations.
Index size sums the three `.bin` files. Build peak memory includes construction
and serialization. JSON stores individual shard build/read/save timings and
memory metrics; `nlist` in the CSV means **per shard**, while JSON records both
per-shard and global list counts. Reused indexes retain their original build
metrics rather than reporting zero or counting load time as build time.

Logs for the full sequential experiment go under `result/logs/`.
Do not plot a partial CSV as a completed curve; check JSON `status=complete`
and the expected 42 rows per method before using results.
