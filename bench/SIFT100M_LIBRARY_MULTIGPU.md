# cuVS and Faiss GPU: SIFT100M on three independent shards

The coordinator and common tests use `conda run -n tensorvs`. The installed
native libraries require Python 3.12, so isolated workers use the existing
`cuvs` and `faiss` environments. No packages are installed or replaced. The
user explicitly authorized this worker-only exception to the project's
Python environment rule in this session.

From `/home/songyitong/TensorVS`:

```bash
# One library/method, building then querying on GPUs 0, 1, 2.
bash bench/run_sift100m_library_bench.sh --library cuvs --kind ivfflat --phase all
bash bench/run_sift100m_library_bench.sh --library cuvs --kind ivfpq --phase all
bash bench/run_sift100m_library_bench.sh --library faiss --kind ivfflat --phase all
bash bench/run_sift100m_library_bench.sh --library faiss --kind ivfpq --phase all

# Sequential Flat/PQ runs. Start only the requested library.
bash bench/run_sift100m_library_suite.sh cuvs
bash bench/run_sift100m_library_suite.sh faiss

# Detached server execution, with durable logs.
bash bench/run_sift100m_library_suite.sh cuvs result/logs/my_cuvs_run --detach
```

`--phase build` only constructs/saves indexes; `--phase query` loads existing
indexes. `--resume` verifies original build/source/data/library provenance
before index reuse or skipping completed rows. The worker processes run on
one visible GPU each. All runners use the same GPU lock as the TensorVS
three-shard runner. Faiss explicitly sets `use_cuvs=False`, so it benchmarks
Faiss's own GPU implementation.

## Workload and training

Input is `/storage/vector_data/sift100M/{base,query,groundtruth}.bin`, using
the uint32 `(rows, columns)` header. Source vectors are uint8 and converted
to FP32 before native training/add, matching TensorVS vector precision.
Each of three contiguous shards has about 33.33M vectors and its own
quantizer/codebooks, with nlist=1024 **per shard** (global list count 3072).
Native local IDs are translated to original global IDs before global Top-K.
K=10; batches=32/128/512; up to 1024 queries (the existing files contain 1000);
nprobe=5/10/15/20/30/40/50/60/70/80/90/100/120/150 per shard. There are 42
measurements per library and method. Defaults are two warmup sweeps and three
timing sweeps. PQ uses M=16, nbits=8 and native FP32 query LUTs. Raw SIFT's
0-255 range can overflow cuVS FP16 LUTs (confirmed on a synthetic range
check), so FP16 is not silently substituted for the native FP32 default.
The optional FP16 setting uses distinct `_LUTfp16` query filenames and
explicitly fails on non-finite candidate scores. Current TensorVS results
used its own FP16 LUT implementation; precision settings remain explicit.

Native builders train fresh indexes. Both request 25 coarse iterations and
50% coarse training data. Faiss exposes separate PQ controls: 50 iterations,
two restarts and 512 training points per centroid. cuVS exposes a single
`kmeans_n_iters`, so its native coarse/PQ iterations both use 25, with
`max_train_points_per_pq_code=512`. cuVS does not expose the same public seed
or restart controls. JSON records the actual native parameters; it does not
claim that the training algorithms or every setting are identical to TensorVS.

Train-only build followed by additions in 65,536-row batches avoids retaining
the full FP32 GPU input beside the final index. cuVS Flat's device-only build
API receives the same regular-stride 50% training sample used by its native
sampler, then releases that GPU sample before bulk add. Its effective native
fraction is 1.0 over this pre-sampled matrix; JSON records both the original
target fraction and actual sample rows/stride. cuVS uses conservative list
allocation and fixed coarse centers. Faiss uses 64-bit IDs, FP32 Flat storage,
and a 128-MiB per-worker scratch reservation. Native saves run one at a time
to limit host memory spikes. Saved indexes include actual vector/code data.

## Files

Library label is `cuVS` or `FaissGPU`:

```text
result/CUDA_cuVS_MultiGPU3_IVFFLAT_sift100m_nlist1024.{csv,json}
result/CUDA_cuVS_MultiGPU3_IVFPQ_sift100m_nlist1024_M16_nbits8_train512.{csv,json}
result/CUDA_FaissGPU_MultiGPU3_IVFFLAT_sift100m_nlist1024.{csv,json}
result/CUDA_FaissGPU_MultiGPU3_IVFPQ_sift100m_nlist1024_M16_nbits8_train512.{csv,json}
```

Build files use the same stem plus `_build`, under `result/Build/`.
Index files are `shard0of3.bin` through `shard2of3.bin` for cuVS, or `.index`
for Faiss, under the dataset's
`GPUIndex/{cuVS,FaissGPU}_MultiGPU3/<build-stem>/` directory, accompanied by
per-shard JSON and `build.json`. Worker logs go under `result/logs/`.
Files for another library, timing convention or run are not combined.

Query CSV columns are exactly:

```text
nprobes,q_batch,K,qps,recall,visited_vectors_per_query,peak_gpu_memory
```

Build CSV columns are exactly the existing AMD/native build schema:

```text
dataset,method,nlist,n_vectors,dim,build_seconds,save_seconds,index_size_bytes,index_size_mib,peak_build_gpu_memory_mib
```

Build time is parallel native train-plus-add wall time, including native data
transfers/allocations; it excludes host input conversion and serialization.
Save time includes Faiss GPU-to-CPU cloning. Size sums all three native index
files. Original build measurements are retained when indexes are reused.

cuVS uses a dedicated training Resources handle, releases that handle and the
temporary GPU sample, then losslessly saves/reloads the empty trained Flat
index before adding vectors. Coarse-centroid bytes are verified unchanged.
One persistent Resources handle is reused throughout add/search; every call
is explicitly synchronized before temporary input buffers are released.
The trained-state round trip is included in measured construction time and
recorded separately in each shard's JSON. Device-free-memory progress samples
are diagnostic fields; reported peak metrics still use worker-process NVML.

QPS counts each query once, using the median full-query wall time from
dispatch to three workers through local GPU results, device-to-host copies,
response IPC, CPU global Top-K merge and final result concatenation. It does
not sum individual GPU QPS. The default `--query-input resident-gpu` uploads
queries before timing, matching current TensorVS measurements. Optional
`--query-input host` includes CPU payload distribution and H2D upload and
uses distinct `_HostInput` result filenames. Initialization, warmup, recall,
list-visit accounting and the separate memory sweep are excluded.

`peak_gpu_memory` is the maximum sampled **simultaneous sum** of the three
worker-process NVML memory values during a separate query sweep, in MiB.
`peak_build_gpu_memory_mib` measures construction/saving instead. JSON keeps
per-GPU peaks, original timing samples, actual environment/library versions,
data/index fingerprints and source hashes. These are sampled process peaks,
not instantaneous hardware maxima or the sum of unrelated jobs' memory.

Visited vectors use native list occupancy and nearest coarse lists. For
cuVS IVF-Flat, the reader validates serialization version 4 and reads actual
occupancy from the small RAFT/NumPy prefix of the saved native index. Unknown
formats leave the CSV field empty and explain why in JSON; counts are never
filled with estimates such as `N*nprobe/nlist`. Only complete 42-row curves
should be used in final figures.
