# Faiss Ascend IVF-PQ baseline

Each launcher runs one method and one dataset. TensorVS examples:

```bash
bash bench/run_ascend_tensor_bench.sh ivfflat glove1.2m --resume
bash bench/run_ascend_tensor_bench.sh ivfpq glove1.2m --resume
bash bench/run_ascend_tensor_bench.sh ivfflat hunyuan --resume
bash bench/run_ascend_tensor_bench.sh ivfpq hunyuan --resume
```

Build a new residual **inner-product (IP)** IVFPQ index from raw GloVe base vectors using Faiss Ascend
`useNpuTrain=True`, `train()` and batched `add()`. No TensorVS index, centroids,
codebooks, rotations, assignments or codes are loaded. Parameters are
nlist=1024, M=32, nbits=8. Input dimension 200 is zero-padded to 256, preserving
inner products exactly in real arithmetic. Vectors are not normalized; this
is raw maximum inner product, not cosine similarity. Coarse training uses
spherical centroids. This independent training and
preprocessing differ from the existing rotated TensorVS PQ index; recall and
visited-vector counts may consequently differ.

On the server, first build and run retrieval:

```bash
cd /home/ma-user/work/TensorVS
bash bench/run_ascend_faiss_ivfpq_bench.sh ivfpq glove1.2m
```

Resume retrieval with the cached Faiss-built index and original build timings:

```bash
bash bench/run_ascend_faiss_ivfpq_bench.sh ivfpq glove1.2m --resume
```

To explicitly rebuild and replace the benchmark output:

```bash
bash bench/run_ascend_faiss_ivfpq_bench.sh ivfpq glove1.2m --rebuild-index --overwrite
```

Default training uses seed 1234, 25 coarse iterations and 50 PQ iterations,
with one restart each. All base vectors are supplied to `train()`; Faiss's
internal sampling limits are 512 points per coarse centroid and per PQ
centroid (up to 524288 coarse / 131072 PQ samples). Use `--train-size N` to
select a seeded uniform subset before training. All base vectors are added
in batches of 65536; original row numbers are vector IDs.

The built index and its build manifest are cached in
`/home/ma-user/work/TensorVS/indexes/faiss_ascend/`. Dataset signature, training
parameters, builder/library hashes and index file SHA256 are checked before
cache reuse. A mismatch requires `--rebuild-index`. Do not combine
`--rebuild-index` and `--resume`. `--overwrite` alone can reuse a valid cache.

Outputs all share this prefix in `/home/ma-user/work/TensorVS/result/`:

```text
Ascend_Faiss_IVFPQ_glove1.2m_nlist1024_M32_nbits8_train512
```

- `.csv`: existing seven retrieval columns, unchanged:
  `nprobes,q_batch,K,qps,recall,visited_vectors_per_query,peak_gpu_memory`.
- `.json`: settings, per-row details and a top-level `build` object.
- `Build/<prefix>_build.csv`: one row of build time and index size measurements.
  Retrieval CSV/JSON files stay in the parent `result/` directory.

`build_time_seconds` includes index constructor, training, full add/encoding,
and NPU stream synchronization. `train_seconds` and `add_seconds` are also
reported. Input loading/padding/sampling (`preparation_seconds`) and CPU export
plus serialization (`serialize_seconds`) are measured separately and excluded
from build time. `build_with_preparation_seconds` includes preparation.
`index_size_bytes` is the actual saved Faiss file size, including centroids,
PQ books, codes and vector IDs. `index_size_mb` uses MiB (2^20 bytes). It is not
NPU resident-memory consumption. Loading a cached index does not constitute a
new build measurement; JSON marks `index_loaded_from_cache` and preserves the
original build times.

Retrieval defaults: first up to 1024 queries, K=10, batches 32/128/512, the
same 18 GloVe nprobes 50 through 400, two warmup sweeps, three timed sweeps.
The final partial batch is padded by repeating its last query; metrics count
only real queries. QPS uses the median synchronized sweep time. Recall is mean
exact IP ground-truth top-K ID overlap divided by K. Exhaustive CPU
`IndexFlatIP` computes the truth once and caches it under `indexes/faiss_ascend/`;
this preparation is excluded from build time and QPS. The dataset's original
L2 ground truth is not used. JSON records the metric, truth signature and
normalization. Filenames remain unchanged as requested. These recalls measure
a different task from L2 TensorVS runs and should not be compared as L2 recall.
Faiss missing-neighbor IDs (`-1`) count as recall misses, with their count
recorded as `missing_neighbor_slots` in each JSON result. Other invalid IDs
and nonfinite scores for valid IDs abort the run.

Retrieval composes two public Faiss NPU APIs: an explicit NPU Flat coarse
IP quantizer using the library's default FP16 centroid storage, then
`NpuIndexIVFPQ.search_preassigned`. The library's Flat query path uses
FP16 blocks; requesting FP32 storage in that path can crash and is not used.
The actual assignments passed to the scan determine the visited-vector count.
The published residual-IP multi-block path disagrees with its CPU-exported
index at large nprobe in synthetic correctness checks. Each scan therefore
receives at most 64 probes. Disjoint probe-group top-K results are merged by
descending IP score on CPU; the entire merge and all scan calls are timed.
No CPU vector scan or CPU IVFPQ retrieval is used in benchmark sweeps.
This is a two-stage API baseline, not native/fused `.search` throughput.
CPU query padding, both API calls, CPU control and transfers are timed. Recall,
visit accounting and memory sampling are excluded. The NPU library may use
FP16 query representations; `--lut-dtype` controls its LUT option. Tensor uses
resident NPU queries, whereas Faiss accepts/returns host arrays. JSON records
these comparison limits. Training can use host preprocessing/bookkeeping
inside the Faiss library even though the NPU training option is enabled.

`peak_gpu_memory` remains the separate-sweep, sampled current-process HBM peak
from `npu-smi`, in MiB, including runtime, index, workspace and allocator caches.
It is not an instantaneous hardware peak or serialized index size.

Runtime: `/home/ma-user/work/faiss-ascend-runtime`, published Faiss Ascend
1.13.2 build 26.2.0 wheel, isolated CANN 9.0.0 and its matched embedded operators.
This release rejects residual L2; the benchmark uses supported residual IP
instead, as requested. The experimental source L2 backend was not activated.
The launcher executes Python with `conda run -n tensorvs`. System CANN 8.5.2
and Tensor scripts are preserved. `check_ascend_faiss_ip.py` uses synthetic
data to check native training/add/export and the actual IP retrieval pipeline;
it does not execute the GloVe benchmark.

## CPU IVF-Flat reference baseline

The same installed Faiss Ascend package also exposes the regular CPU
`IndexIVFFlat`. Run exactly one dataset per invocation:

```bash
cd /home/ma-user/work/TensorVS
bash bench/run_ascend_faiss_cpu_ivfflat_bench.sh ivfflat glove1.2m --resume
bash bench/run_ascend_faiss_cpu_ivfflat_bench.sh ivfflat hunyuan --resume
```

The CPU baseline independently trains/adds the raw FP32 vectors with squared
L2, nlist=1024, seed=1234, 20 coarse iterations and a training sample target
of half the base vectors. All vectors are supplied to Faiss training; the
per-centroid cap is `ceil(0.5 * N / nlist)`. These match the AMD IVF-Flat build
measurement settings. It does not load or modify TensorVS indexes. Independently
trained centroids/list assignments can differ from TensorVS, so recall and
visited-vector counts may differ. This is a CPU reference, not NPU performance.

The CPU baseline uses the first up to 512 queries by default (`--nq`), with
matching ground-truth rows. K=10, padded batches 32/128/512, two warmup
sweeps and median of three timed sweeps are unchanged. Changing the query
count requires fresh query results (`--overwrite` if results already exist);
the saved index and original build metrics remain reusable. Native CPU
`IndexIVFFlat.search` is timed, including coarse selection and scanning;
recall, visit accounting and memory sampling are outside QPS timing. Actual
list-size accounting is checked against Faiss's native `indexIVF_stats.ndis`,
and excludes padded query lanes.
Coarse assignments for accounting reproduce the native IVF query slices for
the configured OpenMP threads. Whole-batch reassignment can change rounding
near tied centroids and incorrectly fail this check; accounting remains outside
QPS timing. Existing trained indexes and build statistics remain reusable.

`--cpu-threads` defaults to the available CPU count bounded by affinity and
cgroup quota (24 on this Notebook). Faiss OpenMP threads are explicitly set;
OpenBLAS/MKL use one thread to avoid nested threading. Change the CPU count
with `--cpu-threads N`; thread changes require separate results/cache or an
explicit rebuild. JSON records threads, library version/path/hash and `cpu`
execution. The script does not import PyTorch or create NPU resources.

Build progress is printed by default: `[1/4]` loading, `[2/4]` native Faiss
k-means iterations (numbered from 0), `[3/4]` added vectors/total, percentage,
throughput and approximate add-stage ETA, then `[4/4]` serialization and the
actual saved size. The ETA refers to the add stage, not the whole build.
`--quiet-build` disables these progress messages. Training/add progress output
is included in their existing wall-clock timings. Native training verbosity
is reset before adding/searching, so it does not add logs to timed retrieval.
The progress-only update accepts caches from the previous builder and retains
their original build times and provenance; it does not require a rebuild.

Output names retain the existing platform/method/index/dataset convention:

```text
result/Ascend_Faiss_IVFFLAT_<dataset>_nlist1024.csv
result/Ascend_Faiss_IVFFLAT_<dataset>_nlist1024.json
result/Build/Ascend_Faiss_IVFFLAT_<dataset>_nlist1024_build.csv
result/Build/Ascend_Faiss_IVFFLAT_<dataset>_nlist1024_build.json
```

The search CSV has the same seven columns. `peak_gpu_memory=0` because this
baseline does not allocate GPU/NPU memory; **CPU RSS is recorded separately
as `peak_rss_memory_mib` in each JSON result**. CPU RSS must not be interpreted
as HBM. The serialized index size and build statistics are in `Build/`.
`build_time_seconds` includes CPU constructor, training and all adds; data
loading/preparation and serialization have separate timings. Index size is
the actual serialized file size, in bytes and MiB. The cache under
`indexes/faiss_ascend_cpu/` validates input/settings/library/content hashes and
preserves original build times when reused. For a fresh build and output:

```bash
bash bench/run_ascend_faiss_cpu_ivfflat_bench.sh ivfflat glove1.2m --rebuild-index --overwrite
```
