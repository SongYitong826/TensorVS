# TensorVS usage and benchmark notes

[Back to TensorVS](../README.md)

Tensor-native IVF vector search with cross-query data reuse and shared PQ lookup tables.

[Design](#design) · [Quick Start](#quick-start) · [Search Backends](#search-backends) · [Benchmarks](#benchmarks) · [Development](#development)

---

TensorVS is an IVF-based vector search engine built around tensor operations. It reorganizes IVF-Flat scanning to reuse vectors across queries, and reformulates residual IVF-PQ distance computation around a lookup table shared by all lists probed by a query. A portable PyTorch path and a specialized Triton path implement these ideas for different hardware.

## Key Features

- **Reuse IVF-Flat vectors across queries.** List-major scanning groups queries that probe the same list, so a vector tile can be used by multiple queries in batched matrix multiplication. Work is tiled across queries and candidates to handle long or frequently probed lists.
- **Build one PQ lookup table per query.** TensorVS moves the list-dependent residual-PQ term into a precomputed per-vector correction. The resulting query-dependent table is shared across probed lists, avoiding a separate table for every query–list pair.
- **Run the same search logic across tensor backends.** The PyTorch implementation expresses search through standard tensor operations for CPU, CUDA, and Apple Silicon MPS. A complementary Triton implementation fuses scan and local selection on supported CUDA configurations.
- **Bound intermediate search work.** Candidate blocks produce local top-k results that are merged per query, with a configurable PyTorch scan/merge workspace budget. IVF-PQ can optionally rerank candidates using external original vectors.

## Design

### IVF-Flat: share work across queries

Query-major GPU scans commonly process each query or query–list pair independently. When queries probe the same list, they can repeatedly load the same full-precision vectors. TensorVS instead supports **list-major scanning**: it groups queries by their selected lists, tiles both the queries and candidate vectors, and computes distances with batched matrix operations. This shares candidate data across queries and splits long or heavily probed lists into smaller units of work. Local top-k results are then merged for each query.

The PyTorch backend also has an adaptive dense path for suitable CUDA workloads, and exposes `torch_flat_scan` to select `grouped`, `dense`, or `auto`. The Triton path uses an interleaved layout and fuses distance computation with local selection. Which strategy is faster depends on query overlap, list sizes, batch size, and hardware.

### IVF-PQ: share a lookup table across lists

Residual IVF-PQ normally combines a query, a list centroid, and PQ codewords when estimating a candidate's distance. TensorVS separates those terms: it builds one **query-dependent lookup table** shared across the query's probed lists, and stores a precomputed scalar correction with each encoded vector. During scanning it combines the lookup values, correction, and query–centroid distance. This rearranges the residual-PQ distance calculation while trading extra per-vector storage and reads for less list-dependent lookup-table work. Floating-point precision and operation order can affect numerical results; the LUT storage precision is configurable.

The PyTorch implementation uses tensor indexing, reduction, and top-k in bounded blocks. The Triton IVF-PQ path scans compact codes in query-major work units and fuses lookup, accumulation, and local selection; unlike IVF-Flat, its compressed candidates and shared per-query table favor a different scheduling choice. Both paths retain the same index-level distance formulation.

### Portable execution, specialized layouts

The PyTorch path uses contiguous list data and standard tensor operators; supported CUDA configurations can instead use Triton with interleaved layouts and fused kernels. Layouts are derived from the same saved index, allowing backend selection without retraining. The implementation does not require custom Metal kernels to search on MPS. Portability and performance depend on the PyTorch device backend and workload.

## Quick Start

TensorVS requires Python 3.11 or later. Clone the repository with its `flash-kmeans` submodule and install it in an environment with PyTorch:

```bash
git clone --recurse-submodules https://github.com/SongYitong826/TensorVS.git
cd TensorVS
python -m pip install -e .
```

The submodule is configured with a GitHub SSH URL, so cloning it requires GitHub SSH access. If you already cloned the repository, run `git submodule update --init --recursive` before installing. For CUDA configurations using Triton, install the optional dependency with `python -m pip install -e '.[triton]'`.

Build an IVF-Flat index from generated vectors, save it, and search it on CPU:

```python
import torch
from tensorvs import (
    build_ivfflat_index,
    load_ivfflat_index,
    save_ivfflat_index,
    search_ivfflat_index,
)

torch.manual_seed(7)
vectors = torch.randn(1024, 16)
queries = vectors[:5].clone()

build = build_ivfflat_index(
    vectors, nlist=8, niter=5, use_triton=False, device="cpu"
)
save_ivfflat_index(build, "./ivfflat-index")
index = load_ivfflat_index("./ivfflat-index")

distances, ids = search_ivfflat_index(
    index, queries, topk=5, nprobe=4, device="cpu", backend="torch"
)
print(ids)        # shape: (5, 5), IDs refer to rows in vectors
print(distances)  # squared L2 distances
```

For real datasets, `load_dataset` reads `.fvecs`, `.npy`, and `.pt`/`.pth` files. See [IVF-Flat](../examples/ivfflat_search.py) and [IVF-PQ](../examples/ivfpq_search.py) for complete build and search examples; edit their dataset paths and parameters before running them.

## IVF-PQ and Refinement

IVF-PQ trains a coarse quantizer and product quantizers on residuals. `m` sets the number of PQ subspaces and `nbits` sets the codebook size per subspace. The index stores compressed codes rather than original vectors.

```python
from tensorvs import (
    build_ivfpq_index,
    load_ivfpq_index,
    save_ivfpq_index,
    search_ivfpq_index,
)

# vectors and queries are two-dimensional PyTorch tensors with the same dimension.
build = build_ivfpq_index(
    vectors, nlist=8, niter=5, use_triton=False, device="cpu", m=4, nbits=4
)
save_ivfpq_index(build, "./ivfpq-index")
index = load_ivfpq_index("./ivfpq-index")

distances, ids = search_ivfpq_index(
    index, queries, topk=5, nprobe=4, device="cpu", backend="torch",
    refine_vectors=vectors, refine_factor=4,
)
```

Here refinement selects up to 20 approximate candidates per query and returns the best five using exact squared L2 distances. Pass the original vectors in their original row order; omit `refine_vectors` and `refine_factor` to return PQ distances directly. Training options for the coarse and PQ stages can be configured independently; see [`build_ivfpq_index`](../src/tensorvs/index_ivfpq.py).

## Search Backends

| Backend | Devices | Behavior |
| --- | --- | --- |
| `auto` | CPU, MPS, CUDA | Selects Triton for supported configurations; otherwise uses PyTorch. |
| `torch` | CPU, MPS, CUDA, ROCm, Ascend NPU | Uses standard PyTorch operations and bounded scan/merge workspaces. |
| `triton` | Supported CUDA and ROCm configurations | Uses fused Triton search. Unsupported devices or shapes raise an error. |

Set `device` explicitly to choose where search runs. If omitted, queries already on a non-CPU device keep that device; otherwise TensorVS chooses CUDA, then MPS, then CPU. `backend` and `device` are independent settings. The examples train on CPU when using MPS for search because some training operations are unavailable on MPS.

The PyTorch backend exposes `torch_query_batch_size`, `torch_candidate_chunk_size`, and `torch_memory_budget_mb` for managing scan workspaces. The budget does not include the resident index, inputs and outputs, or internal compiler and library workspaces. `use_compile=True` enables `torch.compile` for supported PyTorch search blocks; compilation and performance depend on the device and workload.

Execution layouts are cached by default under `~/.cache/tensorvs/layouts`. Set `TENSORVS_LAYOUT_CACHE_DIR` to change the cache location, or pass `persist_layout=False` to disable persistent layout caching.

## Benchmarks

### Reproduce the benchmark figures

The benchmark figures show GloVe 1.2M and Tencent Hunyuan with top-k 10 and
1,024 IVF lists. IVF-Flat uses query batches 32, 128, and 512 on NVIDIA, AMD,
and Apple Silicon. Ascend IVF-Flat and all IVF-PQ plots use batch 128.
[Benchmark summary and curves](benchmarks.md)

The AMD, Apple, and Ascend plots reuse the paper's plotting scripts and
the completed CSV/JSON pairs in [`result/`](../result/). The NVIDIA plots
use approximate points recovered from the vector paths and axis ticks of
the two paper figures. The coordinates describe the visible curves,
including the cuVS and Faiss GPU baselines. [NVIDIA plot data](benchmark-data/nvidia-paper-curves.csv)

The Apple GloVe IVF-PQ run from the Faiss Metal package used CPU search.
The Ascend GloVe IVF-PQ panel shows Faiss Ascend with inner-product ground
truth and TensorVS with squared-L2 ground truth.

Run the [plotting script](../plot/readme_benchmarks.py) from the repository
root with Matplotlib installed. It writes the eight PNGs to `docs/assets/`.

```bash
conda run -n tensorvs python -m plot.readme_benchmarks
```

Apple Silicon query results are stored in `result/` as one CSV/JSON pair per
method, dataset, and index configuration. Names follow the same platform-first
convention as AMD results: `MPS_TensorVS_Tensor_IVFFLAT_glove1.2m_nlist1024.csv`
and `MPS_FaissMetal_IVFPQ_glove1.2m_nlist1024_M32_nbits8_train256.csv`, for example.
Query CSVs use the same seven columns as AMD and Ascend:
`nprobes,q_batch,K,qps,recall,visited_vectors_per_query,peak_gpu_memory`.
Filenames do not include QBatch. JSON sidecars retain every original field in
`records`, including timing samples, execution device, RSS, and provenance.
Unmeasured `visited_vectors_per_query` values are left empty; they are not
estimated. `bench._mps_results.read_rows` validates and joins the CSV/JSON pair
for plotting and analysis.

The current GloVe/Hunyuan MPS comparisons use QBatch=32/128/512 and K=10:

| Index | Figure | Example TensorVS results |
| --- | --- | --- |
| IVF-Flat | [Figure](../plot/Figures/IVFFlat_MPS_QBatch32_128_512_K10.png) | [GloVe](../result/MPS_TensorVS_Tensor_IVFFLAT_glove1.2m_nlist1024.csv) · [Hunyuan](../result/MPS_TensorVS_Tensor_IVFFLAT_hunyuan_nlist1024.csv) |
| IVF-PQ | [Figure](../plot/Figures/IVFPQ_MPS_QBatch32_128_512_K10.png) | [GloVe](../result/MPS_TensorVS_Tensor_IVFPQ_glove1.2m_nlist1024_M32_nbits8_train256.csv) · [Hunyuan](../result/MPS_TensorVS_Tensor_IVFPQ_hunyuan_nlist1024_M384_nbits8_train256.csv) |

Older SIFT1M/GloVe sweeps at QBatch=128/256/512 are retained in `result/Archive/`
with the same configuration-based filenames, so overlapping measurements from
different runs are not overwritten. `plot/qbatch_comparison.py` reads those
historical sweeps; `plot/mps_kernel_performance.py` reads the current sweeps.
CPU memory checks live under `result/Build/`, use `MPS_FaissCPU_..._cpu_memory`
and `MPS_TensorVS_CPU_..._cpu_memory` names, and retain their specialized
memory columns and CPU execution labels. Interrupted query sweeps are retained
in `result/Archive/incomplete/` and are excluded from figures. Use
`conda run -n tensorvs python bench/organize_mps_results.py --normalize` to
normalize pre-existing MPS files; the command backs up and verifies every row.

Index-related records are in `result/Build/MPS_TensorVS_Tensor_..._build.csv`
and matching JSON files. They preserve the saved source index's original build
time, actual build device, and full metadata. Some indexes were built on CUDA
and imported for Apple queries; these records are not Apple build benchmarks.
Faiss imports the same trained TensorVS index, so no separate Faiss training
measurement is inferred.

These results were measured on Apple M5. TensorVS uses MPS for both indexes; the local Faiss Metal comparison uses GPU search for IVF-Flat and four CPU threads for IVF-PQ. TensorVS inputs and outputs reside on MPS, while Faiss API timing includes its internal transfers. The comparison imports the same trained index into Faiss; it does not retrain it. Results depend on hardware, software, and workload, and the two timed paths have different device and transfer behavior.

The [comparison script](../bench/compare_mps_faiss.py) expects dataset queries and ground truth, prebuilt TensorVS indexes, and a local Faiss Metal build. For example:

```bash
conda run -n tensorvs python bench/compare_mps_faiss.py --dataset /path/to/glove1.2m --faiss-root /path/to/faiss-metal --q-batches 32 128 512
conda run -n tensorvs python plot/mps_kernel_performance.py
conda run -n tensorvs python plot/qbatch_comparison.py
```

The script refuses to overwrite existing result CSVs. Use a different `--output-dir` for a new run. See the script's `--help` for dataset and measurement options.

For Hunyuan IVF-PQ (M=384), `--faiss-cpu` imports the same trained TensorVS
lists, codes, centroids, and IDs directly into the installed Faiss CPU index,
bypassing the Metal wrapper's M<=64 restriction. Its curve retains the
"Faiss Metal" legend entry and records `execution_device=cpu` and zero Metal
allocation in the result files. The supplemental sweep uses 1,000 queries,
QBatch=32/128/512, nprobe=350/500/700/850/1000, four CPU threads, two full
warmup sweeps, and the median of three full timing sweeps per point. This
query comparison reuses the TensorVS index rather than the separately
retrained Faiss build-time index. The Hunyuan supplement is currently
interrupted after two QBatch=32 points; its files are retained under
`result/Archive/incomplete/` and are not plotted as a completed curve.

Native Faiss Metal build measurements use `bench/faiss_metal_build.py`:

```bash
conda run -n tensorvs python -u bench/faiss_metal_build.py --dataset glove1.2m --kind ivfflat
conda run -n tensorvs python -u bench/faiss_metal_build.py --dataset glove1.2m --kind ivfpq
conda run -n tensorvs python -u bench/faiss_metal_build.py --dataset hunyuan --kind ivfflat
conda run -n tensorvs python -u bench/faiss_metal_build.py --dataset hunyuan --kind ivfpq
```

Each command trains a fresh index from the original base vectors, adds all
vectors, serializes the native Faiss index under `indexes/faiss_metal/`, then
reloads and validates its contents. CSV/JSON measurements are saved in
`result/Build/MPS_FaissMetal_..._build.*`. Build time includes Faiss CPU training,
Metal upload where supported, and add; vector reading, GloVe zero padding,
serialization, and validation are timed separately. Defaults are nlist=1024,
four CPU threads, 20 coarse/PQ iterations, a 50% coarse training target, and
256 PQ training points per centroid. GloVe IVF-PQ uses M=32 with D padded from
200 to 224; Hunyuan uses M=384. The installed Metal wrapper supports M<=64,
so Hunyuan IVF-PQ uses the library's CPU train/add path. Build records retain
that distinction. Resource tables combine these native build/size measurements
with the separately recorded query-memory experiments.

### CUDA build statistics

Single-device CUDA statistics for sift1m, sift10m, glove1.2m, deep1m, tiny5m,
hunyuan, gist, and openai1536 are stored in `result/Build/`, one CSV/JSON pair
per method and index configuration. Prefixes are `CUDA_TensorVS_Triton`,
`CUDA_FaissGPU`, and `CUDA_cuVS`; QBatch is omitted from build filenames.
Existing SIFT100M multi-GPU records retain their separate names and schemas.

The CSV columns match the existing AMD/CUDA library build records:
`dataset,method,nlist,n_vectors,dim,build_seconds,save_seconds,index_size_bytes,index_size_mib,peak_build_gpu_memory_mib`.
JSON includes M/nbits, the actual build device, original metadata, exact source
paths, checksums, and matching legacy statistics. TensorVS build times retain
full saved precision and serialized sizes come from the actual file size.
Five older Faiss configurations have multiple equally matching build-time
records; their times are left empty and all candidate rows are retained.
Unmeasured save time and build peak memory are also empty.

```bash
conda run -n tensorvs python bench/organize_cuda_build_results.py --ssh-host YOUR_CUDA_HOST
```

This command only reads existing indexes and statistics; it does not retrain
or run queries. The local `tmp/cuda_build_source.json` export can be reimported
with `--source-json`. Current TensorVS benchmark scripts also write the same
Build pairs when creating or reusing an index. Resource-table scripts read
the local Build pairs, so the CUDA server and its index files are no longer
needed to render tables. Historical paper query peaks are retained separately
as `legacy_table_statistics`; newer per-nprobe query counters are stored in
`query_statistics`. Neither is substituted for build peak memory.

### Ascend NPU Tensor benchmarks

Use an Ascend PyTorch environment with a matching `torch_npu` and CANN stack.
The runner imports `torch_npu` explicitly and calls the existing portable
`backend="torch"` implementation on `npu:0`; it does not use CUDA emulation,
Triton, custom Ascend kernels, or `torch.compile`. FP32 matrix multiplication
uses `allow_hf32=False`. The named Conda environment for this project is
`tensorvs`.

Put each dataset's queries, ground truth, and existing `GPUIndex` directory
under `/home/ma-user/work/data/<dataset>/`, retaining the names used by the
AMD/CUDA scripts. Each invocation runs exactly one method and dataset:

```bash
cd /home/ma-user/work/TensorVS
bash bench/run_ascend_tensor_bench.sh ivfflat hunyuan --resume
bash bench/run_ascend_tensor_bench.sh ivfpq hunyuan --resume
```

The Python runner's baseline defaults match `bench/amd_tensor_bench.py`: query batches 32/128/512,
K=10, up to the first 1024 queries, two warmup sweeps, three measured sweeps,
the same dataset-specific nprobe grids, 128 MiB scan workspace, and FP16 PQ
LUTs. `--flat-scan` and `--pq-scan` select the existing Tensor scan strategies.
Flat `auto` uses dense masked GEMM on NPU for batches >=128 with
`nprobe * 16 >= nlist`, when the workspace permits; smaller/sparser batches
use grouped scanning. PQ `auto` retains grouped scanning on NPU. Timings synchronize
the NPU before and after every query batch, with index and padded queries
resident on NPU. Recall and memory sampling are outside QPS timing.

Grouped Tensor execution can be tuned independently of the persisted index:
`--query-tile-size` (a power of two), `--candidate-chunk-size`,
`--min-query-tile-size`, `--max-tasks-per-batch`, and `--memory-budget-mb`. For PQ, `--lut-chunk-size`
controls the ADC reduction width and `--pq-lookup {flat,gather}` selects the
Tensor lookup organization. An explicit query tile honors the requested
candidate chunk, subject to the workspace budget; the original eager heuristic
otherwise prefers chunks up to 1024. These options leave nlist, M, nbits,
the probe count, K, stored vectors/codes, and precision settings unchanged.

For a short tuning sweep on an idle NPU, run one method at a time:

```bash
conda run -n tensorvs python bench/tune_ascend_tensor_tiles.py ivfpq --q-batch 128 --nprobe 80
```

The tuner checks neighbor/distance agreement against the original grouped
path and writes diagnostic JSON to `diagnostics/`, separately from experiment
CSVs. Its batch QPS is a tuning measurement, not a replacement for full-query
benchmark sweeps.

The shared tuner, `bench/tune_tensor_tiles.py`, also supports **Cambricon MLU**
and **Kunlun**, with the same fixed-index validation and synchronized batch
timing. The Ascend command above remains a compatible entry point. Platform
selection imports the vendor plugin only when needed:

| `--platform` | Default device | Required vendor module |
| --- | --- | --- |
| `ascend` | `npu:0` | `torch_npu` |
| `mlu` | `mlu:0` | `torch_mlu` |
| `kunlun` | `cuda:0` | `torch_xmlir` (xpytorch CUDA compatibility interface) |
| `cuda` / `mps` / `cpu` | `cuda:0` / `mps` / `cpu` | None |

Use the matching vendor PyTorch environment; the tuner does not install or
replace a runtime. For Kunlun stacks exposing another device API, override
`--device` and `--device-module` with the registration provided by that stack.
Stock PyTorch's `xpu` device denotes Intel, so `xpu` alone does not identify
Kunlun. The tuner requires the selected accelerator and its synchronization
API; it does not silently select CPU. Runtime versions, the actual device,
precision flags and the unchanged index metadata are recorded in the report.

Pass explicit input paths to reuse MLU's existing `.bin` indexes and `.npy`
queries/ground truth, including the `glove1m` and `gauss768` datasets. For example
(replace `/path/to` with the existing files):

```bash
conda run -n tensorvs python bench/tune_tensor_tiles.py ivfpq \
  --platform mlu --device mlu:0 --dataset gauss768 \
  --index /path/to/indexes/MLU590_Tensor_ivfpq_gauss768.bin \
  --queries /path/to/data/gauss768/gauss768_query.npy \
  --groundtruth /path/to/data/gauss768/gauss768_groundtruth.npy \
  --q-batch 128 --nprobe 650 --query-tiles 8 16 32 64 \
  --candidate-chunks 512 1024 2048 --task-caps 64 256 \
  --memory-budgets 128 256 --lut-chunks 16 32 --pq-lookup flat gather
```

For Kunlun, use the same command with `--platform kunlun --device cuda:0`
and the paths to that machine's unchanged index and dataset. Tune `ivfflat`
with the same shared entry point; PQ-specific options only apply to `ivfpq`.
Choose the existing experiment's NProbe, batch size and LUT dtype before each
sweep; the example's NProbe is not a new search setting to apply universally.
Only query tiles, candidate chunks, task caps, workspace, LUT reduction chunks
and lookup organization vary. No padding, rotation, codebook, M, nbits, nlist,
probe count, K or LUT precision is changed by tuning. Apply the winning
`best` settings from the diagnostic JSON to the benchmark runner explicitly;
the tuner does not overwrite indexes, experiment results or launcher presets.
MLU/Kunlun batch timings synchronize the device on both sides of the search;
their internal stage profile is skipped where supported stage timers are
unavailable. These adapters have local CPU/mock regression coverage; actual
MLU/Kunlun validation requires the target hardware and vendor runtime.

The GloVe launcher applies the measured Ascend settings: 512 MiB workspace;
PQ uses grouped `gather` lookup, a query tile up to 16, 1024 candidates per
chunk, and 16 subspaces per reduction. The launcher uses one CPU thread.
Trailing CLI options override these presets. Hunyuan IVF-PQ uses grouped
`gather` lookup, a query tile up to 64, 1024 candidates per chunk, 32 subspaces
per reduction, and a 256 MiB workspace. Its M=384 index, nbits=8, nlist=1024,
nprobe grid, K and FP16 LUT storage stay unchanged. Hunyuan IVF-Flat retains
its previous execution defaults. No index is rebuilt.
In short same-process GloVe batch checks, Flat improved from 1425 to 2380
QPS at Q128/P80 and 397 to 4717 at Q512/P400; PQ improved from 321 to 979
at Q128/P80 and 57 to 299 at Q512/P400. Returned neighbor IDs matched the
baseline in these checks. The full experiment must be rerun for final metrics;
previous result files are not overwritten by tuning. Changed settings/source
hashes intentionally prevent resuming old measurements into a new run.

Hunyuan M384 short-query checks improved from 9.43 to 79.25 QPS at Q32/P350
on holdout queries 768--799, and from 5.89 to 58.84 QPS at Q128/P650 with the
selected wider query tile. Full-batch Q32/Q128 neighbor IDs matched the old path.
At Q512/P1000 the selected configuration measured 39.57 QPS; only 32 uniformly
sampled queries were checked against the slow baseline there, so no full-batch
speedup is inferred. The tuner records sampled reference counts and leaves
`speedup_over_default` null when baseline/trial query counts differ. Use
`--query-offset` to validate a holdout query segment; source-index size/mtime
are checked after tuning. These diagnostics do not replace complete runs.

Results are written incrementally to `result/` as CSV plus JSON
metadata, using `Ascend_TensorVS_Tensor_IVFFLAT_*.csv` and
`Ascend_TensorVS_Tensor_IVFPQ_*.csv` filenames. AMD outputs in `result/`
use an `AMD_` prefix. The CSV column order is unchanged:

```text
nprobes,q_batch,K,qps,recall,visited_vectors_per_query,peak_gpu_memory
```

The retained `peak_gpu_memory` column measures **NPU HBM in MiB**, sampled for
the current process through `npu-smi` in a separate search sweep. Like the
AMD-SMI/NVML measurements, it includes resident index/query tensors, allocator
cache and runtime memory; it is a sampled process peak, not an exact hardware
peak. JSON also records Torch allocated/reserved peaks, timings, actual index
metadata, software versions, and source hashes. `--resume` checks that the
configuration, inputs, and sources are unchanged before skipping completed
rows. Use a separate output directory for smoke tests:

```bash
bash bench/run_ascend_tensor_bench.sh ivfpq hunyuan \
  --nq 32 --q-batches 32 --nprobes 10 --warmup 1 --repetitions 1 \
  --output-dir result/ascend_smoke
conda run -n tensorvs python -m pytest -q tests/test_ascend_tensor.py tests/test_ascend_benchmark.py
```

Faiss Ascend independently trains and builds a residual **inner-product** GloVe
IVFPQ index and reports build time and serialized size in addition to the same
retrieval columns. Recall uses exhaustive IP ground truth, so it is a different
search task from the L2 TensorVS runs. Output filenames keep their existing format:

```bash
bash bench/run_ascend_faiss_ivfpq_bench.sh ivfpq glove1.2m --resume
```

See [Faiss Ascend benchmark details](../bench/ASCEND_FAISS.md) for build timing,
cache reuse, result files, padding and comparison limits.

For a CPU IVF-Flat reference using that same Faiss Ascend package:

```bash
bash bench/run_ascend_faiss_cpu_ivfflat_bench.sh ivfflat glove1.2m --resume
```

This independently builds a squared-L2 CPU index and writes retrieval results
to `result/Ascend_Faiss_IVFFLAT_*.csv/json`; build time and serialized size go
to `result/Build/`. The seven CSV columns are unchanged: GPU memory is zero,
while sampled CPU RSS and execution/thread information are in JSON. See the
same benchmark details for CPU settings and the fresh-build command.

## Development

Install development dependencies and run the tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

- [`index_ivfflat.py`](../src/tensorvs/index_ivfflat.py) and [`index_ivfpq.py`](../src/tensorvs/index_ivfpq.py) provide the build, save, load, and search entry points.
- [`_coarse.py`](../src/tensorvs/_coarse.py) computes scores for probe selection. [`_pq_math.py`](../src/tensorvs/_pq_math.py) contains the PQ working-space transform and query lookup table construction.
- [`_torch_search.py`](../src/tensorvs/_torch_search.py) implements the portable search path. [`_ivfflat_triton.py`](../src/tensorvs/_ivfflat_triton.py) and [`_ivfpq_triton.py`](../src/tensorvs/_ivfpq_triton.py) implement the fused scan paths. [`_layout_cache.py`](../src/tensorvs/_layout_cache.py) prepares device-specific layouts from loaded indexes.
- [`examples/`](../examples/) shows how to build and query indexes. [`tests/`](../tests/) covers training, indexes, and search backends. [`bench/`](../bench/) and [`plot/`](../plot/) contain the experiment and plotting tools.
