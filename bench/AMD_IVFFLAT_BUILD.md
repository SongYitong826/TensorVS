# AMD IVF-Flat index build measurements

Run on the AMD server from `/root/TensorVS` after copying the two base `.fvecs`
files to `/root/vector_data/{glove1.2m,hunyuan}/`:

```bash
bash bench/run_amd_ivfflat_build.sh
```

Existing build results are protected. To intentionally rebuild both datasets
and both methods, pass `--overwrite`. To build one method and dataset, invoke
`bench/amd_ivfflat_build.py` directly with `--method` and `--dataset`.

Each run writes a method-prefixed CSV and JSON in `/root/TensorVS/result/Build/`.
The serialized index and a copy of its JSON metadata are saved under the
dataset's `GPUIndex/FaissROCm/` or `GPUIndex/hipVS/` directory. Index names
include `_build`, so they do not conflict with the earlier hipVS search
benchmark's index and list-count metadata pair.

The run uses `nlist=1024`, squared L2, 20 k-means iterations and a training
set target of half the base vectors. Faiss uses a per-centroid sample cap of
`ceil(0.5 * n_vectors / nlist)`; hipVS receives a train fraction of `0.5`.
These settings align the requested training work, though the libraries may
sample vectors and implement clustering differently. Faiss uses 64-bit IDs.

`build_seconds` starts after the `.fvecs` file is read into host memory and
includes GPU input transfer, training and adding all vectors. `save_seconds`
measures index conversion and serialization separately. `index_size_bytes` is
the resulting serialized file size. `peak_build_gpu_memory_mib` is a sampled
AMD-SMI peak for the process during build, including the GPU runtime; a short
peak between samples can be missed. The JSON records library versions and
Faiss source commit. Each result is one run, not a statistical distribution.

The deployment used an MI300X VF (`gfx942`) and ROCm core 10.1. TensorVS uses
PyTorch `2.13.0+rocm10.1.0`. hipVS uses AMD's `amd-hipvs==1.0.0` and
`amd-cupy==13.5.1` wheels in `/root/hipvs-venv`; it was smoke-tested with
the existing `bench/hipvs_python.sh` launcher. Faiss `1.15.1` was built from
source commit `e4500c2252dacdcb2d5288c05cfeb7950379ec50` with
`FAISS_ENABLE_GPU=ON`, `FAISS_ENABLE_ROCM=ON`, and `gfx942` in
`/root/faiss-rocm-venv`. On this ROCm image, Faiss's generated
`faiss/gpu/utils/Float16.cuh` needed the ROCm branch to include
`<hip/hip_fp16.h>` and `<hip/hip_bf16.h>` after CMake's hipification step;
the local source change replaces `#if !defined(USE_AMD_ROCM)` with `#if 1`
in that file. No algorithm code was changed.
