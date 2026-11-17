<p align="center">
  <img src="docs/assets/tensorvs-icon.png" width="112" alt="TensorVS icon">
</p>

<h1 align="center">TensorVS</h1>

<p align="center"><strong>High-performance vector search for any tensor platform.</strong></p>

<p align="center">
  <a href="#platform-support">Platform Support</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="docs/usage-and-benchmarks.md">Usage &amp; Reproduction</a>
</p>

---

- **Vector search on any tensor platform.** Run IVF-Flat and IVF-PQ wherever the required PyTorch tensor operations are available. Search the same saved index across NVIDIA, AMD, Apple Silicon, and Ascend.
- **Two execution paths for portability and performance.** The Tensor path uses standard batched tensor operations to make IVF search portable across accelerator platforms. The Triton path uses fused kernels to deliver high-performance search on CUDA GPUs.
- **IVF-Flat with cross-query vector reuse.** Adaptive list-major scanning reuses full-precision vector tiles across queries probing the same list and splits long or heavily probed lists into parallel tasks.
- **IVF-PQ with a query-global LUT.** Search compact residual codes with one lookup table per query, shared across all probed lists, keeping LUT construction and storage independent of `nprobe`.
- **High performance against Faiss GPU and cuVS.** On NVIDIA RTX 4090, the Triton path delivers approximately 8.6× the QPS of Faiss GPU and 7.4× the QPS of cuVS on Tencent Hunyuan IVF-Flat at 99% Recall@10 and query batch size 128.

## Platform Support

| Platform | Search backend | Verified configuration |
| --- | --- | --- |
| CPU | PyTorch (`cpu`) | IVF-Flat and IVF-PQ automated tests |
| NVIDIA GPU | PyTorch or Triton (`cuda`) | RTX 4090 · IVF-Flat and IVF-PQ benchmarked with Triton |
| AMD GPU | PyTorch or Triton on ROCm (`cuda`) | MI300X VF · IVF-Flat and IVF-PQ benchmarked with both backends |
| Apple Silicon GPU | PyTorch MPS (`mps`) | Apple M5 · IVF-Flat and IVF-PQ benchmarked |
| Huawei Ascend NPU | PyTorch with `torch_npu` and CANN (`npu:0`) | Ascend 910B4 · IVF-Flat and IVF-PQ benchmarked |

Validation is pending for Cambricon MLU and Kunlunxin XPU.

## Quick Start

Python 3.11+ with PyTorch. The `flash-kmeans` submodule currently uses GitHub SSH.

```bash
git clone --recurse-submodules https://github.com/SongYitong826/TensorVS.git
cd TensorVS
python -m pip install -e .
```

### IVF-Flat

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
build = build_ivfflat_index(vectors, nlist=8, niter=5, device="cpu", use_triton=False)
save_ivfflat_index(build, "./ivfflat-index")

index = load_ivfflat_index("./ivfflat-index")
distances, ids = search_ivfflat_index(
    index, vectors[:5], topk=5, nprobe=4, device="cpu", backend="torch",
)
```

### IVF-PQ

Build and search a compressed index with the same vectors.

```python
from tensorvs import (
    build_ivfpq_index,
    load_ivfpq_index,
    save_ivfpq_index,
    search_ivfpq_index,
)

build = build_ivfpq_index(
    vectors, nlist=8, niter=5, m=4, nbits=4, device="cpu", use_triton=False,
)
save_ivfpq_index(build, "./ivfpq-index")

index = load_ivfpq_index("./ivfpq-index")
distances, ids = search_ivfpq_index(
    index, vectors[:5], topk=5, nprobe=4, device="cpu", backend="torch",
)
```

## Benchmarks

We evaluate on GloVe with 1,193,514 text vectors of 200 dimensions and Tencent Hunyuan with 1,000,000 text vectors of 768 dimensions. Each index uses 1,024 IVF lists, and each query returns its top 10 neighbors with a query batch size of 128. The table reports TensorVS throughput relative to each baseline at the Recall@10 shown in each cell. Faiss Ascend uses its CPU fallback for configurations unsupported by its NPU implementation.

| Accelerator | Index | TensorVS path | Baseline | GloVe | Hunyuan |
| --- | --- | --- | --- | ---: | ---: |
| NVIDIA RTX 4090 | IVF-Flat | Triton | cuVS | 2.55× at 99% | 7.4× at 99% |
| NVIDIA RTX 4090 | IVF-Flat | Triton | Faiss GPU | 4.5× at 99% | 8.6× at 99% |
| NVIDIA RTX 4090 | IVF-PQ | Triton | cuVS | 2.4× at 36% | 1.3× at 70% |
| AMD MI300X VF | IVF-Flat | Tensor | hipVS | 1.5× at 99% | 10.8× at 99% |
| AMD MI300X VF | IVF-PQ | Triton | hipVS | 2.3× at 40% | 1.6× at 70% |
| Apple M5 | IVF-Flat | Tensor | Faiss Metal | 4.0× at 99% | 8.6× at 99% |
| Apple M5 | IVF-PQ | Tensor | Faiss Metal | 2.1× at 41% | 2.2× at 70% |
| Huawei Ascend 910B4 | IVF-PQ | Tensor | Faiss Ascend | — | 1.7× at 70% |

More experimental results are available in the [benchmark report](docs/benchmarks.md).
