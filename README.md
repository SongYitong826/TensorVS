<p align="center">
  <img src="docs/assets/tensorvs-icon.png" width="112" alt="TensorVS icon">
</p>

<h1 align="center">TensorVS</h1>

<p align="center"><strong>High-performance vector search for any tensor platform.</strong></p>

<p align="center">
  <a href="#platform-support">Platform Support</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="docs/usage.md">Usage Guide</a>
</p>

---

- **Vector search on any tensor platform.** Run IVF-Flat and IVF-PQ wherever the required PyTorch tensor operations are available. Search the same saved index across NVIDIA, AMD, Apple Silicon, and Ascend.
- **Two execution paths for portability and performance.** The Tensor path uses standard batched tensor operations to make IVF search portable across accelerator platforms. The Triton path uses fused kernels to deliver high-performance search on CUDA GPUs.
- **IVF-Flat with cross-query vector reuse.** Adaptive list-major scanning reuses full-precision vector tiles across queries probing the same list and splits long or heavily probed lists into parallel tasks.
- **IVF-PQ with a query-global LUT.** Search compact residual codes with one lookup table per query, shared across all probed lists, keeping LUT construction and storage independent of `nprobe`.

## Platform Support

| Platform | Search backend | Verified configuration |
| --- | --- | --- |
| CPU | PyTorch (`cpu`) | IVF-Flat and IVF-PQ automated tests |
| NVIDIA GPU | PyTorch or Triton (`cuda`) | RTX 4090 · IVF-Flat and IVF-PQ Triton path |
| AMD GPU | PyTorch or Triton on ROCm (`cuda`) | MI300X VF · IVF-Flat and IVF-PQ Tensor and Triton paths |
| Apple Silicon GPU | PyTorch MPS (`mps`) | Apple M5 · IVF-Flat and IVF-PQ |
| Huawei Ascend NPU | PyTorch with `torch_npu` and CANN (`npu:0`) | Ascend 910B4 · IVF-Flat and IVF-PQ |
| Cambricon MLU | PyTorch with `torch_mlu` (`mlu:0`) | MLU590-M9DG · Tensor path |

Validation is pending for Kunlunxin XPU.

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
