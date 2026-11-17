<p align="center">
  <img src="docs/assets/tensorvs-icon.png" width="112" alt="TensorVS icon">
</p>

<h1 align="center">TensorVS</h1>

<p align="center"><strong>Tensor-native vector search for heterogeneous accelerators.</strong></p>

<p align="center">
  <a href="#platform-support">Platform Support</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="docs/benchmarks.md">Benchmarks</a> ·
  <a href="docs/usage-and-benchmarks.md">Usage &amp; Reproduction</a>
</p>

---

- **Two execution paths.** Run the same saved IVF index with portable PyTorch tensor operations or fused Triton kernels on supported GPUs.
- **IVF-Flat reuses vectors across queries.** Adaptive list-major tiling groups queries probing the same list, balances work across scan tasks, and merges local results with packed-key top-k on the Triton path.
- **IVF-PQ builds one lookup table per query.** A precomputed per-vector correction lets every probed list share the table, so the cost and storage of building it are independent of `nprobe`.

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

[IVF-PQ and exact reranking](docs/usage-and-benchmarks.md#ivf-pq-and-refinement) ·
[Backend setup](docs/usage-and-benchmarks.md#search-backends) ·
[Examples](examples/)
