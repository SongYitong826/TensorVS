<p align="center">
  <img src="docs/assets/tensorvs-icon.png" width="112" alt="TensorVS icon">
</p>

<h1 align="center">TensorVS</h1>

<p align="center"><strong>Tensor-native vector search for heterogeneous accelerators.</strong></p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="docs/usage-and-benchmarks.md">Usage &amp; Reproduction</a>
</p>

---

- **Two execution paths.** Run the same saved IVF index with portable PyTorch tensor operations or fused Triton kernels on supported GPUs.
- **IVF-Flat reuses vectors across queries.** Adaptive list-major tiling groups queries probing the same list, balances work across scan tasks, and merges local results with packed-key top-k on the Triton path.
- **IVF-PQ builds one lookup table per query.** A precomputed per-vector correction lets every probed list share the table, so the cost and storage of building it are independent of `nprobe`.

## Benchmarks

**GloVe 1.2M and Tencent Hunyuan.** QPS versus Recall@10 with 1,024 IVF lists. Each line sweeps the number of probed lists. IVF-Flat shows query batches of 32, 128, and 512 on the three GPU platforms and 128 on Ascend. IVF-PQ uses 128.

**NVIDIA RTX 4090 · CUDA**

**IVF-Flat**

<img src="docs/assets/benchmark-nvidia-ivfflat.png" width="670" alt="RTX 4090 IVF-Flat on GloVe and Hunyuan at query batches of 32, 128, and 512">

**IVF-PQ**

<img src="docs/assets/benchmark-nvidia-ivfpq.png" width="670" alt="RTX 4090 IVF-PQ on GloVe and Hunyuan at query batch 128">

<details>
<summary><strong>AMD MI300X VF · ROCm</strong></summary>

**IVF-Flat**

<img src="docs/assets/benchmark-amd-ivfflat.png" width="830" alt="AMD MI300X VF IVF-Flat on GloVe and Hunyuan at query batches of 32, 128, and 512">

**IVF-PQ**

<img src="docs/assets/benchmark-amd-ivfpq.png" width="670" alt="AMD MI300X VF IVF-PQ on GloVe and Hunyuan at query batch 128">

</details>

<details>
<summary><strong>Apple M5 · Metal / MPS</strong></summary>

**IVF-Flat**

<img src="docs/assets/benchmark-apple-ivfflat.png" width="830" alt="Apple M5 IVF-Flat on GloVe and Hunyuan at query batches of 32, 128, and 512">

**IVF-PQ**

<img src="docs/assets/benchmark-apple-ivfpq.png" width="670" alt="Apple M5 IVF-PQ on GloVe and Hunyuan at query batch 128">

</details>

<details>
<summary><strong>Huawei Ascend 910B4 · NPU</strong></summary>

**IVF-Flat**

<img src="docs/assets/benchmark-ascend-ivfflat.png" width="670" alt="Ascend 910B4 IVF-Flat on GloVe and Hunyuan at query batch 128">

**IVF-PQ**

<img src="docs/assets/benchmark-ascend-ivfpq.png" width="670" alt="Ascend 910B4 IVF-PQ on GloVe and Hunyuan at query batch 128">

The GloVe IVF-PQ panel includes Faiss Ascend with inner-product ground truth and TensorVS with squared-L2 ground truth.

</details>

[Benchmark sources and reproduction](docs/usage-and-benchmarks.md#readme-figures)

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
