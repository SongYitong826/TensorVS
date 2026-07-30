# TensorVS

TensorVS is a tensor-based IVF-Flat and residual IVF-PQ vector search project built on top of `flash-kmeans`.

## 目录结构

- `src/tensorvs/`: 项目 Python 包
- `examples/ivfflat_search.py`: IVF-Flat 建索引和搜索示例
- `examples/ivfpq_search.py`: residual IVF-PQ 建索引、搜索和可选重排示例
- `tests/test_index.py`: 训练一致性、元数据和精确重排测试
- `flash-kmeans/`: 本地 `flash-kmeans` 库目录

## 安装

当前仓库已经内置了本地 `flash-kmeans` 源码，直接安装主项目即可：

```bash
cd /home/songyitong/TensorVS
python -m pip install numpy
python -m pip install torch
python -m pip install -e .
```

这条命令会同时安装：

- `tensorvs`
- 仓库内置的 `flash_kmeans`

如果你使用 GPU 并希望启用 Triton 加速，确保 CUDA 可用且依赖安装成功。

## 示例

```bash
conda run -n tensorvs python examples/ivfflat_search.py
conda run -n tensorvs python examples/ivfpq_search.py
```

## 使用说明

在 `examples/ivfflat_search.py` 和 `examples/ivfpq_search.py` 里，可以集中配置建索引和搜索参数，例如：

```python
DATASET_PATH = Path("/storage/vector_data/sift1m/sift1m_base.fvecs")
OUTPUT_DIR = Path("/storage/vector_data/sift1m/GPUIndex")
NLIST = 1024
NITER = 25
USE_TRITON = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

直接调用 TensorVS 聚类并落盘：

```python
from tensorvs import build_ivfflat_index, load_dataset, save_ivfflat_index

vectors = load_dataset("/storage/vector_data/sift1m/sift1m_base.fvecs")
build_result = build_ivfflat_index(
    vectors=vectors,
    nlist=1024,
    niter=25,
    use_triton=False,
    device="cpu",
    seed=0,
    n_init=2,
)
save_ivfflat_index(build_result, "/storage/vector_data/sift1m/GPUIndex/ivfflat.bin")
```

`load_dataset` 支持 `.fvecs`、`.pt/.pth` 和 `.npy`。`seed` 控制可复现性，
`n_init` 控制独立 KMeans 初始化次数；每个 coarse/PQ 训练阶段会使用独立派生 seed，
并保留 training inertia 最低的量化器。默认 `n_init=2`。

IVF-PQ 可以从更大的近似候选集合中，用外部原始向量进行精确重排：

```python
distances, ids = search_ivfpq_index(
    index,
    queries,
    topk=10,
    nprobe=64,
    refine_vectors=vectors,
    refine_factor=4,
)
```

这会先检索 40 个 PQ 候选，再按原始向量的精确平方 L2 距离返回 top10；
原始向量不会重复存进 PQ 索引。

生成的 IVF-Flat 索引目录会包含：

- `centroids.pt`
- `assignments.pt`
- `ids.pt`
- `metadata.json`
- `list_vectors.pt`
- `list_vector_norms.pt`
- `list_ids_flat.pt`
- `list_prefix.pt`
- `list_count.pt`

为减少磁盘占用，不会额外落盘一份 `vectors.pt`；向量只在按 cluster 排序后的
`list_vectors.pt` 中保存一次。IVF-PQ 索引只保存 PQ codes，不保存原始向量；它会同时
落盘两种 list 布局：`list_codes.pt`、`list_ids_flat.pt`、`list_prefix.pt`、
`list_count.pt` 以及行主序的 `list_pq_static_corrections.pt` 供
`tensor_topk` 使用；后者让 Tensor 路线以 query-global LUT 在查表时重建
残差 ADC 值，而无需物化每个 probe 的完整 LUT。另有
`list_codes_interleaved_by_subspace.pt`、`list_ids_interleaved.pt`、
`list_group_prefix.pt`、`list_pq_static_corrections_interleaved.pt` 供
`triton_topk` 使用。调用
`index.reside_on_device("cuda:0")` 时，两种布局都会常驻到该 GPU。
