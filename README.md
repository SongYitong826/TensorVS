# TensorVS

TensorVS is a tensor-based IVF-Flat, residual IVF-PQ, and residual IVF-SQ vector search project built on top of `flash-kmeans`.

## 目录结构

- `src/tensorvs/`: 项目 Python 包
- `examples/ivfflat_search.py`: IVF-Flat 建索引和搜索示例
- `examples/ivfpq_search.py`: residual IVF-PQ 建索引、搜索和可选重排示例
- `examples/ivfsq_search.py`: residual IVF-SQ 建索引、搜索和可选重排示例
- `tests/test_index.py`: 训练一致性、元数据和精确重排测试
- `flash-kmeans/`: 本地 `flash-kmeans` 库目录
  - `flash_kmeans/training.py`: 普通 Lloyd KMeans、Balanced KMeans 以及训练采样/初始化

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
conda run -n tensorvs python examples/ivfsq_search.py
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
    seed=1234,
    n_init=1,
    max_points_per_centroid=256,
)
save_ivfflat_index(build_result, "/storage/vector_data/sift1m/GPUIndex/ivfflat.bin")
```

IVF-SQ 支持 residual scalar quantization 的 `int8` 和 `int4` 两种格式：

```python
from tensorvs import build_ivfsq_index, load_dataset, save_ivfsq_index

vectors = load_dataset("/storage/vector_data/sift1m/sift1m_base.fvecs")
build_result = build_ivfsq_index(
    vectors=vectors,
    nlist=1024,
    niter=25,
    use_triton=False,
    device="cuda:0",
    bits=4,  # 也可以是 8
)
save_ivfsq_index(build_result, "/storage/vector_data/sift1m/GPUIndex/ivfsq4.bin")
```

IVF-SQ 对每个 coarse list 的 residual 做逐维 affine quantization，`int8` 每个
code 占一个字节，`int4` 将两个 code 打包到一个字节。CUDA 上默认使用
`selection_mode="triton_topk"`，直接在 list-grouped 扫描 tile 内解码；CPU 或不支持的
shape 会回退到 `tensor_topk`。也可以使用 `selection_mode="query_major"` 测试
cuVS 风格的 query-major 扫描。原始向量不保存在索引中，可像 IVF-PQ 一样通过
`refine_vectors` 和 `refine_factor` 做精确重排。

`load_dataset` 支持 `.fvecs`、`.pt/.pth` 和 `.npy`。KMeans 默认参数对齐 Faiss：
`niter=25`、`seed=1234`、`n_init=1`（对应 Faiss 的 `nredo`）、
`max_points_per_centroid=256`，并使用 KMeans++ 初始化。普通 Lloyd KMeans 和
cuVS 风格 Balanced KMeans 都由本地 `flash_kmeans` 包提供；IVF-PQ 默认对 IVF
和 PQ 两部分都使用 Balanced KMeans，也可通过
`coarse_kmeans_method` 和 `pq_kmeans_method` 分别选择实现。`n_init` 控制独立
KMeans 初始化次数；每个 coarse/PQ 训练阶段会使用独立派生 seed，并保留 training
inertia 最低的量化器。对于较大的数据集，
`build_ivfflat_index` 和 `build_ivfpq_index` 还支持 Faiss 风格的
`max_points_per_centroid` 参数：每个 KMeans 阶段最多使用
`k * max_points_per_centroid` 个随机采样向量训练，但最终仍会为全量数据生成
IVF labels 和 PQ codes。例如 `max_points_per_centroid=256` 时，coarse 阶段最多使用
`nlist * 256` 个向量，每个 PQ 子空间最多使用 `2**nbits * 256` 个向量。

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

IVF-PQ 的 CUDA 搜索只保留 `selection_mode="triton_topk"`。不满足 CUDA/Triton
形状限制时直接报错，不会回退到 Tensor 或 List-Major 路径。

IVF-Flat 的 CUDA 搜索只使用 `selection_mode="triton_topk"`；CPU 或不支持的
Triton shape 会直接报错，不再回退到 Tensor TopK。

生成的 IVF-Flat 索引目录会包含：

- `centroids.pt`
- `assignments.pt`
- `ids.pt`
- `metadata.json`
- `list_count.pt`
- `list_vectors_interleaved.pt`
- `list_ids_interleaved.pt`
- `list_group_prefix.pt`
- `list_vector_norms_interleaved.pt`

IVF-Flat 只保存 Triton TopK 使用的 interleaved list 布局；按 cluster 排序的
row-major list 张量只在构建过程中临时存在，不会落盘。不会额外落盘一份
`vectors.pt`。IVF-PQ 索引只保存 PQ codes，不保存原始向量；它只落盘
Triton TopK 使用的 interleaved list 布局：`list_count.pt`、
`list_codes_interleaved_by_subspace.pt`、`list_ids_interleaved.pt`、
`list_group_prefix.pt`、`list_pq_static_corrections_interleaved.pt` 供
`triton_topk` 使用。按 cluster 排序的 row-major list 张量只在构建过程中临时存在，
不会落盘。调用 `index.reside_on_device("cuda:0")` 时，这一套布局会常驻到该 GPU。
