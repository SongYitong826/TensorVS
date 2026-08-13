# TensorVS

TensorVS is a tensor-based IVF-Flat, residual IVF-PQ, and residual IVF-SQ vector search project built on top of `flash-kmeans`.

## 目录结构

- `src/tensorvs/`: 项目 Python 包
- `examples/ivfflat_search.py`: IVF-Flat 建索引和搜索示例
- `examples/ivfpq_search.py`: residual IVF-PQ 建索引、搜索和可选重排示例
- `examples/ivfsq_search.py`: residual IVF-SQ 建索引、搜索和可选重排示例
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
    seed=0,
    n_init=2,
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

IVF-PQ 的 CUDA 搜索还支持 `selection_mode="list_major"`（别名
`"triton_list_major"`）。它先把所有 `(query, probe-list)` 边按 List 排序，
再把访问同一 List 的 Query 按小 tile 一起扫描；同一个 32-vector code tile
会被这个 Query tile 共享访问。Global LUT 以构建友好的 `[M, Q, Ksub]` 布局保存，
kernel 内会把当前 Query tile 临时重排成 `[Ksub, QueryTile]` 的 Vector LUT，
让一份 PQ code 和 candidate correction 同时服务这个 Query tile；
Query tile 和 32/64/128 candidate tile 默认按局部 key/LUT 寄存器占用估计选择，
传入 `triton_autotune=True` 时还会在两种 tile 和 warp 配置中实测并缓存最快配置；
当 Query-list 重叠较低时会自动复用 `triton_topk`，避免 List-Major 的分组开销。
不满足 CUDA/Triton 形状限制时会回退到 `tensor_topk`。

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
