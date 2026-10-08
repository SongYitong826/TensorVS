# TensorVS

TensorVS is a tensor-based IVF-Flat and residual IVF-PQ vector search project built on top of `flash-kmeans`.

## 目录结构

- `src/tensorvs/`: 项目 Python 包
- `examples/ivfflat_search.py`: IVF-Flat 建索引和搜索示例
- `examples/ivfpq_search.py`: residual IVF-PQ 建索引、搜索和可选重排示例
- `tests/test_index.py`: 训练一致性、元数据和精确重排测试
- `flash-kmeans/`: 本地 `flash-kmeans` 库目录
  - `flash_kmeans/training.py`: 普通 Lloyd KMeans、Balanced KMeans 以及训练采样/初始化

## 安装

`flash-kmeans` 通过 Git 子模块提供。首次克隆时一并获取子模块：

```bash
git clone --recurse-submodules git@github.com:SongYitong826/TensorVS.git
cd TensorVS
```

已有仓库先执行 `git submodule update --init --recursive`，然后安装主项目：

```bash
conda run -n tensorvs python -m pip install -e .
```

这条命令会同时安装：

- `tensorvs`
- 仓库内置的 `flash_kmeans`

基础安装不再强制依赖 Triton，适用于 CPU 和 Mac 的 MPS 搜索。
Linux 上需要 Triton 加速时，安装可选依赖：

```bash
conda run -n tensorvs python -m pip install -e '.[triton]'
```

PyTorch 的平台发行包可能自带 Triton 依赖；这里移除的是 TensorVS 的强制依赖。
仓库内置的 `flash_kmeans` 不需要另外安装。`dev` extra 提供 pytest。

## 示例

```bash
conda run -n tensorvs python examples/ivfflat_search.py
conda run -n tensorvs python examples/ivfpq_search.py
```

## 使用说明

### 独立的 Torch Baseline

`src/tensorvs/torch_baseline.py` 提供传统 IVF-Flat 和 residual IVF-PQ 搜索，
仅用于对比，独立于正式方法及 `backend` 路由。复用同一已训练索引，避免
训练差异影响搜索对比：

```python
from tensorvs.torch_baseline import TorchIVFFlatBaseline, TorchIVFPQBaseline

flat_baseline = TorchIVFFlatBaseline.from_index(flat_index, device="cuda:0")
pq_baseline = TorchIVFPQBaseline.from_index(pq_index, device="cuda:0")
distances, ids = flat_baseline.search(queries, topk=10, nprobe=64)
pq_distances, pq_ids = pq_baseline.search(queries, topk=10, nprobe=64)
# CPU / Mac: device="cpu" / "mps"；省略设备时按 CUDA → MPS → CPU 选择。
```

也可以直接用原始向量/PQ codes、coarse assignments 和已训练的 centroids/codebooks
构造 Baseline。`.to(device)` 返回另一个设备上的 Baseline，不改变原实例。
准备和传输索引应在测速前完成；搜索输出为平方 L2 距离及原始 ID。

搜索逐条处理查询：粗距离 → 选取 `nprobe` 个列表 → 收集候选 → 距离/ADC →
一次全局 Top-K。Flat 直接计算候选与查询的平方 L2；PQ 对每个探测列表的残差查询
建立 L2 LUT，再按 codes 查表并以 FP32 累加。PQ 默认 `lut_dtype="fp32"`，可显式
设为 `"fp16"`；支持现有索引的 PQ rotation 和 16-bit codes。
普通候选分块 `candidate_chunk_size=4096`、PQ LUT 分块 `lut_probe_chunk_size=8`
仅限制临时内存，仍保存全部候选距离后做 Top-K。候选 ID/距离内存随实际候选数增长。
不使用全库掩码扫描、跨查询分桶、全局 LUT/静态修正、Triton 或 `torch.compile`。
数值计算和候选索引均使用标准 Torch 算子；变长候选的 `repeat_interleave` 可能在
加速器上内部同步。MPS 的实际支持以当前 PyTorch 发行版为准，测试会在有 MPS 的机器上运行。

### 正式方法

在 `examples/ivfflat_search.py` 和 `examples/ivfpq_search.py` 里，可以集中配置建索引和搜索参数，例如：

```python
DATASET_PATH = Path("/storage/vector_data/sift1m/sift1m_base.fvecs")
OUTPUT_DIR = Path("/storage/vector_data/sift1m/GPUIndex")
NLIST = 1024
NITER = 25
from tensorvs import get_default_device

DEVICE = get_default_device()  # CUDA -> MPS -> CPU
TRAIN_DEVICE = torch.device("cpu") if DEVICE.type == "mps" else DEVICE
USE_TRITON = TRAIN_DEVICE.type == "cuda"
BACKEND = "auto"
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

IVF-Flat 和 IVF-PQ 共用 `backend="auto" | "torch" | "triton"`：

- `auto`（默认）：支持的设备和参数优先使用现有 Triton 融合搜索，其他情况使用 PyTorch。
- `triton`：明确要求原有 Triton 路径；设备或参数不支持时报告错误。
- `torch`：用有内存预算的 PyTorch 块扫描，保留块内 Top-K，再做最终或在线合并。

搜索未指定 `device` 时，查询已在非 CPU 设备上就沿用该设备；查询在 CPU 上则按
CUDA → MPS → CPU 的顺序选择。显式 `device="cpu"` 或其他设备始终优先。
`backend` 选择搜索实现，`device` 选择执行设备：支持的 CUDA 配置且安装 Triton 时
`auto` 使用 Triton；MPS、CPU 或缺少 Triton 时使用 PyTorch。

MacBook 的 M 芯片无需手动指定设备即可使用可用的 MPS；也可显式配置：

```python
distances, ids = search_ivfflat_index(
    index, queries, device="mps", backend="auto", use_compile=False,
)
# search_ivfpq_index 同样设置。
```

MPS 搜索尚需实机验证，先保持默认 `use_compile=False`。两个示例在 MPS 机器上
使用 CPU 建索引、MPS 搜索，避免现有训练中的 float64 和设备随机数生成限制；
已有索引可以直接加载搜索。示例路径需改为本机的数据集位置。

旧参数 `selection_mode="triton_topk"` 保留，后端通过 `backend` 选择。PyTorch
路径支持 CPU、框架支持的 GPU、残差 PQ、PQ 旋转、FP16/FP32 LUT 和精确重排：

```python
distances, ids = search_ivfflat_index(
    index, queries, device="cuda:1", backend="torch",
    torch_query_batch_size=128,
    torch_candidate_chunk_size=2048,
    torch_memory_budget_mb=128,
    use_compile=True,
)
# search_ivfpq_index 使用相同的后端、分块和编译参数。
```

候选块与查询块会根据内存预算缩小；默认预算为 128 MiB，约束扫描及 Top-K 合并的张量工作区，
不包含常驻索引、输入/输出、粗筛/LUT 和编译器或矩阵库的内部工作区。
较大的 CUDA 查询批次采用两种调度方式：

- Flat：批次至少 64 个查询且 `nprobe >= nlist / 16` 时，按连续向量块做较大的 GEMM，
  计算全库向量的距离，然后用 IVF 列表掩码排除未选中的候选。
  但参与 Top-K 的候选集合相同；仍使用 IEEE FP32，不以降低精度换取速度。
- PQ：批次至少 32 个查询时，将每个查询选中的变长列表看作一条连续候选流，
  用 GPU 前缀和与 `searchsorted` 映射位置。每批只同步一个候选数量标量，
  避免 CPU 分桶、任务上传及局部结果散写。

这两条路径的块大小同时受内存预算和 `torch_candidate_chunk_size` 控制：Flat 最多
合并 32 个基础候选块，PQ 最多合并 4 个；单块上限分别为 65536 和 8192 个候选。
预算不足、查询批次较小或 Flat 探测稀疏时，使用按列表共享数据的分块路径。
CPU 和 MPS 也使用该通用分块路径。
需要固定 IVF-Flat 扫描策略（例如比较 QPS–Recall 曲线）时，可设置
`backend="torch", torch_flat_scan="grouped"`，始终只扫描选中的列表。
`torch_flat_scan="dense"` 固定全库距离计算后过滤；默认 `"auto"` 保留上述自动选择。
它每批将选中的列表 ID 传回主机一次，然后按查询/候选数量分桶；小形状会适度
填充并合并，减少算子启动次数。

`use_compile=True` 使用 `torch.compile(fullgraph=True, dynamic=True)` 编译纯张量
扫描块和合并块；列表调度保持 eager。编译配置按设备、dtype、块大小和 K 隔离缓存，避免跨设备/形状搜索触发
共享函数的重编译上限。编译是显式选项，首次编译时间不应计入稳定搜索性能。

Triton 的 packed-code、autotune、预分配输出和 CUDA Graph 行为保留。
这些参数仅作用于 Triton；PyTorch 返回独立输出。`auto` 按能力/参数选择后端，
不会捕获和隐藏执行时的 OOM 或未知错误。跨硬件性能仍需实测。

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

默认索引归档继续保存 interleaved 编码，兼容已有索引；不会额外落盘一份
`vectors.pt`。IVF-PQ 索引只保存 PQ codes，不保存原始向量；归档中的布局为
Triton TopK 使用的 interleaved list 布局：`list_count.pt`、
`list_codes_interleaved_by_subspace.pt`、`list_ids_interleaved.pt`、
`list_group_prefix.pt`、`list_pq_static_corrections_interleaved.pt` 供
`triton_topk` 使用。

逻辑列表与执行布局由独立对象管理：`index.logical_lists` 按需还原未填充的
list-major 数据、ID、列表长度/偏移和范数或 PQ 修正值，不依赖 warp 大小。
PyTorch 使用连续布局；Triton 默认使用归档的 interleaved 布局，宽 PQ codes 使用
packed 缓存。设备、硬件属性、PyTorch/runtime 版本、源文件或布局版本改变时，
会选择新的缓存条目并转换/上传所需布局。相同的 Triton interleaved 编码在设备间
可直接复用，只保存设备清单；需要转换的 PyTorch/packed PQ 张量会持久化。

缓存默认位于 `~/.cache/tensorvs/layouts`，可用 `TENSORVS_LAYOUT_CACHE_DIR` 改位置，
或通过 `persist_layout=False` 禁用持久化。缓存采用原子写入，过期条目不会覆盖原索引。
每个索引内部只保留当前后端的设备布局，切换后端会释放另一种 GPU 布局；
`release_device_residency()` 释放
设备缓存，保留主机逻辑数据。预加载默认保持 Triton：

```python
index.reside_on_device("cuda:1")
index.reside_on_device("cpu", backend="torch")
index.reside_on_device(get_default_device(), backend="auto")
```

可使用 `examples/benchmark_search_backends.py` 对已有索引比较稳定搜索时间、
Top-K 一致性和额外显存，并通过 `--compiled` 单独报告编译后结果。
`--all-paths` 还会比较 `auto` 和 PQ 的工作区、CUDA Graph、autotune 及 unpacked
选项；`--query-batch` 固定每次调用的查询数。PQ 全路径测试要求查询数是批次大小的
整数倍，避免计时期间因尾批形状变化重新捕获 CUDA Graph。
可用 `TENSORVS_TRITON_AUTOTUNE_CACHE` 指定独立缓存，区分默认配置和调优配置。
