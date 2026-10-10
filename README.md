# TensorVS

TensorVS is a tensor-based IVF-Flat and residual IVF-PQ vector search project built on top of `flash-kmeans`.

## 目录结构

- `src/tensorvs/`: 项目 Python 包
- `examples/ivfflat_search.py`: IVF-Flat 建索引和搜索示例
- `examples/ivfpq_search.py`: residual IVF-PQ 建索引、搜索和可选重排示例
- `tests/test_index.py`: 训练一致性、元数据和精确重排测试
- `flash-kmeans/`: 本地 `flash-kmeans` 库目录
  - `flash_kmeans/training.py`: 普通 Lloyd KMeans、Balanced KMeans 以及训练采样/初始化

## Mac 搜索对照与结果文件

最终结果统一使用 `results/<dataset>_ivfflat.csv` 和
`results/<dataset>_ivfpq.csv`，每份 CSV 合并 TensorVS、Faiss Metal 以及
QBatch 128、256、512。CSV 同时记录绘图点、初测三次或复测七次的计时样本和必要的索引/测量配置；
不再保留重复 JSON、日志或历史结果目录。图只输出 PDF 和 PNG，位于 `plot/Figures/`。

```bash
conda run -n tensorvs python -u bench/compare_mps_faiss.py --dataset /Users/yitongsong/Dataset/sift1m
conda run -n tensorvs python -u bench/compare_mps_faiss.py --dataset /Users/yitongsong/Dataset/glove1.2m
conda run -n tensorvs python -u bench/remeasure_comparison.py
conda run -n tensorvs python plot/qbatch_comparison.py
```

查询与 ground truth 默认从数据集目录的 `*_query.fvecs`、`*_groundtruth.ivecs`
读取，也可通过 `--queries`、`--groundtruth` 显式指定（支持 `.npy`）。
已有最终 CSV 时脚本会拒绝覆盖；需要重测时先移除对应的最终 CSV。
每张图用三个子图展示三个 QBatch，横轴 Recall@10、纵轴采用对数 QPS，各子图共用坐标范围。
IVFFlat 的 Recall 从 0.9 开始；纵轴用明确的 $10^n$ 对数刻度。
无标题和底部说明，QBatch 标在子图右上角，数据集名称标在左下角；图例仅显示方法名。
默认曲线连接实际测得的性能前沿：只选取不会同时被其他点以更高/相等 Recall 和更高/相等 QPS
超过的点，不修改数值。CSV 保留全部点；`--curve-mode all` 可连接全部实测点。
最多使用前 1,024 条查询；查询不足时使用全部有效查询（当前 GloVe 为 1,000 条，SIFT 为 1,024 条），
实际数量记录在 CSV 中。不足整批的尾批也参与预热和计时。初测每个配置完整预热两遍，取三遍同步计时的中位数；
索引加载、转换、编译和 recall 计算不计入 QPS。

`remeasure_comparison.py` 检查计时样本波动超过 12%、相邻探测数的 QPS 反升超过 2.5%
以及偏离相邻点趋势超过 10% 的低谷，并复测这些配置和相邻配置。
复测完整预热两遍后，按正序/逆序交替运行七轮；每个样本至少累计 0.4 秒同步查询计时，
将每轮耗时按完整查询扫描次数归一化，再取七轮中位数计算 QPS。
最终 CSV 的 `sample1_seconds`～`sample7_seconds` 是每次完整查询扫描的平均耗时，
`sample_sweep_counts` 记录各样本的扫描次数；未复测点保留原三次样本。
复测只用真实测量值替换对应行，不强制 QPS 或 PQ Recall 单调，不额外保存历史结果或日志。

TensorVS 的 IVFFlat 使用 MPS grouped 扫描，IVFPQ 使用编译后的 MPS bucketed
扫描（FP16 LUT、FP32 累加）。本机 Faiss Metal 的 IVFFlat 在 GPU 上查询，
IVFPQ wrapper 则在 CPU 上查询（4 个 OpenMP 线程、FP32 LUT）。CSV
记录执行设备和精度配置，图例只显示 `TensorVS` 与 `Faiss Metal`。
Faiss 使用原 TensorVS 索引的列表、ID、向量/PQ codes、码本；
带正交变换的 PQ 同时导入原变换和变换后的 coarse centroids，不重新训练。
Faiss API 计时包含内部输入上传/结果读回和 PQ 变换，TensorVS 使用驻留 MPS 的输入/输出。
临时导出文件放在系统临时目录，完成后自动删除；Torch 与 Faiss 在独立进程运行。

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
- `torch`：TensorNative 实现，只组合标准 PyTorch 张量算子；用有内存预算的块扫描，
  保留块内 Top-K，再做最终或在线合并。没有手写 Metal/CUDA 内核，也不调用 Faiss 搜索。

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

MPS 搜索已在 Apple M5 上使用 SIFT1M 验证，保持默认 `use_compile=False`。两个示例在 MPS 机器上
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
  用 GPU 前缀和与 `searchsorted` 映射位置。扫描长度使用索引中 nprobe 个最长列表
  的总长度上界，不读取 GPU 候选数量标量，避免 CPU 分桶、任务上传及局部结果散写。

这两条路径的块大小同时受内存预算和 `torch_candidate_chunk_size` 控制：Flat 最多
合并 32 个基础候选块，PQ 最多合并 4 个；单块上限分别为 65536 和 8192 个候选。
预算不足、查询批次较小或 Flat 探测稀疏时，使用按列表共享数据的分块路径。
CPU 和 MPS 也使用该通用分块路径。
IVF-PQ 可显式设置 `backend="torch", torch_pq_scan="gpu"`，在 MPS/CUDA 等
PyTorch 设备上使用上述连续候选流；候选定位、ADC 和 Top-K 都是设备上的标准张量操作。
索引初始化时从列表长度生成静态扫描上界；查询不下载 probe 编号、不调用 `.item()`
决定循环次数，也不在 CPU 构造查询任务。Python 仍提交固定数量的算子，结果留在设备上。
该策略完整扫描选中列表，尾部通过 GPU mask 排除，不截断候选；其代价是扫描上界的
额外填充和每查询重复读取 PQ codes，因此不保证快于列表共享扫描。
连续候选流还按预算划分固定查询小块，以容纳较长候选块，减少候选定位和局部 Top-K
的调用次数；粗筛和 LUT 仍按 API 查询批次计算。这些小块尺寸不依赖 GPU 查询内容。
开启 `use_compile=True` 后，连续候选流还会根据索引统计和调用参数自动选择：

- 定位间距取最短非空列表长度的二次幂下界，最多 1024。一次 GPU `searchsorted`
  建立稀疏锚点，候选随后用张量索引和比较定位；间距内最多跨越一个非空列表边界。
  有空列表时只在 GPU 重排列所选列表；候选流较短、列表很短或映射内存过大时保留通用二分定位。
  `nprobe=1` 直接定位单个列表，无需二分查找。
- 候选流长且 K 相对较小时，先求连续小块的最小距离，只保留最小值最低的 K 个小块，
  收集这些块的全部分数再做最终 Top-K。被排除块的最小值不小于已选 K 块的最小值，
  因而已选块中至少有 K 个候选不差于任何被排除候选，保证精确 Top-K 距离。
  距离相等时允许选择不同的边界 ID；不承诺稳定排序。块宽度随候选跨度在 32～1024
  的二次幂之间选择，目标约 2048 个最小值，且要求 `4 * K * 块宽 < 候选跨度`。
  短流或较大 K 保留全局/原有两级 Top-K。只读取最终 K 个候选的 ID。
  工作区不足以保持合理查询并行度时，回退到原有的分块扫描和收集/在线合并。
- 最小值筛选只将扫描跨度对齐到上述小块宽度，减少分数填充与非连续 reduction 的开销；
  工作区估算包含最小值、候选收集、地址与选择缓冲。分块流的最后一块只计算剩余上界；
  原有两级 Top-K 若块对齐造成超过 1/8 的额外填充，
  只计算到索引上界，再给 Top-K 补无穷大分数。其余尾部保留对齐以减少复制成本。
  所有无效候选的读取位置都重定向到同一安全行，避免读取无关索引行。

上述规则不包含数据集名称，也不要求 SIFT 的维度、子空间数、列表长度或 K。
映射构建及保留缓冲也计入工作区估算；编译器/设备库的内部临时缓冲仍不受此预算保证。
它们不会改变选中列表或截断候选，但属于形状和内存启发式，不保证在所有后端最快。
特别是，不读取查询数量到 CPU、也不使用自定义动态设备调度时，通用静态张量图仍需
保留索引上界与实际候选数之间的填充；这些优化减少其开销，并未彻底消除这一差额。
`torch_pq_scan="grouped"` 固定列表共享扫描，`"auto"` 保留原有设备选择。
允许少量查询相关 CPU 处理时，可设置 `torch_pq_scan="bucketed"`（建议配合
`use_compile=True`）：每个 API 查询批次只下载每条查询的实际候选数量，再在 CPU
按长度排序，上传查询排列；probe 编号、LUT、PQ codes、距离及 Top-K 留在执行设备上。
长度相近的查询按内存预算组成最多 128 条查询的小块，扫描上界为小块内的实际最大长度，
完整扫描选中列表，不再使用索引中最长列表之和作为查询扫描长度。
先规划所有长度桶，再在 GPU 重排较小的查询矩阵并进行 PQ 旋转和 LUT 构建，
不再先构建 LUT 后重排整张表。LUT 仍按完整 API 查询批次一次生成，保持矩阵乘法
的批量大小；扫描时只对已排好顺序的 LUT 切片，结果 ID/距离恢复原查询顺序。
FP16 LUT 用一次标准 `Tensor.to(..., memory_format=torch.contiguous_format)` 完成
布局和精度转换，避免中间的连续 FP32 表；FP32 LUT 仍使用 `contiguous()`。
它保留小块内的局部填充；预算不足、K 较大或未编译时，按实际长度分块扫描。
这仍是标准 PyTorch TensorNative 实现，不引入手写设备内核，但允许每批一次长度同步
及 CPU 查询分组，与 `"gpu"` 的不读回查询元数据约束不同。
编译后的 anchor ADC 根据索引行数与候选跨度选择地址宽度：安全范围内采用 int32，
超出范围保留 int64，不读取查询张量来作这一判断。byte 编码每次读取一个 int32 字，
通过移位与掩码解出最多 4 个子空间的 code，减少重复编码读取；LUT 存储精度及
按子空间顺序的 FP32 累加保持不变。连续行的子空间数能被 4 整除、且支持小端 dtype
视图时，共享原始编码存储，不复制整份索引；其他宽度/后端用标准张量算子构造一次
缓存。不足 4 个子空间及 16-bit 编码保留逐码读取。短列表的 binary 定位与 eager
扫描仍使用原有实现；这些规则不依赖数据集名称，也不增加手写设备内核。
交替对照可运行 `conda run -n tensorvs python bench/mps_ivfpq_adc_compare.py`；
本机还可加 `--faiss`，以独立进程交替测量 Faiss Metal 的 4 线程 CPU IVF-PQ。
实验中的完全紧密一维 ADC 保存在 `bench/_pq_compact_candidate.py`，可用
`bench/mps_ivfpq_compact_compare.py` 对照：`row` 全部按真实长度选 Top-K；`blocked`
只对尾部 Top-K 分数作填充，ADC 无填充。本机 SIFT 对照中，`bucketed` 更快。
需要固定 IVF-Flat 扫描策略（例如比较 QPS–Recall 曲线）时，可设置
`backend="torch", torch_flat_scan="grouped"`，始终只扫描选中的列表。
`torch_flat_scan="dense"` 固定全库距离计算后过滤；默认 `"auto"` 保留上述自动选择。
它每批将选中的列表 ID 传回主机一次，然后按查询/候选数量分桶；小形状会适度
填充并合并，减少算子启动次数。
Flat 的 eager 扫描将查询分组宽度缩小到实际需要的二次幂，减少稀疏探测时的无效距离计算；
使用 `index_select` 读取向量，复用每条查询的范数，并在块内 Top-K 后才读取邻居 ID。
局部 Top-K 按 `||x||² - 2q·x` 排名，只在选出的 K 个值上加回查询范数并截断至非负距离，
减少大张量上的加法和截断。仍使用 FP32，但运算顺序变化可能改变浮点近似相等候选的顺序。
这些优化仍使用相同的标准张量算子组合，在 CPU、MPS 及框架支持的其他加速器上执行；
具体算子支持和速度取决于 PyTorch 后端，不引入芯片专用搜索路径。

Flat 和 PQ eager 分块自动尝试不超过 1024 个候选的块，减少长列表块的填充计算。
如果更多局部 Top-K 会使结果缓冲超出预算、触发逐块合并，则保留原有较大的候选块。
`torch_candidate_chunk_size` 仍是块大小上限，默认 2048；此策略不改变 compiled 或 dense 扫描。
复测命令：`conda run -n tensorvs python bench/mps_search_bench.py --kind ivfflat --flat-topks 10 20 40 60 80 100 --result-suffix tensor_native_v2`。

PQ 的 eager 按列表扫描同样按实际查询数量使用二次幂分组，不再强制填充到 8 条查询。
每组最多 8 条查询，以减少高探测数下尾部分组的填充；API 查询批次仍由
`torch_query_batch_size` 控制。
查表直接生成全局 LUT 的一维地址，再用 `index_select` 读取，避免为每个列表任务复制整张
查询 LUT 和执行四维广播 `gather`。LUT 地址范围允许时采用 int32，超出范围时保留 int64；
按最多 16 个子空间归约，内存估算同时计入地址、查表值及 FP32 转换临时张量。
子空间值按候选连续排列，直接指定 FP32 求和，省去显式转换张量和零初始化累加。
PQ 编码与静态修正也通过 `index_select` 读取，距离用
FP32 原地累加，邻居 ID 仅在块内 Top-K 后读取。这些操作保留压缩 PQ codes，
无需展开成完整向量，不改变用户选择的 LUT 存储精度，也不调用自定义 GPU 内核。
每个列表任务中，探测偏置对同一查询的所有候选相同，因此在局部 Top-K 后才加回并
截断为非负距离，减少候选块上的广播操作；浮点近似相等候选的顺序可能变化。
`use_compile=True` 的 PQ 块和 Triton 路径保留原有实现。
诊断与交替复测：`bench/mps_ivfpq_diagnose.py`、`bench/mps_ivfpq_compare.py`，
均使用 `conda run -n tensorvs python ...` 执行；交替复测需要保存优化前的源码快照。

`use_compile=True` 使用 `torch.compile(fullgraph=True, dynamic=True)` 编译纯张量
扫描块和合并块；列表调度保持 eager。编译配置按设备、dtype、块大小和 K 隔离缓存，避免跨设备/形状搜索触发
共享函数的重编译上限。编译是显式选项，首次编译时间不应计入稳定搜索性能。
本机的 PyTorch 2.14.1 已验证支持 MPS 编译；这是 PyTorch 对同一套张量表达式的自动编译，
没有新增手写 Metal/CUDA 搜索内核。它在高 nprobe 的 IVF-PQ 上更快，低 nprobe 上可能更慢，
因此默认仍为 eager，测试结果应分别标注。完整预热后的编译测试可使用
`conda run -n tensorvs python bench/mps_search_bench.py --kind ivfpq --pq-topks 10 --pq-batch 256 --warmup-scope all-batches --use-compile --output-dir results/ivfpq_torch_round2/compiled/qbatch256`。

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
