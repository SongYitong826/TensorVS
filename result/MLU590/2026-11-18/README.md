# TensorVS 寒武纪基准 · 2026-11-18

当前 TensorVS 在单张 MLU590-M9DG 80 GiB 上的未优化基准。主实验固定 batch=128、nlist=1024、top-k=10，两个百万条数据集、两种索引，共22个完整的 QPS–Recall 参数点。

## 结果

以下为 nprobe=1024，完整曲线见 [query-results.csv](query-results.csv)。

| 数据集 / 索引 | Build / 秒 | QPS | Recall@10 | 搜索进程采样峰值 / GB |
|---|---:|---:|---:|---:|
| GloVe200-1M IVF-Flat | 0.8435 | 1426.21 | 100.00% | 1.5252 |
| GloVe200-1M IVF-PQ | 2.2264 | 158.96 | 30.32% | 0.4288 |
| Gauss768-1M IVF-Flat | 1.7126 | 957.87 | 100.00% | 4.1402 |
| Gauss768-1M IVF-PQ | 46.7677 | 13.77 | 76.01% | 0.9676 |

搜索进程显存来自独立重放，以上为该 nprobe 的采样最大值；GB=10⁹ bytes。Build不含读文件、布局打包与保存，四组均为单次构建实测。

## 环境与数据

- 2×AMD EPYC 9K84 96核，192物理核/384逻辑线程，主存1.5 TiB；实验CPU线程数16。
- TencentOS 4宿主，Ubuntu 24.04容器，Linux 6.6.98-40.9.tl4.x86_64；MLU驱动6.5.48、固件1.1.2。
- Python3.10.20、NumPy1.26.4、PyTorch2.7.1+cpu、torch_mlu1.29.2+torch2.7.1，实际设备为 `mlu:0`。仓库声明Python≥3.11，本次保留厂商3.10环境，偏差已记录。
- 源码基于 `fe15d623ac6e0493842e685a5dd67cf1f5e70ae9`，flash-kmeans为 `2c3c68481761caea2c6100f9bd5c951dc04271f8`。实际文件校验值及未启用的P800兼容补丁保存在 `environment/`，本分支不修改检索源码。
- GloVe取ANN-benchmarks原始train前100万行、test前1000行，保持float32、不归一化；对这100万条重新计算 squared-L2 exact top100。
- Gauss为自行生成的768维标准高斯数据：NumPy PCG64、seed=20261118，100万条base、1000条query，float32、不归一化。它不是真实Hunyuan embedding，不能直接当作论文工业Hunyuan结果。

## 测量口径

- backend=torch，Triton/compile/TF32关闭；MLU原生uint8选取，P800 int8兼容开关未启用。
- 残差IVF-PQ：GloVe M=25、Gauss M=384，8bit，LUT fp16。完整训练与分块参数保留在JSON中。
- 每个点完整预热2轮、计时3轮1000条查询，逐batch同步，QPS=1000/三轮总耗时中位数。末batch补齐，查询已驻留设备；排除输入传输、结果回传及Recall统计。
- 原始计时记录的allocator峰值与进程物理显存分开保存。进程显存使用CNDEV独立重放采样，目标间隔20ms；实际间隔及样本数保留在 `results/Memory/`，采样最大值可能漏掉更短的瞬时峰值。
- 主实验22点全部完成后主动停止Gauss PQ的额外batch32/512扫描。原始Gauss JSON保留 `failed` / `KeyboardInterrupt`，这是已记录的停止额外扫描动作；22点主实验及四组显存重放均完整。控制记录见 [PAPER_BASELINE_COMPLETE.json](results/PAPER_BASELINE_COMPLETE.json)。

## 文件

| 文件 | 内容 |
|---|---|
| [当前基准说明.txt](当前基准说明.txt) / [report.html](report.html) | 中文报告与全部主实验点 |
| [query-results.csv](query-results.csv) | batch128的22点查询结果、三轮耗时与显存 |
| [build-results.csv](build-results.csv) | 四组构建、保存、索引大小与构建allocator峰值 |
| [paper-resource-values.csv](paper-resource-values.csv) | Build、索引GB、搜索进程与allocator显存 |
| [baseline-summary.json](baseline-summary.json) | 主实验覆盖率、参数、结果与软件清单 |
| [additional-query-results.csv](additional-query-results.csv) | 此前额外测得的34个batch32/512点；不声称66点扩展网格完整 |
| `results/` | 原始JSON/CSV、Build、Memory、采样正确性与控制记录 |
| `environment/` | 软硬件清单、镜像标识、数据/源码/索引校验值 |

本次按要求仅上传报告、结果数据和环境记录。完整复现包、源码快照、运行脚本、日志、向量及索引本体暂不上传。原主机数据位于 `/data/home/petrizhang/tensorvs-mlu`，本地完整归档位于 `/Users/petrizhang/Downloads/TensorVS-MLU590-baseline-20261118`。
