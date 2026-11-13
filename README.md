# MADOS 独立 PyTorch 语义分割实验框架

使用 MADOS 官方原始 TIFF、原始划分和 15 类稀疏标注，提供统一的数据读取、训练、验证、评估与结果导出。模型为可替换的小型编码器—解码器，输出原始 logits。项目不含 MariNeXt、MMCV 或 MMSegmentation。

## 1. 数据分析依据

检查的原作者仓库版本：[`c20a7e972bdf111126540d8e6caa45808352f138`](https://github.com/gkakogeorgiou/mados/tree/c20a7e972bdf111126540d8e6caa45808352f138)。主要依据：

- [README 和原环境要求](https://github.com/gkakogeorgiou/mados/tree/c20a7e972bdf111126540d8e6caa45808352f138)。
- [原数据读取代码](https://github.com/gkakogeorgiou/mados/blob/c20a7e972bdf111126540d8e6caa45808352f138/utils/dataset.py)、[类别与归一化常量](https://github.com/gkakogeorgiou/mados/blob/c20a7e972bdf111126540d8e6caa45808352f138/utils/assets.py)。
- [波段堆叠](https://github.com/gkakogeorgiou/mados/blob/c20a7e972bdf111126540d8e6caa45808352f138/utils/stack_patches.py)、[光谱提取](https://github.com/gkakogeorgiou/mados/blob/c20a7e972bdf111126540d8e6caa45808352f138/utils/spectral_extraction.py)。
- [训练](https://github.com/gkakogeorgiou/mados/blob/c20a7e972bdf111126540d8e6caa45808352f138/marinext/train.py)、[评估](https://github.com/gkakogeorgiou/mados/blob/c20a7e972bdf111126540d8e6caa45808352f138/marinext/evaluation.py)、[指标](https://github.com/gkakogeorgiou/mados/blob/c20a7e972bdf111126540d8e6caa45808352f138/utils/metrics.py)。
- [官方数据集与下载](https://zenodo.org/records/10664073)。检查了官方 ZIP 的全部文件目录、三个原始划分文件，并实际读取其中 6 个 patch。

数据集含 174 个场景、2803 个 patch。官方划分为训练 **1433**、验证 **642**、测试 **728** 个 patch。程序直接读取官方 TXT，不重新随机划分；启动时检查三个清单是否存在样本 ID 重叠。

### 1.1 原始目录与文件

`dataset.root` 指向直接包含 `Scene_*` 和 `splits` 的目录。例如：

```text
D:/Datasets/MADOS/splits/train_X.txt
D:/Datasets/MADOS/splits/val_X.txt
D:/Datasets/MADOS/splits/test_X.txt
D:/Datasets/MADOS/Scene_0/10/Scene_0_L2R_rhorc_492_1.tif
D:/Datasets/MADOS/Scene_0/10/Scene_0_L2R_cl_1.tif
D:/Datasets/MADOS/Scene_0/20/Scene_0_L2R_rhorc_704_1.tif
D:/Datasets/MADOS/Scene_0/60/Scene_0_L2R_rhorc_442_1.tif
```

清单中的 `Scene_0_1` 对应场景 `Scene_0` 的第 `1` 个 patch。每个样本读取 **11 个单波段反射率 TIFF 和 1 个标签 TIFF**。标签始终取自 `10` 目录。其他 RGB、置信度、代表性、浊度等辅助文件不作为模型输入。

该读取器针对官方原始目录，不读取原作者另外导出的 `MADOS_nearest`、HDF5 或自定义 RGB/mask 数据布局。

### 1.2 波段顺序、分辨率与数值

顺序与原作者按文件名波长数值排序的结果一致。文件名波长存在以下变体，已核对官方压缩包目录，不能只匹配其中一套。

| Tensor 通道索引 | 波段 | 文件名中的波长 nm | 原始分辨率 | 已核验样本尺寸 |
|---|---|---|---|---|
| 0 | B1 | 442 / 443 | 60 m | 40×40 |
| 1 | B2 | 492 | 10 m | 240×240 |
| 2 | B3 | 559 / 560 | 10 m | 240×240 |
| 3 | B4 | 665 | 10 m | 240×240 |
| 4 | B5 | 704 | 20 m | 120×120 |
| 5 | B6 | 739 / 740 | 20 m | 120×120 |
| 6 | B7 | 780 / 783 | 20 m | 120×120 |
| 7 | B8 | 833 | 10 m | 240×240 |
| 8 | B8A | 864 / 865 | 20 m | 120×120 |
| 9 | B11 | 1610 / 1614 | 20 m | 120×120 |
| 10 | B12 | 2186 / 2202 | 20 m | 120×120 |

实际检查的 6 个样本所有波段均为 `float32`，数值是已处理的反射率；标签为 `uint8`。程序统一将模型输入转换为 `float32`，不假设原始 TIFF 必须是某一存储 dtype。

以下只是这 **6 个样本** 的各波段有限值极值，不能作为整个数据集的取值边界，也不用于裁剪数据：

| 波段 | 样本最小值 | 样本最大值 |
|---|---:|---:|
| B1 | 0.029534 | 0.123755 |
| B2 | 0.017235 | 0.360099 |
| B3 | 0.009735 | 0.386981 |
| B4 | 0.004244 | 0.424189 |
| B5 | 0.002423 | 0.296338 |
| B6 | 0.005589 | 0.363850 |
| B7 | 0.000479 | 0.410297 |
| B8 | 0.005791 | 0.484352 |
| B8A | 0.006006 | 0.413414 |
| B11 | 0.005900 | 0.422667 |
| B12 | 0.003455 | 0.284217 |

没有遍历完整数据集的像素，因此全量各波段的精确极值和所有文件的 dtype 分布尚未确认。**输入不除以 255 或 10000，不擅自限制到 `[0,1]`。**

### 1.3 标签映射与稀疏标注

原始值 **0 是未标注区域，不是背景类别**。这些位置保留在图像中，但不参与损失或任何评价指标。模型预测可覆盖整幅图，只有有标注的位置能够用于定量评价。

| 原始 TIFF 值 | 训练标签 | 原作者类别名 | 中文释义 |
|---:|---:|---|---|
| 0 | -1（默认 ignore_index） | Unlabelled | 未标注，忽略 |
| 1 | 0 | Marine Debris | 海洋垃圾 |
| 2 | 1 | Dense Sargassum | 密集马尾藻 |
| 3 | 2 | Sparse Floating Algae | 稀疏漂浮藻类 |
| 4 | 3 | Natural Organic Material | 天然有机物 |
| 5 | 4 | Ship | 船舶 |
| 6 | 5 | Oil Spill | 溢油 |
| 7 | 6 | Marine Water | 海水 |
| 8 | 7 | Sediment-Laden Water | 富含泥沙的水体 |
| 9 | 8 | Foam | 泡沫 |
| 10 | 9 | Turbid Water | 浑浊水体 |
| 11 | 10 | Shallow Water | 浅水 |
| 12 | 11 | Waves & Wakes | 波浪与尾流 |
| 13 | 12 | Oil Platform | 石油平台 |
| 14 | 13 | Jellyfish | 水母 |
| 15 | 14 | Sea snot | 海洋黏液 |

Dataset 返回 `image: float32[11,H,W]`、`mask: int64[H,W]` 和字符串 `id`。默认标签范围为 `{-1,0,...,14}`，可直接用于 `CrossEntropyLoss(ignore_index=-1)`。忽略值可配置，但不能落入 `0..14`。

### 1.4 必要预处理及依赖取舍

1. 按上述波段顺序读 TIFF。
2. 保留原作者的最近邻重采样：20 m 波段放大 2 倍、60 m 波段放大 6 倍，对齐 10 m 标签；检查每个波段与标签的最终尺寸。
3. 堆叠成 11 通道。这里是同一 patch 内的波段堆叠，不需要跨场景影像拼接。
4. 以对应通道均值填充 NaN。Inf、非法标签、缺失或重复波段直接报错。
5. 训练时执行同步空间增强，再按 `(image - mean) / std` 归一化。验证和评估不做随机增强。

`config.yaml` 完整保留原作者的 11 组均值/标准差；这些数值来自源代码，不是从本次 6 个测试样本重新估算的。

原项目使用 `osgeo.gdal` 读取标签数组，使用 Rasterio 读取影像并重采样。本项目用 Rasterio 完成两种读取，标签结果已与 GDAL 实际读取对照一致，因此不需要单独安装 `osgeo`/GDAL Python 包。**仍然依赖 Rasterio 及其底层 GDAL 栅格能力。**原作者 HDF5 光谱库及 VSCP 光谱合成增强不是读取官方数据的前置条件；本项目不依赖 pandas、PyTables 或下载 `dataset.h5`。

默认启用磁盘缓存，首次访问一个 patch 时完成读取与波段对齐，后续直接读取压缩 NumPy 缓存。缓存内容在 NaN 填充、归一化和随机增强之前；原始文件路径、大小或修改时间变化时生成新缓存。写入采用临时文件加原子替换；损坏缓存会报错，不会悄悄忽略。关闭方式为 `dataset.cache_dir: null`。不需要预先运行原作者 `stack_patches.py`。

## 2. 安装和运行

在 VS Code 选择你当前项目的 Python 3.12 虚拟环境，在该环境的终端内进入本项目目录。无需 Conda。已有可用的 CUDA PyTorch 时，保留它，不要安装原作者旧版 torch/MMCV 依赖。

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`requirements.txt` 不固定 CPU/CUDA 构建，也不强制升级已经满足条件的 PyTorch；`torch` 与 `torchvision` 应为配套版本。如果当前环境尚未安装 GPU 版 PyTorch，先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择适合该电脑的构建。

解压官方完整数据集后，修改 `config.yaml`：

```yaml
dataset:
  root: 'D:/Datasets/MADOS'
```

这里只展示需要修改的字段，**不要用这个片段覆盖整个配置文件**。Windows 路径建议使用 `/` 或 YAML 单引号。数据不必移动到代码目录。

路径解析规则：`dataset.root`、`dataset.cache_dir`、`output_dir` 和配置中的 `evaluation.checkpoint` 相对路径，以该 YAML 所在目录为基准；`dataset.splits` 下的相对路径，以数据根目录为基准。命令行 `--checkpoint` 的相对路径按终端工作目录解析。

```bash
python train.py --config config.yaml
python evaluate.py --config config.yaml --split test
```

评估默认加载 `output_dir/checkpoints/best.pth`。也可以指定：

```bash
python evaluate.py --config config.yaml --split val --checkpoint outputs/checkpoints/best.pth
```

### 先进行小样本检查

可以复制 `config.yaml` 为 `config_smoke.yaml`，在副本中设置 `dataset.max_samples: {train: 2, val: 2, test: 2}`、`training.epochs: 2`、`training.batch_size: 2`、`output_dir: outputs_smoke`，然后对副本执行上述训练和评估命令。

这只读取各官方清单的前 2 个样本，不修改清单，不重新分配样本。正式实验把三个 `max_samples` 全部恢复 `null`，并使用新的 `output_dir`。

默认 `num_workers: 0`，适合先在 Windows 和外接显卡环境验证。`device: auto` 有 CUDA 时使用 CUDA，否则使用 CPU；也支持 `cpu`、`cuda`、`cuda:0`。显存不足先降低 `training.batch_size`。验证和评估的 batch size 统一由 `evaluation.batch_size` 控制。

## 3. 增强、训练和公平比较

默认保留原作者翻转/90°旋转的概率分布：水平、垂直、同时翻转、不翻转分别为 `0.2/0.2/0.2/0.4`；以 `0.8` 概率在 `0/90/180/270` 度中均匀选择。默认关闭额外缩放/裁剪，保持官方 patch 尺寸。

需要裁剪实验时，修改现有配置对应字段，例如：

```yaml
augmentation:
  scale_range: [0.8, 1.2]
  crop_size: [224, 200]
```

这是可选的新实验设置，开启后应对所有对比模型使用相同设置。所有波段共享几何变换；图像随机缩放用双线性插值，标签始终用最近邻；不使用 RGB 颜色扰动。裁剪可为矩形，图像不足时用通道均值填充、标签用忽略值填充。默认尽量保留至少一个标注像素；缩小导致标注完全消失时不编造标签。

可变尺寸样本组成 batch 时，仅在右侧和下侧填充：归一化影像填 0、标签填忽略值；评估导出恢复各样本原始尺寸。整批没有标注时记录 `empty_batches` 且不执行优化器更新；整个 epoch 没有任何标注则报错。

默认损失是加权交叉熵，类别权重使用原作者 `1 / log(1.03 + p_class)`，分布常量保存在 `utils/losses.py`。`class_weights: null` 为等权交叉熵，也可指定 15 个正权重。支持可选 focal loss。每轮 loss 按全轮有效像素及其权重累积，不简单平均各 batch 的 loss，避免稀疏程度不同导致统计失真。

支持 Adam、AdamW、SGD，以及关闭调度、MultiStepLR、CosineAnnealingLR、ReduceLROnPlateau；plateau 监测验证 loss，其他调度器每轮结束后更新。按验证集 **mIoU 最大值** 保存 `best.pth`，相同值保留较早的最佳轮次。每轮都完成训练和验证。训练指标来自增强后的训练批次和训练中的模型，验证指标来自该轮训练结束后的固定模型。

随机种子覆盖 Python、NumPy、PyTorch、DataLoader。增强使用独立随机数生成器，避免模型初始化或 dropout 改变增强序列。开启严格确定性时，遇到不支持确定性实现的运算会报错；跨硬件、库版本或 worker 数量不承诺逐位相同。

每个对比实验使用单独 `output_dir`；已有 `metrics.csv` 或最佳权重时拒绝覆盖。checkpoint 保存完整配置，可追溯预处理和模型参数。项目当前提供最佳权重保存与加载，不提供断点续训。评估会检查类别、波段、归一化和模型配置与权重一致。

本项目不含原作者专用 VSCP、EMA 或测试时增强，也不复现 MariNeXt；默认示例模型、训练设置及其结果不能直接宣称复现论文数值。模型比较时固定数据划分、预处理、增强、损失、指标平均方式和随机种子，并记录模型之间必要的训练策略差异。

## 4. 指标定义

使用 `segmentation_models_pytorch.metrics.get_stats(mode="multiclass", ignore_index=...)`，在整轮/整个评估集上累加每类 TP、FP、FN、TN，然后调用 SMP 的 IoU、F1、Precision、Recall 函数。

- 各类 IoU：`TP / (TP + FP + FN)`；各类 F1：`2TP / (2TP + FP + FN)`。
- `mIoU`、`mF1`、`Precision`、`Recall` 均为类别等权的宏平均，不是按图像平均或按像素频数加权。
- 未标注/填充像素在统计前被排除。未标注区域上的预测不计为假阳性。
- 默认 `absent_class_policy: exclude`：整个评估集中真值和预测均未出现的类别不参与宏平均；逐类值在 JSON 中为 `null`、CSV 中为空。仅预测出现或仅真值出现的类别仍参与，未定义的分母按 0 计分。
- `absent_class_policy: include`：所有 15 类都参与，完全缺席类按 0 计分。对比实验不能混用两种策略。

这与原作者宏平均通常按真值和预测类别并集统计的方式一致，但微小数值精度、模型和训练策略差异仍需区分。详见 [SMP 指标文档](https://smp.readthedocs.io/en/latest/metrics.html)。

## 5. 输出文件

| 路径（相对 output_dir） | 内容 |
|---|---|
| `checkpoints/best.pth` | 最佳模型参数、epoch、验证指标与完整配置 |
| `logs/train.log` | 每轮训练记录 |
| `logs/evaluate.log` | 评估运行记录 |
| `metrics.csv` | 每轮训练/验证 loss、宏平均指标、各类别 IoU/F1、有效像素数 |
| `curves/loss.png` | 训练/验证 loss 曲线 |
| `curves/miou.png` | 训练/验证 mIoU 曲线 |
| `curves/metrics.png` | mIoU、mF1、Precision、Recall 曲线 |
| `predictions/<split>_<时间>/metrics.json` | 完整评估结果及导出清单 |
| `predictions/<split>_<时间>/per_class.csv` | 15 类名称、有效像素数与逐类指标 |

`metrics.csv` 中 `train_iou_00`/`val_f1_00` 的 `00` 是训练类别 ID 0（原始值 1：Marine Debris），依次到 `14`。所有指标保存为 `0..1` 数值。

每个导出样本的目录包含：

| 文件 | 内容 |
|---|---|
| `image.tif` | 最近邻对齐后的 11 通道原始反射率；未归一化，原 NaN 保留 |
| `target.tif` | 原始标签值 `0..15`，0 为未标注 |
| `prediction.tif` | 预测标签转换回原始类别值 `1..15` |
| `rgb.png` | B4/B3/B2 真彩色预览，默认各通道 2%～98% 拉伸 |
| `target.png`、`prediction.png` | 类别配色图，真实标签的黑色表示未标注 |
| `comparison.png` | RGB、真实标签、预测、叠加图及类别图例 |

RGB 拉伸只用于显示，完全不改变模型输入。`image.tif` 是对齐后的多通道导出，不是原始 11 个不同分辨率 TIFF 的字节副本。`max_saved_predictions` 只限制导出图片数量，不限制评估样本数量；大量完整 TIFF 导出会占用较多空间。

检查的真实 TIFF 有 CRS 信息，但缺少有效的仿射定位变换；因此不能仅凭 CRS 声称导出图像能准确地理定位。程序保留可用的原始信息，并在 TIFF 与 JSON 中记录 `georeferenced`，不捏造坐标。Rasterio 对此发出的 `NotGeoreferencedWarning` 不影响按像素读取。

## 6. 增加网络与文件职责

在 `models/` 新增一个 `torch.nn.Module`，构造参数接受 `in_channels`、`num_classes`，输出 `[B,num_classes,H,W]` 的 logits。在 `models/__init__.py` 导入并加入 `MODEL_REGISTRY`，然后修改 YAML 的 `model.name` 和 `model.kwargs`。

模型构造器支持可配置的通道/类别数量；此 MADOS 读取器明确要求官方 11 波段、15 类，防止误配为 RGB 或二分类。自定义网络如返回字典、低分辨率 logits，应在该网络的封装内取出结果并恢复输入空间尺寸。模型内不要加 Softmax。

| 文件 | 职责 |
|---|---|
| `datasets/mados_dataset.py` | 官方划分、TIFF、波段对齐、缓存、NaN/标签处理、batch 填充 |
| `models/example_model.py` | 简单可替换的分割网络，支持非方形和奇数尺寸 |
| `models/__init__.py` | 模型注册与构建 |
| `utils/metrics.py` | SMP 统计、宏平均和 CSV 字段 |
| `utils/losses.py` | CE/focal、类别权重、有效像素平均 |
| `utils/transforms.py` | 多光谱影像与标签同步增强 |
| `utils/visualize.py` | 曲线、RGB、TIFF/PNG 预测导出 |
| `utils/seed.py` | 全局和 worker 随机种子 |
| `train.py` | 配置、DataLoader、原生优化器/调度器、每轮训练验证与权重保存 |
| `evaluate.py` | 加载权重、指定 split 评估与结果导出 |

## 7. 本次实际验证范围

验证环境：Linux、Python **3.12.14**、PyTorch **2.9.1+cpu**、torchvision **0.24.1+cpu**、SMP **0.6.0**、Rasterio **1.5.2**、NumPy **2.5.4**。另使用 PyYAML 6.0.3、Matplotlib 3.11.2、Pillow 13.0.0、tqdm 4.70.1。`pip check` 通过。这些是本次实际安装的版本，requirements 采用下限约束以允许使用当前兼容的 PyTorch 环境。

真实数据来自官方 ZIP：训练 `Scene_0_1`、`Scene_0_2`；验证 `Scene_100_1`、`Scene_100_10`；测试 `Scene_132_1`、`Scene_132_10`，共 72 个 TIFF。测试时保留完整官方划分清单，仅限制每个 split 读取前 2 项。

已验证：

- 六个真实样本的预处理与原作者 Rasterio 读取/最近邻/归一化流程逐像素一致；标签与实际 GDAL 读取结果一致；首次读取和缓存读取一致。
- Dataset 的 shape/dtype/标签范围；真实 batch 前向、有限 CE loss、反向传播与非零梯度；加权 CE 对照 PyTorch 标准实现。
- 手算样例核对 SMP IoU/F1/Precision/Recall、忽略区域、缺席类别策略；全忽略标签的零梯度处理。
- 同步翻转/旋转、矩形裁剪、缩放标签最近邻、稀疏标注裁剪、可变尺寸 batch 填充。
- 使用真实训练/验证样本执行 **2 个 epoch**，保存最佳权重、CSV、日志和曲线；加载最佳权重评估真实测试样本并导出 TIFF/PNG。
- 额外合成边界用例检查 NaN 填充、缓存原始值保留、源文件变更后的缓存失效、Inf/标签 16/缺失波段/损坏缓存报错，以及无标注 batch 不更新优化器。

未执行完整 2803 样本训练、Windows 本机测试或 CUDA/eGPU 测试。本运行环境禁止多进程 Tensor 共享所需的本地 IPC socket，因此未完成 `num_workers>0` 的运行验证；默认 `num_workers=0` 已通过。少量样本的指标仅用于确认计算和文件流程，不能作为模型性能结论。

压缩包不包含数据集、环境目录、临时测试数据或经过两轮检查的权重。你的正式结果由本机训练产生。

## 8. 来源与许可

数据处理常量、类别定义和必要处理逻辑参考 MADOS 原作者项目，保留其 MIT 许可和版权声明于 `LICENSE`。数据集按 [Zenodo 记录中的许可与引用说明](https://zenodo.org/records/10664073) 使用；论文比较实验应引用 MADOS 原论文，并说明自己的模型和训练协议。
