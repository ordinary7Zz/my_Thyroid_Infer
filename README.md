# Thyroid Inference Toolkit

甲状腺超声图像推理工具集，包含多种深度学习模型的统一推理代码，支持**分割**和**分类**两类任务。

## 目录

- [概述](#概述)
- [目录结构](#目录结构)
- [任务与模型一览](#任务与模型一览)
- [从 HuggingFace 下载并运行（完整流程）](#从-huggingface-下载并运行完整流程)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [数据集说明](#数据集说明)
- [统一指标计算](#统一指标计算)

---

## 概述

本项目汇集了 8 个模型的推理代码，用于甲状腺超声图像的自动分析。

项目提供两个层次的运行入口：

| 入口 | 说明 |
|---|---|
| `pipeline.py` | **端到端流水线**：数据预处理（ROI 裁剪 + 掩码对齐 + 标签提取）→ 生成配置 → 运行推理 |
| `run_all.py` | **统一推理**：一键运行全部四个任务的所有模型，并汇总性能指标 |

各推理子目录共享项目根目录下的两个统一指标模块：

| 模块 | 作用 |
|---|---|
| `seg_metrics.py` | 分割指标：Dice / HD95 / ECE + Bootstrap CI95 |
| `cls_metrics.py` | 分类指标：AUROC / AUPRC / Accuracy / Precision / Recall / F1 + Bootstrap CI95 |

---

## 目录结构

```
my_Thyroid_infer/
├── config.yaml                       # 统一配置（数据集路径、权重、预训练模型等）
├── pipeline.py                       # 端到端流水线脚本
├── run_all.py                        # 统一推理脚本
├── seg_metrics.py                    # 分割指标统一模块（5 个分割模型共享）
├── cls_metrics.py                    # 分类指标统一模块（4 个分类模型共享）
├── metrics.py                        # 旧版分割指标（兼容用，推荐使用 seg_metrics）
├── unified_requirements.txt           # 统一依赖
├── README.md                         # 本文件
│
├── my_ThyroidROI/                    # ROI 提取工具（pipeline.py 调用）
│   └── newcode/prepare_data.py       #   数据预处理：ROI 裁剪 + 掩码对齐 + 标签提取
│
├── datasets/                         # 测试数据集
│
├── infer_biomedclip/                 # BiomedCLIP 分类推理
├── infer_dinov3_unet/                # DINOv3-UNet 分割推理
├── infer_dinov3_unet_multitask/      # DINOv3-UNet 多任务分类推理
├── infer_medsam2/                    # MedSAM2 分割推理
├── infer_medsegx/                    # MedSegX 分割推理
├── infer_medsiglip/                  # MedSigLIP 分类推理
├── infer_transunet/                  # TransUNet 分割推理
├── infer_ultrafedfm/                 # UltraFedFM 分割 + 分类推理
├── infer_seg_agent/                  # 分割 Agent（多模型掩码选择/融合）
├── infer_cls_agent/                  # 分类 Agent（多模型预测选择/融合）
├── infer_explain/                    # 可解释性分析（SHAP 分类 + GradCAM 分割）
│
└── results/                          # 推理结果（掩码、预测 CSV、指标日志、汇总报告）
```

---

## 任务与模型一览

### 分割模型（腺体分割 / 结节分割）

| 模型 | 骨干网络 | 入口脚本 | 特点 |
|---|---|---|---|
| **DINOv3-UNet** | DINOv3 ViT-S | `infer_dinov3_unet/infer.py` | UNet + DINOv3 预训练编码器 |
| **MedSAM2** | SAM2 Hiera | `infer_medsam2/infer.py` | 基于 SAM2 的 2D 分割，全图 box prompt |
| **MedSegX** | SAM ViT-B | `infer_medsegx/inference.py` | 基于 SAM 的医学分割，支持 GT box / full box |
| **TransUNet** | R50-ViT-B_16 | `infer_transunet/infer.py` | CNN+Transformer 混合架构 |
| **UltraFedFM** | MAE ViT-B | `infer_ultrafedfm/segment.py` | 联邦预训练 + UNet 解码器 |

### 分类模型（良恶性二分类 / TIRADS 五分类）

| 模型 | 骨干网络 | 入口脚本 | 特点 |
|---|---|---|---|
| **BiomedCLIP** | BiomedCLIP ViT | `infer_biomedclip/infer.py` | CLIP 视觉编码器 + 分类头 |
| **MedSigLIP** | MedSigLIP-448 | `infer_medsiglip/inference.py` | SigLIP 视觉编码器 + 分类头 |
| **DINOv3-UNet (多任务)** | DINOv3 ViT-S | `infer_dinov3_unet_multitask/infer_classification.py` | 分割模型的多任务分类头 |
| **UltraFedFM** | ViT-B | `infer_ultrafedfm/classify.py` | 联邦预训练 ViT 分类 |

### Agent 模型（多模型选择/融合）

| Agent | 任务 | 入口脚本 | 特点 |
|---|---|---|---|
| **SegAgent** | 分割掩码选择/融合 | `infer_seg_agent/infer.py` | LLM 从多模型掩码中选择最佳或加权融合 |
| **ClsAgent** | 分类预测选择/融合 | `infer_cls_agent/infer.py` | LLM 从多模型预测中选择最佳或 soft voting |

> Agent 默认使用 `local_gpt_oss`（本地 GPT-OSS 模型），不调用外部 API。也可通过配置切换为云端 LLM（如阿里百炼 Qwen）。Agent 任务是后处理，需先运行对应基础任务（如 `nodule` / `binary`）。

---

## 从 HuggingFace 下载并运行

本项目托管在 HuggingFace：https://huggingface.co/instincts7Zz/my_Thyroid_infer （含全部代码和权重，约 14.7 GB，不含 `datasets/`）。

```bash
# 1. 安装并登录 HuggingFace CLI
pip install -U huggingface_hub
hf auth login

# 2. 下载项目
hf download instincts7Zz/my_Thyroid_infer --repo-type model --local-dir ./my_Thyroid_infer
cd my_Thyroid_infer

# 3. 安装环境（详见下方"环境安装"）
#    方式 A：在线安装（需联网，从 PyPI 下载）
conda create -n thyroid_infer python=3.11 -y && conda activate thyroid_infer
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r unified_requirements.txt
#
#    方式 B：离线安装（使用仓库内 wheels/，无需联网）
conda create -n thyroid_infer python=3.11 -y && conda activate thyroid_infer
pip install --no-index --find-links=./wheels/ torch==2.4.1 torchvision==0.19.1
pip install --no-index --find-links=./wheels/ -r unified_requirements.txt
#
#    方式 C：conda-pack 离线环境（解压即用，无需安装任何包）
mkdir -p ~/envs/thyroid_infer
tar -xzf thyroid_infer_env.tar.gz -C ~/envs/thyroid_infer
source ~/envs/thyroid_infer/bin/activate && conda-unpack

# 4. 准备数据集（详见下方"数据集说明"，放入 datasets/ 对应目录）

# 5. 运行推理
python run_all.py          # 统一推理（已有数据集和权重）
python pipeline.py         # 或端到端流水线（含预处理）
```

> 国内用户若访问慢，可设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 从 ModelScope 下载并运行

本项目也托管在 ModelScope（魔搭社区）：https://www.modelscope.cn/instincts/my_Thyroid_infer （内容与 HuggingFace 同步）。

```bash
# 1. 安装 ModelScope CLI
pip install modelscope

# 2. 下载项目
ms download instincts/my_Thyroid_infer --local_dir ./my_Thyroid_infer
cd my_Thyroid_infer

# 3. 安装环境（详见下方"环境安装"）
#    方式 A：在线安装（需联网，从 PyPI 下载）
conda create -n thyroid_infer python=3.11 -y && conda activate thyroid_infer
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r unified_requirements.txt
#
#    方式 B：离线安装（使用仓库内 wheels/，无需联网）
conda create -n thyroid_infer python=3.11 -y && conda activate thyroid_infer
pip install --no-index --find-links=./wheels/ torch==2.4.1 torchvision==0.19.1
pip install --no-index --find-links=./wheels/ -r unified_requirements.txt
#
#    方式 C：conda-pack 离线环境（解压即用，无需安装任何包）
mkdir -p ~/envs/thyroid_infer
tar -xzf thyroid_infer_env.tar.gz -C ~/envs/thyroid_infer
source ~/envs/thyroid_infer/bin/activate && conda-unpack

# 4. 准备数据集（详见下方"数据集说明"，放入 datasets/ 对应目录）

# 5. 运行推理
python run_all.py          # 统一推理（已有数据集和权重）
python pipeline.py         # 或端到端流水线（含预处理）
```

> ModelScope 为国内平台，无需镜像加速，下载速度通常优于 HuggingFace。

---

## 环境安装

### 基础环境

- Python 3.11
- CUDA 11.8
- PyTorch 2.4.1

### 方式一：在线安装（需联网）

```bash
# 1. 创建 conda 环境
conda create -n thyroid_infer python=3.11 -y
conda activate thyroid_infer

# 2. 安装 PyTorch (CUDA 11.8)
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其余依赖
pip install -r unified_requirements.txt
```

### 方式二：离线安装（使用 wheels/，无需联网）

仓库内 `wheels/` 目录包含了所有依赖的预编译 wheel 文件（Linux x86_64 + Python 3.11 + CUDA 11.8），适合网络受限环境或需要精确复现环境的场景。

```bash
# 1. 创建 conda 环境
conda create -n thyroid_infer python=3.11 -y
conda activate thyroid_infer

# 2. 离线安装 PyTorch（--no-index 表示不访问 PyPI，只从 wheels/ 找包）
pip install --no-index --find-links=./wheels/ \
    torch==2.4.1 torchvision==0.19.1

# 3. 离线安装其余依赖
pip install --no-index --find-links=./wheels/ \
    -r unified_requirements.txt
```

> **注意**：`wheels/` 中的 wheel 文件是平台相关的，仅适用于 **Linux x86_64 + Python 3.11**。其他平台请使用在线安装。

### 方式三：conda-pack 离线环境（解压即用，无需安装任何包）

仓库内 `thyroid_infer_env.tar.gz` 是用 `conda-pack` 打包的完整 conda 环境（含 Python 解释器 + 所有依赖，约 3.4 GB），解压后直接激活即可使用，无需 `pip install`。

```bash
# 1. 解压环境（解压到任意目录，不需要 conda）
mkdir -p ~/envs/thyroid_infer
tar -xzf thyroid_infer_env.tar.gz -C ~/envs/thyroid_infer

# 2. 激活环境
source ~/envs/thyroid_infer/bin/activate

# 3. 修复路径（首次激活后执行一次）
conda-unpack

# 4. 运行推理
cd my_Thyroid_infer
python run_all.py
```

> **注意**：
> - 该环境仅适用于 **Linux x86_64**（与打包机器的 OS / glibc 版本相关）
> - 无需预装 conda 或 Python，解压即用
> - 激活方式为 `source ~/envs/thyroid_infer/bin/activate`（不是 `conda activate`）

详见各子目录 `README.md` 中的模型权重准备和使用说明。

---

## 快速开始

### 方式一：端到端流水线（从原始数据到推理结果）

```bash
# 1. 编辑 config.yaml 中的 prepare 段配置输入目录和 ROI 权重
#    prepare:
#      input_dir:      ./datasets/甲状腺私有数据/新建文件夹
#      output_dir:     ./datasets/processed
#      roi_checkpoint: ./my_ThyroidROI/outputs/best_dice_model.pth

# 2. 运行完整流水线
python pipeline.py

# 也可通过命令行覆盖参数
python pipeline.py --input_dir /path/to/raw_data --skip_roi

# 跳过预处理，直接用已有 processed/ 数据生成配置并运行
python pipeline.py --skip_prepare

# 只运行特定任务
python pipeline.py --tasks gland nodule
```

### 方式二：统一推理（已有数据集和权重）

```bash
# 1. 编辑 config.yaml 配置数据集路径、权重、预训练模型

# 2. 运行全部任务（分割 + 分类）
python run_all.py

# 只运行分割任务
python run_all.py --tasks gland nodule

# 只运行分类任务
python run_all.py --tasks binary tirads

# 只运行特定模型
python run_all.py --models dinov3_unet medsam2

# 只打印命令不执行
python run_all.py --dry_run

# 列出所有任务和模型
python run_all.py --list

# 只运行 Agent 任务（需先运行基础任务）
python run_all.py --tasks seg_agent cls_agent

# 单独运行分割 Agent
python infer_seg_agent/infer.py \
    --task_dir results/nodule \
    --models dinov3_unet medsam2 medsegx transunet ultrafedfm \
    --gt_dir /path/to/gt_masks \
    --output_dir results/nodule/seg_agent

# 单独运行分类 Agent
python infer_cls_agent/infer.py \
    --task_dir results/binary \
    --models biomedclip medsiglip dinov3_unet_multitask ultrafedfm autogluon \
    --label_json /path/to/labels.json \
    --label_field malignancy \
    --output_dir results/binary/cls_agent
```

### 方式三：单独运行某个模型

```bash
# DINOv3-UNet 腺体分割
python infer_dinov3_unet/infer.py \
    --checkpoint /path/to/model.pth \
    --input_dir /path/to/images \
    --gt_dir /path/to/masks \
    --log_dir ./logs

# BiomedCLIP 良恶性二分类
python infer_biomedclip/infer.py \
    --ckpt /path/to/model.pth \
    --model_dir /path/to/biomedclip \
    --folder /path/to/images \
    --num_classes 2 \
    --class_names 0 1 \
    --label_json /path/to/labels.json \
    --label_field malignancy \
    --output ./results.csv
```

---

## 数据集说明

### 测试数据集

数据集路径在 `config.yaml` 中配置。支持两类数据来源：

**公共数据集**（直接放入对应目录）：

| 数据集 | 任务 | config.yaml 中的路径 |
|---|---|---|
| **TGVideo_PNG** | 腺体分割 | `datasets/TGVideo_PNG/test/` |
| **TN3K** | 结节分割 + 良恶性分类 | `datasets/TN3K/test/` |
| **Cine-Clip** | TIRADS 五分类 | `datasets/Cine-Clip/test/` |

**私有数据**（通过 `pipeline.py` 预处理）：

```bash
# 原始数据放在 datasets/甲状腺私有数据/新建文件夹/ 下
# 每个样本包含：原图、_ORG1 腺体掩码、_ROI1 结节掩码、.ini 标签文件
python pipeline.py  # 自动裁剪 ROI + 对齐掩码 + 提取标签
```

### 标签文件格式

分类任务需要提供 JSON 标签文件（`config.yaml` 中 `labels` 段配置）：

```json
[
    {"filename": "a.jpg", "malignancy": 0, "tirads": 2},
    {"filename": "b.jpg", "malignancy": 1, "tirads": 4}
]
```

- `malignancy`：良恶性（0=良性, 1=恶性, -1=缺失将被过滤）
- `tirads`：TI-RADS 分级（1-5, -1=缺失将被过滤）

GT mask 与图像按**文件名**匹配，灰度图，前景 > 0 视为正类。

### 私有数据集结构（pipeline.py 输入）

`pipeline.py` 的输入是未经处理的甲状腺私有数据，默认放在 `datasets/甲状腺私有数据/新建文件夹/` 下（可通过 `config.yaml` 的 `prepare.input_dir` 或命令行 `--input_dir` 修改）。

**每个样本由 4 个同名文件组成**（扩展名相同）：

```
datasets/甲状腺私有数据/新建文件夹/
├── THYB_S_AN01_ND000091_202052842015.png        # 原始超声图像
├── THYB_S_AN01_ND000091_202052842015_ORG1.png  # 腺体掩码（_ORG1 后缀）
├── THYB_S_AN01_ND000091_202052842015_ROI1.png  # 结节掩码（_ROI1 后缀）
└── THYB_S_AN01_ND000091_202052842015.ini       # 标签文件（INI 格式）
```

**文件命名规则**：

| 文件 | 后缀 | 说明 |
|---|---|---|
| 原图 | 无 | 超声图像，支持 `.png` `.jpg` `.jpeg` `.bmp` `.tiff` `.tif` `.webp` |
| 腺体掩码 | `_ORG1` | 灰度图，前景像素 > 0 为甲状腺腺体 |
| 结节掩码 | `_ROI1` | 灰度图，前景像素 > 0 为结节区域 |
| 标签文件 | `.ini` | INI 格式，包含良恶性与 TI-RADS 分级 |

> 掩码文件必须与原图**同名 + 后缀**（`_ORG1` / `_ROI1`）+ **同扩展名**。例如原图为 `case001.png`，则腺体掩码必须为 `case001_ORG1.png`，结节掩码为 `case001_ROI1.png`。

**INI 标签文件格式**：

标签从 `[ROI1]` 节（第一个非空的 ROI 节）中读取两个字段：

```ini
[ROI1]
pathologic=良性          ; 良恶性标签
birads=2类               ; TI-RADS 分级
```

字段映射规则：

| 字段 | 取值 | 映射结果 |
|---|---|---|
| `pathologic` | `良` / `良性` / `0` / `benign` | `malignancy=0`（良性）|
| | `恶` / `恶性` / `1` / `malignant` / `癌` | `malignancy=1`（恶性）|
| | 空 / 未知 | `malignancy=-1`（缺失）|
| `birads` | `1类` | `tirads=1` |
| | `2类` | `tirads=2` |
| | `3类` | `tirads=3` |
| | `4a类` / `4b类` / `4c类` | `tirads=4` |
| | `5类` | `tirads=5` |
| | 空 / 未知 | `tirads=-1`（缺失）|

### 预处理输出结构（pipeline.py 输出）

`pipeline.py` 预处理后的数据默认输出到 `datasets/processed/`，结构如下：

```
datasets/processed/
├── images/          # ROI 裁剪后的原图（文件名保持不变）
├── gland_masks/     # 对齐裁剪后的腺体掩码
├── nodule_masks/    # 对齐裁剪后的结节掩码
├── labels.json      # 分类标签（malignancy + tirads）
└── config.yaml      # 自动生成的配置文件，路径指向本目录
```

- 若启用 ROI 提取（提供 `roi_checkpoint`），原图和掩码会被裁剪到甲状腺区域并对齐；否则直接复制
- `labels.json` 的格式见上方[标签文件格式](#标签文件格式)小节
- 自动生成的 `config.yaml` 会将四个任务（gland / nodule / binary / tirads）的数据路径都指向本目录

---

## 统一指标计算

项目通过 `seg_metrics.py` 和 `cls_metrics.py` 两个模块统一所有模型的指标计算方式，确保跨模型公平比较。

### 分割指标（seg_metrics.py）

所有 5 个分割模型共享相同的指标计算方式：

| 指标 | 计算方式 | 边界情况 |
|---|---|---|
| **Dice** | `2\|P∩G\| / (\|P\|+\|G\|)`，无 smooth | TN（两空）= 1.0，FP/FN = 0.0 |
| **HD95** | `max(p95(d(pred→gt)), p95(d(gt→pred)))`，scipy EDT | 任一侧为空 = 0.0 |
| **CI95** | Bootstrap 百分位法，`n_boot=2000`，`seed=42` | — |

GT 二值化：`> 0`（任意非零像素视为前景）。

> **注意**：`seg_metrics.py` 同时提供类式 API（`Dice`、`HD95`、`ECE`，供 `dinov3_unet` 使用），在 GT 为空时返回 `None`（跳过该样本），与函数式 API 略有不同。`infer_dinov3_unet/metrics.py` 为旧版本，已由项目级 `seg_metrics.py` 替代。

### 分类指标（cls_metrics.py）

所有 4 个分类模型共享相同的指标计算方式：

| 指标 | 说明 |
|---|---|
| AUROC | ROC 曲线下面积 |
| AUPRC | PR 曲线下面积（macro 平均） |
| Accuracy | 准确率 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1 | F1 分数 |

- 二分类：`average="binary"`（默认正类 index=1）
- 多分类：`average="macro"`
- CI95：Bootstrap 百分位法，`n_boot=2000`，`seed=42`
- 标签为 `-1`（缺失）的样本会被自动过滤

### 性能汇总

`run_all.py` 运行结束后自动生成汇总报告：

- 终端打印：每个任务的性能表格（含 CI95）
- `results/summary.log`：完整的运行状态 + 性能指标

汇总时自动解析各模型输出的 `metrics.log` 文件，统一格式：

```
MetricName:  0.1234  (95% CI: [0.1000, 0.2000])
```

### 配置参数

`config.yaml` 中的关键参数：

```yaml
n_bootstrap: 2000    # Bootstrap CI95 迭代次数（所有模型共享）
device: cuda         # 推理设备
output_root: ./results   # 输出根目录
save_masks: false        # 是否保存分割预测掩码
```
