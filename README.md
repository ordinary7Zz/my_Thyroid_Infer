# Thyroid Inference Toolkit

甲状腺超声图像推理工具集，包含多种深度学习模型的独立推理代码，支持**分割**和**分类**两类任务。

## 目录

- [概述](#概述)
- [目录结构](#目录结构)
- [任务与模型一览](#任务与模型一览)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [数据集说明](#数据集说明)
- [指标计算方式分析](#指标计算方式分析)

---

## 概述

本项目汇集了 8 个模型的推理代码，用于甲状腺超声图像的自动分析。每个子目录均为**独立可运行**的推理包，不依赖项目内其他文件。

| 任务 | 说明 |
|---|---|
| **分割** | 甲状腺腺体分割（Gland）和结节分割（Nodule），输出二值掩码 |
| **分类** | 良恶性二分类（Binary）和 TIRADS 五分类（Multi-class） |

---

## 目录结构

```
my_Thyroid_infer/
├── datasets/                        # 测试数据集
├── infer_biomedclip/                # BiomedCLIP 分类推理
├── infer_dinov3_unet/               # DINOv3-UNet 分割推理
├── infer_dinov3_unet_multitask/     # DINOv3-UNet 多任务分类推理
├── infer_medsam2/                   # MedSAM2 分割推理
├── infer_medsegx/                   # MedSegX 分割推理
├── infer_medsiglip/                 # MedSigLIP 分类推理
├── infer_transunet/                 # TransUNet 分割推理
├── infer_ultrafedfm/                # UltraFedFM 分割 + 分类推理
├── unified_requirements.txt         # 统一依赖
└── README.md                        # 本文件
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

---

## 环境安装

### 基础环境

- Python 3.11
- CUDA 11.8
- PyTorch 2.4.1

### 安装步骤

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

详见各子目录 `README.md` 中的模型权重准备和使用说明。

---

## 快速开始

### 分割推理示例

```bash
# DINOv3-UNet 腺体分割
python infer_dinov3_unet/infer.py \
    --checkpoint /path/to/model.pth \
    --input_dir /path/to/images \
    --gt_dir /path/to/masks \
    --log_dir ./logs

# MedSAM2 结节分割
python infer_medsam2/infer.py \
    --image_dir /path/to/images \
    --checkpoint /path/to/medsam2.pt \
    --gt_dir /path/to/masks \
    --log_dir ./logs
```

### 分类推理示例

```bash
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

# MedSigLIP TIRADS 五分类
python infer_medsiglip/inference.py \
    --checkpoint /path/to/model.pt \
    --model_path /path/to/medsiglip-448 \
    --input /path/to/images \
    --output ./results.csv \
    --label_file /path/to/labels.json \
    --label_field tirads
```

---

## 数据集说明

| 数据集 | 任务 | 说明 |
|---|---|---|
| **TGVideo** | 腺体分割 | 甲状腺视频截帧，PNG 格式 |
| **TN3K** | 结节分割 + 良恶性分类 | 甲状腺结节 3000 例 |
| **Cine-Clip** | TIRADS 五分类 | 甲状腺视频片段 |

标签 JSON 格式：

```json
[
    {"filename": "a.jpg", "malignancy": 0, "FTCPTC": 1, "LNM_CN01": 1, "tirads": 2},
    {"filename": "b.jpg", "malignancy": 1, "FTCPTC": 0, "LNM_CN01": 0, "tirads": 4}
]
```

GT mask 与图像按**文件名 stem** 匹配（扩展名不需相同），灰度图，前景 > 0 或 > 128 视为正类。

---

## 指标计算方式分析

> 以下详细分析各模型在计算分割指标（Dice、HD95）和分类指标（AUROC、AUPRC 等）时是否存在差异。

### 一、分割指标

#### 1.1 Dice

| 模型 | 公式 | smooth 项 | 两空（TN） | 预空 GT 非空（FN） | 预非空 GT 空（FP） |
|---|---|---|---|---|---|
| **dinov3_unet** | `2\|A∩B\|+s / (\|A\|+\|B\|+s)` | s=1.0 | 跳过（GT 空→None） | ≈0 | ≈0 |
| **medsam2** | `2\|A∩B\|+s / (\|A\|+\|B\|+s)` | s=1.0 | ≈1.0¹ | ≈0 | ≈0 |
| **medsegx** | `2\|A∩B\| / (\|A\|+\|B\|)` | 无 | 1.0 | 0.0 | 0.0 |
| **transunet** | `medpy.metric.binary.dc()` | 无 | 0.0² | 0.0 | 1.0³ |
| **ultrafedfm** | `2\|A∩B\| / (\|A\|+\|B\|)` | 无 | 1.0 | 0.0 | 0.0 |

> ¹ medsam2 两空时 intersection=0, pred_sum=0, gt_sum=0 → (0+1)/(0+0+1)=1.0
> ² transunet 两空时走 `else` 分支 → dice=0.0
> ³ transunet 预非空 GT 空时 → dice=1.0（视为"假阳性不算错"）

**关键差异**：
- **smooth 项**：dinov3_unet 和 medsam2 使用 `smooth=1.0`，对小目标略有影响；medsegx、ultrafedfm、transunet 不使用。
- **两空（TN）处理**：medsam2/medsegx/ultrafedfm 返回 1.0（完美匹配），transunet 返回 0.0，dinov3_unet 跳过。
- **假阳性（FP）处理**：transunet 返回 dice=1.0（不惩罚假阳性），其余返回 0.0。

#### 1.2 HD95

| 模型 | 计算方式 | 预空 GT 非空（FN） | 预非空 GT 空（FP） | 两空（TN） |
|---|---|---|---|---|
| **dinov3_unet** | `max(p95(d(pred→gt)), p95(d(gt→pred)))` | 图像对角线长度⁴ | 跳过 | 跳过 |
| **medsam2** | `max(p95(d(pred→gt)), p95(d(gt→pred)))` | 在 [0,0] 设点⁵ | 在 [0,0] 设点 | 在 [0,0] 设点 |
| **medsegx** | `max(p95(d(pred→gt)), p95(d(gt→pred)))` | 0.0 | 0.0 | 0.0 |
| **transunet** | `medpy.metric.binary.hd95()` | 0.0 | 0.0 | 0.0 |
| **ultrafedfm** | `max(p95(d(pred→gt)), p95(d(gt→pred)))` | 0.0 | 0.0 | 0.0 |

> ⁴ 返回 `sqrt(H² + W²)`，惩罚假阴性（大距离）
> ⁵ 在 [0,0] 处设置一个点后正常计算，距离取决于预测位置

**核心算法相同**：除 transunet 使用 `medpy` 库外，其余 4 个均采用相同的对称 EDT 方法：`d = max(p95(d(pred→gt)), p95(d(gt→pred)))`。

**关键差异在边界情况**：
- **假阴性（FN，预空 GT 非空）**：dinov3_unet 返回最大距离（最严格），medsam2 在原点设点（中间值），medsegx/transunet/ultrafedfm 返回 0.0（最宽松）。
- **假阳性（FP，预非空 GT 空）**：dinov3_unet 跳过，medsam2 在原点设点，medsegx/transunet/ultrafedfm 返回 0.0。

#### 1.3 CI95 置信区间

| 模型 | 方法 | 默认迭代次数 |
|---|---|---|
| **dinov3_unet** | Bootstrap（百分位法） | 5000 |
| **medsam2** | 正态近似（`1.96σ/√n`） | — |
| **medsegx** | Bootstrap（百分位法） | 2000 |
| **transunet** | 正态近似（`1.96σ/√n`） | — |
| **ultrafedfm** | Bootstrap（百分位法） | 2000 |

**差异**：medsam2 和 transunet 使用正态近似（速度快但假设正态分布），dinov3_unet/medsegx/ultrafedfm 使用 Bootstrap（更稳健但计算更慢）。

---

### 二、分类指标

#### 2.1 指标集合（相同）

所有 4 个分类模型计算**完全相同**的 6 个指标：

| 指标 | 说明 |
|---|---|
| AUROC | ROC 曲线下面积 |
| AUPRC | PR 曲线下面积 / Average Precision |
| Accuracy | 准确率 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1 | F1 分数 |

均使用 `sklearn.metrics` 库计算。

#### 2.2 二分类的平均方式（有差异）

| 模型 | 二分类 Precision/Recall/F1 的 average 参数 |
|---|---|
| **biomedclip** | `average="macro"` |
| **medsiglip** | `average="binary"`（默认正类） |
| **dinov3_unet_multitask** | `average="binary"`（默认正类） |
| **ultrafedfm** | `average="macro"` |

**差异**：`binary` 仅报告正类（index=1）的指标，`macro` 对两个类取宏平均。对于平衡数据集差异小，对于不平衡数据集 macro 会拉低正类指标。

> **biomedclip 额外输出正类指标**：除了 macro 指标外，biomedclip 还单独计算 `Precision_pos`、`Recall_pos`、`F1_pos`，因此信息最全面。

#### 2.3 多分类 AUPRC 计算方式（有差异）

| 模型 | 多分类 AUPRC 输入 |
|---|---|
| **biomedclip** | `average_precision_score(y_true, y_prob, average="macro")` — 直接传整数标签 |
| **medsiglip** | `average_precision_score(labels, probs, average="macro")` — 直接传整数标签 |
| **dinov3_unet_multitask** | `average_precision_score(y_onehot, y_probs, average="macro")` — 先 one-hot |
| **ultrafedfm** | `average_precision_score(np.eye(C)[y_true], y_score, average="macro")` — 先 one-hot |

**差异**：sklearn 的 `average_precision_score` 对于整数标签输入会内部 one-hot，因此**结果在数值上等价**。但传 one-hot 矩阵更明确，避免 sklearn 版本差异。

#### 2.4 Bootstrap CI95（有差异）

| 模型 | 点估计来源 | Bootstrap 迭代 | 缺类处理 | 随机数生成器 |
|---|---|---|---|---|
| **biomedclip** | 全量数据 | 2000 | 跳过该次 | `np.random.RandomState(42)` |
| **medsiglip** | 全量数据 | 2000 | 跳过该次 | `np.random.RandomState(42)` |
| **dinov3_unet_multitask** | Bootstrap 均值⁶ | 2000 | 返回 nan | `np.random.default_rng(0)` |
| **ultrafedfm** | 全量数据 | 2000 | 跳过该次 | `np.random.RandomState(42)` |

> ⁶ dinov3_unet_multitask 报告的 "mean" 是 **Bootstrap 采样均值的均值**，而非全量数据的点估计。这在 Bootstrap 分布有偏时会与全量点估计略有不同。

**差异**：
- **点估计来源**：dinov3_unet_multitask 使用 Bootstrap 均值作为点估计，其余 3 个使用全量数据点估计。
- **随机数生成器**：dinov3_unet_multitask 使用 `default_rng`（新版 API），其余使用 `RandomState`（旧版 API），相同 seed 下结果不同。
- **默认 seed**：dinov3_unet_multitask 用 seed=0，其余用 seed=42。

---

### 三、差异汇总

#### 分割指标差异影响

| 差异点 | 影响程度 | 受影响模型 |
|---|---|---|
| Dice smooth 项 | 低（仅对小目标有微小影响） | dinov3_unet, medsam2 |
| Dice 两空返回值 | **高**（直接影响均值） | 全部 |
| HD95 假阴性处理 | **高**（0 vs 最大距离差异巨大） | 全部 |
| HD95 假阳性处理 | **高** | 全部 |
| CI95 方法 | 中（Bootstrap vs 正态近似在小样本时差异大） | 全部 |

#### 分类指标差异影响

| 差异点 | 影响程度 | 受影响模型 |
|---|---|---|
| 二分类 average 参数 | **高**（binary vs macro 结果不同） | 全部 |
| 点估计来源 | 中（Bootstrap 均值 vs 全量点估计） | dinov3_unet_multitask |
| 随机数生成器 | 低（影响 CI 区间但非点估计） | dinov3_unet_multitask |
| AUPRC one-hot 方式 | 无（数值等价） | 无 |

### 四、结论

**分割指标不完全相同**：虽然 Dice 和 HD95 的核心计算公式相同，但在**边界情况处理**（空预测、空 GT）上存在显著差异，这会导致不同模型在同一数据集上的指标**不可直接比较**。特别是 HD95 的假阴性处理，dinov3_unet 返回最大距离而 medsegx/ultrafedfm 返回 0.0，差异可达数百像素。

**分类指标在二分类下不完全相同**：biomedclip 和 ultrafedfm 使用 `macro` 平均，medsiglip 和 dinov3_unet_multitask 使用 `binary` 平均。多分类下指标计算方式基本一致（均 macro），但 dinov3_unet_multitask 的点估计来自 Bootstrap 均值而非全量数据，略有不同。

**建议**：如需跨模型公平比较，应统一指标计算的边界处理策略和平均方式。
