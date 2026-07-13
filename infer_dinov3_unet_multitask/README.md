# 分类推理（独立可运行版本）

本目录包含 DINOv3-UNet 多任务模型的分类推理代码，可独立运行，不依赖外部文件。

## 目录结构

```
classification_inference/
├── infer_classification.py   # 主推理脚本
├── model.py                  # 模型定义（DINOv3_S_UNet_MULTITASK）
├── metrics.py                # 分类指标计算 + bootstrap CI95
├── requirements.txt          # Python 依赖
├── README.md                 # 本文件
└── scripts/                  # 示例运行脚本
    ├── infer_binary.sh       # 二分类推理示例
    └── infer_tirads.sh       # TIRADS 五分类推理示例（含标签文件）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 二分类推理（无标签文件）

```bash
python infer_classification.py \
    --image_dir /path/to/images \
    --checkpoint /path/to/model.pth \
    --num_classes 2 \
    --output results/binary_preds.csv
```

输出 CSV：

| filename | predicted_class | confidence |
|----------|----------------|------------|
| img1.jpg | 1              | 0.8732     |
| img2.jpg | 0              | 0.6541     |

### 2b. 二分类推理 + 验证集 Youden 阈值（推荐）

在独立验证集上计算 Youden 最优阈值，再用于测试集预测与指标统计：

```bash
python infer_classification.py \
    --image_dir /path/to/test/images \
    --checkpoint /path/to/model.pth \
    --num_classes 2 \
    --output results/binary_preds.csv \
    --label_file /path/to/test/labels.json \
    --label_field malignancy \
    --val_image_dir /path/to/val/images \
    --val_label_file /path/to/val/labels.json \
    --log_file results/binary_metrics.log
```

终端会打印验证集上的 Youden 阈值、灵敏度、特异度，并在 `.log` 中记录所用阈值及其来源 `youden(val)`。如需手动指定阈值，用 `--threshold 0.4` 替代（优先级最高）。

### 3. TIRADS 五分类推理（含标签文件，输出指标）

```bash
python infer_classification.py \
    --image_dir /path/to/images \
    --checkpoint /path/to/model.pth \
    --num_classes 5 \
    --output results/tirads_preds.csv \
    --label_file /path/to/labels.json \
    --label_field tirads \
    --log_file results/tirads_metrics.log
```

输出 CSV：

| filename | predicted_class | confidence | true_label |
|----------|----------------|------------|------------|
| img1.jpg | 3              | 0.7234     | 2          |
| img2.jpg | 4              | 0.8912     | 4          |

输出 `.log` 文件示例：

```
======================================================================
分类推理指标报告
======================================================================
生成时间:     2025-07-04 22:30:00
任务字段:     tirads
分类数:       5
平均方式:     macro-average（宏平均）
样本总数:     200
有效标签数:   200
  类别 0: 30 个样本
  类别 1: 45 个样本
  类别 2: 50 个样本
  类别 3: 40 个样本
  类别 4: 35 个样本
Bootstrap:    n_boot=2000, ci=0.95, seed=0

指标结果 (mean + CI95):
  ACCURACY   mean=0.7500  CI95=(0.6800, 0.8150)
  PRECISION  mean=0.7230  CI95=(0.6500, 0.7900)
  RECALL     mean=0.7100  CI95=(0.6400, 0.7800)
  F1         mean=0.7150  CI95=(0.6450, 0.7800)
  AUROC      mean=0.8800  CI95=(0.8300, 0.9200)
  AUPRC      mean=0.7900  CI95=(0.7300, 0.8400)
======================================================================
```

## 参数说明

### 必填参数

| 参数 | 说明 |
|------|------|
| `--image_dir` | 待推理图像所在目录（支持递归扫描子目录） |
| `--checkpoint` | 模型权重文件路径 (.pth) |
| `--num_classes` | 分类类别数：`2`（二分类）或 `5`（TIRADS 五分类） |
| `--output` | 输出 CSV 文件路径 |

### 标签相关（可选）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--label_file` | 无 | 标签 JSON 文件路径。提供后将计算指标并输出 `.log` |
| `--label_field` | 无 | 标签文件中对应的任务字段名（如 `malignancy`、`tirads`、`LNM_CN01` 等） |
| `--label_offset` | `-1` | 标签偏移量。`-1`=自动检测, `0`=不偏移, `1`=标签减1（如 TIRADS 1-5 → 0-4） |
| `--log_file` | 与CSV同名`.log` | 指标日志输出路径 |

### 阈值选择（仅二分类）

二分类预测默认使用 0.5 作为正类概率阈值。可通过以下参数选择阈值策略（优先级从高到低）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--threshold` | 无 | 显式指定二分类阈值。**优先级最高**，设置后跳过 Youden 计算 |
| `--val_image_dir` | 无 | 验证集图像目录。与 `--val_label_file` 同时提供时，在验证集上计算 Youden 最优阈值，再用于测试集预测与指标 |
| `--val_label_file` | 无 | 验证集标签 JSON 文件，字段复用 `--label_field` |

阈值决策逻辑（仅 `num_classes=2` 生效）：

1. 若提供 `--threshold` → 使用该值
2. 否则若同时提供 `--val_image_dir` 和 `--val_label_file` → 在验证集上推理并按 **Youden 指数**（`J = sensitivity + specificity - 1`）求最优阈值
3. 否则 → 使用默认 `0.5`

> Youden 阈值应在独立验证集上计算，避免过拟合到测试集。日志会记录所用阈值及其来源（`default` / `youden(val)` / `user`）。五分类不受影响，仍使用 `argmax`。

### 模型与硬件配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--img_size` | `224` | 输入图像尺寸，需与训练时一致 |
| `--dino_pretrained` | `False` | DINO backbone 是否使用预训练权重（推理时建议 `False`，因为权重从 checkpoint 恢复） |
| `--use_dilation` | `False` | 模型是否使用 dilation 层，需与训练时一致 |
| `--cuda_device` | `0` | CUDA 设备索引 |
| `--batch_size` | `16` | 推理批大小 |
| `--num_workers` | `4` | DataLoader 子进程数 |

### Bootstrap 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_boot` | `2000` | Bootstrap 采样次数 |
| `--ci` | `0.95` | 置信区间水平 |
| `--seed` | `0` | Bootstrap 随机种子 |

## 标签文件格式

标签文件为 JSON 列表，每项包含 `filename` 和一个或多个任务字段：

```json
[
  {"filename": "path/to/image1.jpg", "malignancy": 0, "tirads": 2, "LNM_CN01": 0},
  {"filename": "path/to/image2.jpg", "malignancy": 1, "tirads": 4, "LNM_CN01": 1}
]
```

- `filename`：相对于 `--image_dir` 的图像路径（也支持仅用 basename 匹配）
- 任务字段：通过 `--label_field` 指定当前使用的字段

### 标签偏移说明

- **二分类**：标签值应为 `0`/`1`，无需偏移
- **TIRADS 五分类**：标签值通常为 `1`-`5`，脚本会自动检测并偏移为 `0`-`4`
- 如自动检测不正确，可通过 `--label_offset` 手动指定

## 指标说明

### 二分类

| 指标 | 说明 |
|------|------|
| accuracy | 准确率 |
| precision | 精确率（正类） |
| recall | 召回率（正类） |
| f1 | F1 分数（正类） |
| auroc | ROC 曲线下面积 |
| auprc | PR 曲线下面积 |

### 五分类（TIRADS）

所有指标均使用 **macro-average（宏平均）**：

| 指标 | 说明 |
|------|------|
| accuracy | 准确率 |
| precision | 宏平均精确率 |
| recall | 宏平均召回率 |
| f1 | 宏平均 F1 分数 |
| auroc | 宏平均 AUROC（OvR） |
| auprc | 宏平均 AUPRC（OvR） |

所有指标均通过 **bootstrap**（默认 2000 次）估计 **CI95 置信区间**。

## 分类头选择

脚本根据 `--num_classes` 自动选择分类头：

- `num_classes=2`：使用 `benign_malignant_head`（sigmoid），阈值由上述"阈值选择"策略决定（默认 0.5 / Youden / 用户指定）
- `num_classes=5`：使用 `tirads_head`（softmax + argmax）

## 注意事项

1. `--dino_pretrained` 设为 `False` 时，模型权重完全从 checkpoint 恢复，不需要联网下载 backbone。
2. `--img_size`、`--use_dilation` 需与训练时保持一致。
3. 图像目录支持递归扫描子目录，标签文件中的 `filename` 可以是相对路径或仅 basename。
4. CSV 输出**不包含**每个类别的概率，仅包含 `filename`、`predicted_class`、`confidence`（和可选的 `true_label`）。
