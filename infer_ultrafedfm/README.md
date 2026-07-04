# UltraFedFM 推理（独立版本）

本目录是 UltraFedFM 的**自包含**推理包，支持**分类**和**分割**两种任务，不依赖任何外部文件即可运行。

## 目录结构

```
infer_ultrafedfm/
├── classify.py                        # 分类推理
├── segment.py                         # 分割推理
├── models_vit.py                      # ViT 模型定义
├── segmentation_models_pytorch/       # SMP 库（含 MAE encoder）
├── requirements.txt
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

## 分类推理 (`classify.py`)

支持**二分类**（`--nb_classes 2`）和**多分类**（如 TIRADS 五分类 `--nb_classes 5`）。

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--data_path` | 是 | — | 图像所在目录（扁平结构，无子目录） |
| `--resume` | 是 | — | 模型权重 `.pth` 路径 |
| `--nb_classes` | 否 | 2 | 类别数量 |
| `--label_file` | 否 | None | 标签 JSON 文件路径 |
| `--label_field` | 否 | None | JSON 中的标签字段名（如 `malignancy`、`tirads`） |
| `--output_csv` | 否 | `predictions_<时间戳>.csv` | 输出 CSV 路径 |
| `--output_log` | 否 | `metrics_<时间戳>.log` | 输出指标日志路径 |
| `--model` | 否 | `vit_base_patch16` | 模型架构 |
| `--input_size` | 否 | 224 | 输入图像尺寸 |
| `--batch_size` | 否 | 16 | 批大小 |
| `--device` | 否 | `cuda` | 运行设备 |
| `--n_bootstrap` | 否 | 2000 | CI95 置信区间的 bootstrap 迭代次数 |

### 使用示例

**无标签** — 仅输出预测 CSV：

```bash
python classify.py --data_path /path/to/images --resume /path/to/ckpt.pth \
    --nb_classes 2 --output_csv predictions.csv
```

**有标签** — 输出预测 CSV + 指标日志：

```bash
# 良恶性二分类
python classify.py \
    --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images \
    --resume /mnt/wangbd8/workspace/ThyroidAgent/UltraFedFM/output_dir/dataset_3_cls_experiment/checkpoint-best_auroc.pth \
    --nb_classes 2 \
    --label_file /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/TN3K_test_label.json \
    --label_field malignancy \
    --output_csv ./logs/binary_predictions.csv \
    --output_log ./logs/binary_metrics.log

# TIRADS五分类
python classify.py \
    --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/images \
    --resume /mnt/wangbd8/workspace/ThyroidAgent/UltraFedFM/output_dir/Cine-Clip_TIRADS/checkpoint-best_auroc.pth \
    --nb_classes 5 \
    --label_file /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/Cine-Clip_test_label.json \
    --label_field tirads \
    --output_csv ./logs/multi_predictions.csv \
    --output_log ./logs/multi_metrics.log
```

### 标签 JSON 格式

```json
[
    {"filename": "a.jpg", "malignancy": 0, "FTCPTC": 1, "LNM_CN01": 1, "tirads": 2},
    {"filename": "b.jpg", "malignancy": 1, "FTCPTC": 0, "LNM_CN01": 0, "tirads": 4}
]
```

通过 `--label_field` 指定使用哪个字段作为评估标签。

### 输出 CSV 格式

| image_name | predicted_class | confidence | prob_class_0 | ... | prob_class_N | true_label |
|---|---|---|---|---|---|---|

仅当提供 `--label_file` 时，才会包含 `true_label` 列。

### 指标日志

当提供 `--label_file` 时，会生成 `.log` 文件，包含以下指标（多分类时采用宏平均）：

- **AUROC**
- **AUPRC**
- **Accuracy**（准确率）
- **Precision**（精确率）
- **F1**
- **Recall**（召回率）

每个指标均附带 **95% 置信区间**（CI95），通过 bootstrap 重采样计算。

---

## 分割推理 (`segment.py`)

输出预测掩码（可选）和/或 Dice/HD95 指标（需提供 GT）。

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--data_path` | 是 | — | 图像所在目录（扁平结构，无子目录） |
| `--resume` | 是 | — | 模型权重 `.pth` 路径 |
| `--output_dir` | 否 | None | 预测掩码输出目录（省略则不输出掩码） |
| `--gt_dir` | 否 | None | GT 掩码目录（省略则不计算指标） |
| `--output_log` | 否 | `seg_metrics_<时间戳>.log` | 输出指标日志路径 |
| `--img_size` | 否 | 224 | 输入图像尺寸 |
| `--batch_size` | 否 | 1 | 批大小 |
| `--threshold` | 否 | 0.5 | 二值化阈值 |
| `--device` | 否 | `cuda` | 运行设备 |
| `--n_bootstrap` | 否 | 2000 | CI95 置信区间的 bootstrap 迭代次数 |

### 使用示例

**仅输出掩码：**

```bash
python segment.py --data_path /path/to/images --resume /path/to/ckpt.pth \
    --output_dir /path/to/pred_masks
```

**输出掩码 + 计算指标：**

```bash
python segment.py --data_path /path/to/images --resume /path/to/ckpt.pth \
    --output_dir /path/to/pred_masks --gt_dir /path/to/gt_masks \
    --output_log metrics.log
```

**仅计算指标（不输出掩码）：**

```bash
# 腺体分割
python segment.py \
    --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/image \
    --resume /mnt/wangbd8/workspace/ThyroidAgent/UltraFedFM/my_pth/gland_seg/epoch_bestDice.pth \
    --gt_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/mask \
    --output_log ./logs/gland_metrics.log

# 结节分割
python segment.py \
    --data_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images \
    --resume /mnt/wangbd8/workspace/ThyroidAgent/UltraFedFM/my_pth/nodule_seg/epoch_bestDice.pth \
    --gt_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/masks \
    --output_log ./logs/nodule_metrics.log
```

**无任何输出**（允许的情况）：

```bash
python segment.py --data_path /path/to/images --resume /path/to/ckpt.pth
```

### GT 掩码匹配方式

GT 掩码与图像按**文件名主干**（stem）匹配，后缀不需要相同。例如图像 `a.jpg` 可匹配 GT `a.png`。GT 掩码会进行二值化处理（非零像素 → 前景）。

### HD95 边界情况处理

| 预测有前景 | GT 有前景 | HD95 |
|---|---|---|
| 是 | 是 | 正常计算 |
| 是 | 否（假阳性） | 0.0 |
| 否（假阴性） | 是 | 0.0 |
| 否 | 否（真阴性） | 0.0 |

### 指标日志

当提供 `--gt_dir` 时，会生成 `.log` 文件，包含：

- **Dice**（均值 ± CI95）
- **HD95**（均值 ± CI95，单位为像素）
- 每个样本的 Dice 和 HD95 详细数值
