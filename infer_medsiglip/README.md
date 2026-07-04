# MedSigLIP 分类推理（独立可运行版本）

本目录是 **最小可运行的分类推理工具包**，不依赖外部代码文件，只需安装 `requirements.txt` 中的 Python 依赖即可运行。

---

## 目录结构

```
infer_medsiglip/
├── inference.py       ← 主推理脚本（入口）
├── model.py            ← MedSigLIP 分类模型定义
├── transforms.py       ← 图像预处理（推理专用）
├── metrics.py          ← 分类指标 + Bootstrap 95% CI
├── requirements.txt    ← Python 依赖
└── README.md           ← 本文件
```

## 安装

```bash
pip install -r requirements.txt
```

> 还需要 MedSigLIP 预训练权重和训练好的 checkpoint（通过命令行参数传入，不包含在本目录中）。

## 用法

### 1. 基本推理（仅输出 CSV）

```bash
python inference.py \
    --checkpoint /path/to/best_model.pt \
    --model_path /path/to/medsiglip-448 \
    --input /path/to/images/ \
    --output predictions.csv
```

### 2. 带标签文件的推理（输出 CSV + 性能指标 .log）

```bash
# 良恶性二分类
python inference.py \
    --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_MedSigLIP/checkpoints/binary_cls/best_model.pt \
    --model_path /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_MedSigLIP/pretrained/medsiglip-448 \
    --input /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images/ \
    --output ./pred/binary_cls.csv \
    --label_file /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/TN3K_test_label.json \
    --label_field malignancy

# TIRADS五分类
python inference.py \
    --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_MedSigLIP/checkpoints/multi_cls/best_model.pt \
    --model_path /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_MedSigLIP/pretrained/medsiglip-448 \
    --input /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/images/ \
    --output ./pred/tirads_cls.csv \
    --label_file /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/Cine-Clip_test_label.json \
    --label_field tirads
```

提供标签文件后，会额外输出一个 `predictions_metrics.log` 文件，包含以下指标及 95% 置信区间：
- **AUROC**
- **AUPRC**
- **Accuracy**
- **Precision**
- **F1**
- **Recall**

> 二分类使用 `binary` 平均，TIRADS 五分类使用 `macro` 宏平均。

### 3. 完整参数

```bash
python inference.py \
    --checkpoint /path/to/best_model.pt \
    --model_path /path/to/medsiglip-448 \
    --input /path/to/images/ \
    --output predictions.csv \
    --label_file /path/to/labels.json \
    --label_field tirads \
    --device cuda:0 \
    --batch_size 32 \
    --n_bootstrap 2000 \
    --metrics_output /path/to/metrics.log
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `--checkpoint` | 是 | 模型 checkpoint 路径 (.pt) |
| `--model_path` | 是 | 预训练 MedSigLIP 权重目录路径 |
| `--input` | 是 | 输入图像文件或目录 |
| `--output` | 是 | 输出 CSV 路径 |
| `--label_file` | 否 | 标签 JSON 文件路径 |
| `--label_field` | 否* | 标签 JSON 中的任务字段名 |
| `--device` | 否 | 设备 (cuda/cuda:0/cpu)，默认自动 |
| `--batch_size` | 否 | 批量大小，默认 32 |
| `--n_bootstrap` | 否 | Bootstrap 迭代次数，默认 2000 |
| `--metrics_output` | 否 | 指标 .log 输出路径 |
| `--label_offset` | 否 | 标签偏移量（1-indexed 标签自动减 1，或手动指定） |

> *`--label_field` 在提供 `--label_file` 时必填。

## 标签文件格式

JSON 数组，每个元素包含 `filename` 和若干任务字段：

```json
[
    {
        "filename": "a.jpg",
        "malignancy": 0,
        "FTCPTC": 1,
        "LNM_CN01": 1,
        "tirads": 2
    },
    {
        "filename": "b.jpg",
        "malignancy": 1,
        "FTCPTC": 0,
        "LNM_CN01": 0,
        "tirads": 4
    }
]
```

- `filename`: 图像文件名（basename，如 `a.jpg`），与输入图像的 basename 匹配
- 任务字段（如 `malignancy`、`tirads`）：整数值，对应类别索引（从 0 开始）
- 通过 `--label_field` 指定当前使用哪个字段

## 输出说明

### CSV 文件

| 列名 | 说明 |
|------|------|
| `filename` | 图像文件名 |
| `predicted_class` | 预测类别名称 |
| `confidence` | 预测置信度（最大概率值） |
| `true_label` | 真实标签名称（仅当提供标签文件时存在） |

### 性能指标 .log 文件

仅当提供 `--label_file` 时生成，默认路径为 `<output>_metrics.log`。包含：
- 所有指标的点估计和 95% CI
- 混淆矩阵
- 详细分类报告 (classification report)

## 注意事项

1. **模型配置**：checkpoint 中保存了完整的训练配置（包括 num_classes 等），无需额外提供 config 文件；预训练权重路径通过 `--model_path` 手动指定
2. **灰度图处理**：自动将灰度超声图转为三通道 RGB
3. **二分类 vs 多分类**：自动根据 checkpoint 中的 `num_classes` 判断
4. **CI95 计算**：使用 Bootstrap 重采样方法，默认 2000 次迭代
