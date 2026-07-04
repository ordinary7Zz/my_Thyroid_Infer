# BiomedCLIP 分类推理（独立版，带 Bootstrap CI95 评估）

本目录是最小可运行版本，**不依赖项目中的任何其他文件**，可直接复制到任意位置运行。

支持二分类和多分类（如 TIRADS 五分类）推理，输出 CSV 结果。
若提供标签 JSON 文件，额外计算分类性能指标（含 95% Bootstrap 置信区间）。

---

## 评估指标

提供标签文件时，计算以下指标（均含 95% CI）：

| 指标 | 说明 |
|------|------|
| **AUROC** | ROC 曲线下面积（二分类用正类概率，多分类用 macro OvR） |
| **AUPRC** | PR 曲线下面积 / Average Precision（二分类用正类概率，多分类用 macro） |
| **Accuracy** | 准确率 |
| **Precision (macro)** | 宏平均精确率 |
| **Recall (macro)** | 宏平均召回率 |
| **F1 (macro)** | 宏平均 F1 |

二分类额外输出正类（positive class）的 Precision / Recall / F1。

置信区间通过 Bootstrap 重采样计算（默认 2000 次），取 2.5% ~ 97.5% 百分位区间。

---

## 目录结构

```
infer_biomedclip/
├── infer.py                    # 推理主脚本（含模型定义，无外部依赖）
├── requirements.txt            # 依赖包
├── sample_labels.json          # 示例标签文件
└── README.md                   # 本说明
```

> 预训练骨干权重和微调后的分类权重均通过命令行参数指定路径，无需放在本目录内。

---

## 安装依赖

```bash
pip install -r requirements.txt
```

---

## 准备预训练模型

推理时需要 BiomedCLIP 预训练骨干权重（用于初始化模型结构，分类权重由 `--ckpt` 提供）。

模型目录中必须包含以下文件：

```
biomedclip/
├── open_clip_config.json          # 模型结构配置
├── open_clip_pytorch_model.bin    # 预训练权重（或 .safetensors）
└── ...                            # tokenizer 等其他文件
```

通过 `--model_dir` 指定本地模型目录：

```bash
python infer.py --model_dir /path/to/biomedclip ...
```

> **注意**: 本脚本完全使用本地权重，不依赖 HuggingFace 在线下载。

---

## 用法

### 1. 仅推理（输出 CSV）

```bash
# 二分类
python infer.py \
    --ckpt /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_BiomedCLIP/output/BM/best_model.pth \
    --model_dir /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_BiomedCLIP/pretrained_models/biomedclip \
    --folder /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images/ \
    --num_classes 2 \
    --class_names 0 1 \
    --output ./logs/binary_results.csv

# TIRADS 五分类
python infer.py \
    --ckpt /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_BiomedCLIP/output/TIRADS/best_model.pth \
    --model_dir /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_BiomedCLIP/pretrained_models/biomedclip \
    --folder /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/images/ \
    --num_classes 5 \
    --class_names 1 2 3 4 5 \
    --output ./logs/multi_results.csv
```

### 2. 推理 + 性能评估（提供标签文件）

```bash
# 二分类
python infer.py \
    --ckpt /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_BiomedCLIP/output/BM/best_model.pth \
    --model_dir /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_BiomedCLIP/pretrained_models/biomedclip \
    --folder /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images/ \
    --num_classes 2 \
    --class_names 0 1 \
    --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/TN3K_test_label.json \
    --label_field malignancy \
    --output ./logs/binary_results.csv \
    --eval_output ./logs/eval_result.log

# TIRADS 五分类
python infer.py \
    --ckpt /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_BiomedCLIP/output/TIRADS/best_model.pth \
    --model_dir /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_BiomedCLIP/pretrained_models/biomedclip \
    --folder /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/images/ \
    --num_classes 5 \
    --class_names 1 2 3 4 5 \
    --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/Cine-Clip_test_label.json \
    --label_field tirads \
    --output ./logs/multi_results.csv \
    --eval_output ./logs/eval_result.log
```

---

## 参数说明

| 参数 | 必选 | 说明 |
|------|------|------|
| `--ckpt` | ✅ | 训练好的分类模型权重路径 (.pth) |
| `--folder` | ✅ | 待推理的图片文件夹路径 |
| `--num_classes` | ✅ | 类别数（二分类填 2，TIRADS 五分类填 5） |
| `--class_names` | ✅ | 类别名称列表，顺序与训练时一致 |
| `--model_dir` | ✅ | 本地 BiomedCLIP 预训练模型目录（须含 `open_clip_config.json` 和权重文件） |
| `--device` | | 推理设备，`cuda` 或 `cpu`（默认 `cuda`，无 GPU 自动回退） |
| `--batch_size` | | 批推理大小（默认 32） |
| `--output` | | CSV 结果输出路径（默认 `results.csv`） |
| `--label_json` | | 标签 JSON 文件路径（可选，提供后进行性能评估） |
| `--label_field` | | JSON 中的标签字段名（提供 `--label_json` 时必填） |
| `--eval_output` | | 评估结果保存路径（.log，未指定时自动生成） |
| `--n_bootstrap` | | Bootstrap 迭代次数（默认 2000） |
| `--seed` | | 随机种子（默认 42，确保 Bootstrap 可复现） |

---

## 标签 JSON 格式

参考 `sample_labels.json`：

```json
[
    {"filename": "a.jpg", "malignancy": 0, "FTCPTC": 1, "LNM_CN01": 1, "tirads": 2},
    {"filename": "b.jpg", "malignancy": 1, "FTCPTC": 0, "LNM_CN01": 0, "tirads": 4}
]
```

- `filename`：与图片文件夹中的文件名对应（仅文件名，不含路径）
- 标签值会自动映射为 0-based 索引，支持以下格式：
  - **标签值 = class_names 中的名称**：例如 `--class_names 1 2 3 4 5`，`tirads=3` → 索引 2
  - **标签值已是 0-based**：例如 `malignancy=0` → 索引 0，`malignancy=1` → 索引 1
  - **标签值是 1-based**：例如 `tirads=3` 且 `--class_names benign malignant` 无匹配时 → 索引 2

> 示例：`--class_names 1 2 3 4 5` 时，JSON 中 `tirads=1` → 索引 0，`tirads=5` → 索引 4，自动映射。

---

## 输出文件说明

### results.csv

| 列名 | 说明 |
|------|------|
| `filename` | 图片文件名 |
| `predict_label` | 预测类别名 |
| `predict_confidence` | 预测置信度 |
| `prob_<class>` | 每个类别的 softmax 概率 |

### eval_result.log（仅提供标签文件时生成）

包含以下评估指标（均含 95% Bootstrap CI）：

```
  📊 平均性能指标 (95% CI):
  ----------------------------------------------------------
  AUROC                 : 0.9234  (0.8812 - 0.9567)
  AUPRC                 : 0.8945  (0.8421 - 0.9356)
  Accuracy              : 0.8765  (0.8301 - 0.9123)
  Precision (macro)     : 0.8701  (0.8200 - 0.9100)
  Recall (macro)        : 0.8650  (0.8150 - 0.9050)
  F1 (macro)            : 0.8625  (0.8120 - 0.9000)
```

另含混淆矩阵、每类准确率、多分类 per-class 分类报告。
