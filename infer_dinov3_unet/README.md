# DINOv3-UNet 独立推理

最小可运行的分割推理工具包，不依赖项目其他任何文件。

## 目录结构

```
infer_dinov3_unet/
├── model.py          # DINOv3-UNet 模型定义
├── metrics.py        # Dice / HD95 / ECE / Bootstrap CI95
├── infer.py          # 主推理脚本（入口）
├── requirements.txt  # Python 依赖
└── README.md         # 本文件
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 1. 仅推理（不输出掩码、不计算指标）

```bash
python infer.py \
    --checkpoint /path/to/model.pth \
    --input_dir /path/to/images
```

### 2. 推理 + 输出掩码

```bash
python infer.py \
    --checkpoint /path/to/model.pth \
    --input_dir /path/to/images \
    --output_dir /path/to/preds
```

### 3. 推理 + 输出掩码 + 计算指标（Dice/HD95/ECE + CI95）

```bash
python infer.py \
    --checkpoint /path/to/model.pth \
    --input_dir /path/to/images \
    --gt_dir /path/to/gt_masks \
    --output_dir /path/to/preds \
    --log_dir /path/to/logs
```

### 4. 推理 + 仅计算指标（不输出掩码）

```bash
# 腺体分割
python infer.py \
    --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/baseline/gland/train_TGVideo/dino_unet_train_TGVideo_epoch_30.pth \
    --input_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/image \
    --gt_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/mask \
    --log_dir ./logs


# 结节分割
python infer.py \
    --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/train_Nodule/train_dataset_4/dino_unet_train_dataset_4_epoch_50.pth \
    --input_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images/ \
    --gt_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/masks \
    --log_dir ./logs
```

## 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | 是 | - | 模型权重文件 (.pth) |
| `--input_dir` | 是 | - | 输入图像目录 |
| `--gt_dir` | 否 | None | GT mask 目录；提供后计算指标，按文件名 stem 匹配 |
| `--output_dir` | 否 | None | 预测掩码输出目录；不提供则不输出掩码 |
| `--log_dir` | 否 | `./logs` | 指标 log 保存目录；仅当提供 `--gt_dir` 时生成 |
| `--img_size` | 否 | 224 | 模型输入尺寸 |
| `--batch_size` | 否 | 4 | 推理 batch size |
| `--num_workers` | 否 | 4 | DataLoader 工作进程数 |
| `--device` | 否 | 自动 | 设备 (cuda/cpu) |
| `--dino_pretrained` | 否 | false | 是否加载 DINO 预训练权重 |
| `--use_dilation` | 否 | false | 是否使用 dilation 模块 |
| `--threshold` | 否 | 0.5 | 二值化阈值 |
| `--save_orig_size` | 否 | true | 掩码是否 resize 回原始尺寸 |
| `--n_boot` | 否 | 5000 | Bootstrap CI95 迭代次数 |
| `--ci` | 否 | 0.95 | 置信区间水平 |

## GT 匹配规则

输入图像与 GT mask 按**文件名 stem** 匹配，格式不需要相同。

例如：
- 输入图像 `case_001.jpg` ↔ GT mask `case_001.png` ✓
- 输入图像 `case_002.png` ↔ GT mask `case_002.png` ✓
- 输入图像 `case_003.bmp` ↔ GT mask `case_003.tif` ✓

## 输出说明

### 预测掩码（`--output_dir`）

- 二值 PNG 图像，前景 255、背景 0
- 文件名与输入图像同名，后缀统一为 `.png`
- 默认 resize 回原始图像尺寸（可用 `--save_orig_size false` 关闭）

### 指标日志（`--gt_dir` 时生成）

保存在 `--log_dir` 目录下，包含两个文件：

1. **`infer_<timestamp>.log`** — 人类可读文本日志（与终端输出一致）
2. **`infer_<timestamp>_metrics.json`** — 结构化 JSON，包含：
   - Dice / HD95 / ECE 的 mean 和 CI95 置信区间
   - 逐病例的指标值
   - 跳过的样本及原因

## 特殊行为

- **GT 全空（无前景）**：跳过该 case，不计入 mean/CI95
- **无匹配 GT**：跳过该 case 的指标计算
- **HD95 计算异常**：打印警告，跳过该 case 的 HD95
- **无输出**：若不提供 `--output_dir` 且不提供 `--gt_dir`，脚本无任何输出产物
