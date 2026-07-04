# MedSAM2 2D 推理（自包含版本）

本目录是一个**最小可运行**的 MedSAM2 2D 图像分割推理工具，不依赖外部项目文件。

---

## 目录结构

```
infer_medsam2/
├── sam2/                       # MedSAM2 模型包（含 configs/modeling/utils）
├── infer.py                    # 主推理脚本
├── metrics.py                  # 评估指标（Dice、HD95、CI95）
├── README.md                   # 本文件
└── requirements.txt            # Python 依赖
```

**使用前需要准备：**
- 模型权重文件（`.pt`），放到任意位置，通过 `--checkpoint` 指定
- （可选）GT mask 目录，通过 `--gt_dir` 指定

---

## 环境安装

```bash
# 1. 创建 conda 环境
conda create -n medsam2 python=3.12 -y
conda activate medsam2

# 2. 安装 PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3. 安装其他依赖
cd infer_medsam2
pip install -r requirements.txt
```

---

## 使用方法

### 基本参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--image_dir` | 是 | - | 输入图像目录 |
| `--checkpoint` | 是 | - | MedSAM2 权重路径 (.pt) |
| `--gt_dir` | 否 | - | GT mask 目录，提供后计算 Dice/HD95 |
| `--output_dir` | 否 | - | 输出预测 mask 目录，不提供则不保存 |
| `--config` | 否 | `sam2.1_hiera_t512.yaml` | 模型配置文件 |
| `--device` | 否 | `cuda` | 推理设备 |
| `--log_dir` | 否 | `./logs` | 日志输出目录 |

### 使用场景

**1. 仅推理（不保存、不评估）**

```bash
python infer.py \
    --image_dir ./images/ \
    --checkpoint ./medsam2.pt
```

**2. 推理 + 保存预测 mask**

```bash
python infer.py \
    --image_dir ./images/ \
    --checkpoint ./medsam2.pt \
    --output_dir ./predictions/
```

**3. 推理 + 评估指标（需要 GT）**

```bash
# 腺体分割
python infer.py \
    --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/image/ \
    --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/MedSAM2/my_finetune/MedSAM2_TG_Video/checkpoints/checkpoint_10.pt \
    --gt_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/mask/ \
    --log_dir ./logs/gland

# 结节分割
python infer.py \
    --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images/ \
    --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/MedSAM2/my_finetune/MedSAM2_Noudle_FullBox/checkpoints/checkpoint_5.pt \
    --gt_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/masks/ \
    --log_dir ./logs/nodule
```

**4. 推理 + 保存 + 评估**

```bash
python infer.py \
    --image_dir ./images/ \
    --checkpoint ./medsam2.pt \
    --gt_dir ./masks/ \
    --output_dir ./predictions/
```

**5. 指定 CPU 推理**

```bash
python infer.py \
    --image_dir ./images/ \
    --checkpoint ./medsam2.pt \
    --device cpu
```

---

## 输入格式

### 图像目录
- 支持 `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`
- 图像会自动转为 RGB（灰度图复制为三通道）
- 所有图像会 resize 到 512×512 送入模型，输出 mask 映射回原始尺寸

### GT mask 目录（可选）
- 按文件名 stem 与图像一一对应（如 `case_001.png` ↔ `case_001.png`）
- 灰度图，前景 > 128 视为正类，否则为背景
- 尺寸不需要与图像一致，会自动 resize

---

## 输出说明

### 预测 mask（`--output_dir`）
- 二值 PNG，前景 = 255，背景 = 0
- 保持原图分辨率
- 文件名与输入图像 stem 一致

### 日志文件（`--log_dir`）
- 文件名格式：`infer_YYYYMMDD_HHMMSS.log`
- 包含运行配置、推理进度、评估结果

### 评估指标（`--gt_dir` 提供时）
- **Dice**：范围 [0, 1]，越大越好
- **HD95**：95% Hausdorff 距离（像素），越小越好
- **CI95**：基于正态近似的 95% 置信区间
- 日志中包含每个样本的明细和整体统计

---

## 原理说明

- **Box Prompt 策略**：始终使用全图作为 box prompt（`[0, 0, W-1, H-1]`），GT mask 仅用于计算指标，不影响推理过程
- **Video Predictor**：MedSAM2 基于 video model 训练，2D 图像被视为单帧"视频"进行推理
- **预处理**：ImageNet 均值/标准差归一化，resize 到模型期望分辨率（512×512）

---

## 常见问题

### Q: 没有 GPU 能用吗？
可以，添加 `--device cpu`。但推理速度会显著下降。

### Q: 不想保存 mask 只想看指标？
不提供 `--output_dir`，只提供 `--gt_dir` 即可。

### Q: 不想计算指标只想保存 mask？
不提供 `--gt_dir`，只提供 `--output_dir` 即可。

### Q: 什么都不输出可以吗？
可以。不提供 `--output_dir` 也不提供 `--gt_dir`，脚本仅执行推理。日志文件仍会生成，记录运行信息。
