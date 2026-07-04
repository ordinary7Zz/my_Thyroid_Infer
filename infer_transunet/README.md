# TransUNet 独立推理目录

自包含的 TransUNet 分割推理代码，不依赖项目根目录的其他文件。

## 目录结构

```
infer_transunet/
├── README.md               # 本文件
├── requirements.txt        # pip 依赖
├── infer.py                # 主推理脚本
└── networks/               # TransUNet 网络定义 (自包含)
    ├── __init__.py
    ├── vit_seg_configs.py
    ├── vit_seg_modeling.py
    └── vit_seg_modeling_resnet_skip.py
```

## 环境配置

```bash
cd infer_transunet

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 仅推理（不保存、不计算指标）

```bash
python infer.py \
    --ckpt /path/to/model.pth \
    --img_dir /path/to/images
```

### 2. 推理 + 保存掩码

```bash
python infer.py \
    --ckpt /path/to/model.pth \
    --img_dir /path/to/images \
    --out_dir ./preds
```

### 3. 推理 + 保存掩码 + 计算指标

```bash
python infer.py \
    --ckpt /path/to/model.pth \
    --img_dir /path/to/images \
    --gt_dir /path/to/masks \
    --out_dir ./preds \
    --log ./eval_log.txt
```

### 4. 推理 + 仅计算指标（不保存掩码）

```bash
# 腺体分割
python infer.py \
    --ckpt /mnt/wangbd8/workspace/ThyroidAgent/TransUNet/my_model/TG_Video/epoch_49.pth \
    --img_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/image \
    --gt_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/mask \
    --log ./eval_log.txt

# 结节分割
python infer.py \
    --ckpt /mnt/wangbd8/workspace/ThyroidAgent/TransUNet/my_model/Nodule/epoch_49.pth \
    --img_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images \
    --gt_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/masks \
    --log ./eval_log.txt
```

## 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--ckpt` | 是 | — | 模型权重路径 (.pth) |
| `--img_dir` | 是 | — | 输入图像目录 (png/jpg/jpeg) |
| `--gt_dir` | 否 | None | GT mask 目录，提供则计算 Dice/HD95 指标 |
| `--out_dir` | 否 | None | 输出掩码目录，提供则保存推理掩码 PNG |
| `--log` | 否 | `./eval_log.txt` | 指标 log 文件路径 (纯文本)，仅在有 `--gt_dir` 时写入 |
| `--img_size` | 否 | 224 | 网络输入尺寸 (需与训练一致) |
| `--num_classes` | 否 | 2 | 类别数 (二分类=2) |
| `--vit_name` | 否 | `R50-ViT-B_16` | ViT 骨干名称 (需与训练一致) |
| `--n_skip` | 否 | 3 | skip 连接数量 (需与训练一致) |
| `--device` | 否 | `cuda` | 设备: `cuda` 或 `cpu`，无 CUDA 时自动回退 |

## 输出说明

- **掩码 PNG** (当提供 `--out_dir` 时)：单通道，原图尺寸，前景=255，背景=0
- **指标 log** (当提供 `--gt_dir` 时)：纯文本，包含时间戳、模型配置、逐例结果和汇总统计
  - **Dice**：均值及 95% 置信区间 (正态近似)
  - **HD95**：均值及 95% 置信区间 (正态近似)

## 注意事项

1. 图像和 GT mask 通过文件名 stem 自动匹配（如 `image_001.png` 对应 `image_001.png`）
2. 图像以灰度模式读入，resize 使用三次样条插值 (order=3)
3. 掩码 resize 使用最近邻插值 (order=0)
4. 推理时网络输出经过 softmax + argmax 得到预测类别
