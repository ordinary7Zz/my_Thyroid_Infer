# MedSegX Inference Toolkit

自包含的 MedSegX 推理工具包，不依赖项目其他目录。

## 目录结构

```
infer_medsegx/
├── inference.py            # 主推理脚本
├── requirements.txt
├── README.md
├── segment_anything/       # SAM 模型代码
├── model/                  # MedSegX / MedSAM 模型
├── data/
│   └── datainfo.py         # 模态 & 器官映射
└── utils/
    └── metrics.py          # DSC、HD95（纯 scipy）、bootstrap CI95
```

## 安装

```bash
conda create -n medsegx python=3.10 -y
conda activate medsegx
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 模型权重

本工具包只包含代码，权重文件需单独下载并自行指定路径。需要两个权重文件：

1. **SAM backbone** — 从 [SAM](https://github.com/facebookresearch/segment_anything#model-checkpoints) 下载，如 `sam_vit_b_01ec64.pth`。放在一个目录中，通过 `--checkpoint` 指定该**目录**。
2. **MedSegX weight** — 预训练或微调的 `.pth` 文件，通过 `--model_weight` 指定**文件路径**。

例如，你可以把权重放在任意位置：

```
/path/to/weights/
├── SAM/
│   └── sam_vit_b_01ec64.pth      ← --checkpoint 指向这个目录
└── medsegx_vit_b.pth             ← --model_weight 指向这个文件
```

## 使用

以下示例均包含 GT mask，运行推理的同时计算 DSC、HD95 及其 CI95 置信区间。

### Box prompt 模式

脚本支持两种 box prompt 模式，通过 `--box_mode` 参数选择：

| 模式 | 说明 | 是否需要 `--gt_dir` |
|------|------|:-------------------:|
| `full`（默认） | 使用整张图像作为 box `[0, 0, W, H]`，无需 GT | ❌ |
| `gt` | 从 GT mask 中提取最小外接矩形作为 box，模拟用户框选 | ✅ |

**GT box 模式**会从 GT mask 中提取前景区域的最小外接矩形，并可选地加入随机扰动（每边扩展 0~`perturb` 像素），与训练时 `data/dataset_copy.py` 的数据增强逻辑一致，用于模拟真实场景中用户框选不精确的情况。

### 1. 甲状腺腺体分割（`US_GlndThyroid`）

#### 1a. Full-image box（默认）

```bash
python inference.py \
    --input_dir    /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/image \
    --gt_dir       /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/mask \
    --task_name    US_GlndThyroid \
    --checkpoint   /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/SAM \
    --model_weight /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/MedSegX/finetune/cross_site/US_GlndThyroid/TG_Video/checkpoint_epoch_29.pth \
    --log_file     ./logs/glnd_thyroid_full.log \
    --box_mode     full \
    --device       cuda:0
```

#### 1b. GT box（从 GT mask 提取 box）

```bash
python inference.py \
    --input_dir    /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/image \
    --gt_dir       /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/mask \
    --task_name    US_GlndThyroid \
    --checkpoint   /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/SAM \
    --model_weight /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/MedSegX/finetune/cross_site/US_GlndThyroid/TG_Video/checkpoint_epoch_29.pth \
    --log_file     ./logs/glnd_thyroid_gtbox.log \
    --box_mode     gt \
    --box_perturb  20 \
    --box_seed     42 \
    --device       cuda:0
```

### 2. 甲状腺结节分割（`US_ThyroidNodule`）

#### 2a. Full-image box（默认）

```bash
python inference.py \
    --input_dir    /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images \
    --gt_dir       /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/masks \
    --task_name    US_ThyroidNodule \
    --checkpoint   /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/SAM \
    --model_weight /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/MedSegX/finetune/cross_site/US_ThyroidNodule/NoduleData/model_best.pth \
    --log_file     ./logs/nodule_full.log \
    --box_mode     full \
    --device       cuda:0
```

#### 2b. GT box（从 GT mask 提取 box）

```bash
python inference.py \
    --input_dir    /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images \
    --gt_dir       /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/masks \
    --task_name    US_ThyroidNodule \
    --checkpoint   /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/SAM \
    --model_weight /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/MedSegX/finetune/cross_site/US_ThyroidNodule/NoduleData/model_best.pth \
    --log_file     ./logs/nodule_gtbox.log \
    --box_mode     gt \
    --box_perturb  20 \
    --box_seed     42 \
    --device       cuda:0
```

> **提示**：
> - `--box_mode gt` 时，`--gt_dir` 为必填项（用于提取 box 和计算指标）。
> - `--box_perturb 0` 表示使用精确 GT box（无扰动），适合评估模型上界性能。
> - `--box_perturb 20`（默认）模拟用户框选不精确的场景，与训练一致。
> - `--box_seed` 控制随机扰动的可复现性。
> - 如需同时保存预测掩码，加上 `--output_dir /path/to/pred_masks`。
> - Full-image box 模式下，去掉 `--gt_dir` 即可仅做推理不评估。

## 参数说明

| 参数 | 必须 | 说明 |
|------|:----:|------|
| `--input_dir` | ✅ | 输入图像目录（PNG/JPG/BMP/TIFF） |
| `--task_name` | ✅ | 任务名，如 `US_GlndThyroid`、`US_ThyroidNodule` |
| `--checkpoint` | ✅ | SAM 权重目录 |
| `--model_weight` | ✅ | MedSegX 权重文件路径 |
| `--log_file` | ✅ | 日志保存路径 |
| `--gt_dir` | ❌ | GT mask 目录，提供则计算 DSC + HD95 + CI95；`--box_mode gt` 时必填 |
| `--output_dir` | ❌ | 预测掩码保存目录，提供则输出 PNG |
| `--model_type` | ❌ | SAM 规模：`vit_b`（默认）/ `vit_l` / `vit_h` |
| `--method` | ❌ | `medsegx`（默认）/ `medsam` |
| `--box_mode` | ❌ | Box prompt 模式：`full`（默认，整图 box）/ `gt`（从 GT mask 提取） |
| `--box_perturb` | ❌ | GT box 每边随机扩展像素数，默认 20；设为 0 则使用精确 GT box |
| `--box_seed` | ❌ | GT box 随机扰动的种子，默认 42 |
| `--device` | ❌ | `cuda:0`（默认） |
| `--n_boot` | ❌ | Bootstrap 迭代次数，默认 2000 |
| `--ci` | ❌ | 置信区间水平，默认 95 |

## 输出

### 预测掩码（`--output_dir` 提供）

- 格式：PNG，二值（0 = 背景，255 = 前景）
- 文件名：`{原图名}_mask.png`

### 日志文件（`--log_file`）

包含：运行参数、指标摘要（DSC/HD95 的 mean、std、CI95）、逐样本详细结果。

示例输出：

```
============================================================
MedSegX Inference Log
Time: 2026-07-04 17:00:00
============================================================
Input dir:    /path/to/test_images
GT dir:       /path/to/gt_masks
Output dir:   (none)
Task name:    US_GlndThyroid
Model weight: /path/to/medsegx_vit_b.pth
Model type:   vit_b
Method:       medsegx
Device:       cuda:0
Box mode:     gt (perturb=20, seed=42)
Total images: 50
Evaluated:    50
Skipped (no GT): 0

Per-sample results:
  File                                     DSC       HD95
  ----                                     ---       ----
  case_001.png                          0.920100    1.500000
  case_002.png                          0.780500    5.200000
  ...

============================================================
Metrics Summary (Mean ± std with CI)
============================================================
  DSC  : 0.852300  [95% CI: 0.830000 – 0.872000]
  HD95 : 3.214700  [95% CI: 2.800000 – 3.700000]
  DSC  std: 0.067100
  HD95 std: 1.890200
  N samples: 50
============================================================
```

## 指标说明

- **DSC (Dice Similarity Coefficient)**：衡量预测与 GT 的重叠度，范围 [0, 1]，越高越好。
- **HD95 (95th-percentile Hausdorff Distance)**：衡量预测与 GT 边界之间的最大距离的 95 百分位（单位：像素），越低越好。
- **CI95**：通过 Bootstrap（2000 次重采样）计算的 95% 置信区间。

## GT mask 匹配

图像和 GT mask 按文件名（不含扩展名）匹配。例如：
- `case_001.png` → 匹配 `case_001.png`（或 `.jpg`、`.bmp` 等）
- 如果找不到对应 GT，该样本会被跳过并在日志中记录。
