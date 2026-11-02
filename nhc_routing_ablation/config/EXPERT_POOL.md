# NHC 七策略 / B7 级联 — 专家池（与 B3 主表一致）

**权重路径约定**：相对 **my_Thyroid_Infer** 根目录（`./infer_*`）。NHC 脚本在调用 Cascade 前会 materialize 为绝对路径（见 `nhc_routing_ablation/paths.py`）。

## 分割（5）

| `name` | 权重 |
|--------|------|
| `all_datasets` | `./infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_all_datasets_epoch_40.pth` |
| `TN3K` | `./infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_TN3K_epoch_50.pth` |
| `ThyroidXL` | `./infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_ThyroidXL_epoch_50.pth` |
| `PKTN_comprehensive` | `./infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_PKTN_comprehensive_epoch_50.pth` |
| `TN5K` | `./infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_TN5K_epoch_50.pth` |

## 分类（4）

| `name` | 权重 |
|--------|------|
| `all_datasets` | `./infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_all_datasets_epoch_40.pth` |
| `TN3K` | `./infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_TN3K_epoch_50.pth` |
| `TN5K` | `./infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_TN5K_epoch_50.pth` |
| `ThyroidXL` | `./infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_ThyroidXL_epoch_50.pth` |

（分类池 **不含** PKTN multitask。）

## Radiomics

- `radiomics_judge.enabled: true`
- `model_dir`: `./infer_autogluon/checkpoints/binary/autogluon_model_20260204_140346`
- `radiomics_params`: `./infer_autogluon/radiomics_2d.yaml`

## Phase 1–2 intermediate 必须包含

- `intermediate/seg/`：**5 个** `.npz`（上表 seg 名各一）
- `intermediate/cls/`：**4 个** `.json`（上表 cls 名各一）

## 策略 (6) Learned

- 路由器 **joblib** 当前默认：`artifacts/b3/tn5k/learned_router_{seg,cls}.joblib`（相对 **ThyroidCascadeAgent**，非 infer 权重）
- 含义：在 **TN5K validation** 上 fit 的逻辑回归，**专家名列表与上表相同**（见 joblib 旁 `.meta.json`）
- **不是**只加载 TN5K 两个 checkpoint
