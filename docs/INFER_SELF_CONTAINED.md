# my_Thyroid_Infer 推理自包含约定

**目标**：`run_all` / `run_all_extended` / B2 批跑所需的**推理**代码与权重均在 `my_Thyroid_Infer/` 内（`infer_<model>/`），不 import、不 subprocess 到 `Classification_Models/`、`dino_unet_multitask/` 等兄弟目录。

**仍依赖外部**（属数据，非代码仓库）：

- `cohorts.yaml` / config 中的 `data_root` 测试图像与标签 JSON  
- Python 环境（torch、timm、sklearn 等）

## B2 分类扩展三模型

| 目录 | 入口 | 权重（默认） |
|------|------|----------------|
| `infer_repvit/` | `infer_binary.py` | `checkpoints/BM/.../checkpoint_best.pth` |
| `infer_lsnet/` | `infer_binary.py` | `checkpoints/checkpoint_best.pth` |
| `infer_resnet50/` | `infer_binary.py` | `checkpoints/resnet50_thyroid_cls_epoch_50.pth` |

## 其它基线

TransUNet / MedSAM2 / MedSegX / UltraFedFM / BiomedCLIP / MedSigLIP 等已在各自 `infer_*` 下带 `checkpoints/` 与 `pretrained_models/`（见各目录 README）。

**训练脚本**不保证自包含；仅推理路径需满足上表。
