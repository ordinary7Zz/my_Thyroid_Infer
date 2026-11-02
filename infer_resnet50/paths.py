"""ResNet-50 推理路径（代码与默认权重在 infer_resnet50/ 下）。"""

from pathlib import Path

INFER_DIR = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT = INFER_DIR / "checkpoints/resnet50_thyroid_cls_epoch_50.pth"
