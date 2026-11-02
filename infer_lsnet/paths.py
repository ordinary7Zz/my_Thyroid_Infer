"""LSNet 推理路径（代码与默认权重在 infer_lsnet/ 下）。"""

from pathlib import Path

INFER_DIR = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT = INFER_DIR / "checkpoints/checkpoint_best.pth"
