"""RepViT 推理路径（代码与权重均在 infer_repvit/ 下）。"""

from pathlib import Path

INFER_DIR = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT = (
    INFER_DIR
    / "checkpoints/BM/repvit_m1_0_224_0.025_0.001_0.25/checkpoint_best.pth"
)
