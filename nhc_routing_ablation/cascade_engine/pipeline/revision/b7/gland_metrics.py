"""B7：全腺体分割逐图 Dice/HD95（独立 GT/预测 mask 目录）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from shared.image_io import ImageIO
from segmentation.metrics import compute_dice, compute_hd95


def _stem_map(dir_path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not dir_path.is_dir():
        return out
    for p in dir_path.iterdir():
        if p.is_file():
            out[p.stem] = p
    return out


def compute_gland_per_image(
    *,
    gt_gland_dir: Path,
    pred_gland_dir: Path,
    image_io: Optional[ImageIO] = None,
    image_stems: Optional[list[str]] = None,
) -> list[dict]:
    """
    对 stem 交集计算腺体 mask Dice/HD95。

    pred 与 gt 按 stem 匹配；若提供 image_stems 则仅评该列表。
    """
    io = image_io or ImageIO()
    gt_map = _stem_map(gt_gland_dir)
    pred_map = _stem_map(pred_gland_dir)
    stems = image_stems if image_stems is not None else sorted(set(gt_map) & set(pred_map))
    rows: list[dict] = []
    for stem in stems:
        gt_path = gt_map.get(stem)
        pred_path = pred_map.get(stem)
        if gt_path is None or pred_path is None:
            continue
        gt_mask = io.binarize_mask(io.load_mask(gt_path))
        pred_mask = io.binarize_mask(io.load_mask(pred_path))
        dice = compute_dice(pred_mask, gt_mask)
        hd95 = compute_hd95(pred_mask, gt_mask)
        rows.append({
            "stem": stem,
            "image_name": stem,
            "dice": float(dice) if not np.isnan(dice) else None,
            "hd95": float(hd95) if not (np.isinf(hd95) or np.isnan(hd95)) else None,
            "task": "gland",
        })
    return rows


def write_gland_jsonl(output_dir: Path, rows: list[dict]) -> Path:
    path = output_dir / "per_image_gland_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
