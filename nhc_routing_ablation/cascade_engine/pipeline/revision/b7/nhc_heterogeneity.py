"""B7：NHC 按中心聚合分割/分类性能。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pipeline.evaluate import _label_to_binary


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _auroc_auprc(y_true: list[int], y_score: list[float]) -> dict[str, Optional[float]]:
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {"auroc": None, "auprc": None, "n": len(y_true), "estimable": False}
    from sklearn.metrics import average_precision_score, roc_auc_score

    yt = np.array(y_true)
    ys = np.array(y_score)
    return {
        "auroc": float(roc_auc_score(yt, ys)),
        "auprc": float(average_precision_score(yt, ys)),
        "n": len(y_true),
        "estimable": True,
    }


def aggregate_by_center(
    *,
    per_image_seg: list[dict],
    per_image_cls: list[dict],
    results: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """
    按 center_id 汇总。center_id 优先来自逐图行，否则从 results 的 stem 映射。
    """
    stem_to_center: dict[str, str] = {}
    if results:
        for r in results:
            stem = Path(r.get("image_name", "")).stem
            cid = r.get("center_id")
            if stem and cid:
                stem_to_center[stem] = str(cid)

    centers: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "n_images_seg": 0,
        "dice_values": [],
        "hd95_values": [],
        "cls_y_true": [],
        "cls_y_score": [],
    })

    for row in per_image_seg:
        stem = row.get("stem") or Path(row.get("image_name", "")).stem
        cid = row.get("center_id") or stem_to_center.get(stem) or "unknown"
        block = centers[cid]
        block["n_images_seg"] += 1
        if row.get("dice") is not None:
            block["dice_values"].append(float(row["dice"]))
        if row.get("hd95") is not None:
            block["hd95_values"].append(float(row["hd95"]))

    for row in per_image_cls:
        stem = row.get("stem") or Path(row.get("image_name", "")).stem
        cid = row.get("center_id") or stem_to_center.get(stem) or "unknown"
        block = centers[cid]
        if row.get("true_label") is not None and row.get("score_malignant") is not None:
            block["cls_y_true"].append(int(row["true_label"]))
            block["cls_y_score"].append(float(row["score_malignant"]))

    out_centers: dict[str, Any] = {}
    dice_macro: list[float] = []
    auroc_macro: list[float] = []

    for cid, block in sorted(centers.items()):
        dice_vals = block["dice_values"]
        hd95_vals = block["hd95_values"]
        cls_metrics = _auroc_auprc(block["cls_y_true"], block["cls_y_score"])
        dice_mean = float(np.mean(dice_vals)) if dice_vals else None
        hd95_mean = float(np.mean(hd95_vals)) if hd95_vals else None
        if dice_mean is not None:
            dice_macro.append(dice_mean)
        if cls_metrics["estimable"] and cls_metrics["auroc"] is not None:
            auroc_macro.append(cls_metrics["auroc"])
        out_centers[cid] = {
            "n_images_seg": block["n_images_seg"],
            "dice_mean": dice_mean,
            "hd95_mean": hd95_mean,
            "n_cls": cls_metrics["n"],
            "auroc": cls_metrics["auroc"],
            "auprc": cls_metrics["auprc"],
            "auroc_estimable": cls_metrics["estimable"],
        }

    return {
        "n_centers": len(out_centers),
        "macro_mean_dice": float(np.mean(dice_macro)) if dice_macro else None,
        "macro_mean_auroc": float(np.mean(auroc_macro)) if auroc_macro else None,
        "centers": out_centers,
    }


def score_values_by_center(
    per_image_rows: list[dict],
    results: Optional[list[dict]] = None,
    *,
    field: str = "score_malignant",
) -> dict[str, list[float]]:
    stem_to_center: dict[str, str] = {}
    if results:
        for r in results:
            stem = Path(r.get("image_name", "")).stem
            cid = r.get("center_id")
            if stem and cid:
                stem_to_center[stem] = str(cid)
    out: dict[str, list[float]] = {}
    for row in per_image_rows:
        val = row.get(field)
        if val is None:
            continue
        stem = row.get("stem") or Path(row.get("image_name", "")).stem
        cid = row.get("center_id") or stem_to_center.get(stem) or "unknown"
        out.setdefault(str(cid), []).append(float(val))
    return out


def dice_values_by_center(
    per_image_seg: list[dict],
    results: Optional[list[dict]] = None,
) -> dict[str, list[float]]:
    stem_to_center: dict[str, str] = {}
    if results:
        for r in results:
            stem = Path(r.get("image_name", "")).stem
            cid = r.get("center_id")
            if stem and cid:
                stem_to_center[stem] = str(cid)
    out: dict[str, list[float]] = {}
    for row in per_image_seg:
        d = row.get("dice")
        if d is None:
            continue
        stem = row.get("stem") or Path(row.get("image_name", "")).stem
        cid = row.get("center_id") or stem_to_center.get(stem) or "unknown"
        out.setdefault(str(cid), []).append(float(d))
    return out


def aggregate_gland_by_center(
    per_image_gland: list[dict],
    results: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """全腺体逐图 Dice 按中心汇总（结构同结节，仅 dice/hd95）。"""
    stem_to_center: dict[str, str] = {}
    if results:
        for r in results:
            stem = Path(r.get("image_name", "")).stem
            cid = r.get("center_id")
            if stem and cid:
                stem_to_center[stem] = str(cid)

    centers: dict[str, dict[str, Any]] = defaultdict(lambda: {"dice_values": [], "hd95_values": []})
    for row in per_image_gland:
        stem = row.get("stem") or Path(row.get("image_name", "")).stem
        cid = row.get("center_id") or stem_to_center.get(stem) or "unknown"
        block = centers[cid]
        if row.get("dice") is not None:
            block["dice_values"].append(float(row["dice"]))
        if row.get("hd95") is not None:
            block["hd95_values"].append(float(row["hd95"]))

    out_centers: dict[str, Any] = {}
    dice_macro: list[float] = []
    for cid, block in sorted(centers.items()):
        dice_vals = block["dice_values"]
        hd95_vals = block["hd95_values"]
        dice_mean = float(np.mean(dice_vals)) if dice_vals else None
        if dice_mean is not None:
            dice_macro.append(dice_mean)
        out_centers[cid] = {
            "n_images": len(dice_vals),
            "dice_mean": dice_mean,
            "hd95_mean": float(np.mean(hd95_vals)) if hd95_vals else None,
        }
    return {
        "n_centers": len(out_centers),
        "macro_mean_dice": float(np.mean(dice_macro)) if dice_macro else None,
        "centers": out_centers,
    }


def load_center_metrics_from_output_dir(output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    per_seg = _load_jsonl(out / "per_image_seg_metrics.jsonl")
    per_cls = _load_jsonl(out / "per_image_cls_metrics.jsonl")
    results_path = out / "results.json"
    results = None
    if results_path.is_file():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    return aggregate_by_center(
        per_image_seg=per_seg,
        per_image_cls=per_cls,
        results=results,
    )
