"""B7：汇总中心表、异质性与 meta 回归为单一 JSON 报告。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pipeline.revision.b7.meta_regression import heterogeneity_i2_tau2, meta_regress_image_dice
from pipeline.revision.b7.nhc_heterogeneity import (
    _load_jsonl,
    aggregate_by_center,
    aggregate_gland_by_center,
    dice_values_by_center,
    score_values_by_center,
)
from pipeline.revision.b7.per_image_enrich import build_meta_regression_rows
from pipeline.revision.b7.within_center import within_center_summary


def build_b7_report(
    output_dir: Path,
    *,
    results: list[dict],
    center_descriptive: Optional[dict[str, Any]] = None,
    meta_covariate_keys: Optional[list[str]] = None,
) -> dict[str, Any]:
    per_seg = _load_jsonl(output_dir / "per_image_seg_metrics.jsonl")
    per_cls = _load_jsonl(output_dir / "per_image_cls_metrics.jsonl")
    per_gland = _load_jsonl(output_dir / "per_image_gland_metrics.jsonl")

    center_block = aggregate_by_center(
        per_image_seg=per_seg,
        per_image_cls=per_cls,
        results=results,
    )
    gland_block = aggregate_gland_by_center(per_gland, results=results) if per_gland else None

    nodule_dice_by_c = dice_values_by_center(per_seg, results)
    gland_dice_by_c = dice_values_by_center(per_gland, results) if per_gland else {}
    cls_score_by_c = score_values_by_center(per_cls, results, field="score_malignant")

    hetero = {
        "nodule_dice": heterogeneity_i2_tau2(nodule_dice_by_c),
        "gland_dice": heterogeneity_i2_tau2(gland_dice_by_c) if gland_dice_by_c else None,
        "classification_score": heterogeneity_i2_tau2(cls_score_by_c) if cls_score_by_c else None,
    }
    within = {
        "nodule_dice": within_center_summary(nodule_dice_by_c),
        "gland_dice": within_center_summary(gland_dice_by_c) if gland_dice_by_c else None,
        "classification_score": within_center_summary(cls_score_by_c) if cls_score_by_c else None,
    }

    meta: dict[str, Any] = {}
    if meta_covariate_keys and per_seg:
        center_mr: dict[str, float] = {}
        if center_descriptive:
            for cid, block in (center_descriptive.get("centers") or {}).items():
                ratio = block.get("malignant_ratio")
                if ratio is not None:
                    center_mr[str(cid)] = float(ratio)
        rows = build_meta_regression_rows(
            per_seg,
            center_malignant_ratio=center_mr or None,
        )
        meta["nodule_dice"] = meta_regress_image_dice(rows, covariate_keys=meta_covariate_keys)

    return {
        "output_dir": str(output_dir),
        "n_centers": center_block.get("n_centers"),
        "center_performance": center_block,
        "gland_center_performance": gland_block,
        "heterogeneity": hetero,
        "within_center": within,
        "meta_regression": meta or None,
        "center_descriptive": center_descriptive,
    }
