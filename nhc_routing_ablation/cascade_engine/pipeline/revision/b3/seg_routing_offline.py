"""
仅读 intermediate 的分割路由评测（波次 1：不调 LLM / 不跑 cascade）。

适用于 soft_voting / majority / max_confidence / best_single / oracle。
static_rule（B3 默认 agreement-only）请用 static_b3_bundle / cascade；含 radiomics 时设 static_rule_use_radiomics_judge。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pipeline.cascade import CascadePipeline
from pipeline.compare_models import (
    eval_seg_masks,
    load_cls_model_json,
    load_seg_model_masks,
)
from routing.seg_strategies import select_mask_for_mode
from classification.base_model import ClsModelOutput
from segmentation.base_model import SegModelOutput
from shared.image_io import ImageIO


def _stem(n: str) -> str:
    return Path(n).stem


def _load_cls_preds_by_image(intermediate_dir: Path, image_names: list[str]) -> dict[str, list]:
    cls_dir = intermediate_dir / "cls"
    by_model: dict[str, dict] = {}
    for jp in sorted(cls_dir.glob("*.json")):
        img_map, model_name = load_cls_model_json(jp)
        by_model[model_name] = img_map

    out: dict[str, list] = {n: [] for n in image_names}
    for model_name, img_map in by_model.items():
        for img_name in image_names:
            pred = img_map.get(img_name) or img_map.get(_stem(img_name))
            if pred is None:
                continue
            out[img_name].append(
                ClsModelOutput(
                    model_name=model_name,
                    predictions=pred.get("predictions", {}),
                    top_class=pred.get("top_class", ""),
                    top_confidence=float(pred.get("top_confidence", 0)),
                    requires_mask=False,
                    metadata=pred.get("metadata", {}),
                )
            )
    return out


def _load_seg_preds_for_image(intermediate_dir: Path, img_name: str) -> list[SegModelOutput]:
    seg_dir = intermediate_dir / "seg"
    preds: list[SegModelOutput] = []
    target_stem = _stem(img_name)
    for npz in sorted(seg_dir.glob("*.npz")):
        names, masks, model_name = load_seg_model_masks(npz)
        j = None
        for i, n in enumerate(names):
            if n == img_name or _stem(n) == target_stem:
                j = i
                break
        if j is None:
            continue
        preds.append(
            SegModelOutput(
                model_name=model_name,
                mask=masks[j],
                confidence_map=None,
                metadata={},
            )
        )
    return preds


def evaluate_seg_routing_mode_offline(
    intermediate_dir: Path,
    gt_mask_dir: Path,
    routing_mode: str,
    *,
    best_single_model: Optional[str] = None,
    image_io: Optional[ImageIO] = None,
) -> dict:
    image_io = image_io or ImageIO()
    inter = Path(intermediate_dir)
    ref_npz = next((inter / "seg").glob("*.npz"))
    all_names, _, _ = load_seg_model_masks(ref_npz)

    cls_by_img = _load_cls_preds_by_image(inter, all_names)
    selected_names: list[str] = []
    selected_masks = []
    methods: set[str] = {f"routing_{routing_mode}_seg"}

    gt_dir = Path(gt_mask_dir)

    for img_name in all_names:
        seg_preds = _load_seg_preds_for_image(inter, img_name)
        if not seg_preds:
            continue
        indie = cls_by_img.get(img_name, [])
        consensus, anchor = CascadePipeline._check_classification_consensus(indie)
        gt_mask = None
        gt_path = gt_dir / img_name
        if not gt_path.exists():
            gt_path = gt_dir / f"{_stem(img_name)}.png"
        if not gt_path.exists():
            for p in gt_dir.iterdir():
                if p.is_file() and p.stem == _stem(img_name):
                    gt_path = p
                    break
        if gt_path.exists():
            gt_mask = image_io.binarize_mask(image_io.load_mask(gt_path))

        decision = select_mask_for_mode(
            routing_mode,
            seg_preds,
            anchor if consensus else None,
            gt_mask=gt_mask,
            best_single_model=best_single_model,
        )
        selected_names.append(img_name)
        selected_masks.append(decision.selected_mask)

    return {
        "routing_mode": routing_mode,
        "methods": sorted(methods),
        "metrics": eval_seg_masks(selected_names, selected_masks, gt_dir, image_io),
    }


def evaluate_seg_mode_ladder(
    intermediate_dir: Path,
    gt_mask_dir: Path,
    modes: list[str],
    best_single_model: Optional[str] = None,
) -> list[dict]:
    rows = []
    for mode in modes:
        if mode in ("static_rule", "llm", "learned"):
            rows.append({
                "routing_mode": mode,
                "skipped": True,
                "reason": "需要 cascade（judge/LLM）",
            })
            continue
        rows.append(
            evaluate_seg_routing_mode_offline(
                intermediate_dir,
                gt_mask_dir,
                mode,
                best_single_model=best_single_model,
            )
        )
    return rows
