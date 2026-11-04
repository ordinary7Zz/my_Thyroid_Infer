"""B8：从专家预测与分割一致性提取不确定度特征（不调用 LLM）。"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from classification.base_model import ClsModelOutput
from segmentation.quality_evaluator import SegmentationQualityEvaluator

_DEFAULT_EVALUATOR = SegmentationQualityEvaluator()


def _class_probs_from_output(pred: ClsModelOutput) -> list[float]:
    probs = list((pred.predictions or {}).values())
    if not probs:
        return []
    total = sum(probs)
    if total <= 0:
        return []
    return [p / total for p in probs]


def cls_disagreement_features(indie_preds: Sequence[ClsModelOutput]) -> dict[str, Any]:
    """独立分类模型分歧：vote 熵、margin、投票一致性。"""
    preds = list(indie_preds)
    n = len(preds)
    if n == 0:
        return {
            "num_indie_cls_models": 0,
            "vote_entropy": 0.0,
            "margin_top2_min": 0.0,
            "margin_top2_mean": 0.0,
            "num_models_same_class": 0,
            "classification_unanimous": False,
        }

    votes_per_class: dict[str, int] = {}
    for pred in preds:
        votes_per_class[pred.top_class] = votes_per_class.get(pred.top_class, 0) + 1

    vote_entropy = 0.0
    if n > 0:
        class_probs = [c / n for c in votes_per_class.values()]
        vote_entropy = -sum(p * math.log(p + 1e-12, 2) for p in class_probs if p > 0)

    margins: list[float] = []
    for pred in preds:
        norm = _class_probs_from_output(pred)
        if not norm:
            continue
        sorted_p = sorted(norm, reverse=True)
        margin = (sorted_p[0] - sorted_p[1]) if len(sorted_p) >= 2 else sorted_p[0]
        margins.append(float(margin))

    majority = max(votes_per_class.values()) if votes_per_class else 0
    top_class = max(votes_per_class, key=votes_per_class.get) if votes_per_class else None

    return {
        "num_indie_cls_models": n,
        "vote_entropy": round(float(vote_entropy), 6),
        "margin_top2_min": round(min(margins), 6) if margins else 0.0,
        "margin_top2_mean": round(float(np.mean(margins)), 6) if margins else 0.0,
        "num_models_same_class": int(majority),
        "majority_top_class": top_class,
        "classification_unanimous": majority == n and n > 0,
    }


def seg_disagreement_from_masks(masks: Sequence[np.ndarray]) -> dict[str, Any]:
    """多专家 mask 两两 IoU 分歧（与返修「专家 IoU」一致）。"""
    mask_list = [np.asarray(m) for m in masks]
    if len(mask_list) < 2:
        return {
            "num_seg_models": len(mask_list),
            "seg_min_pairwise_iou": 1.0,
            "seg_mean_pairwise_iou": 1.0,
            "seg_volume_cv": 0.0,
        }

    agreement = _DEFAULT_EVALUATOR.evaluate_model_agreement(mask_list)
    iou_mat = agreement.get("pairwise_iou_matrix")
    min_iou = 1.0
    if iou_mat is not None:
        iou_mat = np.asarray(iou_mat)
        n = iou_mat.shape[0]
        upper = iou_mat[np.triu_indices(n, k=1)]
        if upper.size > 0:
            min_iou = float(np.min(upper))

    return {
        "num_seg_models": len(mask_list),
        "seg_min_pairwise_iou": round(min_iou, 6),
        "seg_mean_pairwise_iou": round(float(agreement.get("overall_agreement", 0.0)), 6),
        "seg_volume_cv": round(float(agreement.get("volume_cv") or 0.0), 6),
    }


def seg_disagreement_from_stored(stored: Optional[dict]) -> dict[str, Any]:
    """从 Phase 3 写入的 seg_agreement 摘要读取（避免 Phase 5 重载 npz）。"""
    if not stored:
        return {
            "num_seg_models": 0,
            "seg_min_pairwise_iou": 1.0,
            "seg_mean_pairwise_iou": 1.0,
            "seg_volume_cv": 0.0,
        }
    return {
        "num_seg_models": int(stored.get("num_seg_models", 0)),
        "seg_min_pairwise_iou": float(stored.get("seg_min_pairwise_iou", 1.0)),
        "seg_mean_pairwise_iou": float(stored.get("seg_mean_pairwise_iou", 1.0)),
        "seg_volume_cv": float(stored.get("seg_volume_cv", 0.0)),
    }


def routing_uncertainty_flags(
    *,
    final_confidence: float,
    seg_confidence: Optional[float],
    llm_used_seg: bool,
    llm_used_cls: bool,
    low_confidence_threshold: float = 0.6,
) -> dict[str, Any]:
    """LLM 路由队列：低置信视为不确定；NHC 无 LLM 时均为 false。"""
    seg_unc = bool(llm_used_seg and seg_confidence is not None and seg_confidence < low_confidence_threshold)
    cls_unc = bool(llm_used_cls and final_confidence < low_confidence_threshold)
    return {
        "routing_uncertain_seg": seg_unc,
        "routing_uncertain_cls": cls_unc,
        "routing_uncertain_flag": seg_unc or cls_unc,
        "low_confidence_threshold": low_confidence_threshold,
    }


def build_uncertainty_block(
    indie_preds: Sequence[ClsModelOutput],
    seg_agreement: Optional[dict],
    *,
    final_confidence: float,
    seg_confidence: Optional[float],
    llm_used_seg: bool,
    llm_used_cls: bool,
    classification_consensus: bool,
) -> dict[str, Any]:
    block = {
        **cls_disagreement_features(indie_preds),
        **seg_disagreement_from_stored(seg_agreement),
        **routing_uncertainty_flags(
            final_confidence=final_confidence,
            seg_confidence=seg_confidence,
            llm_used_seg=llm_used_seg,
            llm_used_cls=llm_used_cls,
        ),
        "classification_consensus": classification_consensus,
    }
    return block
