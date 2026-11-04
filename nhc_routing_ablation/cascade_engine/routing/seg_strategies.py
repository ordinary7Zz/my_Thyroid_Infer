"""Phase 3 分割路由（读缓存候选，不经 LLM）。"""

from __future__ import annotations

from typing import Optional

import numpy as np

from segmentation.agent import SegAgentDecision
from segmentation.base_model import SegModelOutput
from segmentation.metrics import compute_dice
from shared.routing_modes import normalize_routing_mode


def _path_label(anchor_class: Optional[str]) -> str:
    return "A" if anchor_class else "B"


def select_mask_soft_voting(
    predictions: list[SegModelOutput],
    anchor_class: Optional[str],
    threshold: float = 0.5,
) -> SegAgentDecision:
    if not predictions:
        raise ValueError("无分割候选")
    h, w = predictions[0].mask.shape[:2]
    acc = np.zeros((h, w), dtype=np.float64)
    for p in predictions:
        acc += p.mask.astype(np.float64)
    acc /= len(predictions)
    final = (acc >= threshold).astype(np.uint8)
    return SegAgentDecision(
        selected_model="ensemble_soft_voting",
        selected_mask=final,
        confidence=float(acc.mean()),
        reasoning=f"掩膜平均 (n={len(predictions)}, thr={threshold})",
        all_predictions=[p.to_dict() for p in predictions],
        is_ensemble=True,
        selected_models=[p.model_name for p in predictions],
        classification_anchor=anchor_class,
        path=_path_label(anchor_class),
    )


def select_mask_majority(
    predictions: list[SegModelOutput],
    anchor_class: Optional[str],
) -> SegAgentDecision:
    if not predictions:
        raise ValueError("无分割候选")
    stacks = np.stack([p.mask.astype(np.uint8) for p in predictions], axis=0)
    vote = (stacks.sum(axis=0) > (stacks.shape[0] / 2)).astype(np.uint8)
    return SegAgentDecision(
        selected_model="ensemble_majority",
        selected_mask=vote,
        confidence=float(vote.mean()),
        reasoning=f"像素多数票 (n={len(predictions)})",
        all_predictions=[p.to_dict() for p in predictions],
        is_ensemble=True,
        classification_anchor=anchor_class,
        path=_path_label(anchor_class),
    )


def select_mask_max_confidence(
    predictions: list[SegModelOutput],
    anchor_class: Optional[str],
) -> SegAgentDecision:
    best = max(
        predictions,
        key=lambda p: float(
            (p.metadata or {}).get("mean_confidence")
            or (p.confidence_map.mean() if p.confidence_map is not None else 0.0)
        ),
    )
    conf = float((best.metadata or {}).get("mean_confidence", 0.5))
    return SegAgentDecision(
        selected_model=best.model_name,
        selected_mask=best.mask,
        confidence=conf,
        reasoning=f"最高 mean_confidence → {best.model_name}",
        all_predictions=[p.to_dict() for p in predictions],
        classification_anchor=anchor_class,
        path=_path_label(anchor_class),
    )


def select_mask_best_single(
    predictions: list[SegModelOutput],
    model_name: str,
    anchor_class: Optional[str],
) -> SegAgentDecision:
    pred = next((p for p in predictions if p.model_name == model_name), predictions[0])
    return SegAgentDecision(
        selected_model=pred.model_name,
        selected_mask=pred.mask,
        confidence=0.5,
        reasoning=f"best_single 固定专家 {pred.model_name}",
        all_predictions=[p.to_dict() for p in predictions],
        classification_anchor=anchor_class,
        path=_path_label(anchor_class),
    )


def select_mask_oracle(
    predictions: list[SegModelOutput],
    gt_mask: np.ndarray,
    anchor_class: Optional[str],
) -> SegAgentDecision:
    best_pred = predictions[0]
    best_dice = -1.0
    for p in predictions:
        d = compute_dice(p.mask, gt_mask)
        if d > best_dice:
            best_dice = d
            best_pred = p
    return SegAgentDecision(
        selected_model=best_pred.model_name,
        selected_mask=best_pred.mask,
        confidence=float(best_dice),
        reasoning=f"Oracle Dice={best_dice:.4f} → {best_pred.model_name}",
        all_predictions=[p.to_dict() for p in predictions],
        dice_score=best_dice,
        classification_anchor=anchor_class,
        path=_path_label(anchor_class),
    )


def select_mask_for_mode(
    mode: str,
    predictions: list[SegModelOutput],
    anchor_class: Optional[str],
    *,
    gt_mask: Optional[np.ndarray] = None,
    best_single_model: Optional[str] = None,
    static_selector=None,
) -> SegAgentDecision:
    """
    static_selector: 可选 Callable → SegAgentDecision，用于 static_rule（cascade._select_mask_static）。
    """
    m = normalize_routing_mode(mode) or mode
    if m in ("soft_voting", "average"):
        return select_mask_soft_voting(predictions, anchor_class)
    if m == "majority":
        return select_mask_majority(predictions, anchor_class)
    if m == "max_confidence":
        return select_mask_max_confidence(predictions, anchor_class)
    if m == "best_single":
        name = best_single_model or predictions[0].model_name
        return select_mask_best_single(predictions, name, anchor_class)
    if m == "oracle":
        if gt_mask is None:
            raise ValueError("oracle 分割需要 gt_mask")
        return select_mask_oracle(predictions, gt_mask, anchor_class)
    if m == "static_rule":
        if static_selector is None:
            raise ValueError("static_rule 需要 static_selector")
        return static_selector()
    if m in ("llm", "learned"):
        raise ValueError(f"select_mask_for_mode 不支持 {m}")
    raise ValueError(f"未知分割 routing_mode: {mode}")
