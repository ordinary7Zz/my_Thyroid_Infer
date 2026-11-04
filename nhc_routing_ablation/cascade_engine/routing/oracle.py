"""Oracle 路由（仅评测，上界标尺）。"""

from __future__ import annotations

from typing import Optional

from classification.agent import ClsAgentDecision
from classification.base_model import ClsModelOutput
from shared.routing_modes import normalize_routing_mode


def _label_binary(label) -> Optional[int]:
    if isinstance(label, (int, float)):
        return int(label)
    if isinstance(label, str):
        if label in ("恶性", "malignant", "1"):
            return 1
        if label in ("良性", "benign", "0"):
            return 0
    return None


def resolve_cls_oracle(
    indie_predictions: list[ClsModelOutput],
    true_label: int,
) -> ClsAgentDecision:
    """每例选预测类别与 GT 一致且置信度最高的独立模型。"""
    correct = []
    for p in indie_predictions:
        pred_bin = _label_binary(p.top_class)
        if pred_bin is not None and pred_bin == int(true_label):
            correct.append(p)
    if correct:
        best = max(correct, key=lambda p: p.top_confidence)
        return ClsAgentDecision(
            selected_model=best.model_name,
            selected_class=best.top_class,
            confidence=float(best.top_confidence),
            reasoning=f"Oracle cls 正确专家 {best.model_name}",
            all_predictions=[p.to_dict() for p in indie_predictions],
            method="path_b_oracle",
        )
    # 无正确专家：取与 GT 概率最接近的恶性概率
    target = "恶性" if int(true_label) == 1 else "良性"
    best = max(indie_predictions, key=lambda p: p.top_confidence)
    return ClsAgentDecision(
        selected_model=best.model_name,
        selected_class=target,
        confidence=float(best.top_confidence) * 0.5,
        reasoning="Oracle cls 无正确专家，回退指定 GT 类",
        all_predictions=[p.to_dict() for p in indie_predictions],
        method="path_b_oracle",
    )
