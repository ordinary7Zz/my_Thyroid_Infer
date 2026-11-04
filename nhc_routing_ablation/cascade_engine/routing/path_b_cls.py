"""Path B 分类裁决（B3：独立模型-only，不含 AutoGluon）。"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Optional

from classification.agent import ClsAgentDecision
from classification.base_model import ClsModelOutput
from classification.soft_voting import average_class_probabilities, winning_class_from_avg_probs
from shared.routing_modes import normalize_routing_mode

if TYPE_CHECKING:
    from routing.learned_router import LearnedExpertRouter


def _normalize_top_class(label: Any) -> Optional[int]:
    if label is None:
        return None
    if isinstance(label, (int, float)):
        return int(label)
    s = str(label)
    if s in ("恶性", "malignant", "M", "1", "1.0"):
        return 1
    if s in ("良性", "benign", "B", "0", "0.0"):
        return 0
    return None


def _class_to_str(binary: int) -> str:
    return "恶性" if binary == 1 else "良性"


def resolve_path_b_indie_only(
    mode: str,
    indie_predictions: list[ClsModelOutput],
    *,
    best_single_model: Optional[str] = None,
    learned_router: Optional["LearnedExpertRouter"] = None,
) -> ClsAgentDecision:
    """
    B3 策略 1–6、8（分类）：仅使用 Phase 2 独立模型，不并入 AutoGluon。
    """
    if not indie_predictions:
        raise ValueError("没有独立分类模型预测")

    m = normalize_routing_mode(mode) or mode
    method = f"path_b_{m}"

    if m == "best_single":
        if not best_single_model:
            best_single_model = indie_predictions[0].model_name
        pred = next(
            (p for p in indie_predictions if p.model_name == best_single_model),
            indie_predictions[0],
        )
        return ClsAgentDecision(
            selected_model=pred.model_name,
            selected_class=pred.top_class,
            confidence=float(pred.top_confidence),
            reasoning=f"best_single 固定专家 {pred.model_name}",
            all_predictions=[p.to_dict() for p in indie_predictions],
            method=method,
        )

    if m in ("soft_voting", "average"):
        avg = average_class_probabilities(indie_predictions)
        cls, conf = winning_class_from_avg_probs(avg)
        return ClsAgentDecision(
            selected_model="indie_soft_voting",
            selected_class=cls,
            confidence=float(conf),
            reasoning=f"独立模型概率平均 → {cls}({conf:.2f})",
            all_predictions=[p.to_dict() for p in indie_predictions],
            method=method,
        )

    if m == "majority":
        votes = []
        for p in indie_predictions:
            b = _normalize_top_class(p.top_class)
            if b is not None:
                votes.append(b)
        if not votes:
            best = max(indie_predictions, key=lambda x: x.top_confidence)
            return ClsAgentDecision(
                selected_model=best.model_name,
                selected_class=best.top_class,
                confidence=float(best.top_confidence),
                reasoning="多数票无效，回退最高置信度",
                all_predictions=[p.to_dict() for p in indie_predictions],
                method=method,
            )
        majority_bin = Counter(votes).most_common(1)[0][0]
        majority_cls = _class_to_str(majority_bin)
        supporters = [p for p in indie_predictions if _normalize_top_class(p.top_class) == majority_bin]
        conf = sum(p.top_confidence for p in supporters) / len(supporters)
        return ClsAgentDecision(
            selected_model="indie_majority",
            selected_class=majority_cls,
            confidence=float(conf),
            reasoning=f"独立模型多数票 → {majority_cls}",
            all_predictions=[p.to_dict() for p in indie_predictions],
            method=method,
        )

    if m == "max_confidence":
        best = max(indie_predictions, key=lambda p: p.top_confidence)
        return ClsAgentDecision(
            selected_model=best.model_name,
            selected_class=best.top_class,
            confidence=float(best.top_confidence),
            reasoning=f"最高置信度专家 {best.model_name}",
            all_predictions=[p.to_dict() for p in indie_predictions],
            method=method,
        )

    if m == "static_rule":
        # 与分类 prompt 决策序近似：置信度优先，次看 entropy（若有）
        def score(p: ClsModelOutput) -> tuple:
            meta = p.metadata or {}
            ent = meta.get("entropy")
            ent_val = float(ent) if ent is not None else 1.0
            return (float(p.top_confidence), -ent_val)

        best = max(indie_predictions, key=score)
        return ClsAgentDecision(
            selected_model=best.model_name,
            selected_class=best.top_class,
            confidence=float(best.top_confidence),
            reasoning=f"static_rule 决策序 → {best.model_name}",
            all_predictions=[p.to_dict() for p in indie_predictions],
            method=method,
        )

    if m == "oracle":
        raise ValueError("oracle 分类请使用 routing.oracle.resolve_cls_oracle")

    if m == "learned":
        if learned_router is None:
            raise ValueError("routing_mode=learned 需要已加载的 LearnedExpertRouter（fit 或 random_init）")
        return learned_router.resolve_cls(indie_predictions)

    if m == "llm":
        raise ValueError("llm Path B 应走 LLMClassificationAgent.resolve_path_b")

    raise ValueError(f"不支持的 Path B routing_mode: {mode}")
