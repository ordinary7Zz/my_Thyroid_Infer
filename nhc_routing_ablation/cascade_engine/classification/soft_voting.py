"""
Soft voting：对 top_k 个模型的类别概率做加权平均。
"""

from __future__ import annotations

from .base_model import ClsModelOutput


def average_class_probabilities(predictions: list[ClsModelOutput]) -> dict[str, float]:
    """对多个模型的各类别概率取算术平均（缺失类别视为 0）。"""
    if not predictions:
        return {}
    sums: dict[str, float] = {}
    for p in predictions:
        for cls, prob in p.predictions.items():
            sums[cls] = sums.get(cls, 0.0) + float(prob)
    n = len(predictions)
    return {c: v / n for c, v in sums.items()}


def winning_class_from_avg_probs(avg_probs: dict[str, float]) -> tuple[str, float]:
    if not avg_probs:
        return "", 0.0
    return max(avg_probs.items(), key=lambda x: x[1])


def resolve_topk_model_outputs(
    predictions: list[ClsModelOutput],
    requested_names: list[str],
    top_k: int,
) -> list[ClsModelOutput]:
    """
    按 LLM 给出的顺序选取模型名；无效或重复则跳过，
    不足 top_k 时用剩余模型按 top_confidence 补齐。
    """
    k = min(max(1, top_k), len(predictions))
    name_to_pred = {p.model_name: p for p in predictions}
    picked: list[ClsModelOutput] = []
    seen: set[str] = set()
    for name in requested_names:
        if name in name_to_pred and name not in seen:
            picked.append(name_to_pred[name])
            seen.add(name)
        if len(picked) >= k:
            break
    if len(picked) < k:
        rest = sorted(
            [p for p in predictions if p.model_name not in seen],
            key=lambda p: p.top_confidence,
            reverse=True,
        )
        for p in rest:
            picked.append(p)
            if len(picked) >= k:
                break
    return picked[:k]


def soft_voting(predictions: list[ClsModelOutput], top_k: int = 5) -> ClsModelOutput:
    """
    取 top_confidence 最高的 top_k 个模型，对 predictions 做加权平均。

    权重 = top_confidence（归一化）。
    """
    if not predictions:
        raise ValueError("没有预测结果")
    k = min(top_k, len(predictions))
    sorted_preds = sorted(predictions, key=lambda p: p.top_confidence, reverse=True)[:k]

    # 加权平均
    weights = [p.top_confidence for p in sorted_preds]
    total_w = sum(weights)
    if total_w <= 0:
        weights = [1.0 / k] * k
    else:
        weights = [w / total_w for w in weights]

    all_classes = set()
    for p in sorted_preds:
        all_classes.update(p.predictions.keys())

    avg_probs: dict[str, float] = {}
    for cls in all_classes:
        avg_probs[cls] = sum(
            p.predictions.get(cls, 0.0) * w for p, w in zip(sorted_preds, weights)
        )

    top_cls = max(avg_probs, key=avg_probs.get)
    return ClsModelOutput(
        model_name="soft_voting",
        predictions=avg_probs,
        top_class=top_cls,
        top_confidence=avg_probs[top_cls],
        requires_mask=any(p.requires_mask for p in sorted_preds),
        metadata={
            "voting_models": [p.model_name for p in sorted_preds],
            "voting_weights": weights,
            "method": "weighted_soft_voting",
        },
    )
