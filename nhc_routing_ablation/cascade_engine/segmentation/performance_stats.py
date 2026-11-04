"""
批次性能统计聚合：Dice/HD95/ECE 的 mean/std/min/max + bootstrap CI95。

重写自 Segmentation_Agent/utils/performance_stats.py，逻辑一致。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def extract_ece_scores(results: list[dict[str, Any]]) -> list[float]:
    """从结果 dict 列表提取每样本 ECE。"""
    scores: list[float] = []
    for r in results:
        if r.get("ece_mean") is not None:
            try:
                scores.append(float(r["ece_mean"]))
            except (TypeError, ValueError):
                pass
        else:
            em = r.get("ece_metrics")
            if isinstance(em, dict) and em.get("ece") is not None:
                try:
                    scores.append(float(em["ece"]))
                except (TypeError, ValueError):
                    pass
    return scores


def bootstrap_mean_ci95(
    values: list[float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Optional[tuple[float, float]]:
    """
    Bootstrap 95% 置信区间（百分位法）。

    Returns:
        (lower, upper)，空列表返回 None，单值返回 (v, v)。
    """
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        v = float(arr[0])
        return (v, v)

    rng = np.random.default_rng(seed)
    n = arr.size
    sample_idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = arr[sample_idx].mean(axis=1)
    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))
    return (lower, upper)


def build_performance_stats(results: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """
    聚合 Dice/HD95/ECE 统计 + CI95。

    Args:
        results: 每个样本的结果 dict，含 dice_score / hd95_score / ece_mean 等字段。

    Returns:
        含 mean/std/min/max/ci95 的统计 dict，无可用数据返回 None。
    """
    dice = [float(r["dice_score"]) for r in results if r.get("dice_score") is not None]
    hd95 = [
        float(r["hd95_score"])
        for r in results
        if r.get("hd95_score") is not None and r["hd95_score"] != float("inf")
    ]
    ece = extract_ece_scores(results)

    if not (dice or hd95 or ece):
        return None

    stats: dict[str, Any] = {
        "num_samples_with_gt": max(len(dice), len(ece), len(hd95)),
    }

    def _fill(key: str, vals: list[float]) -> None:
        if not vals:
            return
        ci95 = bootstrap_mean_ci95(vals)
        stats[f"mean_{key}"] = float(np.mean(vals))
        stats[f"std_{key}"] = float(np.std(vals))
        stats[f"min_{key}"] = float(np.min(vals))
        stats[f"max_{key}"] = float(np.max(vals))
        if ci95 is not None:
            stats[f"mean_{key}_ci95"] = [float(ci95[0]), float(ci95[1])]

    _fill("dice", dice)
    _fill("hd95", hd95)
    _fill("ece", ece)
    return stats
