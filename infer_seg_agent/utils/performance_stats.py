"""
Aggregate Dice / HD95 / ECE statistics from batch result dicts (shared by main and merge tools).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def extract_ece_scores(results: List[Dict[str, Any]]) -> List[float]:
    """Extract per-sample ECE values from result dicts."""
    ece_scores: List[float] = []
    for r in results:
        if r.get("ece_mean") is not None:
            try:
                ece_scores.append(float(r["ece_mean"]))
            except (TypeError, ValueError):
                pass
        else:
            em = r.get("ece_metrics")
            if isinstance(em, dict) and em.get("ece") is not None:
                try:
                    ece_scores.append(float(em["ece"]))
                except (TypeError, ValueError):
                    pass
    return ece_scores


def bootstrap_mean_ci95(
    values: List[float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Optional[Tuple[float, float]]:
    """
    Bootstrap 95% CI for mean using percentile interval.
    Returns (lower, upper). If values is empty, returns None.
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


def build_performance_stats(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build aggregate performance stats for Dice/HD95/ECE, including CI95 for means."""
    dice_scores = [float(r["dice_score"]) for r in results if r.get("dice_score") is not None]
    hd95_scores = [
        float(r["hd95_score"])
        for r in results
        if r.get("hd95_score") is not None and r["hd95_score"] != float("inf")
    ]
    ece_scores = extract_ece_scores(results)

    if not (dice_scores or hd95_scores or ece_scores):
        return None

    stats: Dict[str, Any] = {
        "num_samples_with_gt": max(len(dice_scores), len(ece_scores), len(hd95_scores))
    }

    def _fill_metric(metric_key: str, values: List[float]) -> None:
        if not values:
            return
        mean_v = float(np.mean(values))
        std_v = float(np.std(values))
        min_v = float(np.min(values))
        max_v = float(np.max(values))
        ci95 = bootstrap_mean_ci95(values)
        stats[f"mean_{metric_key}"] = mean_v
        stats[f"std_{metric_key}"] = std_v
        stats[f"min_{metric_key}"] = min_v
        stats[f"max_{metric_key}"] = max_v
        if ci95 is not None:
            stats[f"mean_{metric_key}_ci95"] = [float(ci95[0]), float(ci95[1])]

    _fill_metric("dice", dice_scores)
    _fill_metric("hd95", hd95_scores)
    _fill_metric("ece", ece_scores)
    return stats
