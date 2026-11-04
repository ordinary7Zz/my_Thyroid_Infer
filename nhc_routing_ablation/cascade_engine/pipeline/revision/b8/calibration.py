"""B8：分类概率校准指标（ECE、Brier、可靠性分箱）。"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def malignant_probability(final_label: int, final_confidence: float) -> float:
    """与 evaluate.py 一致：score 表示 P(恶性=1)。"""
    conf = float(final_confidence)
    if int(final_label) == 1:
        return conf
    return 1.0 - conf


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[dict[str, Any]]]:
    """等宽分箱 ECE + 各 bin 统计（可靠性图）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    n = y_true.size
    if n == 0:
        return float("nan"), []

    bin_ids = np.minimum(np.floor(y_prob * n_bins).astype(np.int32), n_bins - 1)
    ece = 0.0
    bins: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            bins.append({
                "bin": b,
                "count": 0,
                "avg_confidence": None,
                "avg_accuracy": None,
            })
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece += abs(avg_acc - avg_conf) * (count / n)
        bins.append({
            "bin": b,
            "count": count,
            "avg_confidence": avg_conf,
            "avg_accuracy": avg_acc,
        })
    return float(ece), bins


def calibration_from_results(
    results: list[dict],
    *,
    n_bins: int = 15,
    label_to_binary,
) -> dict[str, Any]:
    """从 pipeline results.json 条目计算校准指标。"""
    y_true_list: list[int] = []
    y_prob_list: list[float] = []
    for r in results:
        if "true_label" not in r or "final_label" not in r:
            continue
        tb = label_to_binary(r["true_label"])
        pb = label_to_binary(r["final_label"])
        if tb is None or pb is None:
            continue
        conf = float(r.get("final_confidence", 0.5))
        y_true_list.append(int(tb))
        y_prob_list.append(malignant_probability(int(pb), conf))

    if not y_true_list:
        return {"n": 0, "ece": None, "brier": None, "bins": []}

    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)
    ece, bins = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    return {
        "n": int(len(y_true_list)),
        "ece": ece,
        "brier": brier_score(y_true, y_prob),
        "bins": bins,
    }
