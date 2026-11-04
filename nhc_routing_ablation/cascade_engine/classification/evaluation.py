"""
分类评估指标：ROC/AUC、Accuracy、bootstrap CI95。
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def compute_roc_auc(labels: np.ndarray, probs: np.ndarray) -> dict:
    """计算 ROC AUC + 曲线。"""
    from sklearn.metrics import roc_auc_score, roc_curve

    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=np.float64)

    if len(np.unique(labels)) < 2:
        return {"auc": 0.5, "fpr": [0, 1], "tpr": [0, 1], "thresholds": [1, 0]}

    auc = float(roc_auc_score(labels, probs))
    fpr, tpr, thresholds = roc_curve(labels, probs)
    return {
        "auc": auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def compute_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算准确率。"""
    from sklearn.metrics import accuracy_score

    return float(accuracy_score(labels, preds))


def bootstrap_ci95(
    labels: np.ndarray,
    probs: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Optional[tuple[float, float]]:
    """
    Bootstrap 95% 置信区间。

    Args:
        labels: 真实标签。
        probs: 预测概率或类别。
        metric_fn: 评估函数，如 roc_auc_score。
        n_bootstrap: bootstrap 次数。

    Returns:
        (lower, upper)，数据不足返回 None。
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return None

    rng = np.random.default_rng(seed)
    n = len(labels)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            scores.append(metric_fn(labels[idx], probs[idx]))
        except Exception:
            continue

    if len(scores) < 10:
        return None

    lower = float(np.percentile(scores, 2.5))
    upper = float(np.percentile(scores, 97.5))
    return (lower, upper)


def bootstrap_auc_ci95(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 2000,
) -> Optional[tuple[float, float]]:
    """Bootstrap AUC 的 95% CI。"""
    from sklearn.metrics import roc_auc_score

    return bootstrap_ci95(labels, probs, roc_auc_score, n_bootstrap)
