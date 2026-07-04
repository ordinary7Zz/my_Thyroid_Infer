"""
分类指标计算工具。

支持二分类和多分类（TIRADS 五分类），使用 bootstrap 估计 CI95 置信区间。

指标列表：
  - accuracy
  - precision  （多分类用 macro-average）
  - recall     （多分类用 macro-average）
  - f1         （多分类用 macro-average）
  - auroc      （多分类用 macro-average, OvR）
  - auprc      （多分类用 macro-average, OvR）
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# 指标名称统一顺序
METRIC_ORDER = ("accuracy", "precision", "recall", "f1", "auroc", "auprc")


# ---------------------------------------------------------------------------
# 单次指标计算
# ---------------------------------------------------------------------------

def _safe_metric(func, *args, **kwargs) -> float:
    """安全调用指标函数，出错或类别不足时返回 nan。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(func(*args, **kwargs))
    except Exception:
        return float("nan")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """计算单次二分类指标。y_prob 为正类概率。"""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int32)

    unique = np.unique(y_true)
    has_two_classes = unique.size >= 2

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": _safe_metric(roc_auc_score, y_true, y_prob) if has_two_classes else float("nan"),
        "auprc": _safe_metric(average_precision_score, y_true, y_prob) if has_two_classes else float("nan"),
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """计算单次多分类指标（macro-average）。y_probs 为 (N, C) 概率矩阵。"""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_probs = np.asarray(y_probs, dtype=np.float64)
    y_pred = y_probs.argmax(axis=1)

    unique = np.unique(y_true)
    has_two_classes = unique.size >= 2

    # AUROC / AUPRC 需要 one-hot 编码
    if has_two_classes:
        y_onehot = np.zeros((y_true.size, num_classes), dtype=np.float64)
        y_onehot[np.arange(y_true.size), y_true] = 1.0
        auroc = _safe_metric(
            roc_auc_score, y_true, y_probs, multi_class="ovr", average="macro"
        )
        auprc = _safe_metric(
            average_precision_score, y_onehot, y_probs, average="macro"
        )
    else:
        auroc = float("nan")
        auprc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": auroc,
        "auprc": auprc,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI95
# ---------------------------------------------------------------------------

def _summarize_bootstrap(
    metrics_samples: dict[str, list[float]],
    ci: float,
) -> dict[str, tuple[float, tuple[float, float]]]:
    """汇总 bootstrap 采样结果，返回 {metric: (mean, (lower, upper))}。"""
    results: dict[str, tuple[float, tuple[float, float]]] = {}
    alpha = 1.0 - ci
    for key, vals in metrics_samples.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr_valid = arr[~np.isnan(arr)]
        if arr_valid.size == 0:
            results[key] = (0.0, (0.0, 0.0))
            continue
        mean = float(arr_valid.mean())
        lower = float(np.percentile(arr_valid, 100 * alpha / 2))
        upper = float(np.percentile(arr_valid, 100 * (1 - alpha / 2)))
        results[key] = (mean, (lower, upper))
    return results


def binary_bootstrap_metrics(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, tuple[float, tuple[float, float]]]:
    """二分类 bootstrap 指标 + CI95。"""
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)

    valid = y_true != -1
    y_prob = y_prob[valid]
    y_true = y_true[valid]

    if y_true.size == 0:
        zero = (0.0, (0.0, 0.0))
        return {k: zero for k in METRIC_ORDER}

    rng = np.random.default_rng(seed)
    n = y_true.size
    metrics_samples: dict[str, list[float]] = {k: [] for k in METRIC_ORDER}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        probs_s = y_prob[idx]
        labels_s = y_true[idx]

        m = compute_binary_metrics(labels_s, probs_s, threshold)
        for key in METRIC_ORDER:
            metrics_samples[key].append(m[key])

    return _summarize_bootstrap(metrics_samples, ci)


def multiclass_bootstrap_metrics(
    y_probs: np.ndarray,
    y_true: np.ndarray,
    num_classes: int,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, tuple[float, tuple[float, float]]]:
    """多分类 bootstrap 指标 + CI95（macro-average）。"""
    y_probs = np.asarray(y_probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)

    valid = y_true != -1
    y_probs = y_probs[valid]
    y_true = y_true[valid]

    if y_true.size == 0:
        zero = (0.0, (0.0, 0.0))
        return {k: zero for k in METRIC_ORDER}

    rng = np.random.default_rng(seed)
    n = y_true.size
    metrics_samples: dict[str, list[float]] = {k: [] for k in METRIC_ORDER}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        probs_s = y_probs[idx]
        labels_s = y_true[idx]

        m = compute_multiclass_metrics(labels_s, probs_s, num_classes)
        for key in METRIC_ORDER:
            metrics_samples[key].append(m[key])

    return _summarize_bootstrap(metrics_samples, ci)


# ---------------------------------------------------------------------------
# 日志格式化
# ---------------------------------------------------------------------------

def format_metrics_log(
    metrics: dict[str, tuple[float, tuple[float, float]]],
    metric_order: tuple[str, ...] = METRIC_ORDER,
) -> str:
    """将指标字典格式化为可读的文本行列表。"""
    lines = []
    for key in metric_order:
        if key not in metrics:
            continue
        mean_v, (low_v, high_v) = metrics[key]
        lines.append(
            f"  {key.upper():<10} mean={mean_v:.4f}  CI95=({low_v:.4f}, {high_v:.4f})"
        )
    return "\n".join(lines)
