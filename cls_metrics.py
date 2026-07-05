"""
统一分类指标计算模块
====================
为所有分类推理脚本提供一致的指标计算 + Bootstrap CI95。

统一规范:
  - 指标集: AUROC, AUPRC, Accuracy, Precision, F1, Recall
  - 二分类: binary average
  - 多分类: macro average (AUROC/AUPRC 用 OvR)
  - CI95:   Bootstrap percentile, n_boot=2000, seed=0
  - 点估计: Bootstrap 采样均值

依赖: scikit-learn, numpy

API:
  compute_all_metrics(y_true, y_pred, y_prob, num_classes, n_boot)
    → dict {metric_name: {"value": float, "ci_lower": float, "ci_upper": float}}

  format_metrics_report(metrics, ...)
    → str (统一格式的文本报告)
"""

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
METRIC_ORDER = ("AUROC", "AUPRC", "Accuracy", "Precision", "F1", "Recall")


# ============================================================================
# 单次指标计算
# ============================================================================

def _safe_metric(func, *args, **kwargs) -> float:
    """安全调用指标函数，出错或类别不足时返回 nan。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(func(*args, **kwargs))
    except Exception:
        return float("nan")


def compute_point_metrics(y_true, y_pred, y_prob, num_classes):
    """计算单次（点估计）分类性能指标。

    二分类用 binary average，多分类用 macro average。

    Args:
        y_true: (N,) int 真实标签 (0-based)
        y_pred: (N,) int 预测标签
        y_prob: (N, C) float 概率矩阵
        num_classes: 类别数

    Returns:
        dict {metric_name: float}
    """
    is_binary = (num_classes == 2)
    avg = "binary" if is_binary else "macro"

    return {
        "AUROC": _safe_auroc(y_true, y_prob, num_classes),
        "AUPRC": _safe_auprc(y_true, y_prob, num_classes),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }


def _safe_auroc(y_true, y_prob, num_classes):
    """计算 AUROC。二分类用正类概率，多分类用 macro OvR。"""
    try:
        if num_classes == 2:
            return roc_auc_score(y_true, y_prob[:, 1])
        else:
            return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except (ValueError, IndexError):
        return float("nan")


def _safe_auprc(y_true, y_prob, num_classes):
    """计算 AUPRC (Average Precision)。二分类用正类概率，多分类用 macro。"""
    try:
        if num_classes == 2:
            return average_precision_score(y_true, y_prob[:, 1])
        else:
            y_onehot = np.eye(num_classes)[y_true]
            return average_precision_score(y_onehot, y_prob, average="macro")
    except (ValueError, IndexError):
        return float("nan")


# ============================================================================
# Bootstrap CI95
# ============================================================================

def bootstrap_ci(y_true, y_pred, y_prob, num_classes,
                 n_boot=2000, seed=0, ci=0.95):
    """Bootstrap 95% 置信区间。

    点估计取 Bootstrap 采样均值，CI 取百分位区间。
    多分类 bootstrap 时若采样后类别不足 2 类则跳过该次迭代。

    Args:
        y_true: (N,) int
        y_pred: (N,) int
        y_prob: (N, C) float
        num_classes: 类别数
        n_boot: Bootstrap 迭代次数
        seed: 随机种子
        ci: 置信水平

    Returns:
        results: dict {metric_name: (mean, ci_lower, ci_upper)}
        valid_iters: 有效迭代次数
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    alpha = 1.0 - ci

    boot_values = {name: [] for name in METRIC_ORDER}
    valid_iters = 0

    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bt_true = y_true[idx]
        bt_pred = y_pred[idx]
        bt_prob = y_prob[idx]

        # 多分类 bootstrap 时可能缺少某些类别，跳过
        if len(np.unique(bt_true)) < 2:
            continue

        bt_metrics = compute_point_metrics(bt_true, bt_pred, bt_prob, num_classes)
        for name in METRIC_ORDER:
            val = bt_metrics[name]
            if not np.isnan(val):
                boot_values[name].append(val)
        valid_iters += 1

    if valid_iters == 0:
        valid_iters = 1

    results = {}
    for name in METRIC_ORDER:
        vals = np.array(boot_values[name])
        if len(vals) == 0:
            mean, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")
        else:
            mean = float(vals.mean())
            ci_lo = float(np.percentile(vals, 100 * alpha / 2))
            ci_hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
        results[name] = (mean, ci_lo, ci_hi)

    return results, valid_iters


# ============================================================================
# 便捷接口
# ============================================================================

def compute_all_metrics(y_true, y_pred, y_prob, num_classes, n_boot=2000):
    """计算所有指标及其 95% CI。

    点估计取 Bootstrap 采样均值。

    Args:
        y_true: (N,) int 真实标签
        y_pred: (N,) int 预测标签
        y_prob: (N, C) float 概率矩阵
        num_classes: 类别数
        n_boot: Bootstrap 迭代次数

    Returns:
        dict {metric_name: {"value": float, "ci_lower": float, "ci_upper": float}}
    """
    results, _ = bootstrap_ci(
        y_true, y_pred, y_prob, num_classes, n_boot=n_boot
    )

    out = {}
    for name in METRIC_ORDER:
        mean, ci_lo, ci_hi = results[name]
        out[name] = {
            "value": mean,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        }
    return out


# ============================================================================
# 兼容接口: dinov3_unet_multitask 的二分类/多分类分离调用
# ============================================================================

def binary_bootstrap_metrics(y_prob, y_true, threshold=0.5,
                             n_boot=2000, ci=0.95, seed=0):
    """二分类 bootstrap 指标 + CI95。

    兼容 dinov3_unet_multitask 的调用方式。

    Args:
        y_prob: (N,) 正类概率
        y_true: (N,) int 真实标签 (0/1)
        threshold: 二值化阈值

    Returns:
        dict {metric_name: (mean, (lower, upper))}
    """
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)

    valid = y_true != -1
    y_prob = y_prob[valid]
    y_true = y_true[valid]

    if y_true.size == 0:
        zero = (0.0, (0.0, 0.0))
        return {k: zero for k in METRIC_ORDER}

    # 构造 (N, 2) 概率矩阵
    y_prob_2d = np.column_stack([1.0 - y_prob, y_prob])
    y_pred = (y_prob >= threshold).astype(np.int32)

    results, _ = bootstrap_ci(
        y_true, y_pred, y_prob_2d, num_classes=2,
        n_boot=n_boot, seed=seed, ci=ci,
    )

    # 转换为 (mean, (lower, upper)) 格式
    return {k: (v[0], (v[1], v[2])) for k, v in results.items()}


def multiclass_bootstrap_metrics(y_probs, y_true, num_classes,
                                 n_boot=2000, ci=0.95, seed=0):
    """多分类 bootstrap 指标 + CI95 (macro-average)。

    兼容 dinov3_unet_multitask 的调用方式。

    Args:
        y_probs: (N, C) 概率矩阵
        y_true: (N,) int 真实标签 (0-based)
        num_classes: 类别数

    Returns:
        dict {metric_name: (mean, (lower, upper))}
    """
    y_probs = np.asarray(y_probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)

    valid = y_true != -1
    y_probs = y_probs[valid]
    y_true = y_true[valid]

    if y_true.size == 0:
        zero = (0.0, (0.0, 0.0))
        return {k: zero for k in METRIC_ORDER}

    y_pred = y_probs.argmax(axis=1)

    results, _ = bootstrap_ci(
        y_true, y_pred, y_probs, num_classes=num_classes,
        n_boot=n_boot, seed=seed, ci=ci,
    )

    # 转换为 (mean, (lower, upper)) 格式
    return {k: (v[0], (v[1], v[2])) for k, v in results.items()}


# ============================================================================
# 报告格式化
# ============================================================================

def format_metrics_report(metrics, is_binary=None, class_names=None,
                          labels=None, preds=None, n_bootstrap=2000,
                          label_field=""):
    """将指标格式化为统一格式的报告字符串。

    Args:
        metrics: compute_all_metrics 的返回值，或
                 {metric_name: (mean, ci_lo, ci_hi)} 格式

    Returns:
        str
    """
    lines = []
    lines.append("=" * 60)
    n_samples = len(labels) if labels is not None else 0
    lines.append(f"评估样本数: {n_samples}")

    for name in METRIC_ORDER:
        if name not in metrics:
            continue

        # 兼容两种格式:
        #   {"value": float, "ci_lower": float, "ci_upper": float}  (compute_all_metrics)
        #   (mean, ci_lo, ci_hi)                                    (bootstrap_ci)
        m = metrics[name]
        if isinstance(m, dict):
            val, ci_lo, ci_hi = m["value"], m["ci_lower"], m["ci_upper"]
        else:
            val, ci_lo, ci_hi = m

        if np.isnan(val):
            lines.append(f"{name:<12s}: N/A")
        else:
            ci_str = ""
            if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
                ci_str = f"  (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])"
            lines.append(f"{name:<12s}: {val:.4f}{ci_str}")

    lines.append("=" * 60)
    return "\n".join(lines)
