"""
分类性能指标计算模块
支持二分类和多分类，使用 Bootstrap 计算 95% 置信区间。

指标列表:
  - AUROC (二分类: 正类概率; 多分类: macro-average OvR)
  - AUPRC (二分类: 正类概率; 多分类: macro-average)
  - Accuracy
  - Precision (二分类: binary; 多分类: macro)
  - F1 (二分类: binary; 多分类: macro)
  - Recall (二分类: binary; 多分类: macro)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


METRIC_NAMES = ["auroc", "auprc", "accuracy", "precision", "f1", "recall"]


def compute_metric(name, labels, preds, probs, is_binary):
    """计算单个指标。

    Args:
        name: 指标名称
        labels: (N,) int 真实标签
        preds: (N,) int 预测标签
        probs: (N, C) float 概率矩阵（二分类时 probs[:,1] 为正类概率）
        is_binary: 是否二分类
    """
    if name == "auroc":
        if is_binary:
            return roc_auc_score(labels, probs[:, 1])
        else:
            return roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    elif name == "auprc":
        if is_binary:
            return average_precision_score(labels, probs[:, 1])
        else:
            y_onehot = np.eye(probs.shape[1])[labels]
            return average_precision_score(y_onehot, probs, average="macro")
    elif name == "accuracy":
        return accuracy_score(labels, preds)
    elif name == "precision":
        avg = "binary" if is_binary else "macro"
        return precision_score(labels, preds, average=avg, zero_division=0)
    elif name == "f1":
        avg = "binary" if is_binary else "macro"
        return f1_score(labels, preds, average=avg, zero_division=0)
    elif name == "recall":
        avg = "binary" if is_binary else "macro"
        return recall_score(labels, preds, average=avg, zero_division=0)
    else:
        raise ValueError(f"Unknown metric: {name}")


def bootstrap_ci(name, labels, preds, probs, is_binary,
                 n_bootstrap=2000, seed=0, confidence=0.95):
    """Bootstrap 计算 95% 置信区间。
    点估计取 Bootstrap 采样均值，CI 取百分位区间。
    与 dinov3_unet_multitask 的实现方式一致。

    Returns:
        (mean, lower, upper): 均值和 95% CI 下界、上界
    """
    n = len(labels)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        try:
            score = compute_metric(
                name, labels[idx], preds[idx], probs[idx], is_binary
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue

    if len(scores) == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(np.mean(scores))
    alpha = 1.0 - confidence
    lower = np.percentile(scores, alpha / 2 * 100)
    upper = np.percentile(scores, (1 - alpha / 2) * 100)
    return mean, float(lower), float(upper)


def compute_all_metrics(labels, preds, probs, is_binary, n_bootstrap=2000):
    """计算所有指标及其 95% CI。

    点估计取 Bootstrap 采样均值，与 dinov3_unet_multitask 一致。

    Args:
        labels: (N,) int 真实标签
        preds: (N,) int 预测标签
        probs: (N, C) float 概率矩阵
        is_binary: 是否二分类
        n_bootstrap: Bootstrap 迭代次数

    Returns:
        dict: {metric_name: {"value": float, "ci_lower": float, "ci_upper": float}}
    """
    results = {}
    for name in METRIC_NAMES:
        # Bootstrap CI + 点估计（取 Bootstrap 均值）
        mean, lower, upper = bootstrap_ci(
            name, labels, preds, probs, is_binary,
            n_bootstrap=n_bootstrap,
        )
        results[name] = {"value": mean, "ci_lower": lower, "ci_upper": upper}

    return results


def format_metrics_report(metrics, is_binary, class_names, labels, preds,
                           n_bootstrap, label_field=""):
    """将指标格式化为可读的报告字符串。

    Args:
        metrics: compute_all_metrics 的返回值
        is_binary: 是否二分类
        class_names: 类别名称列表
        labels: (N,) int 真实标签
        preds: (N,) int 预测标签
        n_bootstrap: Bootstrap 迭代次数
        label_field: 标签字段名（用于报告标题）
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Classification Performance Report")
    lines.append("=" * 70)
    lines.append(f"Task field:       {label_field if label_field else 'N/A'}")
    task_type = "Binary" if is_binary else f"Multi-class ({len(class_names)} classes)"
    lines.append(f"Task type:        {task_type}")
    lines.append(f"Classes:          {class_names}")
    lines.append(f"Samples:          {len(labels)}")
    lines.append(f"Bootstrap iters:  {n_bootstrap}")
    avg_method = "binary" if is_binary else "macro"
    lines.append(f"Averaging:        {avg_method}")
    lines.append("")

    lines.append("--- Metrics (point estimate + 95% CI) ---")
    for name in METRIC_NAMES:
        m = metrics[name]
        if np.isnan(m["value"]):
            lines.append(f"  {name:12s}: N/A")
        else:
            ci_str = ""
            if not (np.isnan(m["ci_lower"]) or np.isnan(m["ci_upper"])):
                ci_str = f" (95% CI: {m['ci_lower']:.4f} - {m['ci_upper']:.4f})"
            lines.append(f"  {name:12s}: {m['value']:.4f}{ci_str}")
    lines.append("")

    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    lines.append("--- Confusion Matrix ---")
    header = f"  {'True/Pred':>12s}" + "".join(f"{name:>12s}" for name in class_names)
    lines.append(header)
    for i, name in enumerate(class_names):
        row = f"  {name:>12s}" + "".join(f"{cm[i, j]:>12d}" for j in range(len(class_names)))
        lines.append(row)
    lines.append("")

    # 详细分类报告
    lines.append("--- Detailed Classification Report ---")
    try:
        report = classification_report(
            labels, preds, target_names=class_names, digits=4, zero_division=0
        )
    except Exception:
        report = classification_report(labels, preds, digits=4, zero_division=0)
    lines.append(report)
    lines.append("=" * 70)

    return "\n".join(lines)
