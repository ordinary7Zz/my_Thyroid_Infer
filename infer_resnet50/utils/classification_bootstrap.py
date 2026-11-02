"""Binary classification bootstrap metrics (inference-only; no training/seg deps)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_bootstrap_metrics(
    y_probs, y_labels, threshold=0.5, n_boot=2000, ci=0.95, seed=0
):
    y_probs = np.asarray(y_probs, dtype=np.float32)
    y_labels = np.asarray(y_labels, dtype=np.int32)

    valid_mask = y_labels != -1
    y_probs = y_probs[valid_mask]
    y_labels = y_labels[valid_mask]

    if y_labels.size == 0:
        zero_ci = (0.0, (0.0, 0.0))
        return {
            k: zero_ci
            for k in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auroc",
                "auprc",
                "sensitivity",
                "specificity",
                "youden",
            ]
        }

    rng = np.random.default_rng(seed)
    n = y_labels.size
    metrics_samples = {
        k: []
        for k in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auroc",
            "auprc",
            "sensitivity",
            "specificity",
            "youden",
        ]
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        probs_s = y_probs[idx]
        labels_s = y_labels[idx]
        preds_s = (probs_s >= float(threshold)).astype(int)

        tp = np.sum((preds_s == 1) & (labels_s == 1))
        tn = np.sum((preds_s == 0) & (labels_s == 0))
        fp = np.sum((preds_s == 1) & (labels_s == 0))
        fn = np.sum((preds_s == 0) & (labels_s == 1))

        acc = float((preds_s == labels_s).mean())
        prec = precision_score(labels_s, preds_s, zero_division=0)
        rec = recall_score(labels_s, preds_s, zero_division=0)
        f1 = f1_score(labels_s, preds_s, zero_division=0)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = sens + spec - 1.0

        try:
            auroc = (
                roc_auc_score(labels_s, probs_s)
                if len(np.unique(labels_s)) > 1
                else float("nan")
            )
        except Exception:
            auroc = float("nan")
        try:
            auprc = (
                average_precision_score(labels_s, probs_s)
                if len(np.unique(labels_s)) > 1
                else float("nan")
            )
        except Exception:
            auprc = float("nan")

        metrics_samples["accuracy"].append(acc)
        metrics_samples["precision"].append(float(prec))
        metrics_samples["recall"].append(float(rec))
        metrics_samples["f1"].append(float(f1))
        metrics_samples["sensitivity"].append(float(sens))
        metrics_samples["specificity"].append(float(spec))
        metrics_samples["youden"].append(float(youden))
        metrics_samples["auroc"].append(float(np.nan if np.isnan(auroc) else auroc))
        metrics_samples["auprc"].append(float(np.nan if np.isnan(auprc) else auprc))

    results = {}
    alpha = 1.0 - ci
    for k, vals in metrics_samples.items():
        arr = np.asarray(vals, dtype=np.float32)
        arr_valid = arr[~np.isnan(arr)]
        if arr_valid.size == 0:
            results[k] = (0.0, (0.0, 0.0))
            continue
        mean = float(arr_valid.mean())
        lower = float(np.percentile(arr_valid, 100 * alpha / 2))
        upper = float(np.percentile(arr_valid, 100 * (1 - alpha / 2)))
        results[k] = (mean, (lower, upper))

    return results
