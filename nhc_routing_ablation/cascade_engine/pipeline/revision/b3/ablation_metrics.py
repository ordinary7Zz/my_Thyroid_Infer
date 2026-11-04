"""从 results + evaluate 输出构造 b3_metrics.json。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from pipeline.evaluate import _bootstrap_auc_ci, _bootstrap_auprc_ci, _label_to_binary


def _cls_metrics_from_results(results: list[dict], subset_path_b: bool = False) -> Optional[dict]:
    y_true_list = []
    y_score_list = []
    y_pred_list = []
    for r in results:
        if subset_path_b and r.get("path") != "B":
            continue
        if "true_label" not in r or "final_label" not in r:
            continue
        true_binary = _label_to_binary(r["true_label"])
        pred_label = r["final_label"]
        pred_binary = _label_to_binary(pred_label) if isinstance(pred_label, str) else pred_label
        confidence = r.get("final_confidence", 0.5)
        if true_binary is not None and pred_binary is not None:
            y_true_list.append(true_binary)
            y_pred_list.append(int(pred_binary))
            y_score_list.append(confidence if pred_binary == 1 else 1 - confidence)
    n = len(y_true_list)
    if n < 1:
        return None

    base = {"n": n}
    if y_pred_list:
        base["accuracy"] = float(np.mean(np.array(y_true_list) == np.array(y_pred_list)))

    if n < 2 or len(set(y_true_list)) < 2:
        return base

    y_true = np.array(y_true_list)
    y_score = np.array(y_score_list)
    auc_val, auc_lo, auc_hi = _bootstrap_auc_ci(y_true, y_score)
    auprc_val, auprc_lo, auprc_hi = _bootstrap_auprc_ci(y_true, y_score)
    base.update({
        "auroc": auc_val,
        "auroc_ci_lower": auc_lo,
        "auroc_ci_upper": auc_hi,
        "auprc": auprc_val,
        "auprc_ci_lower": auprc_lo,
        "auprc_ci_upper": auprc_hi,
    })
    return base


def build_b3_metrics(
    results: list[dict],
    evaluate_metrics: dict,
    *,
    routing_mode: str,
    task: str,
    seg_mode: str,
    cls_mode: str,
    metadata_mode: Optional[str] = None,
) -> dict[str, Any]:
    n_consensus = sum(1 for r in results if r.get("path") == "A")
    n_disagreement = sum(1 for r in results if r.get("path") == "B")
    seg_block = {}
    if "dice" in evaluate_metrics:
        seg_block["dice_mean"] = evaluate_metrics["dice"]["mean"]
        seg_block["dice_ci_lower"] = evaluate_metrics["dice"]["ci_lower"]
        seg_block["dice_ci_upper"] = evaluate_metrics["dice"]["ci_upper"]
    if "hd95" in evaluate_metrics:
        seg_block["hd95_mean"] = evaluate_metrics["hd95"]["mean"]
        seg_block["hd95_ci_lower"] = evaluate_metrics["hd95"]["ci_lower"]
        seg_block["hd95_ci_upper"] = evaluate_metrics["hd95"]["ci_upper"]

    cls_all = _cls_metrics_from_results(results, subset_path_b=False)
    cls_pb = _cls_metrics_from_results(results, subset_path_b=True)

    out: dict[str, Any] = {
        "routing_mode": routing_mode,
        "routing_mode_seg": seg_mode,
        "routing_mode_cls": cls_mode,
        "task": task,
        "seg_routing_mode": seg_mode,
        "cls_routing_mode": cls_mode,
        "n_total": len(results),
        "n_consensus": n_consensus,
        "n_disagreement": n_disagreement,
        "segmentation": seg_block,
        "classification": cls_all or {},
        "metrics_path_b_only": cls_pb or {},
        "classification_path_b": cls_pb or {},
    }
    if metadata_mode:
        out["metadata_mode"] = metadata_mode
    return out


def save_b3_metrics(output_dir: Path, payload: dict) -> Path:
    path = Path(output_dir) / "b3_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        import json
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
