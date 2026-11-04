"""B4 L1 单专家指标 JSON（供 b4_summary.build_component_table 读取）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


L1_SCHEMA_VERSION = 1

REQUIRED_TOP_LEVEL = ("dataset_id",)


def segmentation_from_evaluate_metrics(metrics: dict) -> Optional[dict[str, Any]]:
    if not metrics.get("dice") and not metrics.get("hd95"):
        return None
    dice = metrics.get("dice") or {}
    hd95 = metrics.get("hd95") or {}
    return {
        "dice_mean": dice.get("mean"),
        "dice_ci_lower": dice.get("ci_lower"),
        "dice_ci_upper": dice.get("ci_upper"),
        "hd95_mean": hd95.get("mean"),
        "hd95_ci_lower": hd95.get("ci_lower"),
        "hd95_ci_upper": hd95.get("ci_upper"),
    }


def classification_from_evaluate_metrics(metrics: dict, n: Optional[int] = None) -> Optional[dict[str, Any]]:
    if "auroc" not in metrics and metrics.get("accuracy") is None:
        return None
    auroc = metrics.get("auroc") or {}
    auprc = metrics.get("auprc") or {}
    block: dict[str, Any] = {"n": n}
    if auroc:
        block.update(
            {
                "auroc": auroc.get("mean"),
                "auroc_ci_lower": auroc.get("ci_lower"),
                "auroc_ci_upper": auroc.get("ci_upper"),
            }
        )
    if auprc:
        block.update(
            {
                "auprc": auprc.get("mean"),
                "auprc_ci_lower": auprc.get("ci_lower"),
                "auprc_ci_upper": auprc.get("ci_upper"),
            }
        )
    if metrics.get("accuracy") is not None:
        block["accuracy"] = metrics["accuracy"]
    return block


def build_l1_document(
    *,
    dataset_id: str,
    segmentation: Optional[dict[str, Any]] = None,
    classification: Optional[dict[str, Any]] = None,
    classification_path_b: Optional[dict[str, Any]] = None,
    provenance: Optional[dict[str, Any]] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """组装 L1 JSON。单专家无多模型分歧，默认不写 classification_path_b。"""
    doc: dict[str, Any] = {
        "schema_version": L1_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "layer": "L1",
        "description": "单 DINO 专家（完整堆叠 development 训练）；无专家池、无 LLM 路由",
    }
    if note:
        doc["note"] = note
    if provenance:
        doc["provenance"] = provenance
    if segmentation:
        doc["segmentation"] = segmentation
    if classification:
        doc["classification"] = classification
    if classification_path_b:
        doc["classification_path_b"] = classification_path_b
    return doc


def build_l1_from_evaluate(
    *,
    dataset_id: str,
    evaluate_metrics: dict,
    results: Optional[list[dict]] = None,
    provenance: Optional[dict[str, Any]] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    n_cls = None
    if results:
        n_cls = sum(1 for r in results if "true_label" in r and "final_label" in r)
    seg = segmentation_from_evaluate_metrics(evaluate_metrics)
    cls = classification_from_evaluate_metrics(evaluate_metrics, n=n_cls)
    return build_l1_document(
        dataset_id=dataset_id,
        segmentation=seg,
        classification=cls,
        provenance=provenance,
        note=note,
    )


def validate_l1_document(doc: dict) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in doc:
            errors.append(f"缺少字段: {key}")
    if doc.get("segmentation") is None and doc.get("classification") is None:
        errors.append("segmentation 与 classification 至少应有一项非空")
    return errors


def save_l1_document(doc: dict, path: Path) -> Path:
    errs = validate_l1_document(doc)
    if errs:
        raise ValueError("; ".join(errs))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path


def load_l1_document(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
