"""
B6 路由审计：结构化 routing_log.jsonl 与分层/摘要工具。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# 分割：top-1 与 top-2 average_agreement 差值 ≤ 阈值 → close
SEG_DISAGREEMENT_DELTA_THRESHOLD = 0.10


class RoutingLogger:
    """Append-only JSONL logger for routing decisions (R2.9 / B6)."""

    def __init__(
        self,
        path: Path | str,
        enabled: bool = True,
        include_prompts: bool = False,
    ):
        self.path = Path(path)
        self.enabled = enabled
        self.include_prompts = include_prompts
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        row = dict(record)
        row.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def llm_config_dict(llm_client: Any) -> dict[str, Any]:
    if llm_client is None:
        return {}
    return {
        "llm_model": getattr(llm_client, "model_name", None),
        "temperature": getattr(llm_client, "temperature", None),
        "max_tokens": getattr(llm_client, "max_tokens", None),
    }


def seg_disagreement_tier(quality_results: Optional[dict]) -> str:
    """easy: clear winner by inter-model agreement; close: top experts similarly ranked."""
    if not quality_results:
        return "unknown"
    agreement = quality_results.get("agreement_metrics", {}).get("average_agreement")
    if not agreement or len(agreement) < 2:
        return "easy"
    sorted_vals = sorted(float(x) for x in agreement if x is not None)
    if len(sorted_vals) < 2:
        return "easy"
    delta = sorted_vals[-1] - sorted_vals[-2]
    return "close" if delta <= SEG_DISAGREEMENT_DELTA_THRESHOLD else "easy"


def cls_disagreement_tier(predictions: list) -> str:
    """easy: unanimous or large margin; close: disagreeing classes or tight confidences."""
    if not predictions or len(predictions) < 2:
        return "easy"
    classes = {getattr(p, "top_class", None) for p in predictions}
    if len(classes) == 1:
        return "easy"
    confs = sorted(float(p.top_confidence) for p in predictions)
    margin = confs[-1] - confs[-2] if len(confs) >= 2 else 1.0
    return "close" if margin < 0.15 else "easy"


def summarize_seg_candidates(
    predictions: list,
    quality_results: Optional[dict],
) -> list[dict[str, Any]]:
    agreement = (
        (quality_results or {}).get("agreement_metrics", {}).get("average_agreement") or []
    )
    individual = (quality_results or {}).get("individual_quality") or []
    out = []
    for i, p in enumerate(predictions):
        d = p.to_dict() if hasattr(p, "to_dict") else {}
        row: dict[str, Any] = {
            "model_name": d.get("model_name", getattr(p, "model_name", "")),
            "mask_area": d.get("mask_area", int(getattr(p, "mask", [[0]]).sum()) if hasattr(p, "mask") else 0),
        }
        if d.get("has_confidence_map"):
            row["mean_confidence"] = d.get("mean_confidence")
        if i < len(agreement):
            row["average_agreement"] = float(agreement[i])
        if i < len(individual) and isinstance(individual[i], dict):
            row["circularity"] = individual[i].get("circularity")
            row["num_components"] = individual[i].get("num_components")
        out.append(row)
    return out


def summarize_judge_results(judge_results: Optional[list]) -> Optional[list[dict]]:
    if not judge_results:
        return None
    rows = []
    for j in judge_results:
        if not isinstance(j, dict):
            continue
        rows.append({
            "valid": j.get("valid"),
            "malignant_prob": j.get("malignant_prob"),
            "confidence": j.get("confidence"),
        })
    return rows or None


def summarize_cls_candidates(predictions: list) -> list[dict[str, Any]]:
    out = []
    for p in predictions:
        d = p.to_dict() if hasattr(p, "to_dict") else {}
        meta = d.get("metadata") or getattr(p, "metadata", {}) or {}
        out.append({
            "model_name": d.get("model_name", getattr(p, "model_name", "")),
            "top_class": d.get("top_class"),
            "top_confidence": d.get("top_confidence"),
            "entropy": meta.get("entropy"),
            "margin_top2": meta.get("margin_top2"),
            "num_models_same_class": meta.get("num_models_same_class"),
        })
    return out


def build_routing_logger_from_config(config: dict, output_dir: Path | str) -> RoutingLogger:
    """从 pipeline 配置创建 RoutingLogger。"""
    pipe = config.get("pipeline", {})
    enabled = bool(pipe.get("log_routing", False))
    include_prompts = bool(pipe.get("routing_log_include_prompts", False))
    custom = pipe.get("routing_log_path")
    if custom:
        path = Path(custom)
    else:
        path = Path(output_dir) / "routing_log.jsonl"
    return RoutingLogger(path, enabled=enabled, include_prompts=include_prompts)
