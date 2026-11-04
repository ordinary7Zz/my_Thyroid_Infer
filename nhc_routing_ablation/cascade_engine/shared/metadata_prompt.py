"""
B5 路由元数据消融：按 metadata_mode 过滤 LLM 可见字段。
"""

from __future__ import annotations

import copy
import json
from typing import Any, Optional

METADATA_MODES = frozenset({
    "full",
    "drop_device",
    "drop_source",
    "drop_both",
    "quality_only",
})

DEFAULT_METADATA_MODE = "full"


def normalize_metadata_mode(mode: Optional[str]) -> str:
    if not mode:
        return DEFAULT_METADATA_MODE
    m = str(mode).strip().lower()
    if m not in METADATA_MODES:
        raise ValueError(f"metadata_mode 须为 {sorted(METADATA_MODES)}，得到 {m!r}")
    return m


def _drops_device(mode: str) -> bool:
    return mode in ("drop_device", "drop_both", "quality_only")


def _drops_source(mode: str) -> bool:
    return mode in ("drop_source", "drop_both", "quality_only")


def apply_metadata_mode_to_config(config: dict, metadata_mode: Optional[str]) -> dict:
    """深拷贝 config，写入 seg/cls agent.metadata_mode。"""
    import copy as _copy

    mode = normalize_metadata_mode(metadata_mode)
    cfg = _copy.deepcopy(config)
    cfg.setdefault("segmentation", {}).setdefault("agent", {})["metadata_mode"] = mode
    cfg.setdefault("classification", {}).setdefault("agent", {})["metadata_mode"] = mode
    return cfg


def seg_user_prefix_lines(
    mode: str,
    input_device_info: Optional[list[str]],
    input_data_info: Optional[dict],
) -> tuple[str, str]:
    """返回 (device_text, data_text) 两行前缀（可为空字符串）。"""
    mode = normalize_metadata_mode(mode)
    device_text = ""
    if not _drops_device(mode):
        if input_device_info:
            device_text = f"\n**输入设备**: {', '.join(input_device_info)}\n"
        else:
            device_text = "\n**输入设备**: 未知\n"
    data_text = ""
    if not _drops_source(mode):
        payload = input_data_info if input_data_info else {}
        data_text = f"\n**输入数据**: {json.dumps(payload, ensure_ascii=False)}\n"
    return device_text, data_text


def cls_device_line(mode: str, input_device_info: Optional[list[str]]) -> str:
    mode = normalize_metadata_mode(mode)
    if _drops_device(mode):
        return ""
    if input_device_info:
        return f"输入设备: {', '.join(input_device_info)}"
    return "输入设备: 未知"


def filter_seg_routing_payload(data: dict[str, Any], mode: str) -> dict[str, Any]:
    """过滤 format_predictions_for_agent 构造的 dict（return 前调用）。"""
    mode = normalize_metadata_mode(mode)
    if mode == "full":
        return data

    out = copy.deepcopy(data)
    if _drops_device(mode):
        out.pop("input_device_info", None)
    if _drops_source(mode):
        out.pop("input_data_info", None)

    models = []
    for m in out.get("models") or []:
        mc = copy.deepcopy(m)
        if _drops_device(mode):
            mc.pop("training_devices", None)
        if _drops_source(mode):
            mc.pop("dataset_size", None)
        if mode == "quality_only":
            mc = _seg_model_quality_only(mc)
        models.append(mc)
    out["models"] = models

    if mode == "quality_only":
        anchor = out.get("classification_anchor")
        if isinstance(anchor, dict):
            out["classification_anchor"] = anchor.get("class")
        out.pop("reasoning_requirements", None)

    return out


def _seg_model_quality_only(model_info: dict[str, Any]) -> dict[str, Any]:
    keep_keys = {
        "model_name",
        "mask_statistics",
        "agreement_with_others",
        "disagreement",
        "mean_confidence",
        "radiomics_judge",
    }
    slim = {k: v for k, v in model_info.items() if k in keep_keys}
    rj = slim.get("radiomics_judge")
    if isinstance(rj, dict):
        slim["radiomics_judge"] = {
            k: rj[k]
            for k in ("malignant_prob", "confidence")
            if k in rj
        }
    return slim


def filter_cls_prediction_dicts(
    compact: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    mode = normalize_metadata_mode(mode)
    if mode == "full":
        return compact

    out: list[dict[str, Any]] = []
    for row in compact:
        r = copy.deepcopy(row)
        meta = dict(r.get("metadata") or {})
        if _drops_device(mode):
            meta.pop("training_devices", None)
        if _drops_source(mode):
            meta.pop("dataset_size", None)
        if mode == "quality_only":
            meta = {
                k: meta[k]
                for k in (
                    "top_confidence_calibrated",
                    "entropy",
                    "margin_top2",
                    "num_models_same_class",
                )
                if k in meta
            }
        r["metadata"] = meta
        out.append(r)
    return out


def filter_cls_routing_envelope(
    envelope: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """过滤顶层 vote_entropy + predictions 列表。"""
    mode = normalize_metadata_mode(mode)
    if mode == "full":
        return envelope
    out = copy.deepcopy(envelope)
    preds = out.get("predictions") or []
    out["predictions"] = filter_cls_prediction_dicts(preds, mode)
    return out


def merge_per_image_input_data_info(
    base: Optional[dict],
    label_row: Optional[dict],
    *,
    label_keys: Optional[list[str]] = None,
    label_key: str = "malignancy",
) -> Optional[dict]:
    """
    合并静态 pipeline.data.input_data_info 与标签行中的元数据字段。
    label_keys 为 None 时：复制 label_row 中除 filename 与 label_key 外的所有字段。
    """
    merged: dict[str, Any] = dict(base or {})
    if not label_row:
        return merged if merged else None
    skip = {"filename", label_key}
    if label_keys:
        for k in label_keys:
            if k in label_row and label_row[k] is not None:
                merged[k] = label_row[k]
    else:
        for k, v in label_row.items():
            if k in skip or v is None:
                continue
            merged[k] = v
    return merged if merged else None
