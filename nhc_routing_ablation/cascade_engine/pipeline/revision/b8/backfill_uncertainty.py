"""为已有 pipeline 输出补写 uncertainty / center_id（无需重跑 GPU）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from classification.base_model import ClsModelOutput
from pipeline.cascade import CascadePipeline
from shared.center_manifest import (
    build_center_lookup_from_labels,
    load_center_lookup,
    resolve_center_id,
)
from shared.routing_modes import cls_uses_llm_api, get_cls_routing_mode, get_seg_routing_mode, seg_uses_llm_api
from shared.uncertainty_signals import build_uncertainty_block, seg_disagreement_from_masks
def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cls_pred_from_dict(pred_dict: dict) -> ClsModelOutput:
    return ClsModelOutput(
        model_name=pred_dict["model_name"],
        predictions=pred_dict.get("predictions", {}),
        top_class=pred_dict["top_class"],
        top_confidence=float(pred_dict.get("top_confidence", 0.0)),
        requires_mask=bool(pred_dict.get("requires_mask", False)),
        metadata=pred_dict.get("metadata", {}),
    )


def _index_cls_caches(cls_cache_dir: Path) -> dict[str, list[ClsModelOutput]]:
    """image_name -> 各独立模型 ClsModelOutput（按文件顺序）。"""
    by_image: dict[str, list[ClsModelOutput]] = {}
    for json_path in sorted(cls_cache_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data["results"] if "results" in data else data
        for img_name, pred_dict in results.items():
            by_image.setdefault(img_name, []).append(_cls_pred_from_dict(pred_dict))
    return by_image


def _index_seg_caches(seg_cache_dir: Path) -> dict[str, list[np.ndarray]]:
    by_image: dict[str, list[np.ndarray]] = {}
    for npz_path in sorted(seg_cache_dir.glob("*.npz")):
        data = np.load(str(npz_path), allow_pickle=True)
        names = [str(x) for x in data["image_names"]]
        masks = data["masks"]
        for i, name in enumerate(names):
            by_image.setdefault(name, []).append(np.asarray(masks[i]))
    return by_image


def _seg_agreement_from_decisions(
    image_name: str,
    decisions_by_name: dict[str, dict],
) -> Optional[dict]:
    d = decisions_by_name.get(image_name)
    if not d:
        return None
    if "seg_agreement" in d:
        return d["seg_agreement"]
    return None


def backfill_results_metadata(
    output_dir: str | Path,
    config_path: str | Path,
    *,
    center_manifest: Optional[str] = None,
    label_file: Optional[str] = None,
    in_place: bool = True,
    out_path: Optional[str | Path] = None,
) -> list[dict]:
    """
    读取 output_dir/results.json 与 intermediate，补 uncertainty、center_id。

    Returns:
        更新后的 results 列表。
    """
    out = Path(output_dir)
    config = _load_config(Path(config_path))
    seg_cfg = config.get("segmentation", {}).get("agent", {}) or {}
    cls_cfg = config.get("classification", {}).get("agent", {}) or {}
    seg_mode = get_seg_routing_mode(config)
    cls_mode = get_cls_routing_mode(config)
    llm_seg = bool(seg_cfg.get("enable_agent", True) and seg_uses_llm_api(seg_mode))
    llm_cls = bool(cls_cfg.get("enable_agent", True) and cls_uses_llm_api(cls_mode))

    pipe = config.get("pipeline", {}) or {}
    rev = pipe.get("revision") or {}
    data_cfg = pipe.get("data", {}) or {}
    manifest = center_manifest or rev.get("center_manifest") or data_cfg.get("center_manifest")
    labels_path = label_file or data_cfg.get("label_file")

    lookup = load_center_lookup(manifest)
    if not lookup and labels_path:
        lookup = build_center_lookup_from_labels(labels_path)

    results_path = out / "results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        results: list[dict] = json.load(f)

    inter = out / "intermediate"
    cls_by_img = _index_cls_caches(inter / "cls") if (inter / "cls").is_dir() else {}
    seg_by_img = _index_seg_caches(inter / "seg") if (inter / "seg").is_dir() else {}

    decisions_by_name: dict[str, dict] = {}
    sel_npz = inter / "selected_masks.npz"
    if sel_npz.is_file():
        _, _, decisions = CascadePipeline._load_selected_masks_npz(sel_npz)
        for d in decisions:
            decisions_by_name[d.get("img_name", "")] = d

    updated: list[dict] = []
    for r in results:
        row = dict(r)
        img_name = row.get("image_name", "")
        if not img_name:
            updated.append(row)
            continue

        indie = cls_by_img.get(img_name, [])
        seg_dec = row.get("seg_decision") or decisions_by_name.get(img_name)
        seg_ag = None
        if isinstance(seg_dec, dict):
            seg_ag = _seg_agreement_from_decisions(img_name, decisions_by_name) or seg_dec.get(
                "seg_agreement"
            )
        if seg_ag is None and img_name in seg_by_img and len(seg_by_img[img_name]) >= 2:
            seg_ag = seg_disagreement_from_masks(seg_by_img[img_name])

        consensus = bool(row.get("classification_consensus", False))
        conf = float(row.get("final_confidence", seg_dec.get("confidence") if seg_dec else 0.0))

        cid = resolve_center_id(img_name, lookup)
        if cid:
            row["center_id"] = cid

        row["uncertainty"] = build_uncertainty_block(
            indie,
            seg_ag,
            final_confidence=conf,
            seg_confidence=float(seg_dec.get("confidence")) if isinstance(seg_dec, dict) and seg_dec.get("confidence") is not None else None,
            llm_used_seg=llm_seg,
            llm_used_cls=llm_cls,
            classification_consensus=consensus,
        )
        updated.append(row)

    dest = Path(out_path) if out_path else results_path
    if not in_place and out_path is None:
        dest = out / "results_with_uncertainty.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
    return updated
