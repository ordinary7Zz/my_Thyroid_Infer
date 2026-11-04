"""B4 组件主表：从 B3 mode 目录 + L1 指标拼 L1–L3、S4 校验行（路径均由 config 指定）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from pipeline.revision.b3.intermediate_manifest import manifests_match


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_b3_metrics(mode_dir: Path) -> Optional[dict]:
    p = mode_dir / "b3_metrics.json"
    if not p.is_file():
        return None
    doc = _load_json(p)
    doc["_source_dir"] = str(mode_dir)
    return doc


def _metric_block(
    m: Optional[dict],
    *,
    task: str,
) -> dict[str, Any]:
    """task: seg | cls | cls_path_b"""
    if not m:
        return {"available": False}
    if task == "seg":
        seg = m.get("segmentation") or {}
        return {
            "available": True,
            "dice_mean": seg.get("dice_mean"),
            "dice_ci_lower": seg.get("dice_ci_lower"),
            "dice_ci_upper": seg.get("dice_ci_upper"),
            "hd95_mean": seg.get("hd95_mean"),
        }
    if task == "cls":
        cls = m.get("classification") or {}
        return {
            "available": True,
            "auroc": cls.get("auroc"),
            "auroc_ci_lower": cls.get("auroc_ci_lower"),
            "auroc_ci_upper": cls.get("auroc_ci_upper"),
            "auprc": cls.get("auprc"),
            "accuracy": cls.get("accuracy"),
            "n": cls.get("n") or m.get("n_total"),
        }
    if task == "cls_path_b":
        pb = m.get("metrics_path_b_only") or m.get("classification_path_b") or {}
        return {
            "available": bool(pb),
            "auroc": pb.get("auroc"),
            "auprc": pb.get("auprc"),
            "accuracy": pb.get("accuracy"),
            "n": pb.get("n"),
        }
    raise ValueError(task)


def _row_from_layer(
    layer_id: str,
    label: str,
    source: str,
    seg: dict,
    cls_full: dict,
    cls_pb: dict,
    *,
    b3_mode: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "layer": layer_id,
        "label": label,
        "source": source,
        "b3_routing_mode": b3_mode or "",
        "seg_dice": seg.get("dice_mean") if seg.get("available") else None,
        "seg_dice_ci_lower": seg.get("dice_ci_lower"),
        "seg_dice_ci_upper": seg.get("dice_ci_upper"),
        "seg_hd95": seg.get("hd95_mean"),
        "cls_auroc": cls_full.get("auroc") if cls_full.get("available") else None,
        "cls_auroc_ci_lower": cls_full.get("auroc_ci_lower"),
        "cls_auroc_ci_upper": cls_full.get("auroc_ci_upper"),
        "cls_auprc": cls_full.get("auprc"),
        "cls_path_b_n": cls_pb.get("n"),
        "cls_path_b_accuracy": cls_pb.get("accuracy"),
        "cls_path_b_auroc": cls_pb.get("auroc"),
        "metrics_available": seg.get("available") or cls_full.get("available"),
    }


def _delta_row(
    layer_hi: str,
    layer_lo: str,
    row_hi: dict,
    row_lo: dict,
) -> dict[str, Any]:
    def d(key: str) -> Optional[float]:
        a, b = row_hi.get(key), row_lo.get(key)
        if a is None or b is None:
            return None
        return float(a) - float(b)

    return {
        "layer": f"Δ({layer_hi}-{layer_lo})",
        "label": f"{layer_hi} minus {layer_lo}",
        "source": "computed",
        "seg_dice_delta": d("seg_dice"),
        "cls_auroc_delta": d("cls_auroc"),
        "note": "点估计 Δ；CI 差分待 B2 cluster bootstrap",
    }


def validate_b3_manifests(
    b3_root: Path,
    mode_names: list[str],
    reference_manifest: Path,
) -> list[dict]:
    if not reference_manifest.is_file():
        return [{"ok": False, "error": f"缺少 reference manifest: {reference_manifest}"}]
    ref_doc = _load_json(reference_manifest)
    reports = []
    for name in mode_names:
        mpath = b3_root / name / "intermediate_manifest.json"
        if not mpath.is_file():
            reports.append({"mode": name, "ok": False, "error": "missing manifest"})
            continue
        doc = _load_json(mpath)
        reports.append({"mode": name, "ok": manifests_match(ref_doc, doc)})
    return reports


def build_component_table(cfg: dict) -> tuple[list[dict], dict]:
    """
    config 结构见 config/revision/b4/sample.yaml。
    返回 (主表行, 构建报告)。
    """
    b4 = cfg.get("pipeline", {}).get("ablation_b4") or cfg.get("ablation_b4") or {}
    dataset_id = b4.get("dataset_id", "unknown")
    b3_root = Path(b4["b3_root"])
    l2_mode = b4.get("l2_b3_mode", "soft_voting")
    l3_mode = b4.get("l3_b3_mode", "llm")
    s4_mode = b4.get("s4_replica_b3_mode", l3_mode)
    l1_path = Path(b4["l1_metrics"])

    report: dict[str, Any] = {"dataset_id": dataset_id, "warnings": []}

    l1_doc = _load_json(l1_path) if l1_path.is_file() else {}
    l2_m = load_b3_metrics(b3_root / l2_mode)
    l3_m = load_b3_metrics(b3_root / l3_mode)
    s4_m = load_b3_metrics(b3_root / s4_mode) if s4_mode != l3_mode else l3_m

    if l2_m is None:
        report["warnings"].append(f"L2 缺失: {b3_root / l2_mode}/b3_metrics.json")
    if l3_m is None:
        report["warnings"].append(
            f"L3 缺失: {b3_root / l3_mode}/b3_metrics.json（sample 可先跑 B3 llm 或仅占位 L1/L2）"
        )

    rows: list[dict] = []

    if l1_doc.get("segmentation"):
        l1_seg = _metric_block({"segmentation": l1_doc["segmentation"]}, task="seg")
    else:
        l1_seg = {"available": False}
    l1_cls = _metric_block(
        {"classification": l1_doc["classification"]} if l1_doc.get("classification") else None,
        task="cls",
    )
    pb_block = l1_doc.get("classification_path_b") or l1_doc.get("metrics_path_b_only")
    l1_pb = _metric_block(
        {"metrics_path_b_only": pb_block} if pb_block else None,
        task="cls_path_b",
    )

    rows.append(
        _row_from_layer(
            "L1",
            "单 DINO 专家（完整堆叠训练数据）",
            str(l1_path),
            l1_seg,
            l1_cls,
            l1_pb,
        )
    )

    if l2_m:
        rows.append(
            _row_from_layer(
                "L2",
                "专家池 + 固定平均（B3 策略 2）",
                l2_m["_source_dir"],
                _metric_block(l2_m, task="seg"),
                _metric_block(l2_m, task="cls"),
                _metric_block(l2_m, task="cls_path_b"),
                b3_mode=l2_mode,
            )
        )

    if l3_m:
        rows.append(
            _row_from_layer(
                "L3",
                "L2 + LLM 路由（B3 策略 7）",
                l3_m["_source_dir"],
                _metric_block(l3_m, task="seg"),
                _metric_block(l3_m, task="cls"),
                _metric_block(l3_m, task="cls_path_b"),
                b3_mode=l3_mode,
            )
        )

    if l2_m and l3_m:
        rows.append(_delta_row("L3", "L2", rows[-1], rows[-2]))

    if l1_doc and l2_m:
        rows.append(_delta_row("L2", "L1", rows[1], rows[0]))

    if s4_m and l3_m:
        s4_seg = _metric_block(s4_m, task="seg")
        s4_cls = _metric_block(s4_m, task="cls")
        s4_pb = _metric_block(s4_m, task="cls_path_b")
        rows.append(
            _row_from_layer(
                "S4_check",
                "投稿 S4 校验（应与 L3 同配置）",
                s4_m["_source_dir"],
                s4_seg,
                s4_cls,
                s4_pb,
                b3_mode=s4_mode,
            )
        )
        report["s4_l3_match"] = {
            "seg_dice": s4_seg.get("dice_mean") == _metric_block(l3_m, task="seg").get("dice_mean"),
            "cls_auroc": s4_cls.get("auroc") == _metric_block(l3_m, task="cls").get("auroc"),
            "same_source_dir": s4_m["_source_dir"] == l3_m["_source_dir"],
        }

    ref_manifest = b4.get("reference_manifest")
    if ref_manifest:
        modes = [m for m in (l2_mode, l3_mode, s4_mode) if m]
        report["manifest_validation"] = validate_b3_manifests(
            b3_root,
            list(dict.fromkeys(modes)),
            Path(ref_manifest),
        )

    for r in rows:
        r["dataset_id"] = dataset_id

    return rows, report


def write_component_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarize_b4(cfg: dict, output_dir: Path) -> dict:
    from pipeline.revision.b4.b4_pathb_summary import build_pathb_table, write_pathb_csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, report = build_component_table(cfg)
    csv_path = output_dir / "b4_component_table.csv"
    write_component_csv(rows, csv_path)

    pathb_rows, pathb_report = build_pathb_table(cfg)
    pathb_csv = output_dir / "b4_pathb_radiomics_table.csv"
    write_pathb_csv(pathb_rows, pathb_csv)
    report["pathb"] = pathb_report

    report_path = output_dir / "b4_build_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return {
        "csv": str(csv_path),
        "pathb_csv": str(pathb_csv),
        "report": str(report_path),
        "n_rows": len(rows),
        "n_pathb_rows": len(pathb_rows),
        "report_doc": report,
    }
