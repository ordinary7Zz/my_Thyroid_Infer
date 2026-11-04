"""B4 Path B 附录表：B-a / B-b / B-c（指标来自 config 指定的 B3/B4 跑次目录）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from pipeline.revision.b4.b4_summary import load_b3_metrics, _metric_block


def _pathb_row(
    row_id: str,
    label: str,
    source: str,
    m: Optional[dict],
) -> dict[str, Any]:
    pb = _metric_block(m, task="cls_path_b") if m else {"available": False}
    cls = _metric_block(m, task="cls") if m else {"available": False}
    return {
        "pathb_row": row_id,
        "label": label,
        "source": source,
        "path_b_n": pb.get("n"),
        "path_b_accuracy": pb.get("accuracy"),
        "path_b_auroc": pb.get("auroc"),
        "path_b_auprc": pb.get("auprc"),
        "cls_full_auroc": cls.get("auroc"),
        "metrics_available": pb.get("available") or cls.get("available"),
    }


def _delta_rows(rows: list[dict], hi: str, lo: str) -> dict[str, Any]:
    by_id = {r["pathb_row"]: r for r in rows}
    a, b = by_id.get(hi), by_id.get(lo)
    if not a or not b:
        return {"pathb_row": f"Δ({hi}-{lo})", "metrics_available": False}
    acc_a, acc_b = a.get("path_b_accuracy"), b.get("path_b_accuracy")
    return {
        "pathb_row": f"Δ({hi}-{lo})",
        "label": f"{hi} minus {lo}",
        "source": "computed",
        "path_b_accuracy_delta": (
            float(acc_a) - float(acc_b) if acc_a is not None and acc_b is not None else None
        ),
        "note": "点估计；CI 待 B2",
    }


def build_pathb_table(cfg: dict) -> tuple[list[dict], dict]:
    b4 = cfg.get("pipeline", {}).get("ablation_b4") or cfg.get("ablation_b4") or {}
    pathb = b4.get("pathb") or {}
    b3_root = Path(b4["b3_root"])
    dataset_id = b4.get("dataset_id", "unknown")

    b_a_mode = pathb.get("b_a_b3_mode", "soft_voting")
    b_b_mode = pathb.get("b_b_b3_mode", "path_b_autogluon")
    b_b_dir = pathb.get("b_b_metrics_dir")
    b_c_mode = pathb.get("b_c_b3_mode", "llm")

    report: dict[str, Any] = {"dataset_id": dataset_id, "warnings": []}
    rows: list[dict] = []

    def _resolve_dir(mode: str, override: Optional[str]) -> Path:
        if override:
            return Path(override)
        return b3_root / mode

    specs = [
        ("B-a", "仅独立模型软投票（无 AutoGluon）", _resolve_dir(b_a_mode, None)),
        ("B-b", "独立模型 + AutoGluon 加权软投票（无 LLM）", _resolve_dir(b_b_mode, b_b_dir)),
        ("B-c", "全信号 LLM（与 L3 Path B cls 一致）", _resolve_dir(b_c_mode, None)),
    ]
    for rid, label, dir_path in specs:
        m = load_b3_metrics(dir_path)
        if m is None:
            report["warnings"].append(f"{rid} 缺失: {dir_path}/b3_metrics.json")
        rows.append(_pathb_row(rid, label, str(dir_path), m))

    if all(rows[i].get("metrics_available") for i in range(2)):
        rows.append(_delta_rows(rows, "B-b", "B-a"))
    if all(r.get("metrics_available") for r in rows[:3]):
        rows.append(_delta_rows(rows, "B-c", "B-b"))

    for r in rows:
        r["dataset_id"] = dataset_id
    return rows, report


def write_pathb_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
