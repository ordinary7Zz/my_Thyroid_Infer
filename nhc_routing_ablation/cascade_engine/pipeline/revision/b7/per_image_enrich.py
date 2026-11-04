"""B7：为逐图 jsonl 补 center_id，并构建 meta 回归用图像行。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from shared.center_manifest import load_center_lookup, resolve_center_id


def _stem_map_from_results(results: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in results:
        stem = Path(r.get("image_name", "")).stem
        cid = r.get("center_id")
        if stem and cid:
            out[stem] = str(cid)
    return out


def attach_center_id(
    rows: list[dict],
    *,
    results: Optional[list[dict]] = None,
    manifest_lookup: Optional[dict[str, str]] = None,
) -> list[dict]:
    stem_to_center = _stem_map_from_results(results or [])
    lookup = manifest_lookup or {}
    enriched: list[dict] = []
    for row in rows:
        r = dict(row)
        stem = r.get("stem") or Path(r.get("image_name", "")).stem
        cid = (
            r.get("center_id")
            or stem_to_center.get(stem)
            or resolve_center_id(r.get("image_name", stem), lookup)
        )
        if cid:
            r["center_id"] = cid
        enriched.append(r)
    return enriched


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def enrich_output_dir_jsonl(
    output_dir: Path,
    *,
    results: list[dict],
    manifest_lookup: Optional[dict[str, str]] = None,
) -> dict[str, int]:
    """为 per_image_* jsonl 补写 center_id（原地覆盖）。返回各文件行数。"""
    counts: dict[str, int] = {}
    for name in ("per_image_seg_metrics.jsonl", "per_image_cls_metrics.jsonl", "per_image_gland_metrics.jsonl"):
        p = output_dir / name
        if not p.is_file():
            continue
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows = attach_center_id(rows, results=results, manifest_lookup=manifest_lookup)
        write_jsonl(p, rows)
        counts[name] = len(rows)
    return counts


def build_meta_regression_rows(
    per_image_seg: list[dict],
    *,
    center_malignant_ratio: Optional[dict[str, float]] = None,
    center_scanner_code: Optional[dict[str, float]] = None,
) -> list[dict]:
    """图像级行：dice + 可选协变量（中心恶性比例、扫描仪数值编码等）。"""
    out: list[dict] = []
    for row in per_image_seg:
        cid = row.get("center_id")
        if row.get("dice") is None or not cid:
            continue
        item: dict[str, Any] = {
            "stem": row.get("stem"),
            "center_id": cid,
            "dice": float(row["dice"]),
        }
        if center_malignant_ratio and cid in center_malignant_ratio:
            item["malignant_ratio_center"] = center_malignant_ratio[cid]
        if center_scanner_code and cid in center_scanner_code:
            item["scanner_code_center"] = center_scanner_code[cid]
        if row.get("roi_ratio") is not None:
            item["roi_ratio"] = float(row["roi_ratio"])
        out.append(item)
    return out
