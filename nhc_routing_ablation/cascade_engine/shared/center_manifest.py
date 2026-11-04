"""按 stem / image_name 解析 center_id（B7/B8 按中心汇总）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional


def load_center_lookup(manifest_path: Optional[str]) -> dict[str, str]:
    """
    加载 manifest，返回 stem -> center_id。

    支持：
      - CSV：列 image_name 或 stem + center_id（或 centre_id）
      - JSON：list[{stem|image_name, center_id}] 或 dict stem -> center_id
    """
    if not manifest_path:
        return {}
    path = Path(manifest_path)
    if not path.is_file():
        return {}

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        if isinstance(data, list):
            out: dict[str, str] = {}
            for row in data:
                if not isinstance(row, dict):
                    continue
                cid = row.get("center_id") or row.get("centre_id")
                if cid is None:
                    continue
                key = row.get("stem") or row.get("image_name") or row.get("filename")
                if key is None:
                    continue
                out[Path(str(key)).stem] = str(cid)
            return out
        return {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: dict[str, str] = {}
        for row in reader:
            cid = row.get("center_id") or row.get("centre_id")
            if not cid:
                continue
            key = row.get("stem") or row.get("image_name") or row.get("filename")
            if not key:
                continue
            out[Path(key).stem] = str(cid)
        return out


def resolve_center_id(image_name: str, lookup: dict[str, str]) -> Optional[str]:
    if not lookup:
        return None
    stem = Path(image_name).stem
    return lookup.get(stem)


_CENTER_KEYS = (
    "center_id",
    "centre_id",
    "site_id",
    "hospital_id",
    "institution_id",
    "center",
    "centre",
    "site",
)


def _center_from_row(row: dict) -> Optional[str]:
    for k in _CENTER_KEYS:
        if k in row and row[k] not in (None, ""):
            return str(row[k])
    return None


def build_center_lookup_from_labels(label_file: str | Path) -> dict[str, str]:
    """从 labels.json 解析 stem -> center_id（支持 list 或 dict）。"""
    path = Path(label_file)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: dict[str, str] = {}
    if isinstance(data, dict):
        for key, val in data.items():
            stem = Path(str(key)).stem
            if isinstance(val, dict):
                cid = _center_from_row(val)
                if cid:
                    out[stem] = cid
            elif isinstance(val, (str, int)):
                out[stem] = str(val)
        return out

    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            cid = _center_from_row(row)
            if not cid:
                continue
            for name_key in ("image_name", "filename", "file", "path", "stem", "id"):
                if name_key in row and row[name_key]:
                    out[Path(str(row[name_key])).stem] = cid
                    break
    return out


def summarize_centers_from_labels(label_file: str | Path) -> dict[str, Any]:
    """B7 描述表：各中心图像数、恶性比例（若有 malignancy 字段）。"""
    path = Path(label_file)
    lookup = build_center_lookup_from_labels(path)
    if not path.is_file():
        return {"centers": {}, "n_images": 0}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    centers: dict[str, dict[str, Any]] = {}

    def _bump(stem: str, malignancy: Any = None) -> None:
        cid = lookup.get(stem, "unknown")
        c = centers.setdefault(cid, {"n_images": 0, "n_malignant": 0, "n_labeled": 0})
        c["n_images"] += 1
        if malignancy is not None:
            c["n_labeled"] += 1
            mal = str(malignancy).lower()
            if mal in ("1", "1.0", "恶性", "malignant", "m", "true"):
                c["n_malignant"] += 1

    if isinstance(data, dict):
        for key, val in data.items():
            stem = Path(str(key)).stem
            mal = val.get("malignancy") if isinstance(val, dict) else None
            _bump(stem, mal)
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            name = None
            for name_key in ("image_name", "filename", "file", "path", "stem", "id"):
                if name_key in row and row[name_key]:
                    name = Path(str(row[name_key])).stem
                    break
            if name:
                _bump(name, row.get("malignancy") or row.get("label"))

    for cid, c in centers.items():
        n_lab = c["n_labeled"]
        c["malignant_ratio"] = (c["n_malignant"] / n_lab) if n_lab else None

    return {
        "n_images": sum(c["n_images"] for c in centers.values()),
        "n_centers": len(centers),
        "centers": centers,
    }
