"""NHC 原始文件名 stem 解析与中心级汇总（B7 预处理用）。"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Optional


def parse_nhc_stem(stem: str) -> dict[str, Optional[str]]:
    """
    从原始 stem 解析中心与患者键。

    格式: THYB_S_{中心码}_{ND...}_{时间戳...}
    center_id = 前三段，如 THYB_S_AN01；patient_id = 第四段。
    """
    parts = stem.split("_")
    if len(parts) >= 4 and parts[0] == "THYB" and parts[1] == "S":
        return {
            "center_id": "_".join(parts[:3]),
            "patient_id": parts[3],
            "timestamp": "_".join(parts[4:]) if len(parts) > 4 else "",
        }
    return {"center_id": None, "patient_id": None, "timestamp": ""}


def anonymize_filename(stem: str, ext: str) -> str:
    """与 my_Thyroid_Infer prepare_data 一致的匿名化规则。"""
    parts = stem.split("_")

    if len(parts) >= 4:
        center_key = "_".join(parts[:3])
        patient_key = parts[3]
        suffix = "_".join(parts[4:])

        center_hash = hashlib.sha256(center_key.encode("utf-8")).hexdigest()[:12]
        patient_hash = hashlib.sha256(patient_key.encode("utf-8")).hexdigest()[:12]

        new_stem = f"{center_hash}_{patient_hash}"
        if suffix:
            new_stem = f"{new_stem}_{suffix}"
    else:
        new_stem = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:12]

    return f"{new_stem}{ext}"


class CenterStatsCollector:
    """按 center_id 累计图像数与 patient_id 去重数。"""

    def __init__(self) -> None:
        self._images: dict[str, int] = defaultdict(int)
        self._patients: dict[str, set[str]] = defaultdict(set)
        self._all_patients: set[str] = set()

    def record(self, center_id: Optional[str], patient_id: Optional[str]) -> None:
        cid = center_id if center_id else "unknown"
        self._images[cid] += 1
        if patient_id:
            self._patients[cid].add(patient_id)
            self._all_patients.add(patient_id)

    def to_summary(self) -> dict[str, Any]:
        center_ids = sorted(set(self._images) | set(self._patients))
        centers: dict[str, dict[str, int]] = {}
        for cid in center_ids:
            centers[cid] = {
                "n_images": self._images[cid],
                "n_patients": len(self._patients.get(cid, set())),
            }
        known = [c for c in center_ids if c != "unknown"]
        return {
            "n_centers": len(known),
            "n_centers_including_unknown": len(center_ids),
            "n_images": sum(self._images.values()),
            "n_patients_total": len(self._all_patients),
            "centers": centers,
            "patient_id_note": (
                "patient_id 为文件名第 4 段（如 ND000091）；"
                "与临床登记患者数不一致时请与 G3 元数据对账。"
            ),
        }
