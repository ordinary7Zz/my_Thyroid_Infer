"""从 results.json 提取各策略的 method 签名（smoke / 汇总）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def collect_routing_methods(results: list[dict]) -> dict[str, Any]:
    seg_methods: set[str] = set()
    cls_path_a: set[str] = set()
    cls_path_b: set[str] = set()

    for r in results:
        seg_d = r.get("seg_decision") or {}
        if isinstance(seg_d, dict):
            m = seg_d.get("method")
            if m:
                seg_methods.add(str(m))
            else:
                sm = seg_d.get("selected_model", "")
                if sm:
                    seg_methods.add(f"seg_model:{sm}")
        cls_d = r.get("cls_decision") or {}
        if isinstance(cls_d, dict):
            method = cls_d.get("method", "")
            if r.get("path") == "A":
                cls_path_a.add(method or "classification_consensus")
            elif r.get("path") == "B":
                cls_path_b.add(method or "unknown_path_b")

    return {
        "seg_methods": sorted(seg_methods),
        "cls_path_a_methods": sorted(cls_path_a),
        "cls_path_b_methods": sorted(cls_path_b),
        "n_images": len(results),
        "n_path_b": sum(1 for r in results if r.get("path") == "B"),
    }


def write_routing_methods_report(output_dir: Path, results: list[dict]) -> Path:
    report = collect_routing_methods(results)
    path = Path(output_dir) / "routing_methods.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def load_routing_methods(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
