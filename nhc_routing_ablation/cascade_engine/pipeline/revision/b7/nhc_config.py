"""B7：从 revision yaml 读取 NHC 数据路径（含可选全腺体 mask）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from shared.revision_paths import NHC_REVISION_CONFIG


def load_nhc_data_paths(config_path: Path | None = None) -> dict[str, Optional[str]]:
    import yaml

    cfg_path = config_path or NHC_REVISION_CONFIG
    if not cfg_path.is_file():
        return {
            "gt_mask_dir": None,
            "gt_gland_mask_dir": None,
            "gland_pred_mask_dir": None,
            "label_file": None,
            "center_manifest": None,
        }
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    pipe = data.get("pipeline") or {}
    data_cfg = pipe.get("data") or {}
    rev = pipe.get("revision") or {}
    return {
        "gt_mask_dir": data_cfg.get("gt_mask_dir"),
        "gt_gland_mask_dir": data_cfg.get("gt_gland_mask_dir"),
        "gland_pred_mask_dir": data_cfg.get("gland_pred_mask_dir"),
        "label_file": data_cfg.get("label_file"),
        "center_manifest": rev.get("center_manifest"),
    }
