"""返修实验固定路径（NHC 无 LLM 主结果等），脚本默认读此处，无需每次手填。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

# 相对 cascade_engine 根目录的路径（在 engine 根执行 run_pipeline / B3 脚本时）
NHC_PRIMARY_OUTPUT_DIR = Path("output/revision/nhc_no_llm_primary")
NHC_SHARED_INTERMEDIATE_DIR = Path("output/NHC-MISD-TUS/pipeline_run/intermediate")
NHC_REVISION_CONFIG = Path("config/revision/nhc_no_llm_primary.yaml")

B7_DEFAULT_OUT = Path("output/b7/nhc")
B7_DESCRIPTIVE_OUT = Path("output/b7/descriptive")
B8_FIT_OUT = Path("output/b8/fit")
B8_NHC_EVAL_OUT = Path("output/b8/nhc")

VALIDATION_NO_LLM_RESULTS = Path("output/validation/no_llm_primary/results.json")


def load_pipeline_data_paths(config_path: Path | None = None) -> Tuple[Optional[str], Optional[str]]:
    """从 revision config 读取 gt_mask_dir、label_file。"""
    import yaml

    cfg_path = config_path or NHC_REVISION_CONFIG
    if not cfg_path.is_file():
        return None, None
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data_cfg = (data.get("pipeline") or {}).get("data") or {}
    return data_cfg.get("gt_mask_dir"), data_cfg.get("label_file")
