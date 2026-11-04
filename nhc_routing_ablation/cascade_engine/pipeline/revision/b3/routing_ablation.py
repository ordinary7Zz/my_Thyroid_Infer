"""B3 路由消融：准备 output 目录、共享 intermediate、指标汇总。"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from shared.metadata_prompt import apply_metadata_mode_to_config, normalize_metadata_mode
from shared.routing_modes import (
    cls_uses_llm_api,
    get_cls_routing_mode,
    get_seg_routing_mode,
    normalize_routing_mode,
    seg_uses_llm_api,
    validate_routing_mode,
)

AblationRoutingSpec = Union[str, dict[str, Any]]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_mode_name(mode: str) -> str:
    m = normalize_routing_mode(mode.strip()) or mode.strip().lower()
    return validate_routing_mode(m, allow_oracle=True)


@dataclass(frozen=True)
class AblationRoutingEntry:
    """一次 B3 运行的 seg/cls 路由及输出子目录名。"""

    seg: str
    cls: str
    run_id: str

    @property
    def label(self) -> str:
        return self.run_id if self.seg == self.cls else f"{self.seg}+{self.cls}"


def parse_ablation_routing_entry(raw: AblationRoutingSpec) -> AblationRoutingEntry:
    """
    解析 routing_modes 列表项。

    - 字符串 ``soft_voting``：seg/cls 相同，输出子目录 ``soft_voting``
    - 字典::

        name: custom_id          # 可选，输出子目录
        seg: static_rule
        cls: soft_voting
        # 或 routing_mode_seg / routing_mode_cls
        # 或 mode: soft_voting  （两侧默认，可被 seg/cls 覆盖）
    """
    if isinstance(raw, str):
        m = _norm_mode_name(raw)
        return AblationRoutingEntry(seg=m, cls=m, run_id=m)

    if not isinstance(raw, dict):
        raise ValueError(f"routing_modes 项须为 str 或 dict，得到 {type(raw)}")

    both = raw.get("mode") or raw.get("routing_mode")
    seg = raw.get("seg") or raw.get("routing_mode_seg") or raw.get("segmentation")
    cls = raw.get("cls") or raw.get("routing_mode_cls") or raw.get("classification")
    if both:
        bm = _norm_mode_name(str(both))
        seg = seg or bm
        cls = cls or bm
    if not seg or not cls:
        raise ValueError(
            "routing_modes 字典项须含 seg+cls，或 mode/routing_mode，或 routing_mode_seg/cls"
        )
    seg_m = _norm_mode_name(str(seg))
    cls_m = _norm_mode_name(str(cls))
    run_id = raw.get("name") or raw.get("id") or raw.get("run_id")
    if not run_id:
        run_id = seg_m if seg_m == cls_m else f"{seg_m}__{cls_m}"
    run_id = re.sub(r"[^\w.\-]+", "_", str(run_id))
    return AblationRoutingEntry(seg=seg_m, cls=cls_m, run_id=run_id)


def apply_routing_mode_to_config(
    config: dict,
    *,
    routing_mode: Optional[str] = None,
    routing_mode_seg: Optional[str] = None,
    routing_mode_cls: Optional[str] = None,
    task: str = "both",
) -> dict:
    """深拷贝 config 并写入 seg/cls routing_mode；llm 时同步 enable_agent。

    优先级：显式 routing_mode_seg/cls > routing_mode（按 task 施加到一侧或两侧）。
    """
    import copy

    cfg = copy.deepcopy(config)
    seg_m: Optional[str] = None
    cls_m: Optional[str] = None
    if routing_mode_seg is not None:
        seg_m = _norm_mode_name(routing_mode_seg)
    if routing_mode_cls is not None:
        cls_m = _norm_mode_name(routing_mode_cls)
    if routing_mode is not None:
        unified = _norm_mode_name(routing_mode)
        if task in ("seg", "both") and seg_m is None:
            seg_m = unified
        if task in ("cls", "both") and cls_m is None:
            cls_m = unified

    if task in ("seg", "both") and seg_m is not None:
        cfg.setdefault("segmentation", {}).setdefault("agent", {})["routing_mode"] = seg_m
        cfg["segmentation"]["agent"]["enable_agent"] = seg_m == "llm"
    if task in ("cls", "both") and cls_m is not None:
        cfg.setdefault("classification", {}).setdefault("agent", {})["routing_mode"] = cls_m
        cfg["classification"]["agent"]["enable_agent"] = cls_m == "llm"
    return cfg


def link_shared_intermediate(output_dir: Path, shared_dir: Path) -> Path:
    """
    将 shared_dir 下的 seg/cls（及可选 autogluon.json）链接到 output/intermediate。
    已存在真实目录时先备份为 intermediate_local。
    """
    out_inter = output_dir / "intermediate"
    out_inter.mkdir(parents=True, exist_ok=True)
    shared_dir = Path(shared_dir)

    for name in ("seg", "cls"):
        src = shared_dir / name
        dst = out_inter / name
        if not src.is_dir():
            raise FileNotFoundError(f"共享 intermediate 缺少 {src}")
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink():
                dst.unlink()
            elif dst.is_dir():
                backup = out_inter / f"{name}_local_backup"
                if backup.exists():
                    shutil.rmtree(backup)
                shutil.move(str(dst), str(backup))
        os.symlink(src.resolve(), dst, target_is_directory=True)

    ag_src = shared_dir / "autogluon.json"
    if ag_src.is_file():
        ag_dst = out_inter / "autogluon.json"
        if ag_dst.exists() or ag_dst.is_symlink():
            if ag_dst.is_symlink():
                ag_dst.unlink()
            else:
                ag_dst.rename(out_inter / "autogluon.json.bak")
        os.symlink(ag_src.resolve(), ag_dst)

    return out_inter


def clear_phase3_cache(output_dir: Path) -> None:
    inter = Path(output_dir) / "intermediate"
    sel = inter / "selected_masks.npz"
    if sel.exists():
        sel.unlink()
    results = Path(output_dir) / "results.json"
    if results.exists():
        results.unlink()


def get_ablation_batch_settings(config: dict) -> dict:
    """读取 pipeline.ablation 中的批量 B3 设置。"""
    ab = config.get("pipeline", {}).get("ablation", {}) or {}
    modes_raw = ab.get("routing_modes")
    entries: list[AblationRoutingEntry] = []
    if modes_raw is not None:
        if not isinstance(modes_raw, list):
            raise ValueError("pipeline.ablation.routing_modes 须为列表（str 或 dict）")
        for item in modes_raw:
            if item is None:
                continue
            if isinstance(item, str) and not item.strip():
                continue
            entries.append(parse_ablation_routing_entry(item))
    return {
        "routing_entries": entries,
        "shared_intermediate": ab.get("shared_intermediate"),
        "output_root": ab.get("output_root"),
        "task": ab.get("task", "both"),
        "force_phase3": bool(ab.get("force_phase3", False)),
        "summarize": bool(ab.get("summarize", False)),
        "jobs": int(ab.get("jobs", 1) or 1),
    }


def prepare_ablation_run(
    config: dict,
    output_dir: Path,
    *,
    shared_intermediate: Optional[Path],
    routing_mode: Optional[str] = None,
    routing_mode_seg: Optional[str] = None,
    routing_mode_cls: Optional[str] = None,
    metadata_mode: Optional[str] = None,
    task: str = "both",
    force_phase3: bool = False,
    best_single_artifact: Optional[Path] = None,
    run_label: Optional[str] = None,
) -> dict:
    if routing_mode is None and routing_mode_seg is None and routing_mode_cls is None:
        raise ValueError("须指定 routing_mode 或 routing_mode_seg/cls")
    cfg = apply_routing_mode_to_config(
        config,
        routing_mode=routing_mode,
        routing_mode_seg=routing_mode_seg,
        routing_mode_cls=routing_mode_cls,
        task=task,
    )
    if metadata_mode is not None:
        cfg = apply_metadata_mode_to_config(cfg, metadata_mode)
    elif cfg.get("segmentation", {}).get("agent", {}).get("metadata_mode"):
        mode = normalize_metadata_mode(cfg["segmentation"]["agent"]["metadata_mode"])
        cfg = apply_metadata_mode_to_config(cfg, mode)
    if best_single_artifact and Path(best_single_artifact).is_file():
        from routing.best_single_expert import apply_best_single_artifact, load_best_single_artifact
        apply_best_single_artifact(cfg, load_best_single_artifact(Path(best_single_artifact)))
    cfg.setdefault("pipeline", {}).setdefault("output", {})["output_dir"] = str(output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if shared_intermediate:
        link_shared_intermediate(output_dir, Path(shared_intermediate))
    if force_phase3:
        clear_phase3_cache(output_dir)

    seg_mode = get_seg_routing_mode(cfg)
    cls_mode = get_cls_routing_mode(cfg)
    if seg_mode == "learned" or cls_mode == "learned":
        cfg.setdefault("pipeline", {})["phase3_expert_judge"] = False
    meta_mode = normalize_metadata_mode(
        cfg.get("segmentation", {}).get("agent", {}).get("metadata_mode")
    )
    label = run_label or (
        routing_mode
        if routing_mode and routing_mode_seg is None and routing_mode_cls is None
        else (seg_mode if seg_mode == cls_mode else f"{seg_mode}__{cls_mode}")
    )
    meta_path = output_dir / "ablation_run.json"
    run_meta = {
        "routing_mode": label,
        "routing_mode_seg": seg_mode,
        "routing_mode_cls": cls_mode,
        "metadata_mode": meta_mode,
        "task": task,
        "shared_intermediate": str(shared_intermediate) if shared_intermediate else None,
        "force_phase3": force_phase3,
        "seg_mode": seg_mode,
        "cls_mode": cls_mode,
        "uses_llm": seg_uses_llm_api(seg_mode) or cls_uses_llm_api(cls_mode),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    return cfg
