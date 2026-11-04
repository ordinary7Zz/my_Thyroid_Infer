#!/usr/bin/env python3
"""B3 静态策略 bundle：一次扫图，输出 config 中全部 routing_modes。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

def __revision_bootstrap() -> None:
    p = Path(__file__).resolve()
    for d in [p.parent, *p.parents]:
        if d.name == "revision" and (d / "_path_setup.py").is_file():
            spec = importlib.util.spec_from_file_location("_rev_path_setup", d / "_path_setup.py")
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            mod.bootstrap(__file__)
            return
    raise RuntimeError(f"revision _path_setup.py not found above {__file__}")

__revision_bootstrap()

import importlib.util
from pathlib import Path

import argparse
import os
import subprocess
import sys
from pathlib import Path

import repo_path
import yaml

ROOT = repo_path.install()

from pipeline.revision.b3.routing_ablation import get_ablation_batch_settings, parse_ablation_routing_entry
from pipeline.revision.b3.static_b3_bundle import (
    build_pipeline_for_bundle,
    collect_image_paths,
    load_labels,
    run_static_b3_bundle,
)

WAVE1_MODES = (
    "soft_voting",
    "majority",
    "max_confidence",
    "static_rule",
    "best_single",
    "oracle",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="B3 静态策略 bundle（单次扫图）")
    parser.add_argument("--config", default="config/revision/b3/sample.yaml")
    parser.add_argument("--shared-intermediate", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--modes", default=None)
    parser.add_argument("--task", choices=("seg", "cls", "both"), default=None)
    parser.add_argument("--force-phase3", action="store_true")
    parser.add_argument("--no-force-phase3", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    batch = get_ablation_batch_settings(cfg)

    if args.modes:
        entries = [
            parse_ablation_routing_entry(t.strip())
            for t in args.modes.split(",")
            if t.strip()
        ]
    elif batch["routing_entries"]:
        entries = batch["routing_entries"]
    else:
        entries = [parse_ablation_routing_entry(m) for m in WAVE1_MODES]

    shared = args.shared_intermediate or batch.get("shared_intermediate")
    if not shared:
        parser.error("需要 shared_intermediate")
    out_root = args.output_root or batch.get("output_root")
    if not out_root:
        parser.error("需要 output_root")

    task = args.task or batch.get("task") or "both"
    force_phase3 = batch.get("force_phase3", False)
    if args.force_phase3:
        force_phase3 = True
    if args.no_force_phase3:
        force_phase3 = False

    summarize = batch.get("summarize", False)
    if args.summarize:
        summarize = True
    if args.no_summarize:
        summarize = False

    data_cfg = cfg.get("pipeline", {}).get("data", {})
    img_names, image_paths = collect_image_paths(
        data_cfg["image_input"],
        start_index=data_cfg.get("start_index", 0),
        max_images=data_cfg.get("max_images"),
    )
    labels = load_labels(data_cfg.get("label_file"), data_cfg.get("label_key", "malignancy"))

    pipeline = build_pipeline_for_bundle(cfg, args.device)
    run_static_b3_bundle(
        pipeline,
        cfg,
        entries=entries,
        image_dir=data_cfg["image_input"],
        output_root=Path(out_root),
        shared_intermediate=Path(shared).resolve(),
        gt_mask_dir=data_cfg.get("gt_mask_dir"),
        labels=labels,
        img_names=img_names,
        image_paths=image_paths,
        task=task,
        force_phase3=force_phase3,
        skip_eval=args.skip_eval,
        input_device_info=data_cfg.get("device_info"),
    )

    if summarize and entries:
        ref = Path(out_root) / entries[0].run_id / "intermediate_manifest.json"
        sum_cmd = [
            sys.executable,
            "-m",
            "pipeline.revision.b3.ablation_summary",
            "--root",
            str(out_root),
        ]
        if ref.is_file():
            sum_cmd.extend(["--reference-manifest", str(ref)])
        print(f"\n汇总: {' '.join(sum_cmd)}")
        env = os.environ.copy()
        root = str(ROOT)
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"
        subprocess.check_call(sum_cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    main()

import importlib.util
from pathlib import Path


