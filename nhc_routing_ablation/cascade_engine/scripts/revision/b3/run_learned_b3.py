#!/usr/bin/env python3
"""B3 策略 6（learned）：默认走 static bundle 快速路径（无 Phase 4 AutoGluon / 无 radiomics judge）。

learned 为 CPU 逻辑回归；Phase 1–2 已缓存时固定 --device cpu。

示例:
  python scripts/revision/b3/fit_learned_router.py --config config/revision/b3/tn3k.yaml
  python scripts/revision/b3/run_learned_b3.py \\
    --config config/revision/b3/tn3k.yaml \\
    --shared-intermediate output/TN3K/pipeline_run/intermediate \\
    --output-dir output/ablation/b3/tn3k/learned \\
    --force-phase3 --device cpu
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
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

import argparse

import yaml

import repo_path

ROOT = repo_path.install()

from pipeline.revision.b3.routing_ablation import (
    AblationRoutingEntry,
    apply_routing_mode_to_config,
)
from pipeline.revision.b3.static_b3_bundle import (
    build_pipeline_for_bundle,
    collect_image_paths,
    load_labels,
    run_static_b3_bundle,
)


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _task_from_config(cfg: dict) -> str:
    return str(cfg.get("pipeline", {}).get("ablation", {}).get("task") or "both")


def _run_bundle(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="B3 learned（bundle 快速路径）")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shared-intermediate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=("seg", "cls", "both"), default=None)
    parser.add_argument("--force-phase3", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = _load_config(cfg_path)
    task = args.task or _task_from_config(cfg)
    cfg = apply_routing_mode_to_config(cfg, routing_mode="learned", task=task)
    cfg.setdefault("pipeline", {})["phase3_expert_judge"] = False

    data_cfg = cfg.setdefault("pipeline", {}).setdefault("data", {})
    if args.start_index is not None:
        data_cfg["start_index"] = args.start_index
    if args.max_images is not None:
        data_cfg["max_images"] = args.max_images

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    output_root = out_dir.parent
    run_id = out_dir.name

    img_names, image_paths = collect_image_paths(
        data_cfg["image_input"],
        start_index=data_cfg.get("start_index", 0),
        max_images=data_cfg.get("max_images"),
    )
    labels = load_labels(data_cfg.get("label_file"), data_cfg.get("label_key", "malignancy"))

    shared = Path(args.shared_intermediate)
    if not shared.is_absolute():
        shared = ROOT / shared

    pipeline = build_pipeline_for_bundle(cfg, args.device)
    entries = [AblationRoutingEntry(seg="learned", cls="learned", run_id=run_id)]
    print(
        f"learned bundle：{len(img_names)} 张，output_root={output_root}，"
        "phase3_expert_judge=false",
        flush=True,
    )
    run_static_b3_bundle(
        pipeline,
        cfg,
        entries=entries,
        image_dir=data_cfg["image_input"],
        output_root=output_root,
        shared_intermediate=shared.resolve(),
        gt_mask_dir=data_cfg.get("gt_mask_dir"),
        labels=labels,
        img_names=img_names,
        image_paths=image_paths,
        task=task,
        force_phase3=args.force_phase3,
        skip_eval=args.skip_eval,
        input_device_info=data_cfg.get("device_info"),
        input_data_info=data_cfg.get("input_data_info"),
    )
    return 0


def _run_legacy_pipeline(argv: list[str]) -> int:
    if "--routing-mode" not in argv and "--routing-mode-seg" not in argv:
        argv = ["--routing-mode", "learned"] + argv
    if "--task" not in argv:
        cfg_path = None
        for i, a in enumerate(argv):
            if a == "--config" and i + 1 < len(argv):
                cfg_path = argv[i + 1]
                break
        if cfg_path:
            p = Path(cfg_path)
            if not p.is_absolute():
                p = ROOT / p
            argv = argv + ["--task", _task_from_config(_load_config(p))]
    runner = ROOT / "scripts" / "run_routing_ablation.py"
    cmd = [sys.executable, str(runner)] + argv
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> None:
    argv = list(sys.argv[1:])
    if "--legacy-pipeline" in argv:
        argv = [a for a in argv if a != "--legacy-pipeline"]
        raise SystemExit(_run_legacy_pipeline(argv))
    raise SystemExit(_run_bundle(argv))


if __name__ == "__main__":
    main()
