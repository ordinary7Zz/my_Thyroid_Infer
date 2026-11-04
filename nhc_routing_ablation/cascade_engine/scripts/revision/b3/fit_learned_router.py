#!/usr/bin/env python3
"""B3 策略 6：在 val 上拟合 learned 路由，或写入随机初始化占位 artifact。"""

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
import json
from pathlib import Path

import repo_path
import yaml

ROOT = repo_path.install()

from routing.best_single_expert import resolve_val_fit_paths
from routing.learned_router import (
    LearnedRouterParams,
    artifact_path_for_task,
    build_router_random,
    fit_learned_on_val,
    learned_router_config,
    model_names_from_config,
    save_learned_router,
)


def _init_random_artifacts(config: dict) -> dict:
    lr_cfg = learned_router_config(config)
    params = LearnedRouterParams.from_dict(lr_cfg.get("params"))
    params.random_init = True
    out: dict = {"mode": "random_init"}
    for task in ("seg", "cls"):
        names = model_names_from_config(config, task)
        if not names:
            continue
        router = build_router_random(task, names, params)
        path = artifact_path_for_task(config, task)
        if path is None:
            raise ValueError(f"未配置 learned_router.{task}_artifact")
        save_learned_router(router, path)
        out[f"{task}_artifact"] = str(path)
        out[f"{task}_models"] = names
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="拟合或初始化 B3 learned 路由器")
    parser.add_argument("--config", default="config/revision/b3/sample.yaml")
    parser.add_argument(
        "--intermediate",
        default=None,
        help="含 seg/ cls 的 intermediate（拟合 val 时必填）",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="仅按 config 随机系数写 joblib，不使用 val 标签",
    )
    parser.add_argument(
        "--task",
        choices=("seg", "cls", "both"),
        default=None,
        help="覆盖 config pipeline.ablation.task（seg / cls / both）",
    )
    parser.add_argument(
        "--fit-split",
        choices=("val", "test"),
        default=None,
        help="val=验证集(val_list/val_intermediate)；test=测试集(shared_intermediate 内全部或 test_list)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.random_init:
        payload = _init_random_artifacts(config)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    ab = config.get("pipeline", {}).get("ablation", {}) or {}
    fit_split = args.fit_split or ab.get("learned_fit_split") or "val"
    fit_split = str(fit_split).strip().lower()
    if fit_split not in ("val", "test"):
        parser.error(f"未知 fit_split: {fit_split}")

    if fit_split == "test":
        intermediate = (
            args.intermediate
            or ab.get("test_intermediate")
            or ab.get("shared_intermediate")
        )
        if not intermediate:
            parser.error(
                "test 拟合须 --intermediate 或 yaml 中 ablation.test_intermediate / shared_intermediate"
            )
    else:
        intermediate = args.intermediate or ab.get("val_intermediate")
        if not intermediate:
            parser.error("拟合 val 须指定 --intermediate 或在 yaml 配置 ablation.val_intermediate")

    gt_mask_dir, label_file, label_key = resolve_val_fit_paths(config)
    if fit_split == "test":
        data = config.get("pipeline", {}).get("data", {}) or {}
        if data.get("gt_mask_dir"):
            gt_mask_dir = Path(data["gt_mask_dir"])
        if data.get("label_file"):
            label_file = str(data["label_file"])
        label_key = str(data.get("label_key", label_key))

    payload = fit_learned_on_val(
        config,
        Path(intermediate),
        gt_mask_dir=gt_mask_dir,
        label_file=label_file,
        label_key=label_key,
        task=args.task,
        fit_split=fit_split,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

