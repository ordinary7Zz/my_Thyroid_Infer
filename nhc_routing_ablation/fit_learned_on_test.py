#!/usr/bin/env python3
"""
在指定 B3 队列的 **测试集** Phase 1–2 intermediate 上拟合 learned（策略 6）。

用途：向稿面「开发集上拟合路由、外测 frozen」靠拢前的 **试跑**；
本脚本 deliberately 使用 **test intermediate + pipeline.data 标签**，不用 val_list / validation。

⚠️ 在 test 上 fit 再在 **同一 test** 上报告指标会有乐观偏倚；NHC 外测若用此 joblib 仅作协议试跑，
定稿前应改为 **合并 validation** 拟合（见 README「后续：合并 val」）。

示例:
  python fit_learned_on_test.py --cohort tn5k
  python fit_learned_on_test.py --cohort tn5k --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from paths import CASCADE_ROOT, INFER_ROOT  # noqa: E402

COHORT_DEFAULTS = {
    "tn5k": {
        "b3_config": "config/revision/b3/tn5k.yaml",
        "test_intermediate": "output/TN5K/pipeline_run/intermediate",
        "artifact_subdir": "artifacts/b3/learned_fit_tn5k_test",
    },
}


def _load_b3_config(rel_path: str) -> dict:
    path = CASCADE_ROOT / rel_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _patch_for_test_fit(cfg: dict, *, artifact_subdir: str) -> dict:
    out = deepcopy(cfg)
    ab = out.setdefault("pipeline", {}).setdefault("ablation", {})
    ab["learned_fit_split"] = "test"
    ab["test_intermediate"] = ab.get("shared_intermediate") or COHORT_DEFAULTS["tn5k"]["test_intermediate"]
    lr = ab.setdefault("learned_router", {})
    lr["seg_artifact"] = f"{artifact_subdir}/learned_router_seg.joblib"
    lr["cls_artifact"] = f"{artifact_subdir}/learned_router_cls.joblib"
    lr["allow_missing_artifact"] = False
    lr["write_random_artifact"] = False
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="在测试集 intermediate 上拟合 learned")
    parser.add_argument("--cohort", choices=tuple(COHORT_DEFAULTS), default="tn5k")
    parser.add_argument("--dry-run", action="store_true", help="只写临时 config，不拟合")
    parser.add_argument(
        "--intermediate",
        default=None,
        help="覆盖 test intermediate 目录（默认队列 test pipeline_run/intermediate）",
    )
    args = parser.parse_args()

    if not CASCADE_ROOT.is_dir():
        print(f"缺少 ThyroidCascadeAgent: {CASCADE_ROOT}", file=sys.stderr)
        return 1

    meta = COHORT_DEFAULTS[args.cohort]
    cfg = _load_b3_config(meta["b3_config"])
    cfg = _patch_for_test_fit(cfg, artifact_subdir=meta["artifact_subdir"])

    inter_rel = args.intermediate or meta["test_intermediate"]
    inter_path = (CASCADE_ROOT / inter_rel).resolve()
    if not inter_path.is_dir():
        print(
            f"缺少 test intermediate: {inter_path}\n"
            f"请先: cd {CASCADE_ROOT} && python run_pipeline.py --config config/... --stop-after-phase2",
            file=sys.stderr,
        )
        return 1

    fit_cfg_dir = INFER_ROOT / "results/nhc_routing_ablation/_fit_configs"
    fit_cfg_dir.mkdir(parents=True, exist_ok=True)
    fit_cfg_path = fit_cfg_dir / f"{args.cohort}_test_fit.yaml"
    with open(fit_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    artifact_dir = CASCADE_ROOT / meta["artifact_subdir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/revision/b3/fit_learned_router.py",
        "--config",
        str(fit_cfg_path),
        "--fit-split",
        "test",
        "--intermediate",
        str(inter_path),
    ]
    print("临时 B3 config:", fit_cfg_path)
    print("Artifacts →", artifact_dir)
    print("Fit stems: 全部 test intermediate（忽略 val_list）")
    if args.dry_run:
        print("Dry-run:", " ".join(cmd))
        return 0

    env = os.environ.copy()
    root = str(CASCADE_ROOT)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"

    subprocess.check_call(cmd, cwd=str(CASCADE_ROOT), env=env)

    summary_path = artifact_dir / "fit_on_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cohort": args.cohort,
                "fit_split": "test",
                "intermediate": str(inter_path),
                "seg_artifact": str(artifact_dir / "learned_router_seg.joblib"),
                "cls_artifact": str(artifact_dir / "learned_router_cls.joblib"),
                "note": "Prototype fit on test split; replace with merged validation before final NHC table.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"✓ {summary_path}")
    print("NHC 七策略 yaml 可改为:")
    print(f"  seg_artifact: {meta['artifact_subdir']}/learned_router_seg.joblib")
    print(f"  cls_artifact: {meta['artifact_subdir']}/learned_router_cls.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
