#!/usr/bin/env python3
"""Per-cohort batch infer (run_all_extended) and export to workspace b2_exports/."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

PKG_DIR = Path(__file__).resolve().parent
INFER_ROOT = PKG_DIR.parent


def _abs_under_data_root(rel_or_abs: str, data_root: str) -> str:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return str(p)
    return str(Path(data_root) / rel_or_abs)


def build_run_config(cohort_id: str, cohort: dict, meta: dict) -> dict:
    base_path = (PKG_DIR / cohort["base_config"]).resolve()
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cfg = deepcopy(cfg)
    data_root = meta["data_root"]

    if seg := cohort.get("segmentation"):
        cfg["datasets"]["nodule_images"] = _abs_under_data_root(seg["images"], data_root)
        cfg["datasets"]["nodule_masks"] = _abs_under_data_root(seg["masks"], data_root)

    if cls := cohort.get("classification"):
        cfg["datasets"]["binary_images"] = _abs_under_data_root(cls["images"], data_root)
        masks = cls.get("masks", cls["images"])
        cfg["datasets"]["binary_masks"] = _abs_under_data_root(masks, data_root)
        cfg["labels"]["binary_json"] = _abs_under_data_root(cls["label_file"], data_root)
        cfg["labels"]["binary_field"] = cls.get("label_field", "malignancy")
        ext = cfg.setdefault("extended_binary", {})
        if cls.get("test_cls_dir"):
            ext["test_cls_dir"] = _abs_under_data_root(cls["test_cls_dir"], data_root)
        ext["test_cls_name"] = cls.get("test_cls_name", cohort_id)

    results_root = meta.get("results_root", "./results/b2")
    cfg["output_root"] = str((INFER_ROOT / results_root / cohort_id).resolve())
    cfg["use_dino_mask_for_cls"] = cohort.get("use_dino_mask_for_cls", False)
    return cfg


def save_weights_sidecar(cfg: dict, cohort_results: Path) -> None:
    """Record checkpoint paths for manifest.json (relative to ThyroidAgent when possible)."""
    ws = cfg.get("weights", {})
    sidecar = {
        "nodule": ws.get("nodule", {}),
        "binary": ws.get("binary", {}),
    }
    cohort_results.mkdir(parents=True, exist_ok=True)
    (cohort_results / "_b2_weights.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
    )


JSON_CLS_MODELS = frozenset({"repvit", "lsnet", "resnet50"})


def _latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def infer_output_exists(results_dir: Path, runall_task: str, model_key: str) -> bool:
    """True if results/b2/<cohort>/<task>/<model>/ already has usable inference output."""
    task_dir = results_dir / runall_task / model_key
    if not task_dir.is_dir():
        return False

    if runall_task == "binary" and model_key in JSON_CLS_MODELS:
        pred_json = _latest_file(task_dir, "*predictions.json")
        if pred_json and pred_json.stat().st_size > 50:
            return True
        if model_key == "lsnet" and _latest_file(task_dir, "lsnet_*.json"):
            return True
        if model_key == "resnet50":
            if (task_dir / "plot_single_task_auroc_results.json").is_file():
                return True
            if (task_dir / "metrics_summary.json").is_file():
                return True
        return False

    if runall_task == "binary":
        csv_path = _latest_file(task_dir, "predictions*.csv") or _latest_file(
            task_dir, "predictions.csv"
        )
        if csv_path and csv_path.stat().st_size > 20:
            return True
        return False

    # segmentation (nodule)
    log_path = _latest_file(task_dir, "metrics*.log") or _latest_file(task_dir, "metrics.log")
    return log_path is not None and log_path.stat().st_size > 0


def tasks_to_run_all(cohort: dict) -> list[str]:
    tasks = []
    for t in cohort.get("tasks", []):
        if t == "segmentation":
            tasks.append("nodule")
        elif t == "classification":
            tasks.append("binary")
    return tasks


def run_run_all(
    config_path: Path,
    tasks: list[str],
    models: list[str] | None,
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable,
        str(INFER_ROOT / "run_all_extended.py"),
        "--mode",
        "finetune",
        "--config",
        str(config_path),
        "--tasks",
        *tasks,
    ]
    if models:
        cmd.extend(["--models", *models])
    if dry_run:
        cmd.append("--dry_run")
    print("  $", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(INFER_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch baseline inference via run_all_extended.py")
    parser.add_argument("--cohorts", default=None, help="Comma-separated cohort ids")
    parser.add_argument(
        "--cohorts-config",
        type=Path,
        default=PKG_DIR / "cohorts.yaml",
        help="Cohort manifest (default: baseline_export/cohorts.yaml)",
    )
    parser.add_argument("--models", default=None, help="run_all model keys (default: seg+cls lists in yaml)")
    parser.add_argument("--skip-infer", action="store_true", help="Only convert to b2_exports")
    parser.add_argument("--convert-only", action="store_true", help="Alias for --skip-infer")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Per cohort/task/model: skip run_all for outputs already in results/b2",
    )
    args = parser.parse_args()

    meta = yaml.safe_load(args.cohorts_config.read_text(encoding="utf-8"))
    cohort_ids = list(meta["cohorts"].keys())
    if args.cohorts:
        cohort_ids = [c.strip() for c in args.cohorts.split(",") if c.strip()]

    gen_dir = PKG_DIR / "generated_configs"
    gen_dir.mkdir(parents=True, exist_ok=True)

    all_models: list[str] | None = None
    if args.models:
        all_models = [m.strip() for m in args.models.split(",") if m.strip()]

    for cid in cohort_ids:
        cohort = meta["cohorts"][cid]
        print(f"\n======== {cid} ========")
        cfg = build_run_config(cid, cohort, meta)
        cfg_path = gen_dir / f"{cid}.yaml"
        cfg_path.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"  config: {cfg_path}")

        results_dir = Path(cfg["output_root"])
        save_weights_sidecar(cfg, results_dir)

        cohorts_cfg_path = args.cohorts_config

        if not (args.skip_infer or args.convert_only):
            tasks = tasks_to_run_all(cohort)
            print(
                f"  run_all tasks: {tasks}  "
                "(B2: nodule=结节分割, binary=良恶性; 不跑 gland / tirads)"
            )
            seg_m = meta.get("seg_models", [])
            cls_m = meta.get("cls_models", [])
            pairs: list[tuple[str, str]] = []
            if "nodule" in tasks:
                for m in seg_m:
                    if all_models and m not in all_models:
                        continue
                    pairs.append(("nodule", m))
            if "binary" in tasks:
                for m in cls_m:
                    if all_models and m not in all_models:
                        continue
                    pairs.append(("binary", m))

            if args.skip_existing:
                missing_pairs = [
                    (t, m) for t, m in pairs if not infer_output_exists(results_dir, t, m)
                ]
                if not missing_pairs:
                    print("  [skip-existing] 本 cohort 所需模型输出均已存在，跳过推理")
                    continue
                done = len(pairs) - len(missing_pairs)
                if done:
                    print(f"  [skip-existing] 已有 {done} 个 model×task，补跑 {len(missing_pairs)} 个")
                tasks = list(dict.fromkeys(t for t, _ in missing_pairs))
                model_arg = list(dict.fromkeys(m for _, m in missing_pairs))
                print(f"  [skip-existing] tasks={tasks} models={model_arg}")
            else:
                tasks = tasks_to_run_all(cohort)
                model_arg = list(dict.fromkeys(m for _, m in pairs)) if pairs else None

            rc = run_run_all(cfg_path, tasks, model_arg, args.dry_run)
            if rc != 0:
                print(f"  [error] run_all exited {rc} for {cid}", file=sys.stderr)
                if not args.dry_run:
                    sys.exit(rc)

    if args.dry_run and (args.skip_infer or args.convert_only):
        return

    if not args.dry_run or args.skip_infer or args.convert_only:
        convert_cmd = [
            sys.executable,
            str(PKG_DIR / "convert_to_exports.py"),
            "--cohorts",
            ",".join(cohort_ids),
            "--b2-config",
            str(args.cohorts_config),
        ]
        if args.models:
            convert_cmd.extend(["--models", args.models])
        print("\n======== convert → b2_exports ========")
        print("  $", " ".join(convert_cmd))
        if not args.dry_run:
            rc = subprocess.call(convert_cmd, cwd=str(INFER_ROOT))
            if rc != 0:
                sys.exit(rc)


if __name__ == "__main__":
    main()
