#!/usr/bin/env python3
"""
NHC 私有机主编排：七策略 B3 路由消融 + 七策略中心性能/异质性（B7 逻辑）。

Steps (default):
  1. Phase 1–2 cache (optional)
  2. Static B3 bundle (6 modes) + learned (6)
  3. B3 汇总 → ``_summary/b3_routing_table.csv``
  4. 七策略 B7 → ``_center_analysis/<mode>/`` + ``heterogeneity_by_strategy.csv``（结节/分类）
  5. 腺体分割 B7（单路径 ``_center_analysis/gland/``，非七策略）→ 中心 gland + ``gland_dice`` 异质性，并追加到 ``heterogeneity_by_strategy.csv``
  6. 打 zip → 自动删除 Phase1–2 seg/cls、各策略 ``intermediate/``、遗留 ``selected_masks.npz``（``--keep-cache`` 可保留）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from paths import (  # noqa: E402
    get_engine_root,
    resolve_nhc_path,
    resolve_cascade_path,
    CONFIG_DIR,
    INFER_ROOT,
    RESULTS_ROOT,
    STRATEGY_ROWS,
    ANON_MAPPING_FILENAME,
    apply_processed_dataset_to_config,
    b3_config_for_profile,
    cascade_phase12_config_for_profile,
    copy_anonymization_mapping_to_output,
    materialize_cascade_config_for_infer,
    processed_dir_from_root_config,
    processed_run_config_path,
    validate_nhc_dataset_setup,
    NhcDatasetConfigError,
)
from center_analysis import (  # noqa: E402
    GLAND_CENTER_SUBDIR,
    LEGACY_GLAND_CENTER_SUBDIR,
    run_center_analysis_all_strategies,
    run_gland_center_analysis,
)
from disk_cleanup import (  # noqa: E402
    DiskCleanupReport,
    cleanup_nhc_ablation_disk_cache,
    disk_cleanup_enabled,
    format_cleanup_summary,
)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ablation_block(cfg: dict) -> dict:
    return (cfg.get("pipeline") or {}).get("ablation") or {}


def _run(cmd: list[str], *, cwd: Path, env: dict | None = None) -> None:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    prev = merged.get("PYTHONPATH", "")
    root = str(get_engine_root())
    merged["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"
    subprocess.check_call(cmd, cwd=str(cwd), env=merged)


def _write_merged_config(cfg: dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return dest.resolve()


def _maybe_apply_processed(cfg: dict, *, use_processed: bool) -> dict:
    if not use_processed:
        return cfg
    cfg, proc_dir, run_cfg = apply_processed_dataset_to_config(cfg)
    print(
        f"✓ pipeline.data ← {run_cfg}\n"
        f"  prepare.output_dir → {proc_dir}",
        flush=True,
    )
    return cfg


def _prepare_cascade_yaml(src: Path, tag: str, *, use_processed: bool) -> Path:
    cfg = _load_yaml(src)
    cfg = _maybe_apply_processed(cfg, use_processed=use_processed)
    merged = RESULTS_ROOT / "_config_staging" / f"{src.stem}_{tag}_merged.yaml"
    _write_merged_config(cfg, merged)
    dest = RESULTS_ROOT / "_config_staging" / f"{src.stem}_{tag}_for_cascade.yaml"
    return materialize_cascade_config_for_infer(merged, dest)


def _intermediate_ready(shared: Path) -> bool:
    seg = shared / "seg"
    cls = shared / "cls"
    return seg.is_dir() and cls.is_dir() and any(seg.iterdir()) and any(cls.iterdir())


def _write_run_manifest(
    *,
    profile: str,
    b3_config: Path,
    output_root: Path,
    shared_intermediate: Path,
) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "profile": profile,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine_root": str(get_engine_root()),
        "infer_root": str(INFER_ROOT),
        "b3_config": str(b3_config),
        "shared_intermediate": str(shared_intermediate),
        "seg_cache_dir": str(shared_intermediate / "seg"),
        "output_root": str(output_root),
        "strategies": STRATEGY_ROWS,
        "skipped_llm_strategy_7": True,
        "output_subdirs": {
            s["mode"]: str(output_root / s["mode"]) for s in STRATEGY_ROWS if s["kind"] == "static"
        }
        | {"learned": str(output_root / "learned")},
        "center_analysis_dir": str(output_root / "_center_analysis"),
    }
    out = output_root / "_run_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"✓ manifest: {out}")
    return out


def _shell_quote(path: Path) -> str:
    """Single-quoted path safe for copy-paste into bash."""
    return "'" + str(path.resolve()).replace("'", "'\"'\"'") + "'"


def _zip_output_root(root: Path) -> Path | None:
    """Pack output_root into sibling ``<name>.zip`` (skip if SKIP_RESULT_ZIP=1)."""
    if os.environ.get("SKIP_RESULT_ZIP", "").strip().lower() in ("1", "true", "yes"):
        return None
    if not root.is_dir():
        return None
    zip_base = root.parent / root.name
    zip_path = zip_base.with_suffix(".zip")
    if zip_path.is_file():
        zip_path.unlink()
    print(f"\n正在压缩结果目录 → {zip_path} …", flush=True)
    try:
        created = shutil.make_archive(str(zip_base), "zip", root_dir=str(root))
        out = Path(created).resolve()
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"✓ 已生成 zip（约 {size_mb:.1f} MiB）", flush=True)
        return out
    except OSError as e:
        print(f"⚠ 压缩失败 ({e})，请手动打包目录: {root}", flush=True)
        return None


def _record_disk_cleanup_in_manifest(output_root: Path, report: DiskCleanupReport) -> None:
    manifest_path = output_root / "_run_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    data["disk_cleanup"] = {
        "at": datetime.now(timezone.utc).isoformat(),
        "bytes_freed": report.bytes_freed,
        "freed_mib": round(report.freed_mib, 2),
        "shared_seg_removed": report.shared_seg_removed,
        "shared_cls_removed": report.shared_cls_removed,
        "strategy_intermediate_dirs_removed": report.strategy_intermediate_dirs_removed,
        "selected_masks_removed": report.selected_masks_removed,
        "bundle_staging_removed": report.bundle_staging_removed,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _print_result_download_paths(
    *,
    output_root: Path,
    shared_intermediate: Path,
    zip_path: Path | None,
    cleanup_report: DiskCleanupReport | None = None,
) -> None:
    """End-of-run hint: zip download + disk cleanup summary."""
    root = output_root.resolve()
    mapping = root / ANON_MAPPING_FILENAME
    staging = (RESULTS_ROOT / "_config_staging").resolve()
    shared = shared_intermediate.resolve()
    seg_cache = shared / "seg"

    print("\n" + "=" * 60, flush=True)
    print("下载说明（从私有机拷出结果）", flush=True)
    print("=" * 60, flush=True)
    if zip_path is not None:
        print("  ▶ 只需拷贝下面这一个 zip 文件（scp / 网盘 / U 盘均可）:", flush=True)
        print(f"     {_shell_quote(zip_path)}", flush=True)
        print("", flush=True)
        print(f"  （解压后与目录等价，源目录: {_shell_quote(root)}）", flush=True)
    else:
        print("  ▶ 请打包/拷贝下面这一个目录（整目录 scp 即可，不必逐个策略单独拷）:", flush=True)
        print(f"     {_shell_quote(root)}", flush=True)
    print("", flush=True)
    print("  该目录内已包含:", flush=True)
    print("    · _summary/          汇总表 b3_routing_table.csv 等", flush=True)
    print("    · _run_manifest.json  本次 run 元信息", flush=True)
    if mapping.is_file():
        print(f"    · {ANON_MAPPING_FILENAME}           原名→匿名名映射", flush=True)
    else:
        print(f"    · {ANON_MAPPING_FILENAME}           （未复制到结果目录）", flush=True)
    mode_names = ", ".join(row["mode"] for row in STRATEGY_ROWS)
    print(f"    · {mode_names}/", flush=True)
    print("      各策略的 b3_metrics.json、results.json、per_image_*_metrics.jsonl 等", flush=True)
    center_root = root / "_center_analysis"
    if center_root.is_dir():
        print("    · _center_analysis/   七策略中心性能 + I²/τ²（R18–R20 制表用 CSV）", flush=True)
        print("      heterogeneity_by_strategy.csv、各 mode/b7_report.json 等", flush=True)
        gland_hint = center_root / GLAND_CENTER_SUBDIR
        if not gland_hint.is_dir() and (center_root / LEGACY_GLAND_CENTER_SUBDIR).is_dir():
            gland_hint = center_root / LEGACY_GLAND_CENTER_SUBDIR
        if gland_hint.is_dir():
            print(
                f"    · _center_analysis/{GLAND_CENTER_SUBDIR}/  腺体 B7（单 TGVideo 权重，非七策略）",
                flush=True,
            )
    print("", flush=True)
    print("  ▶ 留在私有机、不必下载:", flush=True)
    print(f"     合并用 staging 配置: {staging}", flush=True)
    print("-" * 60, flush=True)
    if cleanup_report is not None:
        print("私有机磁盘清理（已自动执行）", flush=True)
        print(f"  {format_cleanup_summary(cleanup_report)}", flush=True)
        if seg_cache.is_dir():
            print(f"  ⚠ 仍存在 {seg_cache}（未删或部分失败）", flush=True)
    else:
        print("私有机磁盘清理（未执行）", flush=True)
        print(
            "  保留 Phase1–2 缓存；复跑加 --keep-cache 或 NHC_ABLATION_KEEP_CACHE=1。",
            flush=True,
        )
        if seg_cache.is_dir():
            npz_count = sum(1 for p in seg_cache.glob("*.npz") if p.is_file())
            print(f"  分割 npz: {seg_cache} （{npz_count} 个 *.npz）", flush=True)
    print(
        "  表数字来源: 各策略 b3_metrics / per_image_*_metrics.jsonl；不含 selected_masks.npz。",
        flush=True,
    )
    print("=" * 60, flush=True)


def _summarize_all_modes(output_root: Path) -> None:
    """汇总 7 个策略目录的 b3_metrics.json → _summary/（含 learned）。"""
    if not any((output_root / s["mode"] / "b3_metrics.json").is_file() for s in STRATEGY_ROWS):
        print("No b3_metrics.json under output_root; skip summarize", flush=True)
        return
    ref = output_root / "best_single" / "intermediate_manifest.json"
    if not ref.is_file():
        for s in STRATEGY_ROWS:
            cand = output_root / s["mode"] / "intermediate_manifest.json"
            if cand.is_file():
                ref = cand
                break
    cmd = [
        sys.executable,
        "-m",
        "pipeline.revision.b3.ablation_summary",
        "--root",
        str(output_root),
    ]
    if ref.is_file():
        cmd.extend(["--reference-manifest", str(ref)])
    _run(cmd, cwd=get_engine_root())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NHC routing ablation (7 strategies) + per-strategy center / heterogeneity"
    )
    parser.add_argument(
        "--profile",
        choices=("nhc_misd_tus", "nhc_prepared_sample"),
        default="nhc_misd_tus",
        help="Config stem under nhc_routing_ablation/config/",
    )
    parser.add_argument(
        "--skip-phase12",
        action="store_true",
        help="Do not run run_pipeline --stop-after-phase2 (require existing intermediate)",
    )
    parser.add_argument("--skip-static", action="store_true", help="Skip static B3 bundle")
    parser.add_argument("--skip-learned", action="store_true", help="Skip strategy (6) learned")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for static bundle GPU phases (learned bundle uses cpu)",
    )
    parser.add_argument(
        "--b3-config",
        type=Path,
        default=None,
        help="Override B3 yaml (default: config/<profile>.yaml in this package)",
    )
    parser.add_argument(
        "--use-processed-config",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pipeline.data from config.yaml prepare.output_dir/config.yaml (default: on for nhc_misd_tus)",
    )
    parser.add_argument(
        "--skip-center-analysis",
        action="store_true",
        help="Skip B7 center performance / I² for all 7 strategies",
    )
    parser.add_argument(
        "--center-analysis-only",
        action="store_true",
        help="Only run _center_analysis on existing output_root (skip B3 phases)",
    )
    parser.add_argument(
        "--g3-csv",
        type=Path,
        default=None,
        help="Optional G3 center metadata CSV (merge into descriptive block)",
    )
    parser.add_argument(
        "--center-manifest",
        type=Path,
        default=None,
        help="Optional stem→center_id CSV/JSON manifest",
    )
    parser.add_argument(
        "--skip-gland",
        action="store_true",
        help="Skip single-path gland B7 under _center_analysis/gland/",
    )
    parser.add_argument(
        "--skip-gland-smoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gland-reference-mode",
        default="best_single",
        help="B3 subdir copied for results/seg metadata when running gland B7 (default: best_single)",
    )
    parser.add_argument(
        "--gland-smoke-reference-mode",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not delete Phase1–2 seg/cls cache or strategy intermediate/ after run",
    )
    args = parser.parse_args()
    if getattr(args, "skip_gland_smoke", False):
        args.skip_gland = True
    ref_mode = args.gland_reference_mode
    if args.gland_smoke_reference_mode:
        ref_mode = args.gland_smoke_reference_mode

    use_processed = args.use_processed_config
    if use_processed is None:
        use_processed = args.profile == "nhc_misd_tus"

    b3_path = args.b3_config or b3_config_for_profile(args.profile)
    b3_path = b3_path.resolve()
    b3_cfg_for_validate = _load_yaml(b3_path) if not use_processed else None
    try:
        validate_nhc_dataset_setup(use_processed=use_processed, b3_cfg=b3_cfg_for_validate)
    except NhcDatasetConfigError as e:
        print(f"\n[NHC 数据配置检查失败]\n{e}\n", file=sys.stderr)
        return 1

    if use_processed:
        print(
            f"✓ 数据配置: prepare → {processed_dir_from_root_config()}\n"
            f"  pipeline 生成: {processed_run_config_path()}",
            flush=True,
        )

    engine = get_engine_root()
    if not (engine / "run_pipeline.py").is_file():
        print(
            f"未找到 NHC 内置引擎 cascade_engine（缺少 run_pipeline.py）：{engine}",
            file=sys.stderr,
        )
        return 1

    b3_cascade_cfg = _prepare_cascade_yaml(b3_path, "b3", use_processed=use_processed)
    cfg = _load_yaml(b3_path)
    if use_processed:
        cfg = _maybe_apply_processed(cfg, use_processed=True)
    abl = _ablation_block(cfg)
    shared_raw = abl.get("shared_intermediate")
    out_raw = abl.get("output_root")
    if not shared_raw or not out_raw:
        print("B3 config must set pipeline.ablation.shared_intermediate and output_root", file=sys.stderr)
        return 1

    shared = resolve_nhc_path(shared_raw)
    output_root = resolve_nhc_path(out_raw)
    out_env = os.environ.get("NHC_ABLATION_OUTPUT_ROOT", "").strip()
    if out_env:
        output_root = Path(out_env)
        if not output_root.is_absolute():
            output_root = (INFER_ROOT / out_env).resolve()

    _write_run_manifest(
        profile=args.profile,
        b3_config=b3_path,
        output_root=output_root,
        shared_intermediate=shared,
    )

    if not args.center_analysis_only:
        if not args.skip_phase12:
            if _intermediate_ready(shared):
                print(f"✓ Phase 1–2 cache present: {shared}")
            else:
                phase12_src = cascade_phase12_config_for_profile(args.profile)
                phase12_cfg = _prepare_cascade_yaml(phase12_src, "phase12", use_processed=use_processed)
                _run(
                    [
                        sys.executable,
                        "run_pipeline.py",
                        "--config",
                        str(phase12_cfg),
                        "--stop-after-phase2",
                    ],
                    cwd=get_engine_root(),
                )
                if not _intermediate_ready(shared):
                    print(f"Phase 1–2 still missing under {shared}", file=sys.stderr)
                    return 1

        cfg_arg = str(b3_cascade_cfg)

        if not args.skip_static:
            _run(
                [
                    sys.executable,
                    "scripts/revision/b3/run_b3_static_bundle.py",
                    "--config",
                    cfg_arg,
                    "--shared-intermediate",
                    str(shared),
                    "--output-root",
                    str(output_root),
                    "--force-phase3",
                    "--no-summarize",
                    "--device",
                    args.device,
                ],
                cwd=get_engine_root(),
            )

        if not args.skip_learned:
            learned_out = output_root / "learned"
            _run(
                [
                    sys.executable,
                    "scripts/revision/b3/run_learned_b3.py",
                    "--config",
                    cfg_arg,
                    "--shared-intermediate",
                    str(shared),
                    "--output-dir",
                    str(learned_out),
                    "--force-phase3",
                    "--device",
                    "cpu",
                ],
                cwd=get_engine_root(),
            )

        _summarize_all_modes(output_root)

        copy_anonymization_mapping_to_output(
            output_root,
            use_processed=use_processed,
            cfg=cfg,
        )
    else:
        print(f"✓ --center-analysis-only：跳过 B3，使用已有 {output_root}", flush=True)

    if not args.skip_center_analysis:
        data = (cfg.get("pipeline") or {}).get("data") or {}
        label_raw = data.get("label_file")
        if not label_raw:
            print("⚠ 无 pipeline.data.label_file，跳过中心分析", file=sys.stderr)
        else:
            label_file = Path(label_raw)
            if not label_file.is_file():
                print(f"⚠ 标签文件不存在: {label_file}，跳过中心分析", file=sys.stderr)
            else:
                mapping = output_root / ANON_MAPPING_FILENAME
                run_center_analysis_all_strategies(
                    output_root,
                    label_file=label_file,
                    name_mapping_file=mapping if mapping.is_file() else None,
                    center_manifest=args.center_manifest,
                    g3_csv=args.g3_csv,
                )

    if not args.skip_gland:
        data = (cfg.get("pipeline") or {}).get("data") or {}
        label_raw = data.get("label_file")
        proc = processed_dir_from_root_config() if use_processed else INFER_ROOT / "datasets" / "processed"
        gt_gland = proc / "gland_masks"
        pred_as_gland = proc / "nodule_masks"
        pred_placeholder = pred_as_gland.name == "nodule_masks"
        if not (output_root / ref_mode / "results.json").is_file():
            print(
                f"⚠ 腺体 B7 跳过：缺少 {output_root / ref_mode}/results.json",
                file=sys.stderr,
            )
        elif label_raw and Path(label_raw).is_file():
            mapping = output_root / ANON_MAPPING_FILENAME
            try:
                run_gland_center_analysis(
                    output_root,
                    gt_gland_dir=gt_gland,
                    pred_gland_dir=pred_as_gland,
                    label_file=Path(label_raw),
                    name_mapping_file=mapping if mapping.is_file() else None,
                    center_manifest=args.center_manifest,
                    reference_mode=ref_mode,
                    pred_is_placeholder=pred_placeholder,
                )
            except FileNotFoundError as e:
                print(f"⚠ 腺体 B7 失败: {e}", file=sys.stderr)
        else:
            print("⚠ 腺体 B7 跳过：无 label_file", file=sys.stderr)

    cleanup_report: DiskCleanupReport | None = None
    zip_path = _zip_output_root(output_root.resolve())
    if disk_cleanup_enabled(keep_cache=args.keep_cache):
        cleanup_report = cleanup_nhc_ablation_disk_cache(output_root, shared)
        _record_disk_cleanup_in_manifest(output_root, cleanup_report)
        print(f"\n✓ {format_cleanup_summary(cleanup_report)}", flush=True)
    else:
        print("\n✓ 保留 Phase1–2 与 intermediate 缓存（--keep-cache / NHC_ABLATION_KEEP_CACHE=1）", flush=True)

    _print_result_download_paths(
        output_root=output_root,
        shared_intermediate=shared,
        zip_path=zip_path,
        cleanup_report=cleanup_report,
    )

    print("\n策略编号说明:", CONFIG_DIR / "README_STRATEGIES.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
