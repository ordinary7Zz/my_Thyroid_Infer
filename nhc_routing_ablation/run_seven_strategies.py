#!/usr/bin/env python3
"""
NHC: run routing ablation strategies (1)–(6) and (8) — no LLM (7).

Orchestrates ThyroidCascadeAgent B3 entry points from my_Thyroid_Infer so NHC
jobs can be launched on the private-data machine without hunting revision scripts.

Steps (default):
  1. Phase 1–2 cache (optional): run_pipeline.py --stop-after-phase2
  2. Static bundle: best_single, soft_voting, majority, static_rule, max_confidence, oracle
  3. Learned router (6): run_learned_b3.py
  4. Summarize all modes (static + learned) → ``_summary/b3_routing_table.csv``
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from paths import (  # noqa: E402
    CASCADE_ROOT,
    CONFIG_DIR,
    INFER_ROOT,
    RESULTS_ROOT,
    STRATEGY_ROWS,
    b3_config_for_profile,
    cascade_phase12_config_for_profile,
    materialize_cascade_config_for_infer,
    resolve_cascade_path,
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
    root = str(CASCADE_ROOT)
    merged["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"
    subprocess.check_call(cmd, cwd=str(cwd), env=merged)


def _cascade_config_path(src: Path, tag: str) -> Path:
    """Expand ``./infer_*`` weight paths for ThyroidCascadeAgent (cwd = CASCADE_ROOT)."""
    dest = RESULTS_ROOT / "_config_staging" / f"{src.stem}_{tag}_for_cascade.yaml"
    return materialize_cascade_config_for_infer(src, dest)


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
        "cascade_root": str(CASCADE_ROOT),
        "infer_root": str(INFER_ROOT),
        "b3_config": str(b3_config),
        "shared_intermediate": str(shared_intermediate),
        "output_root": str(output_root),
        "strategies": STRATEGY_ROWS,
        "skipped_llm_strategy_7": True,
        "output_subdirs": {
            s["mode"]: str(output_root / s["mode"]) for s in STRATEGY_ROWS if s["kind"] == "static"
        }
        | {"learned": str(output_root / "learned")},
    }
    out = output_root / "_run_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"✓ manifest: {out}")
    return out


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
    _run(cmd, cwd=CASCADE_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="NHC seven-strategy routing ablation (no LLM)")
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
    args = parser.parse_args()

    if not CASCADE_ROOT.is_dir():
        print(f"ThyroidCascadeAgent not found at {CASCADE_ROOT}", file=sys.stderr)
        return 1

    b3_path = args.b3_config or b3_config_for_profile(args.profile)
    b3_path = b3_path.resolve()
    b3_cascade_cfg = _cascade_config_path(b3_path, "b3")
    cfg = _load_yaml(b3_path)
    abl = _ablation_block(cfg)
    shared_raw = abl.get("shared_intermediate")
    out_raw = abl.get("output_root")
    if not shared_raw or not out_raw:
        print("B3 config must set pipeline.ablation.shared_intermediate and output_root", file=sys.stderr)
        return 1

    shared = resolve_cascade_path(shared_raw)
    output_root = Path(out_raw)
    if not output_root.is_absolute():
        output_root = (INFER_ROOT / output_root).resolve()

    _write_run_manifest(
        profile=args.profile,
        b3_config=b3_path,
        output_root=output_root,
        shared_intermediate=shared,
    )

    if not args.skip_phase12:
        if _intermediate_ready(shared):
            print(f"✓ Phase 1–2 cache present: {shared}")
        else:
            phase12_src = cascade_phase12_config_for_profile(args.profile)
            phase12_cfg = _cascade_config_path(phase12_src, "phase12")
            _run(
                [
                    sys.executable,
                    "run_pipeline.py",
                    "--config",
                    str(phase12_cfg),
                    "--stop-after-phase2",
                ],
                cwd=CASCADE_ROOT,
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
            cwd=CASCADE_ROOT,
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
            cwd=CASCADE_ROOT,
        )

    _summarize_all_modes(output_root)

    print("\nDone. Summaries:", output_root / "_summary")
    print("Map modes → reply strategies:", CONFIG_DIR / "README_STRATEGIES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
