#!/usr/bin/env python3
"""仅重跑七策略中心性能 + 异质性（不跑 B3 GPU）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from center_analysis import run_center_analysis_for_all_strategies  # noqa: E402
from paths import INFER_ROOT, b3_config_for_profile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="B7-style center + heterogeneity for all B3 modes")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=INFER_ROOT / "results" / "nhc_routing_ablation" / "nhc_misd_tus",
    )
    parser.add_argument("--label-file", type=Path, default=None)
    parser.add_argument("--name-mapping", type=Path, default=None)
    parser.add_argument("--center-manifest", type=Path, default=None)
    args = parser.parse_args()

    label = args.label_file
    if label is None:
        import yaml

        cfg_path = b3_config_for_profile("nhc_misd_tus")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        from paths import apply_processed_dataset_to_config

        cfg, _, _ = apply_processed_dataset_to_config(cfg)
        raw = (cfg.get("pipeline") or {}).get("data", {}).get("label_file")
        if raw:
            label = Path(raw)

    out = run_center_analysis_for_all_strategies(
        args.output_root,
        label_file=label,
        name_mapping_file=args.name_mapping or (args.output_root / "name_mapping.json"),
        center_manifest=args.center_manifest,
    )
    print(f"Done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
