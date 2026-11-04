#!/usr/bin/env python3
"""用现有 B3 的 per_image_cls 复制扩增，验证 aggregate_by_center 的 auroc_estimable 逻辑。"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

CASCADE = Path(__file__).resolve().parents[1]
if str(CASCADE) not in sys.path:
    sys.path.insert(0, str(CASCADE))

from paths import get_engine_root  # noqa: E402

_engine = get_engine_root()
if str(_engine) not in sys.path:
    sys.path.insert(0, str(_engine))

from pipeline.revision.b7.nhc_heterogeneity import aggregate_by_center  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def replicate_cls_rows(
    rows: list[dict],
    *,
    replicas: int,
    balance_labels: bool,
) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        for rep in range(replicas):
            r = deepcopy(row)
            stem = r.get("stem") or Path(r["image_name"]).stem
            r["stem"] = f"{stem}_rep{rep:02d}"
            r["image_name"] = f"{r['stem']}{Path(r['image_name']).suffix}"
            if balance_labels and rep % 2 == 1 and r.get("true_label") is not None:
                r["true_label"] = 1 - int(r["true_label"])
            out.append(r)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cls-jsonl",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "results/nhc_routing_ablation/nhc_misd_tus/best_single/per_image_cls_metrics.jsonl",
    )
    p.add_argument("--replicas", type=int, default=5)
    p.add_argument("--balance-labels", action="store_true")
    args = p.parse_args()

    base = load_jsonl(args.cls_jsonl)
    expanded = replicate_cls_rows(
        base, replicas=args.replicas, balance_labels=args.balance_labels
    )
    report = aggregate_by_center(per_image_seg=[], per_image_cls=expanded)
    print("=== 原始 per_image_cls（未扩增）===")
    orig = aggregate_by_center(per_image_seg=[], per_image_cls=base)
    for cid, block in orig["centers"].items():
        print(
            f"  {cid}: n_cls={block['n_cls']} estimable={block['auroc_estimable']} "
            f"auroc={block['auroc']}"
        )
    print(f"\n=== 复制 x{args.replicas} balance_labels={args.balance_labels} ===")
    for cid, block in report["centers"].items():
        print(
            f"  {cid}: n_cls={block['n_cls']} estimable={block['auroc_estimable']} "
            f"auroc={block['auroc']}"
        )
    if args.balance_labels:
        ok = all(b["auroc_estimable"] for b in report["centers"].values())
        print("\n结论:", "各中心 AUROC 可估计 — 聚合逻辑正常" if ok else "仍有 NE — 需查代码")
        return 0 if ok else 1
    print(
        "\n结论: 仅复制、未 balance 时 NE 预期不变（单类别中心）。"
        " 请加 --balance-labels 或改 --balance-labels 扩数据集。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
