#!/usr/bin/env python3
"""
复制 NHC sample（processed）图像/掩码若干份，用于排查中心 AUROC=NE 是否为代码问题。

重要：当前 2 中心 sample 里 AH04 全恶性、AN01 全良性；仅复制图像、不改标签时，
     每中心仍只有单一类别，AUROC 仍会 NE（见 pipeline/revision/b7/nhc_heterogeneity._auroc_auprc）。

加 --balance-labels：在副本上交替翻转 malignancy，使各中心同时含 0/1 标签（仅诊断用，非真实队列）。

产出目录默认：my_Thyroid_Infer/datasets/processed_auroc_sanity/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

INFER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = INFER_ROOT / "datasets" / "processed"
DEFAULT_DST = INFER_ROOT / "datasets" / "processed_auroc_sanity"


def _replica_stem(stem: str, rep: int) -> str:
    return f"{stem}_aurocrep{rep:02d}"


def _replica_name(filename: str, rep: int) -> str:
    p = Path(filename)
    return f"{_replica_stem(p.stem, rep)}{p.suffix}"


def expand(
    src: Path,
    dst: Path,
    *,
    replicas: int,
    balance_labels: bool,
) -> None:
    if replicas < 1:
        raise ValueError("replicas 须 >= 1")
    src = src.resolve()
    dst = dst.resolve()
    if dst.exists():
        shutil.rmtree(dst)
    for sub in ("images", "gland_masks", "nodule_masks"):
        (dst / sub).mkdir(parents=True)

    labels = json.loads((src / "labels.json").read_text(encoding="utf-8"))
    label_by_file = {row["filename"]: row for row in labels if row.get("filename")}

    mapping = json.loads((src / "name_mapping.json").read_text(encoding="utf-8"))
    new_mapping: dict[str, str] = {}
    new_labels: list[dict] = []

    image_files = sorted((src / "images").glob("*"))
    if not image_files:
        raise FileNotFoundError(f"无图像: {src / 'images'}")

    for img in image_files:
        for rep in range(replicas):
            out_name = _replica_name(img.name, rep)
            shutil.copy2(img, dst / "images" / out_name)
            for mask_dir in ("gland_masks", "nodule_masks"):
                m = src / mask_dir / img.name
                if m.is_file():
                    shutil.copy2(m, dst / mask_dir / out_name)

            orig_row = label_by_file.get(img.name)
            if orig_row:
                row = dict(orig_row)
                row["filename"] = out_name
                mal = int(row.get("malignancy", 0))
                if balance_labels and rep % 2 == 1:
                    row["malignancy"] = 1 - mal
                new_labels.append(row)

            for orig, anon in mapping.items():
                if anon != img.name:
                    continue
                orig_key = f"{Path(orig).stem}_aurocrep{rep:02d}{Path(orig).suffix}"
                new_mapping[orig_key] = out_name

    # 无 mapping 的图像（若有）仍写入 labels
    labeled_files = {r["filename"] for r in new_labels}
    for img in image_files:
        for rep in range(replicas):
            out_name = _replica_name(img.name, rep)
            if out_name in labeled_files:
                continue
            orig_row = label_by_file.get(img.name)
            if orig_row:
                row = dict(orig_row)
                row["filename"] = out_name
                new_labels.append(row)

    (dst / "labels.json").write_text(
        json.dumps(new_labels, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    (dst / "name_mapping.json").write_text(
        json.dumps(new_mapping, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    cfg_src = src / "config.yaml"
    if cfg_src.is_file():
        text = cfg_src.read_text(encoding="utf-8")
        text = text.replace(str(src), str(dst)).replace(
            "./datasets/processed", str(dst.relative_to(INFER_ROOT))
        )
        (dst / "config.yaml").write_text(text, encoding="utf-8")

    meta = {
        "source": str(src),
        "replicas": replicas,
        "balance_labels": balance_labels,
        "n_images": len(list((dst / "images").glob("*"))),
        "n_labels": len(new_labels),
        "purpose": "auroc_ne_diagnostic_only",
    }
    (dst / "auroc_sanity_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✓ 写入 {dst}  images={meta['n_images']}  labels={meta['n_labels']}")
    if not balance_labels:
        print(
            "  提示: 未加 --balance-labels 时，各中心仍可能 AUROC=NE（单类别），"
            "  不能据此判断代码有误。",
            flush=True,
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=DEFAULT_DST)
    p.add_argument("--replicas", type=int, default=5)
    p.add_argument(
        "--balance-labels",
        action="store_true",
        help="副本交替翻转 malignancy，使各中心含双类别（仅 AUROC 链路诊断）",
    )
    args = p.parse_args()
    expand(
        args.src,
        args.dst,
        replicas=args.replicas,
        balance_labels=args.balance_labels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
