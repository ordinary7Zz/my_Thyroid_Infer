#!/usr/bin/env python3
"""在 NHC sample 上做两中心交叉复制：原数据保留，另把 A 中心图挂到 B 中心名下（标签随原图）。"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

INFER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = INFER_ROOT / "datasets" / "processed"
DEFAULT_DST = INFER_ROOT / "datasets" / "processed_crosscenter"


def _center_from_orig(orig: str) -> str | None:
    parts = Path(orig).stem.split("_")
    if len(parts) >= 3 and parts[0] == "THYB" and parts[1] == "S":
        return "_".join(parts[:3])
    return None


def _copy_asset_triplet(src: Path, dst: Path, src_name: str, dst_name: str) -> None:
    shutil.copy2(src / "images" / src_name, dst / "images" / dst_name)
    for sub in ("gland_masks", "nodule_masks"):
        p = src / sub / src_name
        if p.is_file():
            shutil.copy2(p, dst / sub / dst_name)


def cross_center_expand(src: Path, dst: Path) -> None:
    src, dst = src.resolve(), dst.resolve()
    if dst.exists():
        shutil.rmtree(dst)
    for sub in ("images", "gland_masks", "nodule_masks"):
        (dst / sub).mkdir(parents=True)

    mapping: dict[str, str] = json.loads(
        (src / "name_mapping.json").read_text(encoding="utf-8")
    )
    labels: list[dict] = json.loads((src / "labels.json").read_text(encoding="utf-8"))
    label_by = {r["filename"]: r for r in labels if r.get("filename")}

    # 1) 原样复制全部图像与掩码
    for img in sorted((src / "images").glob("*")):
        _copy_asset_triplet(src, dst, img.name, img.name)

    new_mapping = dict(mapping)
    new_labels = [dict(r) for r in labels]

    by_center: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for orig, anon in mapping.items():
        cid = _center_from_orig(orig)
        if cid:
            by_center[cid].append((orig, anon))

    centers = sorted(by_center.keys())
    if len(centers) != 2:
        raise RuntimeError(f"期望 2 个中心，实际: {centers}")

    c0, c1 = centers[0], centers[1]
    cross_serial = 0

    def _add_cross(from_center: str, to_center: str, orig: str, anon: str) -> None:
        nonlocal cross_serial
        cross_serial += 1
        ext = Path(anon).suffix
        anon_stem = Path(anon).stem
        new_anon = f"{anon_stem}_x{to_center.split('_')[-1]}_{cross_serial:04d}{ext}"
        _copy_asset_triplet(src, dst, anon, new_anon)

        orig_stem = Path(orig).stem
        tail = orig_stem.split("_", 3)[-1] if orig_stem.count("_") >= 3 else orig_stem
        new_orig = f"{to_center}_NDxcross{cross_serial:04d}_{tail}{ext}"
        new_mapping[new_orig] = new_anon

        row = label_by.get(anon)
        if row:
            nr = dict(row)
            nr["filename"] = new_anon
            new_labels.append(nr)

    # 2) 交叉：c0 的图挂到 c1 名下，c1 的图挂到 c0 名下
    for orig, anon in by_center[c0]:
        _add_cross(c0, c1, orig, anon)
    for orig, anon in by_center[c1]:
        _add_cross(c1, c0, orig, anon)

    (dst / "name_mapping.json").write_text(
        json.dumps(new_mapping, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    (dst / "labels.json").write_text(
        json.dumps(new_labels, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    cfg = src / "config.yaml"
    if cfg.is_file():
        text = cfg.read_text(encoding="utf-8")
        rel = dst.relative_to(INFER_ROOT)
        text = text.replace(str(src.resolve()), str(dst))
        text = text.replace("./datasets/processed", str(rel).replace("\\", "/"))
        (dst / "config.yaml").write_text(text, encoding="utf-8")

    # 统计各中心标签
    from collections import Counter

    mal_by_center: dict[str, Counter] = defaultdict(Counter)
    rev = {v: k for k, v in new_mapping.items()}
    for row in new_labels:
        fn = row["filename"]
        orig = rev.get(fn, "")
        cid = _center_from_orig(orig) or "unknown"
        mal_by_center[cid][int(row.get("malignancy", 0))] += 1

    meta = {
        "source": str(src),
        "n_images": len(list((dst / "images").glob("*"))),
        "n_labels": len(new_labels),
        "malignancy_by_center": {k: dict(v) for k, v in mal_by_center.items()},
        "cross_centers": [c0, c1],
    }
    (dst / "crosscenter_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✓ {dst}")
    print(f"  images={meta['n_images']}  labels={meta['n_labels']}")
    for cid, cnt in sorted(mal_by_center.items()):
        print(f"  {cid}: malignancy counts {dict(cnt)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = p.parse_args()
    cross_center_expand(args.src, args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
