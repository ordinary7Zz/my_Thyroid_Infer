#!/usr/bin/env python3
"""Build ImageFolder-style test_cls/0 and test_cls/1 from a malignancy label JSON."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _parse_malignancy(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = int(value)
        return v if v in (0, 1) else None
    s = str(value).strip()
    if s in ("0", "1"):
        return int(s)
    if s.lower() in ("benign", "良性"):
        return 0
    if s.lower() in ("malignant", "恶性"):
        return 1
    try:
        v = int(float(s))
        return v if v in (0, 1) else None
    except ValueError:
        return None


def _load_records(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("images") or raw.get("data")
        if not isinstance(items, list):
            raise ValueError(f"Unsupported label JSON shape: {path}")
        return items
    if isinstance(raw, list):
        return raw
    raise ValueError(f"Unsupported label JSON shape: {path}")


def build_test_cls(
    images_dir: Path,
    label_json: Path,
    output_dir: Path,
    *,
    label_field: str = "malignancy",
    use_symlink: bool = True,
    clean: bool = False,
) -> dict[str, int]:
    images_dir = images_dir.resolve()
    output_dir = output_dir.resolve()

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    for sub in ("0", "1"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    records = _load_records(label_json)
    stats = {"linked": 0, "skipped_label": 0, "missing_image": 0, "class_0": 0, "class_1": 0}

    for rec in records:
        if not isinstance(rec, dict):
            stats["skipped_label"] += 1
            continue
        fn = rec.get("filename") or rec.get("image_name") or rec.get("image")
        if not fn:
            stats["skipped_label"] += 1
            continue
        label = _parse_malignancy(rec.get(label_field, rec.get("malignancy")))
        if label is None:
            stats["skipped_label"] += 1
            continue

        src = images_dir / Path(str(fn)).name
        if not src.is_file():
            stats["missing_image"] += 1
            continue

        dst = output_dir / str(label) / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        if use_symlink:
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)

        stats["linked"] += 1
        stats[f"class_{label}"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Label JSON → test_cls ImageFolder (0/1)")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--label-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-field", default="malignancy")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinks (default: symlink)",
    )
    parser.add_argument("--clean", action="store_true", help="Remove output-dir before build")
    args = parser.parse_args()

    stats = build_test_cls(
        args.images_dir,
        args.label_json,
        args.output_dir,
        label_field=args.label_field,
        use_symlink=not args.copy,
        clean=args.clean,
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **stats}, indent=2))


if __name__ == "__main__":
    main()
