"""intermediate/ 目录指纹，供 B3/B4 交叉校验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_intermediate_files(inter_dir: Path) -> list[Path]:
    files: list[Path] = []
    seg = inter_dir / "seg"
    cls = inter_dir / "cls"
    if seg.is_dir():
        files.extend(sorted(seg.glob("*.npz")))
    if cls.is_dir():
        files.extend(sorted(cls.glob("*.json")))
    return files


def write_intermediate_manifest(
    inter_dir: Path,
    manifest_path: Path | None = None,
    extra_paths: Iterable[Path] | None = None,
) -> dict:
    inter_dir = Path(inter_dir)
    manifest_path = manifest_path or (inter_dir.parent / "intermediate_manifest.json")
    entries = []
    for p in collect_intermediate_files(inter_dir):
        entries.append({
            "path": str(p.relative_to(inter_dir.parent)),
            "sha256": _sha256_file(p),
            "bytes": p.stat().st_size,
        })
    if extra_paths:
        for p in extra_paths:
            p = Path(p)
            if p.is_file():
                entries.append({
                    "path": str(p.name),
                    "sha256": _sha256_file(p),
                    "bytes": p.stat().st_size,
                })
    doc = {"intermediate_dir": str(inter_dir), "files": entries}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return doc


def manifests_match(a: dict, b: dict) -> bool:
    fa = {x["path"]: x["sha256"] for x in a.get("files", [])}
    fb = {x["path"]: x["sha256"] for x in b.get("files", [])}
    return fa == fb
