"""B6 稳定性：从图像目录选取固定子集（排序前 N 或随机 seed）。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal, Optional

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

MANIFEST_NAME = "stability_case_manifest.json"


def list_sorted_image_paths(image_dir: Path) -> list[Path]:
    image_dir = Path(image_dir)
    return sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )


def select_stability_cases(
    image_dir: Path,
    n: int,
    *,
    start_index: int = 0,
    seed: Optional[int] = None,
    mode: Literal["first", "random"] = "first",
) -> list[Path]:
    """
    从目录选取 n 张图。

    - mode=first: start_index 之后取前 n 张（按文件名排序）
    - mode=random 或提供 seed: 在池子内 random.sample（seed 默认 42）
    返回路径按文件名排序，便于日志与复现。
    """
    pool = list_sorted_image_paths(image_dir)
    if start_index > 0:
        pool = pool[start_index:]
    if n <= 0 or not pool:
        return []
    n = min(n, len(pool))

    if mode == "random" or seed is not None:
        rng = random.Random(seed if seed is not None else 42)
        chosen = rng.sample(pool, n)
    else:
        chosen = pool[:n]
    return sorted(chosen, key=lambda p: p.name)


def write_manifest(path: Path, image_paths: list[Path], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **meta,
        "image_names": [p.name for p in image_paths],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> Optional[list[str]]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data.get("image_names")
    if not names:
        return None
    return list(names)


def paths_from_manifest(image_dir: Path, names: list[str]) -> list[Path]:
    image_dir = Path(image_dir)
    out: list[Path] = []
    for name in names:
        p = image_dir / name
        if p.is_file():
            out.append(p)
    return out
