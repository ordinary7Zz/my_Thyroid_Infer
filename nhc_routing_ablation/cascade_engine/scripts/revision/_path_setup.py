"""Moved revision scripts: add repo root + scripts/ to sys.path, then repo_path.install()."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap(caller: str) -> Path:
    """Call at top of ``scripts/revision/**.py`` with ``bootstrap(__file__)``."""
    here = Path(caller).resolve().parent
    scripts_dir: Path | None = None
    for p in [here, *here.parents]:
        if p.name == "scripts" and (p / "repo_path.py").is_file():
            scripts_dir = p
            break
    if scripts_dir is None:
        raise RuntimeError(f"cannot find scripts/repo_path.py above {caller}")

    repo_root = scripts_dir.parent
    for path in (str(repo_root), str(scripts_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import repo_path

    return repo_path.install()


def install_revision_paths() -> Path:
    here = Path(__file__).resolve().parent
    scripts_dir: Path | None = None
    for p in [here, *here.parents]:
        if p.name == "scripts" and (p / "repo_path.py").is_file():
            scripts_dir = p
            break
    if scripts_dir is None:
        raise RuntimeError("cannot locate scripts/repo_path.py from revision package")

    repo_root = scripts_dir.parent
    for path in (str(repo_root), str(scripts_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import repo_path

    return repo_path.install()
