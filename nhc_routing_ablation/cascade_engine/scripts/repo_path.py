"""``python scripts/<tool>.py`` 时把仓库根目录加入 sys.path。"""

from __future__ import annotations

import sys
from pathlib import Path

def find_repo_root(start: Path | None = None) -> Path:
    """从任意 ``scripts/`` 或 ``scripts/revision/...`` 下的文件定位仓库根（含 ``run_pipeline.py``）。"""
    here = (start or Path(__file__).resolve()).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "run_pipeline.py").is_file():
            return parent
    return Path(__file__).resolve().parents[1]


REPO_ROOT = find_repo_root()


def install() -> Path:
    root = str(find_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from shared.sklearn_warnings import configure_sklearn_pickle_warnings

    configure_sklearn_pickle_warnings()
    return REPO_ROOT
