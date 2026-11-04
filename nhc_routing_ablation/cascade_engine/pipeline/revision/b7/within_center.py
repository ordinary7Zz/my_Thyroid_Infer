"""B7：中心内（组内）离散度汇总。"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def within_center_summary(values_by_group: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    """
    每个中心：n、mean、std、cv（%）、min、max、q25、q75。
    """
    out: dict[str, dict[str, Any]] = {}
    for cid, vals in sorted(values_by_group.items()):
        clean = [float(x) for x in vals if x is not None and not np.isnan(x)]
        if not clean:
            out[cid] = {"n": 0}
            continue
        arr = np.array(clean, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        cv = float(100.0 * std / mean) if mean != 0 else None
        out[cid] = {
            "n": int(arr.size),
            "mean": mean,
            "std": std,
            "cv_percent": cv,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }
    return out
