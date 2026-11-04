"""B8：可靠性图（可选 matplotlib）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def plot_reliability_diagram(
    bins: list[dict[str, Any]],
    out_path: str | Path,
    title: str = "Reliability diagram",
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    confs = []
    accs = []
    counts = []
    for b in bins:
        if b.get("count", 0) == 0:
            continue
        confs.append(b["avg_confidence"])
        accs.append(b["avg_accuracy"])
        counts.append(b["count"])

    if not confs:
        return None

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(confs, accs, s=[max(20, c) for c in counts], alpha=0.7)
    ax.set_xlabel("Mean predicted probability (malignant)")
    ax.set_ylabel("Fraction malignant")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
