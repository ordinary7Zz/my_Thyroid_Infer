"""B8：risk–coverage 曲线（可选 matplotlib）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def plot_risk_coverage_curve(
    curve: list[dict[str, Any]],
    out_path: str | Path,
    *,
    title: str = "Risk–coverage",
    label: Optional[str] = None,
    mark_coverage: Optional[float] = None,
    mark_abstention_rate: Optional[float] = None,
) -> Optional[Path]:
    """
    绘制保留子集分类准确率 vs 覆盖率（或 vs 送复核率）。

    curve 元素需含 coverage、retained_accuracy；可选 abstention_rate。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not curve:
        return None

    coverages: list[float] = []
    accs: list[float] = []
    abst: list[float] = []
    for pt in curve:
        acc = pt.get("retained_accuracy")
        if acc is None:
            continue
        coverages.append(float(pt["coverage"]))
        accs.append(float(acc))
        if "abstention_rate" in pt:
            abst.append(float(pt["abstention_rate"]))

    if not coverages:
        return None

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    line_label = label or "Retained accuracy"
    ax.plot(coverages, accs, "-", linewidth=1.5, label=line_label)

    if mark_coverage is not None:
        best_i = min(range(len(coverages)), key=lambda i: abs(coverages[i] - mark_coverage))
        ax.scatter(
            [coverages[best_i]],
            [accs[best_i]],
            s=60,
            zorder=5,
            label=f"Fixed rule (cov≈{coverages[best_i]:.2f})",
        )

    ax.set_xlabel("Coverage (fraction not flagged)")
    ax.set_ylabel("Classification accuracy (retained subset)")
    ax.set_title(title)
    ax.set_xlim(0, 1.02)
    ymin = max(0.0, min(accs) - 0.05)
    ymax = min(1.0, max(accs) + 0.02)
    if ymax - ymin < 0.1:
        ymin, ymax = max(0, ymin - 0.05), min(1, ymax + 0.05)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_risk_coverage_vs_abstention(
    curve: list[dict[str, Any]],
    out_path: str | Path,
    *,
    title: str = "Risk–coverage",
    label: Optional[str] = None,
    mark_abstention_rate: Optional[float] = None,
) -> Optional[Path]:
    """横轴为送复核率、纵轴为保留子集准确率。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not curve:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for pt in curve:
        acc = pt.get("retained_accuracy")
        if acc is None or "abstention_rate" not in pt:
            continue
        xs.append(float(pt["abstention_rate"]))
        ys.append(float(acc))

    if not xs:
        return None

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(xs, ys, "-", linewidth=1.5, label=label or "Retained accuracy")

    if mark_abstention_rate is not None:
        best_i = min(range(len(xs)), key=lambda i: abs(xs[i] - mark_abstention_rate))
        ax.scatter(
            [xs[best_i]],
            [ys[best_i]],
            s=60,
            zorder=5,
            label=f"Fixed rule ({xs[best_i]*100:.1f}% flagged)",
        )

    ax.set_xlabel("Abstention rate (flagged for review)")
    ax.set_ylabel("Classification accuracy (retained subset)")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
