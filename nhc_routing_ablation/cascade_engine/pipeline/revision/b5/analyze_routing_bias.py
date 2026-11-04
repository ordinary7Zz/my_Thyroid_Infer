"""
B5：从 routing_log.jsonl 统计中心 × 专家选择，计算互信息（离散）。
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


def _load_label_centres(
    labels_path: Path,
    centre_key: str,
    filename_key: str = "filename",
) -> dict[str, str]:
    with open(labels_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, str] = {}
    if isinstance(raw, list):
        for item in raw:
            fn = item.get(filename_key, "")
            if not fn:
                continue
            stem = Path(fn).stem
            val = item.get(centre_key)
            if val is not None:
                out[stem] = str(val)
    elif isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict) and centre_key in v:
                out[Path(k).stem] = str(v[centre_key])
    return out


def _mutual_information(joint: dict[tuple[str, str], int], n: int) -> float:
    if n <= 0:
        return 0.0
    cx: Counter[str] = Counter()
    cy: Counter[str] = Counter()
    for (x, y), c in joint.items():
        cx[x] += c
        cy[y] += c
    mi = 0.0
    for (x, y), c in joint.items():
        p_xy = c / n
        p_x = cx[x] / n
        p_y = cy[y] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log(p_xy / (p_x * p_y), 2)
    return mi


def analyze_routing_log(
    log_path: Path,
    *,
    centre_by_stem: Optional[dict[str, str]] = None,
    constant_centre: Optional[str] = None,
    llm_only: bool = True,
    task_filter: Optional[str] = None,
) -> dict[str, Any]:
    joint: dict[tuple[str, str], int] = Counter()
    n_rows = 0
    n_used = 0
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            row = json.loads(line)
            if llm_only and not row.get("llm_invoked"):
                continue
            if task_filter and row.get("task") != task_filter:
                continue
            model = row.get("selected_model")
            if not model:
                continue
            img = row.get("image_name") or ""
            stem = Path(img).stem
            if centre_by_stem and stem in centre_by_stem:
                centre = centre_by_stem[stem]
            elif constant_centre:
                centre = constant_centre
            else:
                centre = row.get("centre_id") or "unknown"
            joint[(centre, str(model))] += 1
            n_used += 1

    freq: dict[str, dict[str, int]] = defaultdict(dict)
    for (centre, model), c in joint.items():
        freq[centre][model] = c

    mi = _mutual_information(dict(joint), n_used)
    return {
        "log_path": str(log_path),
        "n_log_lines": n_rows,
        "n_pairs": n_used,
        "mutual_information_bits": mi,
        "centre_expert_counts": {k: dict(v) for k, v in sorted(freq.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B5 routing bias / centre × expert")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--centre-key", default="centre")
    parser.add_argument("--constant-centre", default=None, help="单中心队列用 dataset_id")
    parser.add_argument("--task", default=None, help="如 seg / cls_path_b")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--heatmap", type=Path, default=None)
    args = parser.parse_args()

    centres = None
    if args.labels and args.labels.is_file():
        centres = _load_label_centres(args.labels, args.centre_key)

    report = analyze_routing_log(
        args.log,
        centre_by_stem=centres,
        constant_centre=args.constant_centre,
        task_filter=args.task,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)

    if args.heatmap:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib 未安装，跳过热图", file=sys.stderr)
            return
        counts = report["centre_expert_counts"]
        centres_sorted = sorted(counts.keys())
        experts = sorted({m for d in counts.values() for m in d})
        mat = np.zeros((len(centres_sorted), len(experts)))
        for i, c in enumerate(centres_sorted):
            for j, e in enumerate(experts):
                mat[i, j] = counts[c].get(e, 0)
        fig, ax = plt.subplots(figsize=(max(6, len(experts) * 0.8), max(4, len(centres_sorted) * 0.35)))
        im = ax.imshow(mat, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(experts)), experts, rotation=45, ha="right")
        ax.set_yticks(range(len(centres_sorted)), centres_sorted)
        ax.set_title("Centre × selected expert (LLM-invoked)")
        fig.colorbar(im, ax=ax)
        args.heatmap.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.heatmap, dpi=150)
        plt.close(fig)
        print(f"wrote heatmap {args.heatmap}")


if __name__ == "__main__":
    main()
