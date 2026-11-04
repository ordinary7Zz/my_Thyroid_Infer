"""聚合多 routing_mode 的 b3_metrics.json → CSV + manifest 校验 + 配对差分。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from pipeline.revision.b3.intermediate_manifest import manifests_match


def load_metrics_files(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("b3_metrics.json")):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        m["_source_dir"] = str(path.parent)
        rows.append(m)
    return rows


def metrics_to_table(rows: list[dict]) -> list[dict]:
    table = []
    for m in rows:
        mode = m.get("routing_mode", "?")
        seg = m.get("segmentation") or {}
        cls = m.get("classification") or {}
        cls_pb = m.get("classification_path_b") or m.get("metrics_path_b_only") or {}
        table.append({
            "routing_mode": mode,
            "seg_routing_mode": m.get("seg_routing_mode"),
            "cls_routing_mode": m.get("cls_routing_mode"),
            "source_dir": m.get("_source_dir", ""),
            "dice_mean": seg.get("dice_mean"),
            "hd95_mean": seg.get("hd95_mean"),
            "auroc": cls.get("auroc"),
            "auprc": cls.get("auprc"),
            "auroc_path_b": cls_pb.get("auroc"),
            "auprc_path_b": cls_pb.get("auprc"),
            "accuracy_path_b": cls_pb.get("accuracy"),
            "n_path_b": cls_pb.get("n") or m.get("n_disagreement"),
        })
    return table


def write_csv(table: list[dict], out_path: Path) -> None:
    if not table:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(table[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)


def validate_manifests(dirs: list[Path], reference: Optional[Path]) -> list[dict]:
    """若提供 reference manifest，检查各 mode 目录的 intermediate_manifest 与之一致。"""
    if reference is None:
        return []
    with open(reference, "r", encoding="utf-8") as f:
        ref_doc = json.load(f)
    reports = []
    for d in dirs:
        mpath = d / "intermediate_manifest.json"
        if not mpath.is_file():
            reports.append({"dir": str(d), "ok": False, "error": "missing manifest"})
            continue
        with open(mpath, "r", encoding="utf-8") as f:
            doc = json.load(f)
        ok = manifests_match(ref_doc, doc)
        reports.append({"dir": str(d), "ok": ok})
    return reports


def paired_metric_diff(rows: list[dict], metric_key: str, a: str, b: str) -> Optional[dict]:
    """简单点估计差分（bootstrap 配对留待 B2/P2）。"""
    def pick(mode: str) -> Optional[float]:
        for m in rows:
            if m.get("routing_mode") != mode:
                continue
            if metric_key == "dice_mean":
                return (m.get("segmentation") or {}).get("dice_mean")
            if metric_key == "auroc":
                return (m.get("classification") or {}).get("auroc")
        return None

    va, vb = pick(a), pick(b)
    if va is None or vb is None:
        return None
    return {
        "metric": metric_key,
        "mode_a": a,
        "mode_b": b,
        "value_a": va,
        "value_b": vb,
        "diff_a_minus_b": va - vb,
        "note": "点估计差分；投稿前须换 B2 cluster bootstrap",
    }


def summarize_ablation_dirs(
    dirs: list[Path],
    summary_dir: Path,
    *,
    reference_manifest: Optional[Path] = None,
    paired_modes: Optional[tuple[str, str]] = None,
) -> dict:
    rows: list[dict] = []
    for d in dirs:
        p = Path(d) / "b3_metrics.json"
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                m = json.load(f)
            m["_source_dir"] = str(d)
            rows.append(m)

    summary_dir.mkdir(parents=True, exist_ok=True)
    table = metrics_to_table(rows)
    csv_path = summary_dir / "b3_routing_table.csv"
    write_csv(table, csv_path)

    manifest_report = validate_manifests(dirs, reference_manifest)
    if manifest_report:
        with open(summary_dir / "manifest_validation.json", "w", encoding="utf-8") as f:
            json.dump(manifest_report, f, ensure_ascii=False, indent=2)

    paired = {}
    if paired_modes and len(paired_modes) == 2:
        for key in ("dice_mean", "auroc"):
            diff = paired_metric_diff(rows, key, paired_modes[0], paired_modes[1])
            if diff:
                paired[key] = diff
    if paired:
        with open(summary_dir / "b3_paired_bootstrap.json", "w", encoding="utf-8") as f:
            json.dump({"paired_point_estimates": paired}, f, ensure_ascii=False, indent=2)

    return {"csv": str(csv_path), "n_modes": len(table)}


def main():
    parser = argparse.ArgumentParser(description="汇总 B3 ablation metrics")
    parser.add_argument("--root", required=True, help="包含各 mode 子目录的根路径")
    parser.add_argument("--out", default=None, help="汇总目录，默认 root/_summary")
    parser.add_argument(
        "--reference-manifest",
        default=None,
        help="基准 intermediate_manifest.json（校验各 mode 共享 seg/cls）",
    )
    parser.add_argument(
        "--paired",
        default=None,
        help="两 mode 点估计差分，如 soft_voting,static_rule",
    )
    args = parser.parse_args()
    root = Path(args.root)
    summary = Path(args.out) if args.out else root / "_summary"
    dirs = [
        p
        for p in root.iterdir()
        if p.is_dir()
        and p.name != "_summary"
        and not p.name.startswith(".")
    ]
    ref = Path(args.reference_manifest) if args.reference_manifest else None
    paired = None
    if args.paired:
        parts = [x.strip() for x in args.paired.split(",")]
        if len(parts) == 2:
            paired = (parts[0], parts[1])
    info = summarize_ablation_dirs(dirs, summary, reference_manifest=ref, paired_modes=paired)
    print(f"已写入 {info['csv']}（{info['n_modes']} modes）")


if __name__ == "__main__":
    main()
