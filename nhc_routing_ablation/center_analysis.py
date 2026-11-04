"""NHC B3：对七策略输出目录跑 B7 中心性能 + 异质性（使用 cascade_engine 内 B7 模块）。"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from paths import get_engine_root, STRATEGY_ROWS  # noqa: E402

# 单权重腺体 B7（非七策略路由）；与 heterogeneity_by_strategy.csv 中 routing_mode=gland 对应
GLAND_CENTER_SUBDIR = "gland"
LEGACY_GLAND_CENTER_SUBDIR = "gland_smoke"


def _ensure_engine_imports():
    root = str(get_engine_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def build_stem_center_lookup(
    *,
    label_file: Optional[Path],
    name_mapping_file: Optional[Path],
    center_manifest: Optional[Path],
    extra_image_names: Optional[list[str]] = None,
) -> dict[str, str]:
    """stem → center_id（labels、manifest、匿名映射、THYB 文件名解析）。"""
    _ensure_engine_imports()
    from pipeline.revision.b7.nhc_stem import parse_nhc_stem
    from shared.center_manifest import build_center_lookup_from_labels, load_center_lookup

    lookup: dict[str, str] = {}
    if label_file and label_file.is_file():
        lookup.update(build_center_lookup_from_labels(label_file))
    lookup.update(load_center_lookup(str(center_manifest) if center_manifest else None))

    if name_mapping_file and name_mapping_file.is_file():
        mapping = json.loads(name_mapping_file.read_text(encoding="utf-8"))
        if isinstance(mapping, dict):
            for orig, anon in mapping.items():
                anon_stem = Path(str(anon)).stem
                orig_stem = Path(str(orig)).stem
                parsed = parse_nhc_stem(orig_stem)
                cid = parsed.get("center_id")
                if cid:
                    lookup[anon_stem] = cid

    for name in extra_image_names or []:
        stem = Path(name).stem
        if stem in lookup:
            continue
        parsed = parse_nhc_stem(stem)
        cid = parsed.get("center_id")
        if cid:
            lookup[stem] = cid

    return lookup


def _inject_center_into_results(results: list[dict], lookup: dict[str, str]) -> None:
    _ensure_engine_imports()
    from pipeline.revision.b7.nhc_stem import parse_nhc_stem
    from shared.center_manifest import resolve_center_id

    for r in results:
        if r.get("center_id"):
            continue
        name = r.get("image_name", "")
        stem = Path(name).stem
        cid = resolve_center_id(name, lookup) or lookup.get(stem)
        if not cid:
            cid = parse_nhc_stem(stem).get("center_id")
        if cid:
            r["center_id"] = cid


def analyze_strategy_output_dir(
    strategy_dir: Path,
    *,
    label_file: Path,
    out_dir: Path,
    name_mapping_file: Optional[Path] = None,
    center_manifest: Optional[Path] = None,
    center_descriptive: Optional[dict[str, Any]] = None,
    g3_csv: Optional[Path] = None,
    meta_covariates: str = "",
) -> dict[str, Any]:
    """对单个 B3 策略目录（含 results.json + per_image_* jsonl）生成 B7 报告与 CSV。"""
    _ensure_engine_imports()
    from pipeline.revision.b7.per_image_enrich import enrich_output_dir_jsonl
    from pipeline.revision.b7.report import build_b7_report

    results_path = strategy_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"缺少 {results_path}")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    image_names = [r.get("image_name", "") for r in results if r.get("image_name")]
    lookup = build_stem_center_lookup(
        label_file=label_file,
        name_mapping_file=name_mapping_file,
        center_manifest=center_manifest,
        extra_image_names=image_names,
    )
    _inject_center_into_results(results, lookup)

    enriched = enrich_output_dir_jsonl(
        strategy_dir,
        results=results,
        manifest_lookup=lookup,
    )

    descriptive = center_descriptive
    if descriptive is None and label_file.is_file():
        from shared.center_manifest import summarize_centers_from_labels

        descriptive = summarize_centers_from_labels(str(label_file))
        if g3_csv and g3_csv.is_file():
            descriptive = _merge_g3_into_descriptive(descriptive, g3_csv)

    cov_keys = [k.strip() for k in meta_covariates.split(",") if k.strip()]
    report = build_b7_report(
        strategy_dir,
        results=results,
        center_descriptive=descriptive,
        meta_covariate_keys=cov_keys or None,
    )
    report["strategy_dir"] = str(strategy_dir.resolve())
    report["center_id_enrichment"] = {"lookup_size": len(lookup), "jsonl_rows": enriched}

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "b7_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    legacy = {
        "n_centers": report["center_performance"]["n_centers"],
        "macro_mean_dice": report["center_performance"].get("macro_mean_dice"),
        "macro_mean_auroc": report["center_performance"].get("macro_mean_auroc"),
        "centers": report["center_performance"]["centers"],
        "heterogeneity": report.get("heterogeneity"),
    }
    (out_dir / "center_metrics.json").write_text(
        json.dumps(legacy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _export_b7_csvs(report, out_dir)
    return report


def _merge_g3_into_descriptive(descriptive: dict, g3_csv: Path) -> dict:
    import copy

    out = copy.deepcopy(descriptive)
    with open(g3_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("center_id") or row.get("centre_id")
            if not cid:
                continue
            block = out.setdefault("centers", {}).setdefault(str(cid), {"n_images": 0})
            for k in (
                "n_patients",
                "n_nodules",
                "scanner",
                "scanner_vendor",
                "hospital",
                "province",
            ):
                if row.get(k) not in (None, ""):
                    block[k] = row[k]
    return out


def _export_b7_csvs(report: dict[str, Any], out_dir: Path) -> None:
    """内联 export_b7_tables（避免七策略 × 子进程 revision bootstrap 过慢）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    perf = report.get("center_performance", {}).get("centers", {})
    gland = (report.get("gland_center_performance") or {}).get("centers", {})
    within_nodule = (report.get("within_center") or {}).get("nodule_dice", {})
    desc_centers = (report.get("center_descriptive") or {}).get("centers", {})

    perf_path = out_dir / "nhc_center_performance.csv"
    with open(perf_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "center_id",
            "n_images_seg",
            "nodule_dice_mean",
            "nodule_hd95_mean",
            "n_cls",
            "auroc",
            "auprc",
            "auroc_estimable",
            "gland_dice_mean",
            "gland_n_images",
            "nodule_dice_std_within",
            "nodule_dice_cv_percent",
        ])
        all_ids = sorted(set(perf) | set(gland) | set(within_nodule))
        for cid in all_ids:
            p = perf.get(cid, {})
            g = gland.get(cid, {})
            wn = within_nodule.get(cid, {})
            w.writerow([
                cid,
                p.get("n_images_seg"),
                p.get("dice_mean"),
                p.get("hd95_mean"),
                p.get("n_cls"),
                p.get("auroc"),
                p.get("auprc"),
                p.get("auroc_estimable"),
                g.get("dice_mean"),
                g.get("n_images"),
                wn.get("std"),
                wn.get("cv_percent"),
            ])

    hetero_path = out_dir / "nhc_heterogeneity_summary.csv"
    hetero = report.get("heterogeneity") or {}
    with open(hetero_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_metric", "i2_percent", "tau2", "n_groups", "q", "df"])
        for name, block in hetero.items():
            if not block:
                continue
            w.writerow([
                name,
                block.get("i2_percent"),
                block.get("tau2"),
                block.get("n_groups"),
                block.get("q"),
                block.get("df"),
            ])

    if desc_centers:
        desc_path = out_dir / "nhc_center_descriptive_merged.csv"
        keys = sorted({k for c in desc_centers.values() for k in c})
        with open(desc_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["center_id"] + keys, extrasaction="ignore")
            w.writeheader()
            for cid, block in sorted(desc_centers.items()):
                w.writerow({"center_id": cid, **block})


def summarize_centers_from_labels_resolved(
    label_file: Path,
    *,
    stem_to_center: dict[str, str],
) -> dict[str, Any]:
    """
    与 B7 summarize_centers_from_labels 相同，但 center 来自 stem_to_center
    （含 name_mapping → 原始 THYB 名 → center_id，而非 labels 内 center_id 字段）。
    """
    data = json.loads(label_file.read_text(encoding="utf-8"))
    centers: dict[str, dict[str, Any]] = {}

    def _bump(stem: str, malignancy: Any = None) -> None:
        _ensure_engine_imports()
        from pipeline.revision.b7.nhc_stem import parse_nhc_stem

        cid = stem_to_center.get(stem) or parse_nhc_stem(stem).get("center_id") or "unknown"
        c = centers.setdefault(str(cid), {"n_images": 0, "n_malignant": 0, "n_labeled": 0})
        c["n_images"] += 1
        if malignancy is not None:
            c["n_labeled"] += 1
            mal = str(malignancy).lower()
            if mal in ("1", "1.0", "恶性", "malignant", "m", "true"):
                c["n_malignant"] += 1

    if isinstance(data, dict):
        for key, val in data.items():
            stem = Path(str(key)).stem
            mal = val.get("malignancy") if isinstance(val, dict) else None
            _bump(stem, mal)
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            name = None
            for name_key in ("image_name", "filename", "file", "path", "stem", "id"):
                if name_key in row and row[name_key]:
                    name = Path(str(row[name_key])).stem
                    break
            if name:
                mal = row.get("malignancy")
                if mal is None:
                    mal = row.get("label")
                _bump(name, mal)

    for c in centers.values():
        n_lab = c["n_labeled"]
        c["malignant_ratio"] = (c["n_malignant"] / n_lab) if n_lab else None

    known = [k for k in centers if k != "unknown"]
    return {
        "n_images": sum(c["n_images"] for c in centers.values()),
        "n_centers": len(known) if known else len(centers),
        "centers": centers,
        "center_id_note": (
            "labels 无 center_id 字段时，通过 name_mapping.json 反查原始 THYB 文件名解析中心；"
            "匿名文件名本身不含 THYB_S_* 明文。"
        ),
    }


def run_center_analysis_all_strategies(
    output_root: Path,
    *,
    label_file: Path,
    name_mapping_file: Optional[Path] = None,
    center_manifest: Optional[Path] = None,
    g3_csv: Optional[Path] = None,
) -> Path:
    """对 STRATEGY_ROWS 中每个 mode 写 output_root/_center_analysis/<mode>/。"""
    center_root = output_root / "_center_analysis"
    center_root.mkdir(parents=True, exist_ok=True)

    descriptive: Optional[dict] = None
    stem_lookup: dict[str, str] = {}
    if label_file.is_file():
        all_names: list[str] = []
        data = json.loads(label_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict) and row.get("filename"):
                    all_names.append(str(row["filename"]))
        stem_lookup = build_stem_center_lookup(
            label_file=label_file,
            name_mapping_file=name_mapping_file,
            center_manifest=center_manifest,
            extra_image_names=all_names,
        )
        descriptive = summarize_centers_from_labels_resolved(
            label_file,
            stem_to_center=stem_lookup,
        )
        if g3_csv and g3_csv.is_file():
            descriptive = _merge_g3_into_descriptive(descriptive, g3_csv)
        (center_root / "center_descriptive_from_labels.json").write_text(
            json.dumps(descriptive, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary_rows: list[dict[str, Any]] = []
    hetero_long: list[dict[str, Any]] = []
    perf_summary: list[dict[str, Any]] = []
    for row in STRATEGY_ROWS:
        mode = row["mode"]
        strategy_dir = output_root / mode
        out_dir = center_root / mode
        if not (strategy_dir / "results.json").is_file():
            print(
                f"\n>> B7 中心分析: 策略 ({row['id']}) {mode} — 跳过（无 results.json）",
                flush=True,
            )
            continue
        print(f"\n>> B7 中心分析: 策略 ({row['id']}) {mode}", flush=True)
        report = analyze_strategy_output_dir(
            strategy_dir,
            label_file=label_file,
            out_dir=out_dir,
            name_mapping_file=name_mapping_file,
            center_manifest=center_manifest,
            center_descriptive=descriptive,
            g3_csv=None,
        )
        hetero = report.get("heterogeneity") or {}
        nd = hetero.get("nodule_dice") or {}
        cl = hetero.get("classification_score") or {}
        perf = report.get("center_performance") or {}
        summary_rows.append(
            {
                "strategy_id": row["id"],
                "routing_mode": mode,
                "label": row["label"],
                "n_centers": perf.get("n_centers"),
                "macro_mean_dice": perf.get("macro_mean_dice"),
                "macro_mean_auroc": perf.get("macro_mean_auroc"),
                "nodule_dice_i2_percent": nd.get("i2_percent"),
                "nodule_dice_tau2": nd.get("tau2"),
                "nodule_dice_n_groups": nd.get("n_groups"),
                "cls_score_i2_percent": cl.get("i2_percent") if cl else None,
                "cls_score_tau2": cl.get("tau2") if cl else None,
                "cls_score_n_groups": cl.get("n_groups") if cl else None,
            }
        )
        perf_summary.append(
            {
                "strategy_id": row["id"],
                "routing_mode": mode,
                "n_centers": perf.get("n_centers"),
                "macro_mean_dice": perf.get("macro_mean_dice"),
                "macro_mean_auroc": perf.get("macro_mean_auroc"),
            }
        )
        for metric_key, block in hetero.items():
            if not block or block.get("i2_percent") is None:
                continue
            hetero_long.append(
                {
                    "routing_mode": mode,
                    "strategy_id": row["id"],
                    "metric": metric_key,
                    "i2_percent": block.get("i2_percent"),
                    "tau2": block.get("tau2"),
                    "q": block.get("q"),
                    "df": block.get("df"),
                    "n_groups": block.get("n_groups"),
                }
            )
        if nd.get("i2_percent") is not None:
            print(
                f"   结节 Dice I² ≈ {nd['i2_percent']:.1f}%, "
                f"τ² ≈ {nd.get('tau2')}; "
                f"中心数 {perf.get('n_centers')}",
                flush=True,
            )

    if hetero_long:
        hetero_path = center_root / "heterogeneity_by_strategy.csv"
        fields = list(hetero_long[0].keys())
        with open(hetero_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(hetero_long)
        print(f"\n✓ 七策略异质性汇总: {hetero_path}", flush=True)

    if perf_summary:
        perf_path = center_root / "center_performance_by_strategy.csv"
        fields = list(perf_summary[0].keys())
        with open(perf_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(perf_summary)
        print(f"✓ 七策略宏平均性能: {perf_path}", flush=True)

    if summary_rows:
        wide_path = center_root / "center_analysis_summary_wide.csv"
        fields = list(summary_rows[0].keys())
        with open(wide_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary_rows)

    index = {
        "strategies": [row["mode"] for row in STRATEGY_ROWS],
        "per_strategy_dir": {row["mode"]: str(center_root / row["mode"]) for row in STRATEGY_ROWS},
    }
    (center_root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return center_root


def run_gland_center_analysis(
    output_root: Path,
    *,
    gt_gland_dir: Path,
    pred_gland_dir: Path,
    label_file: Path,
    name_mapping_file: Optional[Path] = None,
    center_manifest: Optional[Path] = None,
    reference_mode: str = "best_single",
    pred_is_placeholder: bool = False,
) -> Path:
    """
    单权重腺体 B7：``_center_analysis/gland/``（不重复七策略）。
    ``reference_mode`` 仅用于拷贝 B3 的 results/seg/cls 元数据；pred 来自 ``pred_gland_dir``。
    """
    _ensure_engine_imports()
    from pipeline.revision.b7.gland_metrics import compute_gland_per_image, write_gland_jsonl
    from pipeline.revision.b7.per_image_enrich import enrich_output_dir_jsonl
    from pipeline.revision.b7.report import build_b7_report

    ref_dir = output_root / reference_mode
    if not (ref_dir / "results.json").is_file():
        raise FileNotFoundError(f"参考策略缺少 results.json: {ref_dir}")

    center_root = output_root / "_center_analysis"
    gland_dir = center_root / GLAND_CENTER_SUBDIR
    gland_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads((ref_dir / "results.json").read_text(encoding="utf-8"))
    stems = [Path(r.get("image_name", "")).stem for r in results if r.get("image_name")]

    if not gt_gland_dir.is_dir():
        raise FileNotFoundError(f"腺体 GT 目录不存在: {gt_gland_dir}")
    if not pred_gland_dir.is_dir():
        raise FileNotFoundError(f"腺体 pred 目录不存在: {pred_gland_dir}")

    print(
        f"\n>> 腺体分割 B7（单路径 _center_analysis/{GLAND_CENTER_SUBDIR}/，参考 B3 子目录 {reference_mode}）",
        flush=True,
    )
    print(f"   GT gland:   {gt_gland_dir}", flush=True)
    print(f"   Pred gland: {pred_gland_dir}", flush=True)
    if pred_is_placeholder:
        print("   （当前 pred 为占位：结节 mask；接上 TGVideo 推理后改 pred 目录即可）", flush=True)

    gland_rows = compute_gland_per_image(
        gt_gland_dir=gt_gland_dir,
        pred_gland_dir=pred_gland_dir,
        image_stems=stems,
    )
    write_gland_jsonl(gland_dir, gland_rows)
    print(f"   per_image_gland_metrics.jsonl: {len(gland_rows)} 条", flush=True)

    for name in ("results.json", "per_image_seg_metrics.jsonl", "per_image_cls_metrics.jsonl"):
        src = ref_dir / name
        if src.is_file():
            shutil.copy2(src, gland_dir / name)

    lookup = build_stem_center_lookup(
        label_file=label_file,
        name_mapping_file=name_mapping_file,
        center_manifest=center_manifest,
        extra_image_names=[r.get("image_name", "") for r in results if r.get("image_name")],
    )
    _inject_center_into_results(results, lookup)
    (gland_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    enrich_output_dir_jsonl(gland_dir, results=results, manifest_lookup=lookup)

    descriptive = summarize_centers_from_labels_resolved(label_file, stem_to_center=lookup)
    report = build_b7_report(
        gland_dir,
        results=results,
        center_descriptive=descriptive,
        meta_covariate_keys=None,
    )
    report["gland_run_meta"] = {
        "pred_gland_dir": str(pred_gland_dir.resolve()),
        "gt_gland_dir": str(gt_gland_dir.resolve()),
        "reference_routing_strategy": reference_mode,
        "pred_is_nodule_mask_placeholder": pred_is_placeholder,
    }
    (gland_dir / "b7_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _export_b7_csvs(report, gland_dir)

    gperf = report.get("gland_center_performance") or {}
    hg = (report.get("heterogeneity") or {}).get("gland_dice") or {}
    print(
        f"   腺体中心块: n_centers={gperf.get('n_centers')}, "
        f"gland I²={hg.get('i2_percent')}",
        flush=True,
    )
    print(f"✓ 腺体 B7 产物: {gland_dir}", flush=True)
    _append_gland_heterogeneity_to_strategy_csv(center_root, report=report)
    _update_center_analysis_index_gland(center_root, gland_dir)
    return gland_dir


def _update_center_analysis_index_gland(center_root: Path, gland_dir: Path) -> None:
    index_path = center_root / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {
            "strategies": [row["mode"] for row in STRATEGY_ROWS],
            "per_strategy_dir": {
                row["mode"]: str(center_root / row["mode"]) for row in STRATEGY_ROWS
            },
        }
    index[GLAND_CENTER_SUBDIR] = str(gland_dir.resolve())
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


# 旧名（文档/脚本曾用 smoke）；新代码请用 run_gland_center_analysis
run_gland_smoke_link_test = run_gland_center_analysis


def _append_gland_heterogeneity_to_strategy_csv(
    center_root: Path,
    *,
    report: dict[str, Any],
) -> None:
    """单权重腺体：把 gland_dice 异质性并入 _center_analysis/heterogeneity_by_strategy.csv。"""
    block = (report.get("heterogeneity") or {}).get("gland_dice")
    if not block or block.get("i2_percent") is None:
        return
    row = {
        "routing_mode": "gland",
        "strategy_id": "",
        "metric": "gland_dice",
        "i2_percent": block.get("i2_percent"),
        "tau2": block.get("tau2"),
        "q": block.get("q"),
        "df": block.get("df"),
        "n_groups": block.get("n_groups"),
    }
    hetero_path = center_root / "heterogeneity_by_strategy.csv"
    fields = list(row.keys())
    existing: list[dict[str, Any]] = []
    if hetero_path.is_file():
        with open(hetero_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or fields
            existing = [
                r
                for r in reader
                if not (
                    r.get("routing_mode") == "gland"
                    and r.get("metric") == "gland_dice"
                )
            ]
    existing.append({k: row.get(k) for k in fields})
    center_root.mkdir(parents=True, exist_ok=True)
    with open(hetero_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(existing)
    print(f"✓ 腺体异质性已写入: {hetero_path} (routing_mode=gland)", flush=True)
