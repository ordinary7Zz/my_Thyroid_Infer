#!/usr/bin/env python3
"""Convert run_all.py outputs under results/b2/<cohort>/ to workspace b2_exports/."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import date
from pathlib import Path

import yaml

B2_ROOT = Path(__file__).resolve().parent
INFER_ROOT = B2_ROOT.parent
WORKSPACE_ROOT = INFER_ROOT.parent

RUNALL_TO_B2_ID = {
    "transunet": "TransUNet",
    "medsam2": "MedSAM2",
    "medsegx": "MedSegX",
    "ultrafedfm": "UltraFedFM",
    "biomedclip": "BiomedCLIP",
    "medsiglip": "MedSigLIP",
    "repvit": "RepViT",
    "lsnet": "LSNet",
    "resnet50": "ResNet-50",
}

# Segmentation external-test model inputs (manifest inference_input codes)
SEG_INFERENCE_BY_MODEL = {
    "MedSAM2": "image_full_box_prompt",
    "MedSegX": "image_full_box_prompt",
    "TransUNet": "image_only",
    "UltraFedFM": "image_only",
}

JSON_CLS_MODELS = frozenset({"repvit", "lsnet", "resnet50"})

TASK_RUNALL = {"segmentation": "nodule", "classification": "binary"}

# B2 export dir id: {yaml_cohort}_{modality}_{target} (see docs/B2_基线模型逐图导出与Cluster_CI要求.md)
EXPORT_TASK_SUFFIX = {
    "segmentation": ("seg", "nodule"),
    "classification": ("cls", "bm"),
}


def export_cohort_id(yaml_cohort_id: str, task: str) -> str:
    modality, target = EXPORT_TASK_SUFFIX[task]
    return f"{yaml_cohort_id}_{modality}_{target}"


def _abs_data(path: str, data_root: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(data_root) / path)


def _latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _build_stem_to_name(image_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not image_dir.is_dir():
        return mapping
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            mapping[p.stem] = p.name
    return mapping


def _resolve_image_name(raw: str, stem_map: dict[str, str]) -> str:
    raw = raw.strip()
    if "." in Path(raw).name and Path(raw).suffix:
        return raw
    if raw in stem_map:
        return stem_map[raw]
    return raw


def _canonical_label_key(filename: str) -> str:
    """Align inference filenames with label JSON (e.g. MedSigLIP `*-checkpoint.jpg`)."""
    p = Path(filename.strip())
    stem = p.stem.replace("-checkpoint", "")
    return f"{stem}{p.suffix}" if p.suffix else stem


def _parse_true_label(value) -> int:
    if value is None or value == "":
        raise ValueError("missing y_true")
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    if s in ("0", "1"):
        return int(s)
    if s.lower() in ("benign", "良性"):
        return 0
    if s.lower() in ("malignant", "恶性", "1"):
        return 1
    return int(float(s))


def load_classification_label_map(label_file: str | Path, label_field: str = "malignancy") -> dict[str, int]:
    path = Path(label_file)
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("images") or raw.get("data") or list(raw.values())
        if not isinstance(items, list):
            items = []
    else:
        items = raw
    out: dict[str, int] = {}
    for rec in items:
        if not isinstance(rec, dict):
            continue
        fn = rec.get("filename") or rec.get("image_name") or rec.get("image")
        if not fn:
            continue
        name = Path(str(fn)).name
        out[name] = _parse_true_label(rec.get(label_field, rec.get("malignancy", rec.get("label"))))
    return out


def _score_malignant_from_row(row: dict) -> float:
    if "prob_1" in row and row["prob_1"] != "":
        return float(row["prob_1"])
    for key in ("score_malignant", "mal_prob", "prob_malignant"):
        if key in row and row[key] != "":
            return float(row[key])
    raise KeyError("no malignant probability column in CSV row")


def _latest_prediction_json(task_dir: Path) -> Path | None:
    if not task_dir.is_dir():
        return None
    pred_only = [p for p in task_dir.glob("*predictions.json") if p.is_file()]
    if pred_only:
        return sorted(pred_only, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    # Fallback: any JSON except summary/metrics sidecars
    skip = {"manifest.json"}
    candidates = [
        p
        for p in task_dir.glob("*.json")
        if p.is_file() and p.name not in skip and "metrics" not in p.name.lower()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def classification_jsonl_from_sample_json(json_path: Path) -> list[dict]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [raw]
    out: list[dict] = []
    for rec in raw:
        if rec.get("record_type") not in (None, "sample"):
            continue
        name = rec.get("image_name") or Path(rec.get("image_file", "")).name
        y_true = _parse_true_label(rec.get("true_label"))
        if "prob_class_1" in rec:
            score = float(rec["prob_class_1"])
        elif "score_malignant" in rec:
            score = float(rec["score_malignant"])
        else:
            raise KeyError(f"no probability in record: {rec}")
        out.append(
            {
                "image_name": name,
                "y_true": y_true,
                "score_malignant": round(score, 6),
            }
        )
    return out


def classification_jsonl_from_csv(
    csv_path: Path,
    stem_map: dict[str, str],
    label_map: dict[str, int] | None = None,
) -> list[dict]:
    out: list[dict] = []
    skipped = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row.get("filename") or row.get("image_name", "")
            name = _resolve_image_name(raw_name, stem_map)
            canon = _canonical_label_key(name)
            y_raw = row.get("true_label", row.get("y_true"))
            if y_raw is None or str(y_raw).strip() == "":
                if label_map:
                    y_true = label_map.get(name)
                    if y_true is None:
                        y_true = label_map.get(canon)
                else:
                    y_true = None
                if y_true is None:
                    skipped += 1
                    continue
            else:
                y_true = _parse_true_label(y_raw)
            if name != canon and label_map and canon in label_map:
                name = canon
            score = _score_malignant_from_row(row)
            out.append(
                {
                    "image_name": name,
                    "y_true": y_true,
                    "score_malignant": round(score, 6),
                }
            )
    if skipped:
        print(f"  [warn] {csv_path.name}: skipped {skipped} rows without label (not in CSV or label JSON)")
    if out:
        by_name: dict[str, dict] = {}
        for rec in out:
            by_name[rec["image_name"]] = rec
        if len(by_name) < len(out):
            print(
                f"  [warn] {csv_path.name}: deduped {len(out) - len(by_name)} "
                "duplicate image_name rows (kept last)"
            )
        out = list(by_name.values())
    return out


def segmentation_jsonl_from_log(log_path: Path, stem_map: dict[str, str]) -> list[dict]:
    text = log_path.read_text(encoding="utf-8")
    marker = "--- Per-Sample Metrics ---"
    if marker not in text:
        raise ValueError(f"No per-sample section in {log_path}")
    section = text.split(marker, 1)[1].strip()
    lines = [ln for ln in section.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    out: list[dict] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        stem, dice_s = parts[0], parts[1]
        name = _resolve_image_name(stem, stem_map)
        rec: dict = {"image_name": name, "dice": round(float(dice_s), 6)}
        if len(parts) >= 3 and parts[2].strip():
            rec["hd95"] = round(float(parts[2]), 6)
        out.append(rec)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def manifest_for(
    *,
    model_b2_id: str,
    task: str,
    cohort_id: str,
    cohort_cfg: dict,
    b2_meta: dict,
    n_images: int,
    checkpoint: str | None,
    label_file: str | None,
    image_dir: str | None,
) -> dict:
    if task == "classification":
        inf_in = b2_meta.get("classification_inference_input", "image_only")
    else:
        inf_in = SEG_INFERENCE_BY_MODEL.get(model_b2_id, "image_only")
    modality, target = EXPORT_TASK_SUFFIX[task]
    m: dict = {
        "model_id": model_b2_id,
        "task": task,
        "cohort_id": cohort_id,
        "cohort_yaml_id": cohort_cfg.get("_yaml_cohort_id"),
        "task_modality": modality,
        "task_target": target,
        "data_root": b2_meta["data_root"],
        "inference_input": inf_in,
        "n_images": n_images,
        "created_at": date.today().isoformat(),
        "source": "my_Thyroid_Infer/run_all.py",
    }
    if checkpoint:
        m["checkpoint"] = checkpoint
    if label_file:
        m["label_file"] = label_file
    if image_dir:
        m["image_list"] = image_dir
    if cohort_cfg.get("patient_level"):
        m["patient_level"] = True
    return m


def _remove_legacy_export_dirs(
    exports_root: Path, model_b2_id: str, yaml_cohort_id: str, export_id: str
) -> None:
    """Drop pre-B naming dirs so scan-all does not double-count exports."""
    legacy_ids = {
        yaml_cohort_id,
        f"{yaml_cohort_id}_seg",
        f"{yaml_cohort_id}_cls",
    }
    legacy_ids.discard(export_id)
    for leg_id in sorted(legacy_ids):
        legacy = exports_root / model_b2_id / leg_id
        if legacy.is_dir():
            shutil.rmtree(legacy)
            print(f"  [cleanup] removed legacy export dir {legacy}")


def convert_cohort(
    cohort_id: str,
    results_root: Path,
    exports_root: Path,
    cohort_cfg: dict,
    b2_meta: dict,
    models_filter: set[str] | None,
) -> list[Path]:
    cohort_cfg = {**cohort_cfg, "_yaml_cohort_id": cohort_id}
    data_root = b2_meta["data_root"]
    written: list[Path] = []
    stem_map: dict[str, str] = {}

    if cohort_cfg.get("classification"):
        cls = cohort_cfg["classification"]
        img_dir = Path(_abs_data(cls["images"], data_root))
        stem_map = _build_stem_to_name(img_dir) or stem_map
    if cohort_cfg.get("segmentation"):
        seg = cohort_cfg["segmentation"]
        img_dir = Path(_abs_data(seg["images"], data_root))
        stem_map = _build_stem_to_name(img_dir) or stem_map

    cls_label_map: dict[str, int] | None = None
    if cohort_cfg.get("classification"):
        cls = cohort_cfg["classification"]
        label_field = cls.get("label_field", "malignancy")
        cls_label_map = load_classification_label_map(
            _abs_data(cls["label_file"], data_root),
            label_field=label_field,
        )

    for task in cohort_cfg.get("tasks", []):
        runall_task = TASK_RUNALL[task]
        if task == "segmentation":
            model_keys = b2_meta.get("seg_models", [])
            image_dir = _abs_data(cohort_cfg["segmentation"]["images"], data_root)
            label_file = None
        else:
            model_keys = b2_meta.get("cls_models", [])
            image_dir = _abs_data(cohort_cfg["classification"]["images"], data_root)
            label_file = _abs_data(cohort_cfg["classification"]["label_file"], data_root)

        for model_key in model_keys:
            if models_filter and model_key not in models_filter:
                continue
            model_b2_id = RUNALL_TO_B2_ID.get(model_key, model_key)
            task_dir = results_root / runall_task / model_key
            if task == "classification":
                if model_key in JSON_CLS_MODELS:
                    pred_file = _latest_prediction_json(task_dir)
                    if pred_file is None:
                        print(f"  [skip] no JSON: {task_dir}")
                        continue
                    rows = classification_jsonl_from_sample_json(pred_file)
                else:
                    pred_file = _latest_file(task_dir, "predictions*.csv") or _latest_file(
                        task_dir, "predictions.csv"
                    )
                    if pred_file is None:
                        print(f"  [skip] no CSV: {task_dir}")
                        continue
                    rows = classification_jsonl_from_csv(pred_file, stem_map, cls_label_map)
            else:
                log_file = _latest_file(task_dir, "metrics*.log") or _latest_file(
                    task_dir, "metrics.log"
                )
                if log_file is None:
                    print(f"  [skip] no metrics log: {task_dir}")
                    continue
                rows = segmentation_jsonl_from_log(log_file, stem_map)

            export_id = export_cohort_id(cohort_id, task)
            export_dir = exports_root / model_b2_id / export_id
            _remove_legacy_export_dirs(exports_root, model_b2_id, cohort_id, export_id)
            pred_path = export_dir / "predictions.jsonl"
            write_jsonl(pred_path, rows)

            ckpt_path = None
            weights_yaml = results_root / "_b2_weights.json"
            if weights_yaml.is_file():
                wmap = json.loads(weights_yaml.read_text(encoding="utf-8"))
                ckpt_path = wmap.get(runall_task, {}).get(model_key)

            manifest = manifest_for(
                model_b2_id=model_b2_id,
                task=task,
                cohort_id=export_id,
                cohort_cfg=cohort_cfg,
                b2_meta=b2_meta,
                n_images=len(rows),
                checkpoint=ckpt_path,
                label_file=label_file,
                image_dir=image_dir,
            )
            manifest_path = export_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  wrote {pred_path} ({len(rows)} lines)")
            written.append(pred_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="run_all results → b2_exports")
    parser.add_argument("--cohorts", default=None, help="Comma-separated cohort ids (default: all in yaml)")
    parser.add_argument("--b2-config", type=Path, default=B2_ROOT / "cohorts.yaml")
    parser.add_argument("--results-root", type=Path, default=None, help="Override results root for single cohort")
    parser.add_argument("--cohort-id", default=None, help="With --results-root, single cohort id")
    parser.add_argument("--models", default=None, help="Comma-separated run_all model keys filter")
    args = parser.parse_args()

    meta = yaml.safe_load(args.b2_config.read_text(encoding="utf-8"))
    exports_root = Path(meta.get("exports_root", "../../b2_exports"))
    if not exports_root.is_absolute():
        exports_root = (B2_ROOT / exports_root).resolve()

    models_filter = None
    if args.models:
        models_filter = {m.strip() for m in args.models.split(",") if m.strip()}

    cohort_ids = list(meta["cohorts"].keys())
    if args.cohorts:
        cohort_ids = [c.strip() for c in args.cohorts.split(",") if c.strip()]

    if args.results_root and args.cohort_id:
        cohort_ids = [args.cohort_id]
        results_map = {args.cohort_id: args.results_root.resolve()}
    else:
        results_base = Path(meta.get("results_root", "./results/b2"))
        if not results_base.is_absolute():
            results_base = (INFER_ROOT / results_base).resolve()
        results_map = {cid: results_base / cid for cid in cohort_ids}

    for cid in cohort_ids:
        print(f"=== convert {cid} ===")
        convert_cohort(
            cid,
            results_map[cid],
            exports_root,
            meta["cohorts"][cid],
            meta,
            models_filter,
        )


if __name__ == "__main__":
    main()
