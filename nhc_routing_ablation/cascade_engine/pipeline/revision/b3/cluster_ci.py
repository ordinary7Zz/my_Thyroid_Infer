"""B3 指标 CI：与 B2 相同的 cluster bootstrap（1000, seed 42）。

无 patient ID 的队列：cluster_id = image_name（一图一簇，与逐图 bootstrap 等价）。
ZJH-8K / ThyroidXL：使用 docs/revision/b2/cluster_maps/*.json。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pipeline.evaluate import _label_to_binary
from segmentation.metrics import compute_dice, compute_hd95
from shared.image_io import ImageIO

_REPO = Path(__file__).resolve().parents[3]
_MAP_DIR = _REPO / "docs" / "revision" / "b2" / "cluster_maps"

COHORT_CLUSTER_MAP: dict[str, str] = {
    "zjh8k": "zjh8k.json",
    "thyroidxl": "thyroidxl_test.json",
}

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_SEED = 42


def _load_b2_cluster_module():
    path = _REPO / "scripts" / "revision" / "b2" / "run_b2_cluster_metrics.py"
    spec = importlib.util.spec_from_file_location("run_b2_cluster_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_B2 = _load_b2_cluster_module()


def resolve_cluster_map(cohort_id: str) -> Optional[dict[str, str]]:
    key = cohort_id.lower().strip()
    if key.startswith("zjh8k"):
        key = "zjh8k"
    elif key.startswith("rjh7k"):
        key = "rjh7k"
    fname = COHORT_CLUSTER_MAP.get(key)
    if not fname:
        return None
    p = _MAP_DIR / fname
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def cohort_id_from_config_path(config_path: Path) -> str:
    return config_path.stem.lower()


def _seg_rows_dice_hd95(
    pipeline_dir: Path,
    gt_mask_dir: Path,
    cluster_map: dict[str, str] | None,
    image_io: ImageIO,
) -> list[dict]:
    npz_path = pipeline_dir / "intermediate" / "selected_masks.npz"
    if not npz_path.is_file():
        return []

    data = np.load(str(npz_path), allow_pickle=True)
    image_names = list(data["image_names"])
    masks = list(data["selected_masks"])
    gt_dir = Path(gt_mask_dir)
    gt_stem_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file()}

    rows: list[dict] = []
    for img_name, pred in zip(image_names, masks):
        stem = Path(img_name).stem
        gt_path = gt_stem_map.get(stem)
        if gt_path is None or not gt_path.exists():
            continue
        gt = image_io.binarize_mask(image_io.load_mask(gt_path))
        pred_mask = pred.astype(bool)
        dice = float(compute_dice(pred_mask, gt))
        hd95 = float(compute_hd95(pred_mask, gt))
        cid = (cluster_map or {}).get(img_name, img_name)
        row = {"image_name": img_name, "cluster_id": cid, "dice": dice}
        if not (np.isinf(hd95) or np.isnan(hd95)):
            row["hd95"] = hd95
        rows.append(row)
    return rows


def _cls_block_from_results(
    results: list[dict],
    cluster_map: dict[str, str] | None,
    *,
    subset_path_b: bool,
    n_bootstrap: int,
    seed: int,
) -> Optional[dict[str, Any]]:
    filtered = results
    if subset_path_b:
        filtered = [r for r in results if r.get("path") == "B"]

    rows = _B2._rows_from_results(filtered, cluster_map)
    n = len(rows)
    if n < 1:
        return None

    base: dict[str, Any] = {"n": n}
    y_true = [r["y_true"] for r in rows]
    y_pred = []
    for r in filtered:
        if "true_label" not in r or "final_label" not in r:
            continue
        tb = _label_to_binary(r["true_label"])
        pl = r["final_label"]
        pb = _label_to_binary(pl) if isinstance(pl, str) else pl
        if tb is not None and pb is not None:
            y_pred.append(int(pb))
    if y_pred:
        base["accuracy"] = float(np.mean(np.array(y_true) == np.array(y_pred)))

    if n < 2 or len(set(y_true)) < 2:
        return base

    auc, auc_lo, auc_hi, _ = _B2._cluster_bootstrap_auc(rows, n_bootstrap, seed, False)
    auprc, auprc_lo, auprc_hi = _B2._cluster_bootstrap_auprc(rows, n_bootstrap, seed, False)
    base.update(
        {
            "auroc": auc,
            "auroc_ci_lower": auc_lo,
            "auroc_ci_upper": auc_hi,
            "auprc": auprc,
            "auprc_ci_lower": auprc_lo,
            "auprc_ci_upper": auprc_hi,
        }
    )
    return base


def evaluate_metrics_cluster(
    results: list[dict],
    mode_dir: Path,
    gt_mask_dir: str | Path,
    cohort_id: str,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    image_io: ImageIO | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """返回 (evaluate_pipeline 形似的 metrics, bootstrap 元数据)。"""
    cluster_map = resolve_cluster_map(cohort_id)
    io = image_io or ImageIO()
    metrics: dict[str, Any] = {}

    seg_rows = _seg_rows_dice_hd95(mode_dir, Path(gt_mask_dir), cluster_map, io)
    if seg_rows:
        d_mean, d_lo, d_hi, n_c = _B2._cluster_bootstrap_mean(
            seg_rows, "dice", n_bootstrap, seed, False
        )
        metrics["dice"] = {
            "mean": d_mean,
            "ci_lower": d_lo,
            "ci_upper": d_hi,
            "values": [r["dice"] for r in seg_rows],
            "n_clusters": n_c,
        }
        hd_rows = [r for r in seg_rows if "hd95" in r]
        if hd_rows:
            h_mean, h_lo, h_hi, _ = _B2._cluster_bootstrap_mean(
                hd_rows, "hd95", n_bootstrap, seed, False
            )
            metrics["hd95"] = {
                "mean": h_mean,
                "ci_lower": h_lo,
                "ci_upper": h_hi,
                "values": [r["hd95"] for r in hd_rows],
            }

    meta = {
        "method": "cluster",
        "n": n_bootstrap,
        "seed": seed,
        "cohort_id": cohort_id,
        "cluster_map": str(_MAP_DIR / COHORT_CLUSTER_MAP[cohort_id])
        if cohort_id in COHORT_CLUSTER_MAP and (_MAP_DIR / COHORT_CLUSTER_MAP[cohort_id]).is_file()
        else None,
        "n_clusters_classification": len({r["cluster_id"] for r in _B2._rows_from_results(results, cluster_map)})
        if results
        else 0,
        "n_clusters_segmentation": len({r["cluster_id"] for r in seg_rows}) if seg_rows else 0,
    }
    return metrics, meta


def patch_b3_classification_from_cluster(
    results: list[dict],
    cohort_id: str,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> tuple[Optional[dict], Optional[dict]]:
    cluster_map = resolve_cluster_map(cohort_id)
    cls_all = _cls_block_from_results(
        results, cluster_map, subset_path_b=False, n_bootstrap=n_bootstrap, seed=seed
    )
    cls_pb = _cls_block_from_results(
        results, cluster_map, subset_path_b=True, n_bootstrap=n_bootstrap, seed=seed
    )
    return cls_all, cls_pb
