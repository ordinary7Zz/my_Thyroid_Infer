"""
Pipeline 评估指标：分割（Dice、HD95）和分类（AUROC、AUPRC）+ 95% CI。

CI 使用 bootstrap 重采样（1000 次）计算。
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional

from segmentation.metrics import compute_dice, compute_hd95


def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 计算 95% CI。

    Returns:
        (mean, ci_lower, ci_upper)
    """
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = values[idx].mean()

    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_means, 100 * alpha))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    mean_val = float(values.mean())
    return mean_val, ci_lower, ci_upper


def _bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 计算 AUC 的 95% CI。

    Returns:
        (auc_mean, ci_lower, ci_upper)
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    n = len(y_true)

    boot_aucs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_boot = y_true[idx]
        if len(set(y_boot)) < 2:
            boot_aucs[i] = np.nan
            continue
        y_score_boot = y_score[idx]
        try:
            boot_aucs[i] = roc_auc_score(y_boot, y_score_boot)
        except ValueError:
            boot_aucs[i] = np.nan

    boot_aucs = boot_aucs[~np.isnan(boot_aucs)]
    if len(boot_aucs) == 0:
        return float("nan"), float("nan"), float("nan")

    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_aucs, 100 * alpha))
    ci_upper = float(np.percentile(boot_aucs, 100 * (1 - alpha)))
    auc_val = float(roc_auc_score(y_true, y_score))
    return auc_val, ci_lower, ci_upper


def _bootstrap_auprc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 计算 AUPRC 的 95% CI。

    Returns:
        (auprc_mean, ci_lower, ci_upper)
    """
    from sklearn.metrics import average_precision_score

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    n = len(y_true)

    boot_auprcs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_boot = y_true[idx]
        if len(set(y_boot)) < 2:
            boot_auprcs[i] = np.nan
            continue
        y_score_boot = y_score[idx]
        try:
            boot_auprcs[i] = average_precision_score(y_boot, y_score_boot)
        except ValueError:
            boot_auprcs[i] = np.nan

    boot_auprcs = boot_auprcs[~np.isnan(boot_auprcs)]
    if len(boot_auprcs) == 0:
        return float("nan"), float("nan"), float("nan")

    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_auprcs, 100 * alpha))
    ci_upper = float(np.percentile(boot_auprcs, 100 * (1 - alpha)))
    auprc_val = float(average_precision_score(y_true, y_score))
    return auprc_val, ci_lower, ci_upper


def evaluate_pipeline(
    results: list[dict],
    output_dir: str,
    image_io=None,
    gt_mask_dir: Optional[str] = None,
) -> dict:
    """评估 pipeline 结果，打印并保存指标。

    Args:
        results: run_batch 返回的 results 列表
        output_dir: 输出目录（用于读取 selected_masks.npz）
        image_io: ImageIO 实例（用于加载 GT mask）
        gt_mask_dir: GT mask 目录，优先使用；未提供时回退到从 config 读取

    Returns:
        dict: 评估指标
    """
    print(f"\n{'='*60}")
    print("Pipeline 评估")
    print(f"{'='*60}")

    metrics: dict = {}

    # ===== 分割指标 =====
    seg_evals = []
    selected_masks_path = Path(output_dir) / "intermediate" / "selected_masks.npz"

    has_gt_mask = any("true_label" in r for r in results)
    if not selected_masks_path.exists():
        print("⚠️ selected_masks.npz 不存在，跳过分割评估")
    elif not image_io:
        print("⚠️ 未提供 ImageIO，跳过分割评估")
    else:
        # 加载选定 mask
        data = np.load(str(selected_masks_path), allow_pickle=True)
        sel_img_names = list(data["image_names"])
        sel_masks = list(data["selected_masks"])

        # 从 config 中获取 gt_mask_dir（通过 results 不可用，需要外部传入）
        # 这里尝试从 results 的 image_name 匹配 GT mask
        from shared.image_io import ImageIO as _ImageIO
        if image_io is None:
            image_io = _ImageIO()

        dice_values = []
        hd95_values = []
        per_image_seg: list[dict] = []

        # GT mask 目录需要从外部传入，这里先检查 results 中是否有
        # 实际上 gt_mask_dir 在 run_batch 中使用了，但没存到 results
        # 我们需要另一种方式获取 gt_mask_dir
        print("  分割指标计算中...")

        # 尝试从 output_dir 的 metadata 获取
        # 如果没有 gt_mask_dir，则跳过
        if gt_mask_dir is None:
            gt_mask_dir = _get_gt_mask_dir(output_dir)

        if gt_mask_dir is None:
            print("⚠️ 未找到 gt_mask_dir 配置，跳过分割评估")
        else:
            # 构造 stem -> gt_path 映射，兼容图像与 mask 后缀不同的情况
            gt_dir = Path(gt_mask_dir)
            gt_stem_map: dict[str, Path] = {}
            if gt_dir.exists():
                for p in gt_dir.iterdir():
                    if p.is_file():
                        gt_stem_map[p.stem] = p

            for i, img_name in enumerate(sel_img_names):
                img_stem = Path(img_name).stem
                gt_path = gt_stem_map.get(img_stem)
                if gt_path is None or not gt_path.exists():
                    continue
                gt_mask = image_io.binarize_mask(image_io.load_mask(gt_path))
                pred_mask = sel_masks[i].astype(bool)

                dice = compute_dice(pred_mask, gt_mask)
                hd95 = compute_hd95(pred_mask, gt_mask)

                if not np.isnan(dice):
                    dice_values.append(dice)
                if not np.isinf(hd95) and not np.isnan(hd95):
                    hd95_values.append(hd95)
                per_image_seg.append({
                    "image_name": img_name,
                    "stem": img_stem,
                    "dice": float(dice) if not np.isnan(dice) else None,
                    "hd95": float(hd95) if not (np.isinf(hd95) or np.isnan(hd95)) else None,
                })

            if per_image_seg:
                seg_path = Path(output_dir) / "per_image_seg_metrics.jsonl"
                with open(seg_path, "w", encoding="utf-8") as f:
                    for row in per_image_seg:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  逐图分割指标: {seg_path} ({len(per_image_seg)} 条)")

            if dice_values:
                dice_mean, dice_lo, dice_hi = _bootstrap_ci(dice_values)
                print(f"  Dice:  {dice_mean:.4f} (95% CI: {dice_lo:.4f} - {dice_hi:.4f})")
                metrics["dice"] = {"mean": dice_mean, "ci_lower": dice_lo, "ci_upper": dice_hi, "values": dice_values}
            else:
                print("  ⚠️ 无有效 Dice 值")

            if hd95_values:
                hd95_mean, hd95_lo, hd95_hi = _bootstrap_ci(hd95_values)
                print(f"  HD95:  {hd95_mean:.2f} (95% CI: {hd95_lo:.2f} - {hd95_hi:.2f})")
                metrics["hd95"] = {"mean": hd95_mean, "ci_lower": hd95_lo, "ci_upper": hd95_hi, "values": hd95_values}
            else:
                print("  ⚠️ 无有效 HD95 值")

    # ===== 分类指标 =====
    # 需要 true_label 和 final_confidence
    y_true_list = []
    y_score_list = []
    y_pred_list = []
    per_image_cls: list[dict] = []

    for r in results:
        if "true_label" not in r or "final_label" not in r:
            continue
        true_label = r["true_label"]
        pred_label = r["final_label"]
        confidence = r.get("final_confidence", 0.5)

        # 统一标签为 0/1
        true_binary = _label_to_binary(true_label)
        pred_binary = _label_to_binary(pred_label) if isinstance(pred_label, str) else pred_label

        if true_binary is not None and pred_binary is not None:
            y_true_list.append(true_binary)
            score = confidence if pred_binary == 1 else 1 - confidence
            y_score_list.append(score)
            y_pred_list.append(pred_binary)
            img_name = r.get("image_name", "")
            per_image_cls.append({
                "image_name": img_name,
                "stem": Path(img_name).stem if img_name else "",
                "center_id": r.get("center_id"),
                "true_label": true_binary,
                "pred_label": pred_binary,
                "score_malignant": float(score),
                "final_confidence": float(confidence),
                "flagged_for_review": r.get("flagged_for_review"),
            })

    if per_image_cls:
        cls_path = Path(output_dir) / "per_image_cls_metrics.jsonl"
        with open(cls_path, "w", encoding="utf-8") as f:
            for row in per_image_cls:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  逐图分类指标: {cls_path} ({len(per_image_cls)} 条)")

    if len(y_true_list) < 2 or len(set(y_true_list)) < 2:
        print("⚠️ 无足够有效标签或只有单一类别，跳过分类评估")
    else:
        y_true = np.array(y_true_list)
        y_score = np.array(y_score_list)
        y_pred = np.array(y_pred_list)

        from sklearn.metrics import accuracy_score, f1_score, recall_score

        # AUROC
        auc_val, auc_lo, auc_hi = _bootstrap_auc_ci(y_true, y_score)
        print(f"  AUROC: {auc_val:.4f} (95% CI: {auc_lo:.4f} - {auc_hi:.4f})")
        metrics["auroc"] = {"mean": auc_val, "ci_lower": auc_lo, "ci_upper": auc_hi}

        # AUPRC
        auprc_val, auprc_lo, auprc_hi = _bootstrap_auprc_ci(y_true, y_score)
        print(f"  AUPRC: {auprc_val:.4f} (95% CI: {auprc_lo:.4f} - {auprc_hi:.4f})")
        metrics["auprc"] = {"mean": auprc_val, "ci_lower": auprc_lo, "ci_upper": auprc_hi}

        # Accuracy / Sensitivity / Specificity / F1
        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred, pos_label=1)  # sensitivity = recall for positive class
        spec = recall_score(y_true, y_pred, pos_label=0)  # specificity = recall for negative class
        f1 = f1_score(y_true, y_pred, pos_label=1)

        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Sensitivity: {sens:.4f}")
        print(f"  Specificity: {spec:.4f}")
        print(f"  F1:          {f1:.4f}")
        metrics["accuracy"] = acc
        metrics["sensitivity"] = sens
        metrics["specificity"] = spec
        metrics["f1"] = f1

    # 保存指标
    if metrics:
        metrics_path = Path(output_dir) / "metrics.json"
        # 移除 values 列表（太大）
        metrics_clean = {}
        for k, v in metrics.items():
            if isinstance(v, dict) and "values" in v:
                metrics_clean[k] = {kk: vv for kk, vv in v.items() if kk != "values"}
            else:
                metrics_clean[k] = v
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_clean, f, ensure_ascii=False, indent=2)
        print(f"\n指标已保存到 {metrics_path}")

    return metrics


def _label_to_binary(label) -> Optional[int]:
    """将标签转为 0/1。"""
    if isinstance(label, (int, float)):
        return int(label)
    if isinstance(label, str):
        if label in ("恶性", "malignant", "M", "1", "1.0"):
            return 1
        if label in ("良性", "benign", "B", "0", "0.0"):
            return 0
    return None


def _get_gt_mask_dir(output_dir: str) -> Optional[str]:
    """从 config.yaml 中读取 gt_mask_dir。"""
    # 尝试从 run_pipeline 所在目录的 config 读取
    import yaml
    from pathlib import Path

    # 向上查找 config/config.yaml
    current = Path(output_dir).parent
    for _ in range(5):
        config_path = current / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config.get("pipeline", {}).get("data", {}).get("gt_mask_dir")
        current = current.parent
    return None
