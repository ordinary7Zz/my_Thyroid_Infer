#!/usr/bin/env python3
"""
对比评估：pipeline（agent 链路）vs. 每个独立模型。

从已有中间产物加载：
  - 分割：intermediate/seg/*.npz（各模型预测）+ intermediate/selected_masks.npz（agent 选的）
  - 分类：intermediate/cls/*.json（各模型预测）+ results.json（pipeline 最终裁决）

对每个对象计算指标，输出对比表（终端打印 + comparison.json + comparison_report.txt）。

用法:
  python -m pipeline.compare_models \
      --output-dir output/pipeline_run \
      --gt-mask-dir /path/to/TN3K/test/masks \
      --label-file /path/to/TN3K_test_label.json \
      --label-key malignancy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from shared.image_io import ImageIO
from segmentation.metrics import compute_dice, compute_hd95
from pipeline.evaluate import _bootstrap_ci, _bootstrap_auc_ci, _bootstrap_auprc_ci, _label_to_binary


# ============== 数据加载 ==============

def load_seg_model_masks(seg_npz_path: Path) -> tuple[list[str], list[np.ndarray], str]:
    """加载某个分割模型的预测 mask。返回 (image_names, masks, model_name)。"""
    data = np.load(str(seg_npz_path), allow_pickle=True)
    img_names = [str(x) for x in data["image_names"]]
    masks = [np.asarray(m) for m in data["masks"]]
    # 模型名取文件名（去后缀）
    model_name = seg_npz_path.stem
    return img_names, masks, model_name


def load_selected_masks(npz_path: Path) -> tuple[list[str], list[np.ndarray]]:
    """加载 agent 选中的 mask。返回 (image_names, masks)。"""
    data = np.load(str(npz_path), allow_pickle=True)
    img_names = [str(x) for x in data["image_names"]]
    masks = [np.asarray(m) for m in data["selected_masks"]]
    return img_names, masks


def load_cls_model_json(json_path: Path) -> tuple[dict, str]:
    """加载某个分类模型的预测 json。返回 (img_name->pred_dict, model_name)。

    兼容两种格式：
      - 标准包装: {"model_name": "...", "results": {img: {...}}}
      - autogluon 裸格式: {img: {...}, ...}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], dict):
        return data["results"], data.get("model_name", json_path.stem)
    # autogluon 裸格式
    return data, json_path.stem


def load_labels(label_file: str, label_key: str = "malignancy") -> dict[str, int]:
    """加载标签文件，返回 {img_stem: 0/1}。兼容 dict / list 格式。"""
    with open(label_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    labels: dict[str, int] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            b = _label_to_binary(v)
            if b is not None:
                labels[Path(k).stem] = b
    elif isinstance(raw, list):
        for item in raw:
            fname = item.get("filename", "")
            if fname:
                b = _label_to_binary(item.get(label_key))
                if b is not None:
                    labels[Path(fname).stem] = b
    return labels


# ============== 指标计算 ==============

def build_gt_mask_stem_map(gt_mask_dir: Path) -> dict[str, Path]:
    """stem -> GT mask 路径（兼容图像 .jpg 与 mask .png 后缀不一致）。"""
    gt_stem_map: dict[str, Path] = {}
    if gt_mask_dir.exists():
        for p in gt_mask_dir.iterdir():
            if p.is_file():
                gt_stem_map[p.stem] = p
    return gt_stem_map


def resolve_gt_mask_path(
    gt_mask_dir: Path,
    img_name: str,
    stem_map: Optional[dict[str, Path]] = None,
) -> Optional[Path]:
    direct = gt_mask_dir / img_name
    if direct.is_file():
        return direct
    stem_map = stem_map if stem_map is not None else build_gt_mask_stem_map(gt_mask_dir)
    return stem_map.get(Path(img_name).stem)


def eval_seg_masks(
    img_names: list[str],
    pred_masks: list[np.ndarray],
    gt_mask_dir: Path,
    image_io: ImageIO,
) -> dict:
    """对一组 pred mask 计算 Dice / HD95（含 95% CI）。"""
    dice_values: list[float] = []
    hd95_values: list[float] = []
    n_matched = 0

    gt_stem_map = build_gt_mask_stem_map(gt_mask_dir)

    for img_name, pred in zip(img_names, pred_masks):
        gt_path = resolve_gt_mask_path(gt_mask_dir, img_name, stem_map=gt_stem_map)
        if gt_path is None or not gt_path.exists():
            continue
        gt_mask = image_io.binarize_mask(image_io.load_mask(gt_path))
        pred_mask = pred.astype(bool)
        dice = compute_dice(pred_mask, gt_mask)
        hd95 = compute_hd95(pred_mask, gt_mask)
        n_matched += 1
        if not np.isnan(dice):
            dice_values.append(dice)
        if not np.isinf(hd95) and not np.isnan(hd95):
            hd95_values.append(hd95)

    result = {"n_matched": n_matched, "n_total": len(img_names)}
    if dice_values:
        m, lo, hi = _bootstrap_ci(dice_values)
        result["dice"] = {"mean": m, "ci_lower": lo, "ci_upper": hi}
    else:
        result["dice"] = None
    if hd95_values:
        m, lo, hi = _bootstrap_ci(hd95_values)
        result["hd95"] = {"mean": m, "ci_lower": lo, "ci_upper": hi}
    else:
        result["hd95"] = None
    return result


def eval_cls_predictions(
    img_to_pred: dict,
    labels: dict[str, int],
    subset_stems: Optional[set[str]] = None,
) -> dict:
    """对一组分类预测计算 AUROC / AUPRC / Acc / Sens / Spec / F1。

    img_to_pred: {img_name_or_stem: {predictions: {"良性": p, "恶性": q}, top_class, top_confidence}}
    labels: {img_stem: 0/1}
    subset_stems: 若提供，只在该子集上评估
    """
    y_true: list[int] = []
    y_score: list[float] = []  # 恶性的概率
    y_pred: list[int] = []

    for img_name, pred in img_to_pred.items():
        stem = Path(img_name).stem
        if stem not in labels:
            continue
        if subset_stems is not None and stem not in subset_stems:
            continue
        true_b = labels[stem]
        predictions = pred.get("predictions", {})
        # 优先用 predictions 里的"恶性"概率；否则用 top_confidence 配合 top_class
        if "恶性" in predictions:
            score = float(predictions["恶性"])
        elif "malignant" in predictions:
            score = float(predictions["malignant"])
        else:
            top_class = pred.get("top_class", "")
            top_conf = float(pred.get("top_confidence", 0.5))
            top_b = _label_to_binary(top_class)
            score = top_conf if top_b == 1 else 1 - top_conf
        pred_b = 1 if score >= 0.5 else 0
        y_true.append(true_b)
        y_score.append(score)
        y_pred.append(pred_b)

    result = {"n_matched": len(y_true)}
    if len(y_true) < 2 or len(set(y_true)) < 2:
        result["auroc"] = None
        result["auprc"] = None
        result["accuracy"] = None
        result["sensitivity"] = None
        result["specificity"] = None
        result["f1"] = None
        return result

    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)
    y_pred_arr = np.array(y_pred)

    from sklearn.metrics import accuracy_score, f1_score, recall_score
    auc_val, auc_lo, auc_hi = _bootstrap_auc_ci(y_true_arr, y_score_arr)
    auprc_val, auprc_lo, auprc_hi = _bootstrap_auprc_ci(y_true_arr, y_score_arr)
    result["auroc"] = {"mean": auc_val, "ci_lower": auc_lo, "ci_upper": auc_hi}
    result["auprc"] = {"mean": auprc_val, "ci_lower": auprc_lo, "ci_upper": auprc_hi}
    result["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
    result["sensitivity"] = float(recall_score(y_true_arr, y_pred_arr, pos_label=1))
    result["specificity"] = float(recall_score(y_true_arr, y_pred_arr, pos_label=0))
    result["f1"] = float(f1_score(y_true_arr, y_pred_arr, pos_label=1))
    return result


# ============== pipeline 最终结果提取 ==============

def extract_pipeline_cls_from_results(results: list[dict]) -> dict:
    """从 results.json 提取 pipeline 最终分类预测，构造成与其他 cls json 同样的格式。"""
    out = {}
    for r in results:
        img_name = r.get("image_name") or r.get("seg_decision", {}).get("img_name")
        if not img_name:
            continue
        final_label = r.get("final_label")
        final_conf = float(r.get("final_confidence", 0.5))
        # final_label 可能是 "恶性"/"良性" 字符串，也可能是 1/0 int
        # 统一转为 binary，1=恶性
        label_bin = _label_to_binary(final_label)
        if label_bin == 1:
            preds = {"良性": 1 - final_conf, "恶性": final_conf}
        else:
            preds = {"良性": final_conf, "恶性": 1 - final_conf}
        out[img_name] = {
            "predictions": preds,
            "top_class": final_label,
            "top_confidence": final_conf,
        }
    return out


# ============== 打印 / 格式化 ==============

def _fmt_ci(v: Optional[dict], fmt: str = "{:.4f}") -> str:
    if v is None:
        return "  N/A"
    return f"{fmt.format(v['mean'])} (95% CI {fmt.format(v['ci_lower'])}-{fmt.format(v['ci_upper'])})"


def _fmt_scalar(v, fmt: str = "{:.4f}") -> str:
    return "N/A" if v is None else fmt.format(v)


def format_seg_table(rows: list[dict]) -> str:
    """格式化分割对比表，返回完整文本。"""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append("分割对比 (Dice ↑ / HD95 ↓)")
    lines.append(f"{'='*80}")
    header = f"{'来源':<28} {'Dice':<26} {'HD95':<22} {'N'}"
    lines.append(header)
    lines.append("-" * 80)
    for r in rows:
        name = r["name"]
        dice = _fmt_ci(r["metrics"].get("dice"))
        hd95 = _fmt_ci(r["metrics"].get("hd95"), "{:.2f}")
        n = r["metrics"].get("n_matched", 0)
        marker = " ★" if r.get("is_pipeline") else "  "
        lines.append(f"{name:<28} {dice:<26} {hd95:<22} {n}{marker}")
    lines.append("  ★ = pipeline/agent")
    return "\n".join(lines)


def format_cls_table(rows: list[dict]) -> str:
    """格式化分类对比表，返回完整文本。"""
    lines = []
    lines.append(f"\n{'='*100}")
    lines.append("分类对比 (AUROC ↑ / AUPRC ↑ / Acc ↑ / Sens / Spec / F1)")
    lines.append(f"{'='*100}")
    header = f"{'来源':<28} {'AUROC':<22} {'AUPRC':<22} {'Acc':<7} {'Sens':<7} {'Spec':<7} {'F1':<7} {'N'}"
    lines.append(header)
    lines.append("-" * 100)
    for r in rows:
        name = r["name"]
        m = r["metrics"]
        auc = _fmt_ci(m.get("auroc"))
        auprc = _fmt_ci(m.get("auprc"))
        acc = _fmt_scalar(m.get("accuracy"))
        sens = _fmt_scalar(m.get("sensitivity"))
        spec = _fmt_scalar(m.get("specificity"))
        f1 = _fmt_scalar(m.get("f1"))
        n = m.get("n_matched", 0)
        marker = " ★" if r.get("is_pipeline") else "  "
        lines.append(f"{name:<28} {auc:<22} {auprc:<22} {acc:<7} {sens:<7} {spec:<7} {f1:<7} {n}{marker}")
    lines.append("  ★ = pipeline/agent")
    return "\n".join(lines)


# ============== 主流程 ==============

def compare_models(
    output_dir: str | Path,
    gt_mask_dir: str | Path,
    label_file: str | Path,
    label_key: str = "malignancy",
    image_io=None,
) -> dict:
    """对比 pipeline 与各独立模型性能。

    Args:
        output_dir: pipeline 输出目录
        gt_mask_dir: GT mask 目录
        label_file: 标签文件路径
        label_key: list 格式标签文件中字段名
        image_io: ImageIO 实例（未提供则新建）

    Returns:
        dict: {"segmentation": [...], "classification": [...]}
    """
    output_dir = Path(output_dir)
    inter_dir = output_dir / "intermediate"
    seg_dir = inter_dir / "seg"
    cls_dir = inter_dir / "cls"
    gt_mask_dir = Path(gt_mask_dir)
    if image_io is None:
        image_io = ImageIO()

    # 文本报告收集
    report_parts: list[str] = []
    report_parts.append(f"\n{'='*60}")
    report_parts.append("模型对比评估")
    report_parts.append(f"{'='*60}")

    # 标签
    labels = load_labels(str(label_file), label_key)
    _line = f"已加载标签: {len(labels)} 条"
    print(_line)
    report_parts.append(_line)

    # ============== 分割对比 ==============
    seg_rows: list[dict] = []

    # 各独立模型
    if seg_dir.exists():
        for npz_file in sorted(seg_dir.glob("*.npz")):
            img_names, masks, model_name = load_seg_model_masks(npz_file)
            metrics = eval_seg_masks(img_names, masks, gt_mask_dir, image_io)
            seg_rows.append({"name": model_name, "metrics": metrics, "is_pipeline": False})

    # pipeline agent 选的
    selected_path = inter_dir / "selected_masks.npz"
    if selected_path.exists():
        img_names, masks = load_selected_masks(selected_path)
        metrics = eval_seg_masks(img_names, masks, gt_mask_dir, image_io)
        seg_rows.append({"name": "Pipeline (agent selected)", "metrics": metrics, "is_pipeline": True})
    else:
        _line = "⚠️ selected_masks.npz 不存在，pipeline 分割结果将缺失"
        print(_line)
        report_parts.append(_line)

    _seg_text = format_seg_table(seg_rows)
    print(_seg_text)
    report_parts.append(_seg_text)

    # ============== 分类对比（全集） ==============
    cls_rows: list[dict] = []

    # 收集所有分类模型预测，供子集对比复用
    all_cls_preds: dict[str, dict] = {}  # {model_name: img_to_pred}

    # 各独立模型
    if cls_dir.exists():
        for json_file in sorted(cls_dir.glob("*.json")):
            img_to_pred, model_name = load_cls_model_json(json_file)
            all_cls_preds[model_name] = img_to_pred
            metrics = eval_cls_predictions(img_to_pred, labels)
            cls_rows.append({"name": model_name, "metrics": metrics, "is_pipeline": False})

    # autogluon（在 intermediate 根目录下，不在 cls/ 下）
    autogluon_path = inter_dir / "autogluon.json"
    autogluon_model_name = None
    if autogluon_path.exists():
        img_to_pred, model_name = load_cls_model_json(autogluon_path)
        all_cls_preds[model_name] = img_to_pred
        autogluon_model_name = model_name
        metrics = eval_cls_predictions(img_to_pred, labels)
        cls_rows.append({"name": model_name, "metrics": metrics, "is_pipeline": False})

    # pipeline 最终裁决
    results_path = output_dir / "results.json"
    pipeline_results: list[dict] = []
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            pipeline_results = json.load(f)
        pipeline_pred = extract_pipeline_cls_from_results(pipeline_results)
        if pipeline_pred:
            all_cls_preds["Pipeline (final)"] = pipeline_pred
            metrics = eval_cls_predictions(pipeline_pred, labels)
            cls_rows.append({"name": "Pipeline (final)", "metrics": metrics, "is_pipeline": True})
    else:
        _line = "⚠️ results.json 不存在，pipeline 分类结果将缺失"
        print(_line)
        report_parts.append(_line)

    _cls_text = format_cls_table(cls_rows)
    print(_cls_text)
    report_parts.append(_cls_text)

    # ============== 分类对比（PathB 子集：分类模型无共识的难样本） ==============
    # AutoGluon 现在在全集上推理（便于评估其分类性能），
    # 但 PathB（无共识）子集才是其真正发挥作用的场景。
    # 在同一子集上重算各模型，才能判断 AutoGluon 仲裁是否有价值。
    path_b_stems: Optional[set[str]] = None
    if pipeline_results:
        path_b_stems = {
            Path(r.get("image_name", "")).stem
            for r in pipeline_results
            if r.get("path") == "B" and r.get("image_name")
        }
    # 兼容回退：若没有 results.json，使用 autogluon.json 的 keys
    if not path_b_stems and autogluon_model_name and autogluon_model_name in all_cls_preds:
        path_b_stems = {Path(k).stem for k in all_cls_preds[autogluon_model_name].keys()}

    if path_b_stems:
        _pathb_header = (
            f"\n{'='*100}\n"
            f"分类对比 - PathB 子集（无共识难样本 N={len(path_b_stems)}）\n"
            f"{'='*100}"
        )
        print(_pathb_header)
        report_parts.append(_pathb_header)
        subset_rows: list[dict] = []
        for model_name, img_to_pred in all_cls_preds.items():
            metrics = eval_cls_predictions(img_to_pred, labels, subset_stems=path_b_stems)
            is_pipe = model_name == "Pipeline (final)"
            subset_rows.append({"name": model_name, "metrics": metrics, "is_pipeline": is_pipe})
        _subset_text = format_cls_table(subset_rows)
        print(_subset_text)
        report_parts.append(_subset_text)

    # ============== 保存 ==============
    comparison = {
        "segmentation": [{"name": r["name"], "is_pipeline": r.get("is_pipeline", False), **r["metrics"]} for r in seg_rows],
        "classification": [{"name": r["name"], "is_pipeline": r.get("is_pipeline", False), **r["metrics"]} for r in cls_rows],
    }
    # PathB 子集对比
    if path_b_stems:
        comparison["classification_path_b_subset"] = {
            "subset_size": len(path_b_stems),
            "models": [{"name": r["name"], "is_pipeline": r.get("is_pipeline", False), **r["metrics"]} for r in subset_rows],
        }
    out_path = output_dir / "comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    _line = f"\n对比结果已保存到 {out_path}"
    print(_line)
    report_parts.append(_line)

    # 保存文本报告
    report_path = output_dir / "comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_parts))
    print(f"对比报告已保存到 {report_path}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="对比 pipeline 与各独立模型性能")
    parser.add_argument("--output-dir", default="output/pipeline_run", help="pipeline 输出目录")
    parser.add_argument("--gt-mask-dir", required=True, help="GT mask 目录")
    parser.add_argument("--label-file", required=True, help="标签文件路径")
    parser.add_argument("--label-key", default="malignancy", help="list 格式标签文件中字段名")
    args = parser.parse_args()

    compare_models(
        output_dir=args.output_dir,
        gt_mask_dir=args.gt_mask_dir,
        label_file=args.label_file,
        label_key=args.label_key,
    )


if __name__ == "__main__":
    main()
