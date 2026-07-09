#!/usr/bin/env python3
"""
AutoGluon TabularPredictor 分类推理脚本
========================================
对图片 + 掩码文件夹提取 radiomics 特征后，用 AutoGluon 模型进行批量推理，
输出分类结果 CSV。若提供标签 JSON 文件，额外计算分类性能指标（含 95% 置信区间）
并保存到 .log 文件。

与其它分类模型的关键区别:
  - AutoGluon 的输入是 **radimics 特征**，而非原始图像像素。
  - 因此推理前需先用 PyRadiomics 从 (image, mask) 对中提取特征。
  - 模型存储为目录（内含 predictor.pkl），而非单文件权重。

用法:
    # 仅推理（二分类），输出 CSV
    python infer.py \\
        --image_dir /path/to/images/ \\
        --mask_dir /path/to/masks/ \\
        --model_dir /path/to/autogluon_model/ \\
        --num_classes 2 \\
        --class_names benign malignant \\
        --output results.csv

    # 推理 + 评估（二分类）
    python infer.py \\
        --image_dir /path/to/images/ \\
        --mask_dir /path/to/masks/ \\
        --model_dir /path/to/autogluon_model/ \\
        --num_classes 2 \\
        --class_names benign malignant \\
        --label_json /path/to/labels.json \\
        --label_field malignancy \\
        --output results.csv \\
        --eval_output eval_result.log

    # 推理 + 评估（TIRADS 五分类）
    python infer.py \\
        --image_dir /path/to/images/ \\
        --mask_dir /path/to/masks/ \\
        --model_dir /path/to/autogluon_tirads_model/ \\
        --num_classes 5 \\
        --class_names 1 2 3 4 5 \\
        --label_json /path/to/labels.json \\
        --label_field tirads \\
        --output results.csv \\
        --eval_output eval_result.log

标签 JSON 格式示例:
    [
        {"filename": "a.jpg", "malignancy": 0, "tirads": 2},
        {"filename": "b.jpg", "malignancy": 1, "tirads": 4}
    ]

注意:
    - 标签值为整数索引，与 --class_names 的顺序对应
    - 模型目录须包含 predictor.pkl 文件（AutoGluon TabularPredictor）
"""

import argparse
import csv
import json
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

# ============================================================================
# 使用项目级统一分类指标模块
# ============================================================================

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from cls_metrics import bootstrap_ci as _bootstrap_ci, format_metrics_report

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================================
# 常量
# ============================================================================

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _normalize_rel_path(path: str) -> str:
    return os.path.normpath(path.replace("\\", "/").lstrip("/\\"))


def _strip_ext(path: str) -> str:
    stem, _ = os.path.splitext(path)
    return stem


# ============================================================================
# 特征提取（内联自 extract_base_radiomics.py / extract_radiomics_2d.py）
# ============================================================================

def _read_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={arr.shape} ({path})")
    return arr


def _read_mask(path: str, threshold: int) -> np.ndarray:
    mask = Image.open(path).convert("L")
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={arr.shape} ({path})")
    return (arr > threshold).astype(np.uint8)


def _to_sitk(image: np.ndarray, mask: np.ndarray,
             spacing: Tuple[float, float]):
    import SimpleITK as sitk
    img_sitk = sitk.GetImageFromArray(image.astype(np.float32))
    msk_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    img_sitk.SetSpacing(spacing)
    msk_sitk.SetSpacing(spacing)
    return img_sitk, msk_sitk


def _resolve_with_flexible_ext(base_dir: str, rel_path: str) -> str:
    """Flexible path resolution: try exact match, then try all image exts."""
    rel_path = _normalize_rel_path(rel_path)
    direct = os.path.join(base_dir, rel_path)
    if os.path.exists(direct):
        return direct

    rel_stem = _strip_ext(rel_path)
    for ext in _IMAGE_EXTS:
        candidate = os.path.join(base_dir, rel_stem + ext)
        if os.path.exists(candidate):
            return candidate

    base_name = os.path.basename(rel_path)
    base_direct = os.path.join(base_dir, base_name)
    if os.path.exists(base_direct):
        return base_direct

    base_stem = _strip_ext(base_name)
    for ext in _IMAGE_EXTS:
        candidate = os.path.join(base_dir, base_stem + ext)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"File not found under {base_dir} for rel_path={rel_path}"
    )


def _scan_images(image_dir: str) -> List[str]:
    """扫描目录中所有图片文件名（无递归）。"""
    files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
    ])
    return files


def _extract_features_for_samples(
    image_dir: str,
    mask_dir: str,
    filenames: List[str],
    params_path: str,
    mask_threshold: int = 0,
    spacing: Tuple[float, float] = (1.0, 1.0),
    skip_fail: bool = True,
) -> pd.DataFrame:
    """对给定的图片列表提取 radiomics 特征，返回 DataFrame。

    Args:
        image_dir: 图片目录
        mask_dir: 掩码目录
        filenames: 文件名列表（仅文件名，非完整路径）
        params_path: PyRadiomics YAML 参数文件路径
        mask_threshold: 掩码二值化阈值
        spacing: 像素间距 (x, y)
        skip_fail: 提取失败时跳过还是报错

    Returns:
        DataFrame，列: filename, image_path, mask_path, <radiomics 特征...>
    """
    from radiomics import featureextractor

    extractor = featureextractor.RadiomicsFeatureExtractor(params_path)

    rows: List[Dict[str, object]] = []
    failures = 0

    for fname in filenames:
        try:
            image_path = _resolve_with_flexible_ext(image_dir, fname)
            mask_path = _resolve_with_flexible_ext(mask_dir, fname)

            image = _read_gray(image_path)
            mask = _read_mask(mask_path, threshold=mask_threshold)

            if image.shape != mask.shape:
                raise ValueError(
                    f"Image/mask size mismatch: "
                    f"image={image.shape}, mask={mask.shape} ({fname})"
                )
            if int(mask.sum()) == 0:
                raise ValueError(f"Empty mask (no foreground pixels): {fname}")

            img_sitk, msk_sitk = _to_sitk(image, mask, spacing=spacing)
            result = extractor.execute(img_sitk, msk_sitk, label=1)

            features = {
                k: v for k, v in result.items()
                if not str(k).startswith("diagnostics_")
            }
            features["filename"] = fname
            features["image_path"] = image_path
            features["mask_path"] = mask_path
            rows.append(features)

        except Exception as e:
            failures += 1
            msg = f"  ⚠ 特征提取失败: {fname} — {type(e).__name__}: {e}"
            if skip_fail:
                print(msg)
                continue
            raise RuntimeError(msg) from e

    if failures > 0:
        print(f"  ⚠ 特征提取: {failures}/{len(filenames)} 个样本失败")

    if not rows:
        raise RuntimeError(
            f"所有样本特征提取均失败 (0/{len(filenames)})，"
            f"请检查 --image_dir 和 --mask_dir"
        )

    df = pd.DataFrame(rows)
    return df


# ============================================================================
# 模型加载与推理
# ============================================================================

def load_predictor(model_dir: str):
    """加载 AutoGluon TabularPredictor。

    Args:
        model_dir: 包含 predictor.pkl 的目录

    Returns:
        autogluon.tabular.TabularPredictor
    """
    from autogluon.tabular import TabularPredictor

    predictor_path = os.path.join(model_dir, "predictor.pkl")
    if not os.path.isfile(predictor_path):
        raise FileNotFoundError(
            f"模型目录中未找到 predictor.pkl: {model_dir}\n"
            f"请确认 --model_dir 指向 AutoGluon TabularPredictor 输出目录"
        )

    predictor = TabularPredictor.load(model_dir, require_version_match=False, require_py_version_match=False)
    print(f"  模型类型:     {predictor.problem_type}")
    print(f"  类别标签:     {predictor.class_labels}")
    print(f"  评估指标:     {predictor.eval_metric.name}")
    return predictor


def predict_dataframe(predictor, features_df: pd.DataFrame):
    """对特征 DataFrame 进行推理。

    Args:
        predictor: AutoGluon TabularPredictor
        features_df: 包含 radiomics 特征的 DataFrame (含 filename 列)

    Returns:
        filenames: 文件名列表
        pred_labels: 预测标签列表 (原始类别名，如 0/1 或 1/2/3/4/5)
        proba_dict: {class_label: np.array of shape (N,)}
        class_labels: 模型中的类别标签列表 (排序后)
    """
    # 只保留特征列（去掉 metadata）
    meta_cols = ["filename", "image_path", "mask_path"]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    df_clean = features_df[feature_cols].copy()

    # 预测
    proba_df = predictor.predict_proba(df_clean, as_pandas=True, as_multiclass=True)
    pred_series = predictor.predict(df_clean, as_pandas=True)

    filenames = features_df["filename"].tolist()

    # 获取排序后的类别标签
    class_labels = sorted(predictor.class_labels,
                          key=lambda x: (isinstance(x, str), x))

    # 构建概率字典，确保顺序一致
    proba_dict = {}
    for c in class_labels:
        if c in proba_df.columns:
            proba_dict[c] = proba_df[c].to_numpy().astype(np.float64)
        else:
            proba_dict[c] = np.zeros(len(features_df), dtype=np.float64)

    pred_labels = pred_series.tolist()

    return filenames, pred_labels, proba_dict, class_labels


# ============================================================================
# CSV 输出
# ============================================================================

def save_csv(output_path: str, filenames: List[str],
             pred_labels: list, proba_dict: dict,
             class_labels: list, class_names: List[str],
             label_map=None):
    """保存分类结果到 CSV（格式与其它分类模型一致）。"""
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["filename", "predicted_class", "confidence"]
    for cname in class_names:
        fieldnames.append(f"prob_{cname}")
    if label_map is not None:
        fieldnames.append("true_label")

    # 构建 class_label → class_name 映射
    label_to_name = {}
    for label, name in zip(class_labels, class_names):
        # 统一转为字符串比较
        label_to_name[str(label)] = name

    rows = []
    for i, fname in enumerate(filenames):
        pred_label_raw = pred_labels[i]
        pred_name = label_to_name.get(str(pred_label_raw), str(pred_label_raw))

        # 该类别的置信度
        if pred_label_raw in proba_dict:
            pred_conf = float(proba_dict[pred_label_raw][i])
        else:
            pred_conf = 0.0

        row = {
            "filename": fname,
            "predicted_class": pred_name,
            "confidence": round(pred_conf, 6),
        }

        for label in class_labels:
            cname = label_to_name.get(str(label), str(label))
            row[f"prob_{cname}"] = round(float(proba_dict[label][i]), 6)

        if label_map is not None:
            true_idx = label_map.get(fname)
            row["true_label"] = (
                class_names[true_idx]
                if true_idx is not None and 0 <= true_idx < len(class_names)
                else ""
            )

        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


# ============================================================================
# 标签加载与映射（同 infer_biomedclip/infer.py 逻辑）
# ============================================================================

def load_label_json(json_path: str, label_field: str,
                    class_names: list) -> Dict[str, int]:
    """加载标签 JSON 文件，自动将标签值映射为 0-based 索引。

    映射策略:
      1. 标签值（转字符串）能在 class_names 中找到 → 用其索引
      2. 标签值已是 0-based（0 ~ num_classes-1）→ 直接用
      3. 标签值是 1-based（1 ~ num_classes）→ 减 1

    Returns:
        dict {filename: label_index(0-based)}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    name_to_idx = {str(name): i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    label_map = {}
    missing: List[str] = []
    remapped = 0

    for rec in records:
        fname = rec.get("filename")
        if fname is None:
            continue
        if label_field not in rec:
            missing.append(fname)
            continue

        raw_label = rec[label_field]
        label_str = str(raw_label)

        if label_str in name_to_idx:
            label_idx = name_to_idx[label_str]
            remapped += 1
        elif isinstance(raw_label, (int, float)) and 0 <= int(raw_label) < num_classes:
            label_idx = int(raw_label)
        elif isinstance(raw_label, (int, float)) and 1 <= int(raw_label) <= num_classes:
            label_idx = int(raw_label) - 1
            remapped += 1
        else:
            print(f"  ⚠ 无法映射标签: {fname} {label_field}={raw_label}, 跳过")
            continue

        label_map[fname] = label_idx

    if missing:
        print(f"  ⚠ 以下 {len(missing)} 条记录缺少字段 '{label_field}'，将跳过: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if remapped > 0:
        print(f"  标签映射: {remapped} 条标签通过 class_names 映射为 0-based 索引")

    return label_map


# ============================================================================
# 评估
# ============================================================================

def run_evaluation(filenames: List[str], proba_dict: dict,
                   class_labels: list, pred_labels: list,
                   label_map: Dict[str, int], class_names: List[str],
                   num_classes: int, eval_output: str,
                   n_bootstrap: int, seed: int):
    """对有标签的样本进行评估，使用统一的 cls_metrics 模块。"""
    if not SKLEARN_AVAILABLE:
        print("  ⚠ scikit-learn 未安装，无法进行性能评估。")
        return

    # 构建 (N, num_classes) 概率矩阵 (0-based)
    # 建立 class_label → class_idx (0-based) 的映射
    label_to_idx = {}
    for i, label in enumerate(class_labels):
        label_to_idx[str(label)] = i

    y_true_list: List[int] = []
    y_prob_list: List[np.ndarray] = []
    skipped: List[str] = []

    for i, fname in enumerate(filenames):
        if fname not in label_map:
            skipped.append(fname)
            continue

        true_label_0based = label_map[fname]
        if true_label_0based < 0 or true_label_0based >= num_classes:
            print(f"  ⚠ 标签超出范围: {fname} label={true_label_0based}, 跳过")
            continue

        # 构建 (num_classes,) 概率向量
        probs = np.zeros(num_classes, dtype=np.float64)
        for label_raw in class_labels:
            idx = label_to_idx.get(str(label_raw))
            if idx is None:
                continue
            probs[idx] = proba_dict[label_raw][i]

        y_true_list.append(true_label_0based)
        y_prob_list.append(probs)

    if skipped:
        print(f"  ⚠ {len(skipped)} 个样本无对应标签记录，跳过评估")

    if not y_true_list:
        print("  ⚠ 没有找到任何匹配的标签记录，无法评估。")
        return

    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)
    y_pred = y_prob.argmax(axis=1)

    # Bootstrap CI95 (统一模块)
    results, valid_iters = _bootstrap_ci(
        y_true, y_pred, y_prob, num_classes,
        n_boot=n_bootstrap, seed=seed,
    )

    # 生成报告
    report_str = format_metrics_report(results, labels=y_true)

    # 打印到终端
    print(report_str)

    # 保存到 .log 文件
    out_dir = os.path.dirname(os.path.abspath(eval_output))
    os.makedirs(out_dir, exist_ok=True)
    with open(eval_output, "w", encoding="utf-8") as f:
        f.write(report_str)
        f.write("\n")


# ============================================================================
# 命令行参数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoGluon 分类推理（基于 radiomics 特征 + TabularPredictor）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 数据路径
    parser.add_argument("--image_dir", type=str, required=True,
                        help="待推理的图片文件夹路径")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="对应的掩码（结节 ROI）文件夹路径")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="AutoGluon 模型目录（包含 predictor.pkl）")

    # 推理参数
    parser.add_argument("--num_classes", type=int, required=True,
                        help="类别数，二分类填 2，TIRADS 五分类填 5")
    parser.add_argument("--class_names", type=str, nargs="+", required=True,
                        help="类别名称列表，例如: benign malignant 或 1 2 3 4 5")

    # 标签与评估
    parser.add_argument("--label_json", type=str, default=None,
                        help="标签 JSON 文件路径（可选）；提供后将计算性能指标")
    parser.add_argument("--label_field", type=str, default=None,
                        help="JSON 中用于评估的标签字段名（提供 --label_json 时必填）")

    # 输出
    parser.add_argument("--output", type=str, default="results.csv",
                        help="分类结果 CSV 输出路径（默认 results.csv）")
    parser.add_argument("--eval_output", type=str, default=None,
                        help="评估结果保存路径 (.log)；未指定时自动生成")

    # 特征提取参数
    parser.add_argument("--radiomics_params", type=str, default=None,
                        help="PyRadiomics YAML 参数文件路径；默认使用脚本同目录的 radiomics_2d.yaml")
    parser.add_argument("--mask_threshold", type=int, default=0,
                        help="掩码二值化阈值（默认 0）")
    parser.add_argument("--spacing_x", type=float, default=1.0,
                        help="X 方向像素间距（默认 1.0）")
    parser.add_argument("--spacing_y", type=float, default=1.0,
                        help="Y 方向像素间距（默认 1.0）")
    parser.add_argument("--no_skip_fail", action="store_true",
                        help="特征提取失败时报错退出（默认跳过）")

    # 评估参数
    parser.add_argument("--n_bootstrap", type=int, default=2000,
                        help="Bootstrap 迭代次数（默认 2000）")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子（默认 0）")

    return parser.parse_args()


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()

    # 参数校验
    if len(args.class_names) != args.num_classes:
        print(f"错误: --class_names 长度 ({len(args.class_names)}) "
              f"与 --num_classes ({args.num_classes}) 不一致")
        sys.exit(1)

    if args.label_json is not None and args.label_field is None:
        print("错误: 提供了 --label_json 时必须同时指定 --label_field")
        sys.exit(1)

    # 检查目录
    for name, path in [("图片", args.image_dir), ("掩码", args.mask_dir),
                        ("模型", args.model_dir)]:
        if not os.path.isdir(path):
            print(f"错误: {name}目录不存在: {path}")
            sys.exit(1)

    # 放射组学参数文件
    if args.radiomics_params is None:
        args.radiomics_params = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "radiomics_2d.yaml"
        )
    if not os.path.isfile(args.radiomics_params):
        print(f"错误: radiomics 参数文件不存在: {args.radiomics_params}")
        sys.exit(1)

    # ================================================================
    # Step 1: 加载模型
    # ================================================================
    print("=" * 60)
    print(f"模型:     {args.model_dir}")
    print(f"图片:     {args.image_dir}")
    print(f"掩码:     {args.mask_dir}")
    print(f"类别数:   {args.num_classes}")
    if args.label_json:
        print(f"标签字段: {args.label_field}")
    print("=" * 60)

    print("\n[1/3] 加载 AutoGluon 模型...")
    predictor = load_predictor(args.model_dir)

    # ================================================================
    # Step 2: 提取 radiomics 特征
    # ================================================================
    print(f"\n[2/3] 提取 radiomics 特征...")
    filenames = _scan_images(args.image_dir)
    if not filenames:
        print(f"错误: 图片文件夹中未找到图片: {args.image_dir}")
        sys.exit(1)
    print(f"  图片数量: {len(filenames)}")

    spacing = (args.spacing_x, args.spacing_y)
    features_df = _extract_features_for_samples(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        filenames=filenames,
        params_path=args.radiomics_params,
        mask_threshold=args.mask_threshold,
        spacing=spacing,
        skip_fail=not args.no_skip_fail,
    )
    print(f"  成功提取特征: {len(features_df)} 个样本")
    _feat_cols = [c for c in features_df.columns
                  if c not in ('filename', 'image_path', 'mask_path')]
    print(f"  特征维度:     {len(_feat_cols)}")

    # ================================================================
    # Step 3: 推理
    # ================================================================
    print(f"\n[3/3] 模型推理...")
    filenames_extracted, pred_labels, proba_dict, class_labels = \
        predict_dataframe(predictor, features_df)

    # 构建 class_label → class_name 的映射用于打印
    label_to_name_map = {}
    for label, name in zip(class_labels, args.class_names):
        label_to_name_map[str(label)] = name

    # 打印预测分布
    from collections import Counter
    pred_counts = Counter()
    for pl in pred_labels:
        pred_counts[label_to_name_map.get(str(pl), str(pl))] += 1
    print(f"  预测分布: {dict(pred_counts)}")

    # 加载标签（可选，用于 CSV true_label 列和评估）
    label_map = None
    if args.label_json:
        label_map = load_label_json(args.label_json, args.label_field,
                                    args.class_names)

    # 保存 CSV
    save_csv(args.output, filenames_extracted, pred_labels,
             proba_dict, class_labels, args.class_names, label_map)
    print(f"  结果已保存: {args.output}")

    # ================================================================
    # Step 4: 评估（可选）
    # ================================================================
    if args.label_json:
        print(f"\n  性能评估...")

        if args.eval_output:
            eval_output = args.eval_output
        else:
            out_dir = os.path.dirname(os.path.abspath(args.output))
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            eval_output = os.path.join(out_dir,
                                       f"eval_result_{timestamp}.log")

        run_evaluation(
            filenames=filenames_extracted,
            proba_dict=proba_dict,
            class_labels=class_labels,
            pred_labels=pred_labels,
            label_map=label_map,
            class_names=args.class_names,
            num_classes=args.num_classes,
            eval_output=eval_output,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )

    print(f"\n  完成")


if __name__ == "__main__":
    main()
