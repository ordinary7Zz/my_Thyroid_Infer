#!/usr/bin/env python3
"""
可解释性分析统一入口
===================

对图像目录中的每张图像同时生成 SHAP 分析（分类可解释性）和 GradCAM 分析
（分割可解释性），可选地对整个 train_csv 做批量 SHAP 分析。

所有参数通过 YAML 配置文件指定。

用法:
  python run_explain.py --config config_explain.yaml

输出结构:
  <output_dir>/
  ├── per_image/
  │   ├── shap/
  │   │   ├── waterfall/<stem>.png/.svg       # SHAP waterfall 图
  │   │   └── compact_bar/<stem>.png/.svg     # 紧凑 SHAP 条形图
  │   └── gradcam/
  │       ├── original/<filename>             # 原图
  │       ├── overlay/<filename>              # 热力图叠加
  │       ├── overlay_gt/<filename>           # 热力图 + GT 轮廓
  │       ├── original_gt/<filename>          # 原图 + GT 轮廓
  │       └── gradcam_map/<filename>          # 纯热力图
  ├── batch_shap/                             # （可选）
  │   ├── <ModelName>_shap_values.csv
  │   ├── <ModelName>_feature_importance.csv
  │   ├── shap_analysis_summary.txt
  │   ├── beeswarm/
  │   ├── waterfall/
  │   └── compact_shap_bar/
  └── run_explain.log
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# 确保本地模块可导入
# ============================================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# --- SHAP 相关导入 ---
from autogluon_introspection import (
    extract_ensemble_weights_from_log,
    get_main_models,
    load_autogluon_model,
)
from shap_compute import compute_shap_for_model
from shap_local_plots import plot_waterfall_samples, save_compact_shap_bar_plot
from plotting_utils import (
    _ensure_cjk_fonts,
    format_feature_name,
    paper_friendly_name,
    prepare_df,
    save_beeswarm_plot,
    save_waterfall_plot,
)

# --- GradCAM 相关导入 ---
import torch
from gradcam_seg import (
    IMAGE_EXTENSIONS,
    GradCAM,
    build_model as _build_gradcam_model,
    load_image as _load_gradcam_image,
    load_region_mask,
    postprocess_cam,
    resolve_output_types,
    save_sample_outputs,
)


# ============================================================================
# 配置加载
# ============================================================================

def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件，将相对路径解析为绝对路径。"""
    import yaml

    cfg_path = Path(config_path).resolve()
    if not cfg_path.is_file():
        print(f"[错误] 配置文件不存在: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 必填字段校验
    required = ["image_dir", "train_csv", "output_dir"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        print(f"[错误] 配置缺少必填字段: {', '.join(missing)}")
        sys.exit(1)

    # 将相对路径解析为绝对路径（以配置文件所在目录为基准）
    base = cfg_path.parent
    for key in ("image_dir", "mask_dir", "train_csv", "autogluon_model_dir",
                "gradcam_checkpoint", "output_dir"):
        val = cfg.get(key, "")
        if val and not os.path.isabs(val):
            cfg[key] = str((base / val).resolve())

    return cfg


# ============================================================================
# 文件收集
# ============================================================================

def collect_images(image_dir: str) -> List[Path]:
    """收集目录中所有支持的图像文件，按文件名排序。"""
    img_dir = Path(image_dir)
    if not img_dir.is_dir():
        raise NotADirectoryError(f"图像目录不存在: {image_dir}")
    return sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_masks(mask_dir: str) -> Dict[str, Path]:
    """收集掩码文件，按 stem 建立映射。"""
    if not mask_dir:
        return {}
    m_dir = Path(mask_dir)
    if not m_dir.is_dir():
        return {}
    return {
        p.stem: p for p in sorted(m_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


# ============================================================================
# SHAP: 初始化与逐图分析
# ============================================================================

def setup_shap(cfg: dict):
    """加载 AutoGluon predictor，准备特征数据，获取主模型列表。

    Returns:
        (predictor, main_models, X_train, y_train, sample_ids)
    """
    from autogluon.tabular import TabularPredictor

    shap_cfg = cfg.get("shap", {})
    label = shap_cfg.get("label", "label")

    print(f"[SHAP] 加载 AutoGluon 模型: {cfg['autogluon_model_dir']}")
    predictor = TabularPredictor.load(cfg["autogluon_model_dir"])

    print(f"[SHAP] 加载训练数据: {cfg['train_csv']}")
    raw_df = pd.read_csv(cfg["train_csv"])
    train_df = prepare_df(raw_df.copy(), label)

    # 提取样本 ID（filename 或 image_path）
    id_col = "filename" if "filename" in raw_df.columns else (
        "image_path" if "image_path" in raw_df.columns else None
    )
    sample_ids: Optional[pd.Series] = None
    if id_col is not None:
        sample_ids = raw_df.loc[train_df.index, id_col].copy()
        if id_col == "image_path":
            sample_ids = sample_ids.astype(str).apply(os.path.basename)

    main_models_cfg = shap_cfg.get("main_models", [])
    main_models = get_main_models(
        predictor, cfg["autogluon_model_dir"],
        main_models_cfg if main_models_cfg else None,
    )
    print(f"[SHAP] 主模型: {main_models}")

    X_train = train_df.drop(columns=[label]).copy()
    y_train = train_df[label].copy()

    return predictor, main_models, X_train, y_train, sample_ids


def find_target_index(
    image_stem: str,
    sample_ids: Optional[pd.Series],
) -> Optional[Any]:
    """在 train_csv 的样本 ID 中查找图像，返回 train_df 的行索引。

    匹配策略（按优先级）:
      1. 精确匹配 sample_id == image_stem
      2. stem 匹配 (去掉扩展名)
      3. basename 匹配
      4. basename stem 匹配
    """
    if sample_ids is None:
        return None

    sids = sample_ids.astype(str).values

    # 1) 精确匹配
    matches = np.where(sids == image_stem)[0]
    if len(matches):
        return sample_ids.index[matches[0]]

    # 2) stem 匹配
    stems = np.array([os.path.splitext(s)[0] for s in sids])
    matches = np.where(stems == image_stem)[0]
    if len(matches):
        return sample_ids.index[matches[0]]

    # 3) basename 匹配
    basenames = np.array([os.path.basename(s) for s in sids])
    matches = np.where(basenames == image_stem)[0]
    if len(matches):
        return sample_ids.index[matches[0]]

    # 4) basename stem 匹配
    base_stems = np.array([os.path.splitext(b)[0] for b in basenames])
    matches = np.where(base_stems == image_stem)[0]
    if len(matches):
        return sample_ids.index[matches[0]]

    return None


def _get_positive_proba(proba) -> np.ndarray:
    """从 predict_proba 输出中提取正类概率。"""
    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            return proba[1].to_numpy()
        return proba.iloc[:, -1].to_numpy()
    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr[:, 1]
    return arr.reshape(-1)


def shap_analyze_single_image(
    predictor,
    main_models: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_index,
    target_tag: str,
    output_dir: str,
    shap_cfg: dict,
) -> int:
    """为单张图像生成 SHAP waterfall + compact bar 图。

    Returns:
        成功生成图的模型数
    """
    skip_nn = shap_cfg.get("skip_neural_net", True)
    top_features = shap_cfg.get("top_features", 5)
    bg_samples = shap_cfg.get("background_samples", 100)
    positive_class_name = shap_cfg.get("positive_class_name") or None
    negative_class_name = shap_cfg.get("negative_class_name") or None
    output_space = shap_cfg.get("output_space") or None
    feature_label_lang = shap_cfg.get("feature_label_lang", "en")

    x_explain = X_train.loc[[target_index]].copy()

    # 背景样本池（排除目标样本）
    bg_pool = X_train.drop(index=target_index, errors="ignore")
    if len(bg_pool) == 0:
        bg_pool = X_train.copy()

    np.random.seed(42)
    if len(bg_pool) > bg_samples:
        bg_idx = np.random.choice(len(bg_pool), size=bg_samples, replace=False)
        x_background = bg_pool.iloc[bg_idx].copy()
    else:
        x_background = bg_pool.copy()

    # base value: 背景样本的正类概率均值
    try:
        base_value = float(np.mean(_get_positive_proba(
            predictor.predict_proba(x_background)
        )))
    except Exception:
        base_value = 0.5

    # 输出目录
    waterfall_dir = os.path.join(output_dir, "waterfall")
    compact_bar_dir = os.path.join(output_dir, "compact_bar")
    os.makedirs(waterfall_dir, exist_ok=True)
    os.makedirs(compact_bar_dir, exist_ok=True)

    multi_model = len(main_models) > 1
    success_count = 0

    for model_name in main_models:
        result = compute_shap_for_model(
            predictor, model_name, x_background, x_explain, skip_nn
        )
        if result is None:
            continue

        shap_values, shap_df = result
        sample_shap = np.asarray(shap_values[0]).reshape(-1)
        feature_names = shap_df.columns.tolist()

        max_display = max(1, int(top_features))
        top_idx = np.argsort(np.abs(sample_shap))[-max_display:][::-1]
        top_shap = sample_shap[top_idx]
        top_names = [feature_names[i] for i in top_idx]
        display_names = [format_feature_name(n, feature_label_lang) for n in top_names]

        # 提取特征值
        try:
            if hasattr(predictor, "_learner") and hasattr(predictor._learner, "feature_generator"):
                processed = predictor._learner.feature_generator.transform(x_explain)
                if isinstance(processed, pd.DataFrame):
                    sample_features = processed.iloc[0]
                else:
                    sample_features = pd.Series(np.asarray(processed)[0], index=feature_names)
            else:
                sample_features = x_explain.iloc[0]
        except Exception:
            sample_features = x_explain.iloc[0]

        top_values = []
        for name, idx in zip(top_names, top_idx):
            if isinstance(sample_features, pd.Series) and name in sample_features.index:
                top_values.append(sample_features[name])
            elif isinstance(sample_features, pd.Series):
                try:
                    pos = shap_df.columns.get_loc(name)
                    top_values.append(
                        sample_features.iloc[pos] if pos < len(sample_features) else np.nan
                    )
                except (KeyError, IndexError):
                    top_values.append(np.nan)
            else:
                top_values.append(np.nan)

        display_values = np.array(top_values, dtype=float).flatten()

        # 多模型时按模型名分子目录
        if multi_model:
            wf_dir = os.path.join(waterfall_dir, model_name)
            cb_dir = os.path.join(compact_bar_dir, model_name)
            os.makedirs(wf_dir, exist_ok=True)
            os.makedirs(cb_dir, exist_ok=True)
        else:
            wf_dir = waterfall_dir
            cb_dir = compact_bar_dir

        wf_path = os.path.join(wf_dir, f"{target_tag}.png")
        wf_textless = os.path.join(wf_dir, f"{target_tag}_textless.svg")
        cb_path = os.path.join(cb_dir, f"{target_tag}.png")
        cb_textless = os.path.join(cb_dir, f"{target_tag}_textless.svg")

        save_waterfall_plot(
            top_shap, display_values, display_names,
            base_value, wf_path, max_display,
            textless_svg_path=wf_textless,
            title=f"{model_name} SHAP — {target_tag}",
            title_fontsize=22,
            export_formats=("png", "svg"),
            dpi=150,
            figsize=(14.5, 9.0),
            bbox_inches="tight",
            positive_class_name=positive_class_name,
            output_space=output_space,
        )

        save_compact_shap_bar_plot(
            top_shap, top_names, cb_path, max_display,
            positive_class_name=positive_class_name,
            negative_class_name=negative_class_name,
            output_space=output_space,
            feature_label_lang=feature_label_lang,
            textless_svg_path=cb_textless,
            xlabel_fontsize=12.0,
            ytick_fontsize=11.0,
            export_formats=("png", "svg"),
            dpi=300,
            figsize=(3.35, 3.35 * 4 / 3),
        )

        success_count += 1

    return success_count


# ============================================================================
# SHAP: 批量分析
# ============================================================================

def run_batch_shap(cfg: dict, output_dir: str) -> None:
    """对整个 train_csv 做批量 SHAP 分析。

    复用 shap_analyze.py 的核心逻辑，生成:
      - 各模型的 SHAP 值 CSV + 特征重要性 CSV
      - ensemble 加权 SHAP 值
      - 分析摘要
      - （可选）beeswarm 图、代表性样本 waterfall 图
    """
    from autogluon.tabular import TabularPredictor
    from sklearn.model_selection import train_test_split

    shap_cfg = cfg.get("shap", {})
    batch_cfg = shap_cfg.get("batch", {})
    label = shap_cfg.get("label", "label")

    print(f"\n[Batch SHAP] 开始批量分析")

    predictor = TabularPredictor.load(cfg["autogluon_model_dir"])
    raw_df = pd.read_csv(cfg["train_csv"])
    train_df = prepare_df(raw_df.copy(), label)

    # 样本 ID
    id_col = "filename" if "filename" in raw_df.columns else (
        "image_path" if "image_path" in raw_df.columns else None
    )
    sample_ids_all: Optional[pd.Series] = None
    if id_col is not None:
        sample_ids_all = raw_df.loc[train_df.index, id_col].copy()
        if id_col == "image_path":
            sample_ids_all = sample_ids_all.astype(str).apply(os.path.basename)

    main_models_cfg = shap_cfg.get("main_models", [])
    main_models = get_main_models(
        predictor, cfg["autogluon_model_dir"],
        main_models_cfg if main_models_cfg else None,
    )
    print(f"[Batch SHAP] 主模型: {main_models}")

    X_train = train_df.drop(columns=[label]).copy()
    y_train = train_df[label].copy()

    # 背景样本
    np.random.seed(42)
    bg_n = shap_cfg.get("background_samples", 100)
    if len(X_train) > bg_n:
        bg_idx = np.random.choice(len(X_train), size=bg_n, replace=False)
        X_background = X_train.iloc[bg_idx].copy()
    else:
        X_background = X_train.copy()

    # 解释样本
    explain_n = batch_cfg.get("explain_samples", 500)
    if explain_n is not None and explain_n < len(X_train):
        X_explain, _, y_explain, _ = train_test_split(
            X_train, y_train,
            test_size=1 - explain_n / len(X_train),
            stratify=y_train, random_state=42,
        )
        X_explain = X_explain.copy()
    else:
        X_explain = X_train.copy()
        y_explain = y_train.copy()

    sample_ids_explain: Optional[pd.Series] = None
    if sample_ids_all is not None:
        try:
            sample_ids_explain = sample_ids_all.loc[X_explain.index]
        except Exception:
            sample_ids_explain = sample_ids_all

    print(f"[Batch SHAP] 背景: {len(X_background)}, 解释: {len(X_explain)}")

    skip_nn = shap_cfg.get("skip_neural_net", True)
    top_features = shap_cfg.get("top_features", 5)

    os.makedirs(output_dir, exist_ok=True)
    plot_dirs = {
        "beeswarm": os.path.join(output_dir, "beeswarm"),
        "waterfall": os.path.join(output_dir, "waterfall"),
        "compact_shap_bar": os.path.join(output_dir, "compact_shap_bar"),
    }
    for d in plot_dirs.values():
        os.makedirs(d, exist_ok=True)

    results: Dict[str, dict] = {}
    shap_summary: List[dict] = []

    for model_name in main_models:
        print(f"\n  [Batch SHAP] 分析模型: {model_name}")
        result = compute_shap_for_model(
            predictor, model_name, X_background, X_explain, skip_nn
        )
        if result is None:
            continue

        shap_values, shap_df = result
        results[model_name] = {"shap_values": shap_values, "shap_df": shap_df}

        # 保存 SHAP 值
        shap_df.to_csv(
            os.path.join(output_dir, f"{model_name}_shap_values.csv"), index=False
        )

        # 特征重要性
        mean_abs = shap_df.abs().mean().sort_values(ascending=False)
        pd.DataFrame(
            {"feature": mean_abs.index, "mean_abs_shap": mean_abs.values}
        ).to_csv(
            os.path.join(output_dir, f"{model_name}_feature_importance.csv"),
            index=False,
        )

        shap_summary.append({
            "model": model_name,
            "top_features": mean_abs.head(20).to_dict(),
            "mean_abs_shap_sum": float(shap_df.abs().sum().sum()),
        })

        # beeswarm
        beeswarm_models = batch_cfg.get("plot_beeswarm_for", [])
        if beeswarm_models and model_name in beeswarm_models:
            try:
                bs_path = os.path.join(plot_dirs["beeswarm"], f"{model_name}_beeswarm.png")
                bs_textless = os.path.join(plot_dirs["beeswarm"], f"{model_name}_beeswarm_textless.svg")
                save_beeswarm_plot(
                    shap_values, shap_df, bs_path, max(1, int(top_features)),
                    textless_svg_path=bs_textless,
                    feature_name_formatter=paper_friendly_name,
                    export_formats=("png", "svg"),
                    dpi=150, figsize=(12, 9), plot_type="dot",
                    positive_class_name=shap_cfg.get("positive_class_name") or None,
                    output_space=shap_cfg.get("output_space") or None,
                )
                print(f"    beeswarm 已保存")
            except Exception as e:
                print(f"    beeswarm 失败: {e}")

    # ensemble 加权 SHAP
    log_path = os.path.join(cfg["autogluon_model_dir"], "logs", "predictor_log.txt")
    weights = extract_ensemble_weights_from_log(log_path)
    if weights and results:
        print(f"\n  [Batch SHAP] 计算 ensemble 加权 SHAP...")
        weighted_list = []
        feature_names = None
        expected_shape = None
        for mn, w in weights.items():
            if mn in results:
                sv = results[mn]["shap_values"]
                if expected_shape is None:
                    expected_shape = sv.shape
                elif sv.shape != expected_shape:
                    continue
                weighted_list.append(sv * w)
                if feature_names is None:
                    feature_names = results[mn]["shap_df"].columns.tolist()

        if weighted_list:
            ensemble_shap = np.sum(weighted_list, axis=0)
            if feature_names is None:
                feature_names = X_explain.columns.tolist()
            ensemble_df = pd.DataFrame(
                ensemble_shap, columns=feature_names, index=X_explain.index
            )
            ensemble_df.to_csv(
                os.path.join(output_dir, "WeightedEnsemble_L3_shap_values.csv"),
                index=False,
            )

            ensemble_imp = ensemble_df.abs().mean().sort_values(ascending=False)
            pd.DataFrame(
                {"feature": ensemble_imp.index, "mean_abs_shap": ensemble_imp.values}
            ).to_csv(
                os.path.join(output_dir, "WeightedEnsemble_L3_feature_importance.csv"),
                index=False,
            )
            print(f"    ensemble SHAP 已保存")

    # 摘要
    summary_file = os.path.join(output_dir, "shap_analysis_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("SHAP Batch Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model directory: {cfg['autogluon_model_dir']}\n")
        f.write(f"Training CSV: {cfg['train_csv']}\n")
        f.write(f"Background samples: {len(X_background)}\n")
        f.write(f"Explained samples: {len(X_explain)}\n")
        f.write(f"Main models analyzed: {len(results)}\n\n")
        if weights:
            f.write("Ensemble weights:\n")
            for mn, w in weights.items():
                f.write(f"  {mn}: {w:.4f}\n")
            f.write("\n")
        for item in shap_summary:
            f.write(f"Model: {item['model']}\n")
            f.write(f"  Mean absolute SHAP sum: {item['mean_abs_shap_sum']:.4f}\n")
            f.write("  Top 10 features:\n")
            for feat, val in list(item["top_features"].items())[:10]:
                f.write(f"    {feat}: {val:.6f}\n")
            f.write("\n")
    print(f"\n  [Batch SHAP] 摘要已保存: {summary_file}")

    # 代表性样本 waterfall
    if batch_cfg.get("plot_waterfall", True) and results:
        print(f"  [Batch SHAP] 生成代表性 waterfall 图...")
        plot_waterfall_samples(
            predictor, results, X_explain, y_explain, output_dir,
            batch_cfg.get("waterfall_samples", 3),
            label,
            sample_ids_explain, None,
            task_name=shap_cfg.get("task_name") or None,
            positive_class_name=shap_cfg.get("positive_class_name") or None,
            negative_class_name=shap_cfg.get("negative_class_name") or None,
            output_space=shap_cfg.get("output_space") or None,
        )

    print(f"  [Batch SHAP] 完成")


# ============================================================================
# GradCAM: 逐图分析
# ============================================================================

def run_gradcam_for_images(
    checkpoint: str,
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    gradcam_cfg: dict,
    device: str = "cuda",
) -> None:
    """对图像目录中的所有图像生成 GradCAM 热力图。"""
    print(f"\n[GradCAM] 开始分析")

    dev = torch.device(
        device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    output_type = gradcam_cfg.get("output_type", "all")
    img_size = gradcam_cfg.get("img_size", 224)
    dino_pretrained = gradcam_cfg.get("dino_pretrained", True)
    target_layer_name = gradcam_cfg.get("target_layer", "reduce4")
    alpha = gradcam_cfg.get("alpha", 0.45)
    smooth_sigma_ratio = gradcam_cfg.get("smooth_sigma_ratio", 0.02)
    gamma = gradcam_cfg.get("gamma", 1.0)
    saturation_scale = gradcam_cfg.get("saturation_scale", 1.3)

    output_types = resolve_output_types(output_type)

    images = collect_images(image_dir)
    if not images:
        print(f"[GradCAM] 图像目录中未找到图像: {image_dir}")
        return

    masks = collect_masks(mask_dir)
    requires_mask = any(ot in {"overlay_gt", "original_gt"} for ot in output_types)
    if requires_mask and not masks:
        print(f"[GradCAM] 警告: 输出类型需要 mask 但 mask_dir 为空，将自动跳过 GT 相关输出")

    print(f"[GradCAM] {len(images)} 张图像, 设备: {dev}, 目标层: {target_layer_name}")
    print(f"[GradCAM] 输出类型: {', '.join(output_types)}")

    # 构建模型（一次性）
    model = _build_gradcam_model(checkpoint, dev, dino_pretrained=dino_pretrained)
    model.eval()

    target_layer = getattr(model, target_layer_name, None)
    if target_layer is None:
        print(f"[GradCAM] 警告: 层 '{target_layer_name}' 未找到，使用 'reduce4'")
        target_layer = model.reduce4

    gradcam = GradCAM(model, target_layer)
    os.makedirs(output_dir, exist_ok=True)

    success, failed = 0, 0

    try:
        for i, img_path in enumerate(images, 1):
            print(f"  [{i}/{len(images)}] {img_path.name}")

            try:
                img_tensor, orig_np = _load_gradcam_image(str(img_path), img_size, dev)

                # 查找匹配的 mask
                mask_path = masks.get(img_path.stem) if masks else None
                region_mask = None
                if mask_path is not None:
                    region_mask = load_region_mask(str(mask_path), img_size)

                # 根据 mask 可用性过滤输出类型
                effective_types = list(output_types)
                if region_mask is None:
                    effective_types = [
                        t for t in effective_types
                        if t not in ("overlay_gt", "original_gt")
                    ]

                img_tensor.requires_grad_(True)
                cam_raw = gradcam.generate(img_tensor, target_mask=region_mask)
                cam = postprocess_cam(
                    cam_raw,
                    smooth_sigma_ratio=smooth_sigma_ratio,
                    gamma=gamma,
                )

                save_sample_outputs(
                    image_path=str(img_path),
                    output_dir=output_dir,
                    output_types=effective_types,
                    orig_np=orig_np,
                    cam=cam,
                    region_mask=region_mask,
                    alpha=alpha,
                    saturation_scale=saturation_scale,
                )
                success += 1
            except Exception as e:
                print(f"    错误: {e}")
                traceback.print_exc()
                failed += 1
    finally:
        gradcam.release()

    print(f"\n[GradCAM] 完成: 成功 {success}, 失败 {failed}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="可解释性分析统一入口: SHAP (分类) + GradCAM (分割)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="YAML 配置文件路径",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    run_shap = cfg.get("run_shap_per_image", True)
    run_gradcam = cfg.get("run_gradcam", True)
    run_batch = cfg.get("run_batch_shap", False)

    print("=" * 60)
    print("  可解释性分析")
    print("=" * 60)
    print(f"  图像目录:    {cfg['image_dir']}")
    print(f"  掩码目录:    {cfg.get('mask_dir') or '(无)'}")
    print(f"  训练 CSV:    {cfg['train_csv']}")
    print(f"  AutoGluon:   {cfg.get('autogluon_model_dir', '(未配置)')}")
    print(f"  GradCAM权重: {cfg.get('gradcam_checkpoint', '(未配置)')}")
    print(f"  输出目录:    {output_dir}")
    print(f"  逐图 SHAP:   {run_shap}")
    print(f"  GradCAM:     {run_gradcam}")
    print(f"  批量 SHAP:   {run_batch}")
    print("=" * 60)

    t_start = time.time()

    # 初始化 CJK 字体（matplotlib 中文支持）
    if run_shap or run_batch:
        _ensure_cjk_fonts()

    # --- 1. 逐图 SHAP ---
    shap_ctx = None  # (predictor, main_models, X_train, y_train, sample_ids)

    if run_shap or run_batch:
        shap_ctx = setup_shap(cfg)

    if run_shap:
        print(f"\n{'─' * 60}")
        print("[Per-image SHAP] 逐图 SHAP 分析")
        print(f"{'─' * 60}")

        predictor, main_models, X_train, y_train, sample_ids = shap_ctx
        shap_cfg = cfg.get("shap", {})

        per_image_shap_dir = os.path.join(output_dir, "per_image", "shap")
        os.makedirs(per_image_shap_dir, exist_ok=True)

        images = collect_images(cfg["image_dir"])
        print(f"  共 {len(images)} 张图像")

        success, skipped, failed = 0, 0, 0

        for i, img_path in enumerate(images, 1):
            stem = img_path.stem
            print(f"  [{i}/{len(images)}] {img_path.name}")

            target_index = find_target_index(stem, sample_ids)
            if target_index is None:
                print(f"    跳过: 在 train_csv 中未找到")
                skipped += 1
                continue

            try:
                n = shap_analyze_single_image(
                    predictor, main_models, X_train, y_train,
                    target_index, stem,
                    per_image_shap_dir, shap_cfg,
                )
                if n > 0:
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"    错误: {e}")
                traceback.print_exc()
                failed += 1

        print(f"\n  [Per-image SHAP] 成功: {success}, 跳过: {skipped}, 失败: {failed}")

    # --- 2. GradCAM ---
    if run_gradcam:
        print(f"\n{'─' * 60}")
        print("[GradCAM] 逐图 GradCAM 分析")
        print(f"{'─' * 60}")

        gradcam_output = os.path.join(output_dir, "per_image", "gradcam")
        gradcam_cfg = cfg.get("gradcam", {})
        device = cfg.get("device", "cuda")

        run_gradcam_for_images(
            checkpoint=cfg["gradcam_checkpoint"],
            image_dir=cfg["image_dir"],
            mask_dir=cfg.get("mask_dir", ""),
            output_dir=gradcam_output,
            gradcam_cfg=gradcam_cfg,
            device=device,
        )

    # --- 3. 批量 SHAP ---
    if run_batch:
        print(f"\n{'─' * 60}")
        print("[Batch SHAP] 批量 SHAP 分析")
        print(f"{'─' * 60}")

        batch_output = os.path.join(output_dir, "batch_shap")
        run_batch_shap(cfg, batch_output)

    # --- 汇总 ---
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  可解释性分析完成! 耗时 {elapsed:.1f}s")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
