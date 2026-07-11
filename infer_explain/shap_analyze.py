"""
稳定版SHAP分析脚本（支持AutoGluon BAG树模型）。

主要功能：
- 加载AutoGluon模型与训练CSV，构建背景样本和解释样本。
- 针对主模型计算SHAP值，输出标准化CSV结果。
- 可选生成beeswarm、waterfall和紧凑版局部SHAP条形图，并保存分析摘要。

输出目录默认为 <model_dir>/shap_analysis。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from plotting_utils import paper_friendly_name, prepare_df, save_beeswarm_plot
from autogluon_introspection import (
    extract_ensemble_weights_from_log,
    get_main_models,
)
from shap_compute import compute_shap_for_model
from shap_local_plots import plot_waterfall_samples

np = None
pd = None
TabularPredictor = None


def _require_numpy():
    global np
    if np is None:
        import numpy as _np

        np = _np
    return np


def _require_pandas():
    global pd
    if pd is None:
        import pandas as _pd

        pd = _pd
    return pd


def _require_tabular_predictor():
    global TabularPredictor
    if TabularPredictor is None:
        from autogluon.tabular import TabularPredictor as _TabularPredictor

        TabularPredictor = _TabularPredictor
    return TabularPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SHAP analysis for AutoGluon TabularPredictor models (fixed for BAG models)."
    )
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="AutoGluon model directory (contains predictor.pkl)",
    )
    p.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Training CSV file (used for SHAP background and analysis)",
    )
    p.add_argument(
        "--label",
        type=str,
        default="label",
        help="Label column name",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for SHAP results (default: <model_dir>/shap_analysis)",
    )
    p.add_argument(
        "--background_samples",
        type=int,
        default=100,
        help="Number of background samples for SHAP (default: 100)",
    )
    p.add_argument(
        "--explain_samples",
        type=int,
        default=None,
        help="Number of samples to explain (default: all test samples or 500 if too large)",
    )
    p.add_argument(
        "--skip_neural_net",
        action="store_true",
        help="Skip neural network models (NeuralNetFastAI) in SHAP analysis",
    )
    p.add_argument(
        "--main_models",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of main model names to analyze (default: auto-detect from ensemble)",
    )
    p.add_argument(
        "--plot_waterfall",
        action="store_true",
        help="Generate SHAP waterfall plots and compact local SHAP bars with balanced positive/negative samples",
    )
    p.add_argument(
        "--waterfall_samples",
        type=int,
        default=3,
        help="Total samples per category, split between positive and negative examples when possible (default: 3)",
    )
    p.add_argument(
        "--sample_filename",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional: additionally plot waterfall figures for one or more samples with these "
            "filenames (should match the CSV 'filename' column or the basename of 'image_path')."
        ),
    )
    p.add_argument(
        "--plot_beeswarm_for",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional: names of model(s) to generate SHAP beeswarm plots for, "
            "e.g. LightGBM_BAG_L1."
        ),
    )
    p.add_argument(
        "--top_features",
        type=int,
        default=5,
        help="Number of top features shown in beeswarm plots (default: 5)",
    )
    p.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Task name shown in the compact SHAP legend title (e.g. Malignancy, LNM, FTC/PTC)",
    )
    p.add_argument(
        "--positive_class_name",
        type=str,
        default=None,
        help="Human-readable name for label 1, shown in the compact SHAP x-axis label and legend",
    )
    p.add_argument(
        "--negative_class_name",
        type=str,
        default=None,
        help="Human-readable name for label 0, shown in the compact SHAP legend",
    )
    p.add_argument(
        "--output_space",
        type=str,
        default=None,
        help="SHAP output space used for labeling, e.g. probability or raw score",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    np_mod = _require_numpy()
    pd_mod = _require_pandas()
    predictor_cls = _require_tabular_predictor()

    print(f"Loading predictor from: {args.model_dir}")
    predictor = predictor_cls.load(args.model_dir)

    print(f"Loading training data from: {args.train_csv}")
    raw_df = pd_mod.read_csv(args.train_csv)
    train_df = prepare_df(raw_df.copy(), args.label)

    sample_ids_all: Optional[pd.Series] = None
    id_col = None
    if "filename" in raw_df.columns:
        id_col = "filename"
    elif "image_path" in raw_df.columns:
        id_col = "image_path"

    if id_col is not None:
        sample_ids_all = raw_df.loc[train_df.index, id_col].copy()
        if id_col == "image_path":
            sample_ids_all = sample_ids_all.astype(str).apply(os.path.basename)

    main_models = get_main_models(predictor, args.model_dir, args.main_models)
    print(f"\nMain models to analyze: {main_models}")

    X_train = train_df.drop(columns=[args.label]).copy()
    y_train = train_df[args.label].copy()

    np_mod.random.seed(42)
    if len(X_train) > args.background_samples:
        background_idx = np_mod.random.choice(len(X_train), size=args.background_samples, replace=False)
        X_background = X_train.iloc[background_idx].copy()
        print(f"Using {len(X_background)} background samples for SHAP")
    else:
        X_background = X_train.copy()
        print(f"Using all {len(X_background)} samples as background")

    from sklearn.model_selection import train_test_split

    if args.explain_samples is not None:
        n_explain = min(args.explain_samples, len(X_train))
        if n_explain < len(X_train):
            X_explain, _, y_explain, _ = train_test_split(
                X_train,
                y_train,
                test_size=1 - n_explain / len(X_train),
                stratify=y_train,
                random_state=42,
            )
            X_explain = X_explain.copy()
        else:
            X_explain = X_train.copy()
            y_explain = y_train.copy()
    else:
        max_explain = 500
        if len(X_train) <= max_explain:
            X_explain = X_train.copy()
            y_explain = y_train.copy()
        else:
            X_explain, _, y_explain, _ = train_test_split(
                X_train,
                y_train,
                test_size=1 - max_explain / len(X_train),
                stratify=y_train,
                random_state=42,
            )
            X_explain = X_explain.copy()

    sample_ids_explain: Optional[pd.Series] = None
    if sample_ids_all is not None:
        try:
            sample_ids_explain = sample_ids_all.loc[X_explain.index]
        except Exception:
            sample_ids_explain = sample_ids_all

    class_counts = y_explain.value_counts().sort_index()
    print(f"Explaining {len(X_explain)} samples")
    print(f"  Class distribution: {dict(class_counts)}")
    for cls, count in class_counts.items():
        pct = count / len(X_explain) * 100
        print(f"    Class {cls}: {count} samples ({pct:.1f}%)")

    if args.output_dir is None:
        output_dir = os.path.join(args.model_dir, "shap_analysis")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    plot_dirs = {
        "beeswarm": os.path.join(output_dir, "beeswarm"),
        "waterfall": os.path.join(output_dir, "waterfall"),
        "compact_shap_bar": os.path.join(output_dir, "compact_shap_bar"),
    }
    for plot_dir in plot_dirs.values():
        os.makedirs(plot_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    results = {}
    shap_summary = []

    for model_name in main_models:
        print(f"\nAnalyzing model: {model_name}")
        result = compute_shap_for_model(
            predictor, model_name, X_background, X_explain, args.skip_neural_net
        )

        if result is None:
            continue

        shap_values, shap_df = result
        results[model_name] = {
            "shap_values": shap_values,
            "shap_df": shap_df,
        }

        model_output_file = os.path.join(output_dir, f"{model_name}_shap_values.csv")
        shap_df.to_csv(model_output_file, index=False)
        print(f"  Saved SHAP values to: {model_output_file}")

        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        shap_summary.append(
            {
                "model": model_name,
                "top_features": mean_abs_shap.head(20).to_dict(),
                "mean_abs_shap_sum": float(shap_df.abs().sum().sum()),
            }
        )

        importance_file = os.path.join(output_dir, f"{model_name}_feature_importance.csv")
        importance_df = pd_mod.DataFrame(
            {
                "feature": mean_abs_shap.index,
                "mean_abs_shap": mean_abs_shap.values,
            }
        )
        importance_df.to_csv(importance_file, index=False)
        print(f"  Saved feature importance to: {importance_file}")

        if args.plot_beeswarm_for and model_name in args.plot_beeswarm_for:
            try:
                import shap  # noqa: F401
                import matplotlib.pyplot as plt  # noqa: F401
            except ImportError:
                print(
                    "  Warning: shap or matplotlib not available. "
                    f"Skipping beeswarm plot for {model_name}."
                )
            else:
                print(f"  Generating SHAP beeswarm plot for: {model_name}")
                beeswarm_max_display = max(1, int(args.top_features))
                beeswarm_dir = plot_dirs["beeswarm"]
                beeswarm_path = os.path.join(beeswarm_dir, f"{model_name}_beeswarm.png")
                beeswarm_textless_path = os.path.join(beeswarm_dir, f"{model_name}_beeswarm_textless.svg")
                saved_paths = save_beeswarm_plot(
                    shap_values,
                    shap_df,
                    beeswarm_path,
                    beeswarm_max_display,
                    textless_svg_path=beeswarm_textless_path,
                    feature_name_formatter=paper_friendly_name,
                    export_formats=("png", "svg"),
                    dpi=150,
                    figsize=(12, 9),
                    plot_type="dot",
                    positive_class_name=args.positive_class_name,
                    output_space=args.output_space,
                )
                print(f"  Saved beeswarm plot to: {', '.join(saved_paths)}")

    log_path = os.path.join(args.model_dir, "logs", "predictor_log.txt")
    weights = extract_ensemble_weights_from_log(log_path)
    if weights and results:
        print("\nComputing weighted ensemble SHAP values...")
        weighted_shap_list = []
        feature_names = None
        expected_shape = None
        for model_name in weights.keys():
            if model_name in results:
                weight = weights[model_name]
                shap_values = results[model_name]["shap_values"]
                if expected_shape is None:
                    expected_shape = shap_values.shape
                elif shap_values.shape != expected_shape:
                    print(f"  Warning: Shape mismatch for {model_name}: {shap_values.shape} vs {expected_shape}")
                    print(f"  Skipping this model in ensemble aggregation")
                    continue
                weighted_shap_list.append(shap_values * weight)
                if feature_names is None:
                    feature_names = results[model_name]["shap_df"].columns.tolist()

        if weighted_shap_list:
            ensemble_shap = np_mod.sum(weighted_shap_list, axis=0)
            if feature_names is None:
                feature_names = X_explain.columns.tolist()
            if len(feature_names) != ensemble_shap.shape[1]:
                print(f"  Warning: Feature count mismatch: {len(feature_names)} names vs {ensemble_shap.shape[1]} values")
                feature_names = [f"feature_{i}" for i in range(ensemble_shap.shape[1])]
            ensemble_shap_df = pd_mod.DataFrame(ensemble_shap, columns=feature_names, index=X_explain.index)
            ensemble_file = os.path.join(output_dir, "WeightedEnsemble_L3_shap_values.csv")
            ensemble_shap_df.to_csv(ensemble_file, index=False)
            print(f"Saved ensemble SHAP values to: {ensemble_file}")

            ensemble_importance = ensemble_shap_df.abs().mean().sort_values(ascending=False)
            ensemble_importance_file = os.path.join(
                output_dir, "WeightedEnsemble_L3_feature_importance.csv"
            )
            ensemble_importance_df = pd_mod.DataFrame(
                {
                    "feature": ensemble_importance.index,
                    "mean_abs_shap": ensemble_importance.values,
                }
            )
            ensemble_importance_df.to_csv(ensemble_importance_file, index=False)
            print(f"Saved ensemble feature importance to: {ensemble_importance_file}")

    summary_file = os.path.join(output_dir, "shap_analysis_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("SHAP Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model directory: {args.model_dir}\n")
        f.write(f"Training CSV: {args.train_csv}\n")
        f.write(f"Background samples: {len(X_background)}\n")
        f.write(f"Explained samples: {len(X_explain)}\n")
        f.write(f"Main models analyzed: {len(results)}\n\n")

        if weights:
            f.write("Ensemble weights:\n")
            for model_name, weight in weights.items():
                f.write(f"  {model_name}: {weight:.4f}\n")
            f.write("\n")

        for item in shap_summary:
            f.write(f"Model: {item['model']}\n")
            f.write(f"  Mean absolute SHAP sum: {item['mean_abs_shap_sum']:.4f}\n")
            f.write("  Top 10 features:\n")
            for feat, val in list(item["top_features"].items())[:10]:
                f.write(f"    {feat}: {val:.6f}\n")
            f.write("\n")

    print(f"\nSaved analysis summary to: {summary_file}")

    if args.plot_waterfall and results:
        print(f"\nGenerating SHAP waterfall plots...")
        plot_waterfall_samples(
            predictor,
            results,
            X_explain,
            y_explain,
            output_dir,
            args.waterfall_samples,
            args.label,
            sample_ids_explain,
            args.sample_filename,
            task_name=args.task_name,
            positive_class_name=args.positive_class_name,
            negative_class_name=args.negative_class_name,
            output_space=args.output_space,
        )

    print(f"\nSHAP analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
