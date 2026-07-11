from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from plotting_utils import (
    _ensure_cjk_fonts,
    build_shap_axis_label,
    format_feature_name,
    hide_text_in_figure,
    paper_friendly_name,
    save_current_figure,
    save_waterfall_plot,
)


def save_compact_shap_bar_plot(
    shap_values,
    feature_names,
    out_path: str,
    max_display: int,
    *,
    task_name: Optional[str] = None,
    positive_class_name: Optional[str] = None,
    negative_class_name: Optional[str] = None,
    output_space: Optional[str] = None,
    feature_label_lang: str = "en",
    textless_svg_path: Optional[str] = None,
    xlabel_fontsize: float = 11.0,
    ytick_fontsize: float = 10.0,
    export_formats=("png", "svg"),
    dpi: int = 300,
    figsize: Optional[tuple[float, float]] = None,
) -> list[str]:
    import matplotlib.pyplot as plt
    _ensure_cjk_fonts()

    shap_arr = np.asarray(shap_values).reshape(-1)
    feature_names = list(feature_names)
    if len(shap_arr) != len(feature_names):
        raise ValueError("shap_values and feature_names must have the same length")

    max_display = max(1, min(int(max_display), len(shap_arr)))
    top_indices = np.argsort(np.abs(shap_arr))[-max_display:][::-1]
    display_values = shap_arr[top_indices]
    display_names = [feature_names[i] for i in top_indices]
    display_labels = [format_feature_name(name, lang=feature_label_lang) for name in display_names]

    if figsize is None:
        height = max(2.2, 0.52 * len(display_labels) + 1.15)
        figsize = (3.15, height)

    def _render(show_text: bool, target_path: str, formats) -> list[str]:
        with plt.rc_context(
            {
                "font.family": "sans-serif",
                "font.sans-serif": [
                    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
                    "Noto Sans CJK SC", "Source Han Sans SC",
                    "Microsoft YaHei", "SimHei",
                    "PingFang SC", "Hiragino Sans GB",
                    "Heiti SC", "STHeiti", "Songti SC",
                    "Arial Unicode MS", "DejaVu Sans", "Arial",
                ],
                "font.size": 11,
                "axes.titlesize": 11,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            }
        ):
            fig, ax = plt.subplots(figsize=figsize, facecolor="white")
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            y_pos = np.arange(len(display_labels))
            ax.set_ylim(len(display_labels) - 0.5, -0.5)

            pos_color = "#c44e52"
            neg_color = "#4c72b0"
            colors = [pos_color if value >= 0 else neg_color for value in display_values]
            ax.barh(y_pos, display_values, color=colors, height=0.88, edgecolor="none", linewidth=0, zorder=2)
            ax.axvline(0, color="#6f6f6f", lw=0.85, zorder=1)

            def _normalize_label(value: Optional[str], fallback: str) -> str:
                cleaned = str(value).strip() if value is not None else ""
                return cleaned if cleaned else fallback

            positive_label = _normalize_label(positive_class_name, "positive class")
            negative_label = _normalize_label(negative_class_name, "negative class")
            x_label = build_shap_axis_label(
                positive_class_name=positive_class_name,
                output_space=output_space,
            )

            from matplotlib.patches import Patch

            legend_handles = [
                Patch(facecolor=pos_color, edgecolor="none", label=f"Toward {positive_label}"),
                Patch(facecolor=neg_color, edgecolor="none", label=f"Toward {negative_label}"),
            ]
            ax.legend(
                handles=legend_handles,
                title=None,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.10),
                ncol=2,
                frameon=False,
                handlelength=1.0,
                handletextpad=0.45,
                columnspacing=1.3,
                borderaxespad=0.0,
                fontsize=max(7, xlabel_fontsize - 3),
                title_fontsize=max(7, xlabel_fontsize - 4),
            )

            max_abs = float(np.max(np.abs(display_values))) if len(display_values) else 0.0
            if max_abs > 0:
                ax.set_xlim(-max_abs * 1.32, max_abs * 1.32)
                x_min, x_max = ax.get_xlim()
                locator = ax.xaxis.get_major_locator()
                locator_ticks = np.asarray(locator.tick_values(x_min, x_max), dtype=float)
                locator_ticks = locator_ticks[np.isfinite(locator_ticks)]
                if len(locator_ticks) >= 2:
                    step = float(np.median(np.diff(locator_ticks)))
                    if step > 0:
                        eps = step * 1e-9
                        tick_min = np.floor((x_min + eps) / step) * step
                        tick_max = np.ceil((x_max - eps) / step) * step
                        tick_count = int(round((tick_max - tick_min) / step)) + 1
                        x_ticks = tick_min + np.arange(tick_count) * step
                        x_ticks = np.round(x_ticks, 10)
                        ax.set_xticks(x_ticks)
                        ax.set_xlim(float(x_ticks[0]), float(x_ticks[-1]))

            if show_text:
                x_label_pad = max_abs * 0.02 if max_abs > 0 else 0.02
                for y, value, label in zip(y_pos, display_values, display_labels):
                    if value >= 0:
                        x_pos = -x_label_pad
                        ha = "right"
                    else:
                        x_pos = x_label_pad
                        ha = "left"
                    ax.text(
                        x_pos,
                        y,
                        label,
                        transform=ax.transData,
                        ha=ha,
                        va="center",
                        fontsize=ytick_fontsize,
                        color="#222222",
                        clip_on=False,
                        zorder=3,
                    )

                ax.set_xlabel(x_label, fontsize=xlabel_fontsize, labelpad=2, color="#222222")
            else:
                hide_text_in_figure(fig)

            ax.set_yticks([])
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.grid(axis="x", linestyle="--", alpha=0.12, linewidth=0.5, color="#9a9a9a")
            for side in ["top", "right", "left"]:
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#b0b0b0")
            ax.spines["bottom"].set_linewidth(0.7)
            ax.tick_params(axis="x", labelsize=max(7, xlabel_fontsize - 1), colors="#222222", length=2.5, width=0.6)
            ax.margins(x=0.01, y=0.03)

            fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.22)
            saved_paths = save_current_figure(
                target_path,
                export_formats=formats,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.22,
            )
            plt.close(fig)
            return saved_paths

    saved_paths: list[str] = []
    if textless_svg_path is not None:
        saved_paths.extend(_render(False, textless_svg_path, ("svg",)))
    saved_paths.extend(_render(True, out_path, export_formats))
    return saved_paths


def plot_waterfall_samples(
    predictor: Any,
    results: dict,
    X_explain: pd.DataFrame,
    y_explain: pd.Series,
    output_dir: str,
    n_samples: int = 3,
    label_col: str = "label",
    sample_ids: Optional[pd.Series] = None,
    sample_filenames: Optional[List[str]] = None,
    n_top_features: int = 5,
    *,
    task_name: Optional[str] = None,
    positive_class_name: Optional[str] = None,
    negative_class_name: Optional[str] = None,
    output_space: Optional[str] = None,
) -> None:
    import csv

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        print("  Warning: shap or matplotlib not available. Skipping waterfall plots.")
        return

    import matplotlib.pyplot as plt
    _ensure_cjk_fonts()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
                "Noto Sans CJK SC", "Source Han Sans SC",
                "PingFang SC", "Hiragino Sans GB",
                "Heiti SC", "STHeiti", "Songti SC",
                "Microsoft YaHei", "SimHei",
                "Arial Unicode MS", "DejaVu Sans", "Arial",
            ],
            "font.size": 20,
            "axes.titlesize": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }
    )

    waterfall_dir = os.path.join(output_dir, "waterfall")
    compact_bar_dir = os.path.join(output_dir, "compact_shap_bar")
    os.makedirs(waterfall_dir, exist_ok=True)
    os.makedirs(compact_bar_dir, exist_ok=True)

    print(f"  Getting predictions for {len(X_explain)} samples...")
    y_pred = predictor.predict(X_explain)
    y_proba = predictor.predict_proba(X_explain)

    if isinstance(y_proba, pd.DataFrame):
        if 1 in y_proba.columns:
            proba_positive = y_proba[1].values
        else:
            proba_positive = y_proba.iloc[:, -1].values
    else:
        proba_positive = y_proba[:, 1] if y_proba.ndim == 2 else y_proba

    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(y_explain, pd.Series):
        y_explain = y_explain.values

    correct_mask = y_pred == y_explain
    incorrect_mask = ~correct_mask

    print(f"  Correct predictions: {correct_mask.sum()}/{len(y_explain)} ({correct_mask.sum()/len(y_explain)*100:.1f}%)")
    print(f"  Incorrect predictions: {incorrect_mask.sum()}/{len(y_explain)} ({incorrect_mask.sum()/len(y_explain)*100:.1f}%)")

    conf_pred = np.where(y_pred == 1, proba_positive, 1 - proba_positive)
    quality_score = np.where(y_pred == y_explain, conf_pred, -conf_pred)

    y_true = np.asarray(y_explain)
    positive_indices = np.where(y_true == 1)[0]
    negative_indices = np.where(y_true == 0)[0]
    positive_target = (n_samples + 1) // 2
    negative_target = n_samples // 2

    def _sort_indices_by_score(indices: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if len(indices) == 0:
            return np.array([], dtype=int)
        return indices[np.argsort(scores[indices])[::-1]]

    def _take_middle(sorted_indices: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(sorted_indices) == 0:
            return np.array([], dtype=int)
        count = min(count, len(sorted_indices))
        center = len(sorted_indices) // 2
        start = max(0, center - count // 2)
        end = start + count
        if end > len(sorted_indices):
            end = len(sorted_indices)
            start = max(0, end - count)
        return sorted_indices[start:end]

    def _take_best(sorted_indices: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(sorted_indices) == 0:
            return np.array([], dtype=int)
        return sorted_indices[: min(count, len(sorted_indices))]

    def _take_worst(sorted_indices: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(sorted_indices) == 0:
            return np.array([], dtype=int)
        count = min(count, len(sorted_indices))
        return sorted_indices[-count:]

    def _sort_by_confidence(indices: np.ndarray) -> np.ndarray:
        if len(indices) == 0:
            return np.array([], dtype=int)
        return indices[np.argsort(conf_pred[indices])[::-1]]

    def _select_balanced_quality(indices: np.ndarray, count: int, mode: str) -> np.ndarray:
        sorted_indices = _sort_indices_by_score(indices, quality_score)
        if mode == "best":
            return _take_best(sorted_indices, count)
        if mode == "medium":
            return _take_middle(sorted_indices, count)
        if mode == "worst":
            return _take_worst(sorted_indices, count)
        raise ValueError(f"Unknown quality mode: {mode}")

    def _select_balanced_correct(indices: np.ndarray, count: int) -> np.ndarray:
        selected = indices[correct_mask[indices]]
        return _take_best(_sort_by_confidence(selected), count)

    pos_best = _select_balanced_quality(positive_indices, positive_target, "best")
    neg_best = _select_balanced_quality(negative_indices, negative_target, "best")
    pos_medium = _select_balanced_quality(positive_indices, positive_target, "medium")
    neg_medium = _select_balanced_quality(negative_indices, negative_target, "medium")
    pos_worst = _select_balanced_quality(positive_indices, positive_target, "worst")
    neg_worst = _select_balanced_quality(negative_indices, negative_target, "worst")
    pos_correct = _select_balanced_correct(positive_indices, positive_target)
    neg_correct = _select_balanced_correct(negative_indices, negative_target)

    best_indices = _sort_indices_by_score(np.concatenate([pos_best, neg_best]), quality_score)
    medium_indices = _sort_indices_by_score(np.concatenate([pos_medium, neg_medium]), quality_score)
    worst_indices = _sort_indices_by_score(np.concatenate([pos_worst, neg_worst]), quality_score)
    correct_indices = _sort_by_confidence(np.concatenate([pos_correct, neg_correct]))

    print(
        f"  Balanced waterfall targets per category - total: {n_samples}, positive: {positive_target}, negative: {negative_target}"
    )
    print(
        f"  Selected counts - best: {len(best_indices)}, medium: {len(medium_indices)}, worst: {len(worst_indices)}, "
        f"correct: {len(correct_indices)}"
    )

    if len(positive_indices) == 0:
        print("  Warning: No positive samples found. Positive half will be empty.")
    if len(negative_indices) == 0:
        print("  Warning: No negative samples found. Negative half will be empty.")
    if len(positive_indices) < positive_target:
        print("  Warning: Not enough positive samples to fill the positive half for every category.")
    if len(negative_indices) < negative_target:
        print("  Warning: Not enough negative samples to fill the negative half for every category.")
    if len(pos_correct) < positive_target or len(neg_correct) < negative_target:
        print("  Warning: Not enough correctly predicted positive/negative samples to fill the balanced correct category.")

    sample_ids_array = None
    if sample_ids is not None:
        try:
            aligned_ids = sample_ids.loc[X_explain.index]
        except Exception:
            aligned_ids = sample_ids
        sample_ids_array = np.array(aligned_ids)
    waterfall_records = []

    specified_indices = None
    if sample_filenames is not None and sample_ids_array is not None:
        if isinstance(sample_filenames, str):
            filenames_list = [sample_filenames]
        else:
            filenames_list = list(sample_filenames)

        collected_indices: List[int] = []
        for fname in filenames_list:
            try:
                direct_matches = np.where(sample_ids_array == fname)[0]
                if len(direct_matches) == 0:
                    target_base = os.path.basename(str(fname))
                    bases = np.array([os.path.basename(str(x)) for x in sample_ids_array])
                    direct_matches = np.where(bases == target_base)[0]

                if len(direct_matches) == 0:
                    print(f"  Warning: sample_filename '{fname}' not found among explained samples.")
                else:
                    idx0 = int(direct_matches[0])
                    collected_indices.append(idx0)
                    print(f"  sample_filename '{fname}' matched explained index {idx0}.")
            except Exception as e:
                print(f"  Warning: failed to locate sample_filename '{fname}': {e}")

        if collected_indices:
            specified_indices = np.array(sorted(set(collected_indices)), dtype=int)

    for model_name, model_results in results.items():
        shap_values = model_results["shap_values"]
        shap_df = model_results["shap_df"]

        X_processed = None
        try:
            if hasattr(predictor, "_learner") and hasattr(predictor._learner, "feature_generator"):
                X_processed = predictor._learner.feature_generator.transform(X_explain)
                if not isinstance(X_processed, pd.DataFrame):
                    X_processed = pd.DataFrame(X_processed, columns=shap_df.columns, index=X_explain.index)
        except Exception as e:
            print(f"    Warning: Could not get preprocessed features: {e}")
            X_processed = X_explain

        def _plot_category(indices, category_label: str, filename_tag: str, title_tag: str) -> None:
            if indices is None or len(indices) == 0:
                return
            print(f"  Plotting {len(indices)} {category_label} samples for {model_name}...")
            for i, idx in enumerate(indices):
                try:
                    sample_shap = shap_values[idx]
                    sample_features = X_processed.iloc[idx] if X_processed is not None else X_explain.iloc[idx]

                    if sample_shap.ndim > 1:
                        sample_shap = sample_shap.flatten()
                    elif sample_shap.ndim == 0:
                        sample_shap = np.array([sample_shap])

                    max_display = max(1, int(n_top_features))
                    abs_shap = np.abs(sample_shap)
                    top_indices = np.argsort(abs_shap)[-max_display:][::-1]

                    top_shap = sample_shap[top_indices]
                    top_feature_names = [shap_df.columns[i] for i in top_indices]

                    top_feature_values = []
                    if isinstance(sample_features, pd.Series):
                        for name in top_feature_names:
                            if name in sample_features.index:
                                top_feature_values.append(sample_features[name])
                            else:
                                try:
                                    idx_in_shap = shap_df.columns.get_loc(name)
                                    if idx_in_shap < len(sample_features):
                                        top_feature_values.append(sample_features.iloc[idx_in_shap])
                                    else:
                                        top_feature_values.append(np.nan)
                                except (KeyError, IndexError):
                                    top_feature_values.append(np.nan)

                    nan_count = sum(1 for v in top_feature_values if pd.isna(v))
                    if nan_count > 0:
                        print(
                            f"    Warning: {nan_count} out of {len(top_feature_values)} "
                            f"top features have nan values"
                        )
                        print(f"    Top feature names: {top_feature_names[:3]}...")
                        if isinstance(sample_features, pd.Series):
                            print(f"    Sample features index (first 5): {list(sample_features.index[:5])}")
                    elif isinstance(sample_features, pd.DataFrame):
                        top_feature_values = [
                            sample_features.iloc[0, sample_features.columns.get_loc(name)]
                            if name in sample_features.columns
                            else np.nan
                            for name in top_feature_names
                        ]
                    else:
                        if isinstance(sample_features, np.ndarray):
                            top_feature_values = (
                                sample_features[top_indices]
                                if len(sample_features) > max(top_indices)
                                else [np.nan] * len(top_indices)
                            )
                        elif isinstance(sample_features, pd.Series):
                            top_feature_values = [
                                sample_features.iloc[j] if j < len(sample_features) else np.nan
                                for j in top_indices
                            ]
                        else:
                            top_feature_values = [np.nan] * len(top_indices)

                    display_shap = top_shap
                    display_feature_names = [paper_friendly_name(name) for name in top_feature_names]
                    compact_feature_names = list(top_feature_names)

                    display_feature_values = np.array(top_feature_values)
                    if display_feature_values.ndim > 1:
                        display_feature_values = display_feature_values.flatten()

                    base_value = float(np.mean(proba_positive))

                    waterfall_plot_file = os.path.join(
                        waterfall_dir, f"{model_name}_waterfall_{filename_tag}_{i+1}.png"
                    )
                    waterfall_textless_path = os.path.join(
                        waterfall_dir, f"{model_name}_waterfall_{filename_tag}_{i+1}_textless.svg"
                    )
                    waterfall_saved_paths = save_waterfall_plot(
                        display_shap,
                        display_feature_values,
                        display_feature_names,
                        base_value,
                        waterfall_plot_file,
                        max_display,
                        textless_svg_path=waterfall_textless_path,
                        title=f"{model_name} {title_tag} SHAP",
                        title_fontsize=24,
                        export_formats=("png", "svg"),
                        dpi=150,
                        figsize=(13.0, 9.0),
                        bbox_inches="tight",
                        positive_class_name=positive_class_name,
                        output_space=output_space,
                    )

                    compact_bar_file = os.path.join(
                        compact_bar_dir, f"{model_name}_compact_shap_bar_{filename_tag}_{i+1}.png"
                    )
                    compact_bar_textless_path = os.path.join(
                        compact_bar_dir, f"{model_name}_compact_shap_bar_{filename_tag}_{i+1}_textless.svg"
                    )
                    compact_bar_saved_paths = save_compact_shap_bar_plot(
                        display_shap,
                        compact_feature_names,
                        compact_bar_file,
                        max_display,
                        task_name=task_name,
                        positive_class_name=positive_class_name,
                        negative_class_name=negative_class_name,
                        output_space=output_space,
                        textless_svg_path=compact_bar_textless_path,
                        xlabel_fontsize=12.0,
                        ytick_fontsize=11.0,
                        export_formats=("png", "svg"),
                        dpi=300,
                        figsize=(3.15, 3.15 * 4 / 3),
                    )
                    print(
                        f"    Saved waterfall: {', '.join(waterfall_saved_paths)}; "
                        f"compact bar: {', '.join(compact_bar_saved_paths)}"
                    )

                    if sample_ids_array is not None:
                        img_name = sample_ids_array[idx] if idx < len(sample_ids_array) else None
                        base, _ = os.path.splitext(waterfall_plot_file)
                        waterfall_records.append(
                            {
                                "model": model_name,
                                "category": category_label,
                                "figure_index": i + 1,
                                "image_name": img_name,
                                "true_label": int(y_explain[idx]),
                                "pred_label": int(y_pred[idx]),
                                "prob_positive": float(proba_positive[idx]),
                                "plot_file": os.path.basename(base),
                            }
                        )
                except Exception as e:
                    print(f"    Error plotting {category_label} sample {i+1}: {e}")
                    import traceback

                    traceback.print_exc()

        _plot_category(best_indices, "best", "best", "Best-quality")
        _plot_category(medium_indices, "medium", "medium", "Medium-quality")
        _plot_category(worst_indices, "worst", "worst", "Worst-quality")
        _plot_category(correct_indices, "correct", "correct", "Correct")

        if specified_indices is not None:
            _plot_category(specified_indices, "specified", "sample", "Specified-sample")

    print(f"  Waterfall and compact bar plots saved to: {output_dir}")

    if sample_ids_array is not None and waterfall_records:
        csv_path = os.path.join(output_dir, "waterfall_sample_images.csv")
        fieldnames = [
            "model",
            "category",
            "figure_index",
            "image_name",
            "true_label",
            "pred_label",
            "prob_positive",
            "plot_file",
        ]
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(waterfall_records)
            print(f"  Saved waterfall sample image names to: {csv_path}")
        except Exception as e:
            print(f"  Warning: failed to save waterfall sample image names: {e}")
