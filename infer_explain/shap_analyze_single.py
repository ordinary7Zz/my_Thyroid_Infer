"""批量/单图生成局部 SHAP 解释图。

这个脚本用于读取已经训练好的 AutoGluon 表格分类模型，以及对应的 radiomics
特征表（`TRAIN_CSV`），为指定样本生成两类局部解释图：

1. `waterfall`：展示单个样本最重要特征对模型输出的逐步贡献。
2. `compact_shap_bar`：展示单个样本 top-k 特征的正负向 SHAP 贡献条形图。

脚本支持两种运行模式：

- 单图模式：通过 `--filename` 指定一个目标文件名，只解释这一个样本。
- 批量模式：通过 `--filename_list` 指定一个文件名清单文件。这个清单文件可以是
  `txt`、`csv` 或其他纯文本格式，只要其中保存了待解释的文件名即可。脚本会读取
  这些文件名，并在 `TRAIN_CSV` 中按 basename（文件名，不含目录）进行匹配，随后
  逐个输出 SHAP 解释图。

输入与匹配规则说明：

- `TRAIN_CSV` 中必须包含 `filename` 或 `image_path` 列，用于定位目标样本。
- 批量模式不会去扫描图像目录，而是完全依赖 `--filename_list` 提供待处理文件名。
- 无论清单文件中给的是完整路径还是纯文件名，最终都只按 basename 匹配。
- 若某个文件名在 `TRAIN_CSV` 中找不到，会打印 warning 并跳过。
- 若同一个 basename 匹配到多行，会沿用旧逻辑：给出 warning，并使用第一条匹配。

输出结构说明：

- 所有 `waterfall` 图保存在 `<output_dir>/waterfall/`。
- 所有 `compact_shap_bar` 图保存在 `<output_dir>/compact_shap_bar/`。
- 无论单图模式还是批量模式，输出文件名都直接使用原图 basename（不再附加模型名或图类型前缀）。
- 如果一次解释多个模型，则会在 `waterfall/` 和 `compact_shap_bar/` 下按模型名额外创建子目录，
  避免不同模型的同名结果互相覆盖。

这样设计的目的，是让局部 SHAP 图的文件名与原图样本一一对应，同时保持多模型导出时的路径稳定。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from plotting_utils import format_feature_name, prepare_df, save_waterfall_plot
from autogluon_introspection import get_main_models, load_autogluon_model
from shap_compute import compute_shap_for_model
from shap_local_plots import save_compact_shap_bar_plot


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate waterfall and compact SHAP bar plots for one image or a filename list from TRAIN_CSV."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="AutoGluon model directory")
    parser.add_argument("--train_csv", type=str, required=True, help="Training CSV file")
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Target filename to explain in single-image mode",
    )
    parser.add_argument(
        "--filename_list",
        type=str,
        default=None,
        help="Text/CSV file that stores multiple target filenames for batch explanation",
    )
    parser.add_argument("--label", type=str, default="label", help="Label column name")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <model_dir>/shap_analysis_single_image)",
    )
    parser.add_argument(
        "--background_samples",
        type=int,
        default=100,
        help="Number of background samples for SHAP",
    )
    parser.add_argument(
        "--main_models",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of main model names to analyze",
    )
    parser.add_argument(
        "--skip_neural_net",
        action="store_true",
        help="Skip neural network models (NeuralNetFastAI) in SHAP analysis",
    )
    parser.add_argument(
        "--top_features",
        type=int,
        default=5,
        help="Number of top features shown in waterfall and compact SHAP bar plots (default: 5)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Reserved task name argument kept for backward compatibility",
    )
    parser.add_argument(
        "--positive_class_name",
        type=str,
        default=None,
        help="Human-readable name for label 1, shown in the compact SHAP x-axis label and legend",
    )
    parser.add_argument(
        "--negative_class_name",
        type=str,
        default=None,
        help="Human-readable name for label 0, shown in the compact SHAP legend",
    )
    parser.add_argument(
        "--output_space",
        type=str,
        default=None,
        help="SHAP output space used for labeling, e.g. probability or raw score",
    )
    parser.add_argument(
        "--feature_label_lang",
        type=str,
        choices=("en", "zh"),
        default="en",
        help="Feature label language used in plots: 'en' or 'zh' (default: en)",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    has_filename = bool(str(args.filename).strip()) if args.filename is not None else False
    has_filename_list = bool(str(args.filename_list).strip()) if args.filename_list is not None else False

    if has_filename == has_filename_list:
        raise ValueError("Exactly one of --filename or --filename_list must be provided.")

    if has_filename_list:
        list_path = Path(str(args.filename_list).strip()).expanduser()
        if not list_path.is_file():
            raise ValueError(f"filename_list file does not exist: {list_path}")


def _basenameish(value: object) -> str:
    return os.path.basename(str(value))


def _safe_token(value: object) -> str:
    token = _basenameish(value)
    token = os.path.splitext(token)[0]
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", token)
    return token.strip("._") or "sample"


def _positive_probability(y_proba: object) -> np.ndarray:
    if isinstance(y_proba, pd.DataFrame):
        if 1 in y_proba.columns:
            return y_proba[1].to_numpy()
        return y_proba.iloc[:, -1].to_numpy()
    if isinstance(y_proba, np.ndarray):
        if y_proba.ndim == 2:
            return y_proba[:, 1]
        return y_proba.reshape(-1)
    return np.asarray(y_proba).reshape(-1)


def _resolve_id_col(raw_df: pd.DataFrame) -> str:
    if "filename" in raw_df.columns:
        return "filename"
    if "image_path" in raw_df.columns:
        return "image_path"
    raise ValueError("TRAIN_CSV must contain a 'filename' or 'image_path' column.")


def _resolve_target_row(raw_df: pd.DataFrame, train_df: pd.DataFrame, filename: str) -> tuple[object, str]:
    id_col = _resolve_id_col(raw_df)

    sample_ids = raw_df.loc[train_df.index, id_col].astype(str)
    normalized_ids = sample_ids.map(_basenameish)
    query = str(filename)
    query_base = _basenameish(filename)
    matches = np.where((sample_ids == query) | (normalized_ids == query_base))[0]

    if len(matches) == 0:
        raise ValueError(f"filename '{filename}' was not found in TRAIN_CSV.")

    if len(matches) > 1:
        print(f"Warning: filename '{filename}' matched multiple rows; using the first match.")

    matched_pos = int(matches[0])
    target_index = train_df.index[matched_pos]
    matched_name = normalized_ids.iloc[matched_pos]
    return target_index, matched_name


def _looks_like_header_token(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {"filename", "file_name", "image_path", "path", "file", "name"}


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def _extract_first_non_empty_cell(row: Sequence[str]) -> Optional[str]:
    for cell in row:
        cleaned = str(cell).strip()
        if cleaned:
            return cleaned
    return None


def _load_csv_filename_list(list_path: Path) -> list[str]:
    with list_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        rows = [[cell.strip() for cell in row] for row in csv.reader(handle, dialect) if any(cell.strip() for cell in row)]

    if not rows:
        return []

    header = [cell.strip().lower() for cell in rows[0]]
    for column_name in ("filename", "image_path", "path", "file", "name"):
        if column_name in header:
            column_index = header.index(column_name)
            values = []
            for row in rows[1:]:
                if column_index < len(row) and row[column_index].strip():
                    values.append(row[column_index].strip())
            return _unique_preserve_order(values)

    data_rows = rows[1:] if len(rows[0]) == 1 and _looks_like_header_token(rows[0][0]) else rows
    return _unique_preserve_order(
        value
        for value in (_extract_first_non_empty_cell(row) for row in data_rows)
        if value is not None
    )


def _load_plaintext_filename_list(list_path: Path) -> list[str]:
    values: list[str] = []
    for raw_line in list_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _looks_like_header_token(line):
            continue
        if "," in line or "\t" in line or ";" in line:
            pieces = re.split(r"[,\t;]", line)
            candidate = _extract_first_non_empty_cell(pieces)
            if candidate is None or _looks_like_header_token(candidate):
                continue
            values.append(candidate)
        else:
            values.append(line)
    return _unique_preserve_order(values)


def load_target_filenames(filename_list_path: str) -> list[str]:
    list_path = Path(filename_list_path).expanduser().resolve()
    if not list_path.is_file():
        raise FileNotFoundError(f"filename list file was not found: {list_path}")

    if list_path.suffix.lower() == ".csv":
        filenames = _load_csv_filename_list(list_path)
    else:
        filenames = _load_plaintext_filename_list(list_path)

    if not filenames:
        raise ValueError(f"No valid filenames were found in: {list_path}")
    return filenames


def _collect_target_rows(
    raw_df: pd.DataFrame,
    train_df: pd.DataFrame,
    args: argparse.Namespace,
) -> list[tuple[object, str]]:
    if args.filename:
        target_index, target_name = _resolve_target_row(raw_df, train_df, args.filename)
        print(f"Selected target sample: {target_name} (row index: {target_index})")
        return [(target_index, target_name)]

    requested_filenames = load_target_filenames(str(args.filename_list))
    print(f"Loaded {len(requested_filenames)} requested filenames from: {args.filename_list}")

    selected_rows: list[tuple[object, str]] = []
    seen_indices: set[object] = set()
    missing_names: list[str] = []

    for requested_name in requested_filenames:
        try:
            target_index, target_name = _resolve_target_row(raw_df, train_df, requested_name)
        except ValueError:
            missing_names.append(_basenameish(requested_name))
            print(f"Warning: requested filename '{requested_name}' was not found in TRAIN_CSV. Skipping.")
            continue

        if target_index in seen_indices:
            print(f"Warning: filename '{requested_name}' resolved to a duplicated sample index. Skipping duplicate.")
            continue

        seen_indices.add(target_index)
        selected_rows.append((target_index, target_name))

    if missing_names:
        print(f"Skipped {len(missing_names)} filenames that were not found in TRAIN_CSV.")
    if not selected_rows:
        raise ValueError("None of the filenames from --filename_list matched TRAIN_CSV.")

    print(f"Selected {len(selected_rows)} samples for batch explanation.")
    return selected_rows


def _get_sample_features(predictor, x_explain: pd.DataFrame, shap_df: pd.DataFrame):
    try:
        if hasattr(predictor, "_learner") and hasattr(predictor._learner, "feature_generator"):
            processed = predictor._learner.feature_generator.transform(x_explain)
            if isinstance(processed, pd.DataFrame):
                return processed.iloc[0]
            return np.asarray(processed)[0]
    except Exception as exc:
        print(f"Warning: could not transform features for display: {exc}")
    return x_explain.iloc[0]


def _extract_feature_values(sample_features, top_indices: np.ndarray, top_feature_names: list[str], shap_columns: list[str]):
    values = []
    if isinstance(sample_features, pd.Series):
        for name, idx in zip(top_feature_names, top_indices):
            if name in sample_features.index:
                values.append(sample_features[name])
            else:
                try:
                    pos = shap_columns.index(name)
                    values.append(sample_features.iloc[pos])
                except Exception:
                    values.append(np.nan)
    elif isinstance(sample_features, np.ndarray):
        for idx in top_indices:
            values.append(sample_features[idx] if idx < len(sample_features) else np.nan)
    else:
        for idx in top_indices:
            try:
                values.append(sample_features[idx])
            except Exception:
                values.append(np.nan)
    return values


def _build_output_paths(
    output_dir: str,
    model_name: str,
    target_name: str,
    *,
    batch_mode: bool,
    multi_model: bool,
) -> tuple[str, str, str, str]:
    waterfall_dir = os.path.join(output_dir, "waterfall")
    compact_bar_dir = os.path.join(output_dir, "compact_shap_bar")

    if multi_model:
        waterfall_dir = os.path.join(waterfall_dir, model_name)
        compact_bar_dir = os.path.join(compact_bar_dir, model_name)

    os.makedirs(waterfall_dir, exist_ok=True)
    os.makedirs(compact_bar_dir, exist_ok=True)

    target_tag = _safe_token(target_name)
    waterfall_path = os.path.join(waterfall_dir, f"{target_tag}.png")
    waterfall_textless_path = os.path.join(waterfall_dir, f"{target_tag}_textless.svg")
    compact_bar_path = os.path.join(compact_bar_dir, f"{target_tag}.png")
    compact_bar_textless_path = os.path.join(compact_bar_dir, f"{target_tag}_textless.svg")

    return waterfall_path, waterfall_textless_path, compact_bar_path, compact_bar_textless_path


def _save_shap_plots_for_target(
    predictor,
    main_models: Sequence[str],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    target_index,
    target_name: str,
    args: argparse.Namespace,
    output_dir: str,
    *,
    batch_mode: bool,
) -> None:
    x_explain = x_train.loc[[target_index]].copy()
    y_explain = y_train.loc[[target_index]].copy()

    background_pool = x_train.drop(index=target_index, errors="ignore")
    if len(background_pool) == 0:
        background_pool = x_train.copy()

    np.random.seed(42)
    if len(background_pool) > args.background_samples:
        background_idx = np.random.choice(len(background_pool), size=args.background_samples, replace=False)
        x_background = background_pool.iloc[background_idx].copy()
    else:
        x_background = background_pool.copy()

    print(f"Using {len(x_background)} background samples for {target_name}")
    print(f"Explaining sample '{target_name}' with label={int(y_explain.iloc[0])}")

    multi_model = len(main_models) > 1
    for model_name in main_models:
        print(f"\nAnalyzing model: {model_name} | sample: {target_name}")
        result = compute_shap_for_model(predictor, model_name, x_background, x_explain, args.skip_neural_net)
        if result is None:
            continue

        shap_values, shap_df = result
        sample_shap = np.asarray(shap_values[0]).reshape(-1)
        feature_names = shap_df.columns.tolist()
        if len(sample_shap) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch for {model_name}: {len(feature_names)} names vs {len(sample_shap)} SHAP values"
            )

        max_display = max(1, int(args.top_features))
        top_indices = np.argsort(np.abs(sample_shap))[-max_display:][::-1]
        top_shap = sample_shap[top_indices]
        top_feature_names = [feature_names[i] for i in top_indices]
        display_feature_names = [format_feature_name(name, args.feature_label_lang) for name in top_feature_names]

        sample_features = _get_sample_features(predictor, x_explain, shap_df)
        top_feature_values = _extract_feature_values(sample_features, top_indices, top_feature_names, feature_names)

        try:
            model = load_autogluon_model(predictor, model_name)
            model_proba = model.predict_proba(x_background)
        except Exception as exc:
            print(f"Warning: could not load model-specific predictor for base value; using predictor output instead: {exc}")
            model_proba = predictor.predict_proba(x_background)

        base_value = float(np.mean(_positive_probability(model_proba)))
        waterfall_path, waterfall_textless_path, compact_bar_path, compact_bar_textless_path = _build_output_paths(
            output_dir,
            model_name,
            target_name,
            batch_mode=batch_mode,
            multi_model=multi_model,
        )

        waterfall_saved = save_waterfall_plot(
            top_shap,
            top_feature_values,
            display_feature_names,
            base_value,
            waterfall_path,
            max_display,
            textless_svg_path=waterfall_textless_path,
            title=f"{model_name} SHAP",
            title_fontsize=24,
            export_formats=("png", "svg"),
            dpi=150,
            figsize=(14.5, 9.0),
            bbox_inches="tight",
            positive_class_name=args.positive_class_name,
            output_space=args.output_space,
        )

        compact_bar_saved = save_compact_shap_bar_plot(
            top_shap,
            top_feature_names,
            compact_bar_path,
            max_display,
            positive_class_name=args.positive_class_name,
            negative_class_name=args.negative_class_name,
            output_space=args.output_space,
            feature_label_lang=args.feature_label_lang,
            textless_svg_path=compact_bar_textless_path,
            xlabel_fontsize=12.0,
            ytick_fontsize=11.0,
            export_formats=("png", "svg"),
            dpi=300,
            figsize=(3.35, 3.35 * 4 / 3),
        )

        print(f"Saved waterfall: {', '.join(waterfall_saved)}")
        print(f"Saved compact bar: {', '.join(compact_bar_saved)}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    validate_args(args)

    batch_mode = args.filename_list is not None and str(args.filename_list).strip() != ""

    print(f"Loading predictor from: {args.model_dir}")
    predictor = __import__("autogluon.tabular", fromlist=["TabularPredictor"]).TabularPredictor.load(args.model_dir)

    print(f"Loading training data from: {args.train_csv}")
    raw_df = pd.read_csv(args.train_csv)
    train_df = prepare_df(raw_df.copy(), args.label)

    target_rows = _collect_target_rows(raw_df, train_df, args)

    x_train = train_df.drop(columns=[args.label]).copy()
    y_train = train_df[args.label].copy()

    main_models = get_main_models(predictor, args.model_dir, args.main_models)
    print(f"Main models to analyze: {main_models}")

    output_dir = args.output_dir or os.path.join(args.model_dir, "shap_analysis_single_image")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    processed_count = 0
    for target_index, target_name in target_rows:
        try:
            _save_shap_plots_for_target(
                predictor,
                main_models,
                x_train,
                y_train,
                target_index,
                target_name,
                args,
                output_dir,
                batch_mode=batch_mode,
            )
            processed_count += 1
        except Exception as exc:
            print(f"Error: failed to generate SHAP plots for '{target_name}': {exc}")
            if not batch_mode:
                raise

    print(f"\nDone. Generated SHAP plots for {processed_count}/{len(target_rows)} samples. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
