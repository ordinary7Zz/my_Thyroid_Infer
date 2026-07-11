from __future__ import annotations

import os
import re
from typing import Callable, Optional, Sequence

np = None
pd = None
plt = None
shap = None

# ---------------------------------------------------------------------------
# CJK font auto-detection: scan common system paths and register fonts so that
# matplotlib can find Chinese fonts even without a fully configured fontconfig.
# ---------------------------------------------------------------------------
_CJK_FONT_INITIALIZED = False


def _ensure_cjk_fonts() -> None:
    """Register CJK fonts with matplotlib's font manager if not yet done.

    After installing system fonts (e.g. fonts-wqy-microhei), matplotlib may
    still use a stale font cache. This function:
    1. Scans common system font directories and registers every .ttf/.ttc/.otf
       file via ``fontManager.addfont()``.
    2. If addfont is unavailable (older matplotlib), it rebuilds the entire
       font manager from scratch.
    """
    global _CJK_FONT_INITIALIZED
    if _CJK_FONT_INITIALIZED:
        return
    _CJK_FONT_INITIALIZED = True

    try:
        import matplotlib
        from matplotlib import font_manager as fm

        # Common directories where CJK fonts may reside (Linux + macOS)
        _FONT_DIRS = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.local/share/fonts"),
            os.path.expanduser("~/.fonts"),
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]

        has_addfont = hasattr(fm.fontManager, "addfont")

        if has_addfont:
            # matplotlib >= 3.2: use addfont for incremental registration
            for font_dir in _FONT_DIRS:
                if not os.path.isdir(font_dir):
                    continue
                for root, _dirs, files in os.walk(font_dir):
                    for fname in files:
                        if fname.lower().endswith((".ttf", ".ttc", ".otf")):
                            fpath = os.path.join(root, fname)
                            try:
                                fm.fontManager.addfont(fpath)
                            except Exception:
                                pass
        else:
            # Older matplotlib: rebuild font manager entirely
            fm._rebuild()

        # Delete the on-disk font cache so future runs also pick up the fonts
        cache_dir = matplotlib.get_cachedir()
        if cache_dir:
            import glob
            for cache_file in glob.glob(os.path.join(cache_dir, "fontlist-*.json")):
                try:
                    os.remove(cache_file)
                except OSError:
                    pass

        # Set global default font fallback list that includes CJK-capable fonts
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [
            "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
            "Noto Sans CJK SC", "Source Han Sans SC",
            "PingFang SC", "Hiragino Sans GB",
            "Heiti SC", "STHeiti", "Songti SC",
            "Microsoft YaHei", "SimHei",
            "Arial Unicode MS", "DejaVu Sans", "Arial",
        ]
        # Prevent minus sign rendering issues with CJK fonts
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

DROP_IF_PRESENT = ["image_path", "mask_path", "filename"]

MANUAL_MAP = {
    "original_shape2D_Sphericity": "SHAPE: Sphericity",
    "original_shape2D_Elongation": "SHAPE: Elongation",
    "original_shape2D_Perimeter": "SHAPE: Perimeter",
    "original_shape2D_PerimeterSurfaceRatio": "SHAPE: Perim/Area",
    "original_shape2D_MajorAxisLength": "SHAPE: Major axis",
    "original_shape2D_MeshSurface": "SHAPE: Area",
    "original_shape2D_MinorAxisLength": "SHAPE: Minor axis",
    "original_firstorder_Skewness": "INT: Skewness",
    "original_firstorder_Minimum": "INT: Minimum",
    "original_firstorder_Median": "INT: Median",
    "original_firstorder_Energy": "INT: Energy",
    "original_firstorder_RobustMeanAbsoluteDeviation": "INT: Robust MAD",
    "original_firstorder_Kurtosis": "INT: Kurtosis",
    "original_firstorder_10Percentile": "INT: 10th pctl",
    "original_glrlm_LongRunHighGrayLevelEmphasis": "RLM: LRHGLE",
    "original_glrlm_ShortRunHighGrayLevelEmphasis": "RLM: SRHGLE",
    "original_glrlm_LongRunLowGrayLevelEmphasis": "RLM: LRLGLE",
    "original_glrlm_ShortRunLowGrayLevelEmphasis": "RLM: SRLGLE",
    "original_glrlm_ShortRunEmphasis": "RLM: SRE",
    "original_glrlm_LongRunEmphasis": "RLM: LRE",
    "original_glrlm_RunVariance": "RLM: RunVar",
    "original_glrlm_RunEntropy": "RLM: RunEnt",
    "original_glrlm_GrayLevelNonUniformity": "RLM: GLNU",
    "original_glrlm_RunLengthNonUniformity": "RLM: RLNU",
    "original_glszm_SmallAreaHighGrayLevelEmphasis": "SZM: SAHGLE",
    "original_glszm_LargeAreaHighGrayLevelEmphasis": "SZM: LAHGLE",
    "original_glszm_SmallAreaLowGrayLevelEmphasis": "SZM: SALGLE",
    "original_glszm_LargeAreaLowGrayLevelEmphasis": "SZM: LALGLE",
    "original_glszm_ZoneVariance": "SZM: ZoneVar",
    "original_glszm_ZoneEntropy": "SZM: ZoneEnt",
    "original_glcm_Correlation": "GLCM: Corr",
    "original_glcm_Contrast": "GLCM: Contrast",
    "original_glcm_Energy": "GLCM: Energy",
    "original_glcm_Homogeneity": "GLCM: Homog",
    "original_gldm_DependenceNonUniformity": "DM: DepNU",
    "original_gldm_DependenceVariance": "DM: DepVar",
    "original_gldm_LargeDependenceHighGrayLevelEmphasis": "DM: LDHGLE",
    "original_gldm_LargeDependenceEmphasis": "DM: LDE",
    "original_ngtdm_Contrast": "NGTDM: Contrast",
    "original_ngtdm_Coarseness": "NGTDM: Coarse",
    "original_ngtdm_Complexity": "NGTDM: Complex",
}

MANUAL_MAP_ZH = {
    # ── Shape2D (形状特征) ──
    "original_shape2D_Elongation": "形状: 伸长度",
    "original_shape2D_MajorAxisLength": "形状: 长轴",
    "original_shape2D_MaximumDiameter": "形状: 最大径",
    "original_shape2D_MeshSurface": "形状: 面积",
    "original_shape2D_MinorAxisLength": "形状: 短轴",
    "original_shape2D_Perimeter": "形状: 周长",
    "original_shape2D_PerimeterSurfaceRatio": "形状: 周面比",
    "original_shape2D_PixelSurface": "形状: 像素面积",
    "original_shape2D_Sphericity": "形状: 球形度",
    # ── First Order (强度特征) ──
    "original_firstorder_10Percentile": "强度: 10分位",
    "original_firstorder_90Percentile": "强度: 90分位",
    "original_firstorder_Energy": "强度: 能量",
    "original_firstorder_Entropy": "强度: 熵",
    "original_firstorder_InterquartileRange": "强度: 四分位距",
    "original_firstorder_Kurtosis": "强度: 峰度",
    "original_firstorder_Maximum": "强度: 最大值",
    "original_firstorder_Mean": "强度: 均值",
    "original_firstorder_MeanAbsoluteDeviation": "强度: 平均绝对偏差",
    "original_firstorder_Median": "强度: 中位数",
    "original_firstorder_Minimum": "强度: 最小值",
    "original_firstorder_Range": "强度: 极差",
    "original_firstorder_RobustMeanAbsoluteDeviation": "强度: 稳健MAD",
    "original_firstorder_RootMeanSquared": "强度: 均方根",
    "original_firstorder_Skewness": "强度: 偏度",
    "original_firstorder_TotalEnergy": "强度: 总能量",
    "original_firstorder_Uniformity": "强度: 均匀度",
    "original_firstorder_Variance": "强度: 方差",
    # ── GLCM (灰度共生矩阵) ──
    "original_glcm_Autocorrelation": "纹理GLCM: 自相关",
    "original_glcm_ClusterProminence": "纹理GLCM: 集群突出",
    "original_glcm_ClusterShade": "纹理GLCM: 集群阴影",
    "original_glcm_ClusterTendency": "纹理GLCM: 集群趋势",
    "original_glcm_Contrast": "纹理GLCM: 对比度",
    "original_glcm_Correlation": "纹理GLCM: 相关性",
    "original_glcm_DifferenceAverage": "纹理GLCM: 差均值",
    "original_glcm_DifferenceEntropy": "纹理GLCM: 差熵",
    "original_glcm_DifferenceVariance": "纹理GLCM: 差方差",
    "original_glcm_Energy": "纹理GLCM: 能量",
    "original_glcm_Homogeneity": "纹理GLCM: 同质性",
    "original_glcm_Id": "纹理GLCM: 逆差",
    "original_glcm_Idm": "纹理GLCM: 逆差矩",
    "original_glcm_Idmn": "纹理GLCM: 归一逆差矩",
    "original_glcm_Idn": "纹理GLCM: 归一逆差",
    "original_glcm_Imc1": "纹理GLCM: 信息度1",
    "original_glcm_Imc2": "纹理GLCM: 信息度2",
    "original_glcm_InverseVariance": "纹理GLCM: 逆方差",
    "original_glcm_JointAverage": "纹理GLCM: 联合均值",
    "original_glcm_JointEnergy": "纹理GLCM: 联合能量",
    "original_glcm_JointEntropy": "纹理GLCM: 联合熵",
    "original_glcm_MaximumProbability": "纹理GLCM: 最大概率",
    "original_glcm_SumEntropy": "纹理GLCM: 和熵",
    "original_glcm_SumSquares": "纹理GLCM: 平方和",
    # ── GLRLM (灰度行程矩阵) ──
    "original_glrlm_GrayLevelNonUniformity": "纹理RLM: 灰度不均",
    "original_glrlm_GrayLevelNonUniformityNormalized": "纹理RLM: 灰度不均(归一)",
    "original_glrlm_GrayLevelVariance": "纹理RLM: 灰度方差",
    "original_glrlm_HighGrayLevelRunEmphasis": "纹理RLM: 高灰度",
    "original_glrlm_LongRunEmphasis": "纹理RLM: 长程",
    "original_glrlm_LongRunHighGrayLevelEmphasis": "纹理RLM: 长程高灰",
    "original_glrlm_LongRunLowGrayLevelEmphasis": "纹理RLM: 长程低灰",
    "original_glrlm_LowGrayLevelRunEmphasis": "纹理RLM: 低灰度",
    "original_glrlm_RunEntropy": "纹理RLM: 行程熵",
    "original_glrlm_RunLengthNonUniformity": "纹理RLM: 行程不均",
    "original_glrlm_RunLengthNonUniformityNormalized": "纹理RLM: 行程不均(归一)",
    "original_glrlm_RunPercentage": "纹理RLM: 行程占比",
    "original_glrlm_RunVariance": "纹理RLM: 行程方差",
    "original_glrlm_ShortRunEmphasis": "纹理RLM: 短程",
    "original_glrlm_ShortRunHighGrayLevelEmphasis": "纹理RLM: 短程高灰",
    "original_glrlm_ShortRunLowGrayLevelEmphasis": "纹理RLM: 短程低灰",
    # ── GLSZM (灰度区域矩阵) ──
    "original_glszm_GrayLevelNonUniformity": "纹理SZM: 灰度不均",
    "original_glszm_GrayLevelNonUniformityNormalized": "纹理SZM: 灰度不均(归一)",
    "original_glszm_GrayLevelVariance": "纹理SZM: 灰度方差",
    "original_glszm_HighGrayLevelZoneEmphasis": "纹理SZM: 高灰度",
    "original_glszm_LargeAreaEmphasis": "纹理SZM: 大区域",
    "original_glszm_LargeAreaHighGrayLevelEmphasis": "纹理SZM: 大区高灰",
    "original_glszm_LargeAreaLowGrayLevelEmphasis": "纹理SZM: 大区低灰",
    "original_glszm_LowGrayLevelZoneEmphasis": "纹理SZM: 低灰度",
    "original_glszm_SizeZoneNonUniformity": "纹理SZM: 区域不均",
    "original_glszm_SizeZoneNonUniformityNormalized": "纹理SZM: 区域不均(归一)",
    "original_glszm_SmallAreaEmphasis": "纹理SZM: 小区域",
    "original_glszm_SmallAreaHighGrayLevelEmphasis": "纹理SZM: 小区高灰",
    "original_glszm_SmallAreaLowGrayLevelEmphasis": "纹理SZM: 小区低灰",
    "original_glszm_ZoneEntropy": "纹理SZM: 区域熵",
    "original_glszm_ZonePercentage": "纹理SZM: 区域占比",
    "original_glszm_ZoneVariance": "纹理SZM: 区域方差",
    # ── GLDM (灰度依赖矩阵) ──
    "original_gldm_DependenceEntropy": "纹理DM: 依赖熵",
    "original_gldm_DependenceNonUniformity": "纹理DM: 依赖不均",
    "original_gldm_DependenceNonUniformityNormalized": "纹理DM: 依赖不均(归一)",
    "original_gldm_DependenceVariance": "纹理DM: 依赖方差",
    "original_gldm_GrayLevelNonUniformity": "纹理DM: 灰度不均",
    "original_gldm_GrayLevelVariance": "纹理DM: 灰度方差",
    "original_gldm_HighGrayLevelEmphasis": "纹理DM: 高灰度",
    "original_gldm_LargeDependenceEmphasis": "纹理DM: 大依赖",
    "original_gldm_LargeDependenceHighGrayLevelEmphasis": "纹理DM: 大依赖高灰",
    "original_gldm_LargeDependenceLowGrayLevelEmphasis": "纹理DM: 大依赖低灰",
    "original_gldm_LowGrayLevelEmphasis": "纹理DM: 低灰度",
    "original_gldm_SmallDependenceEmphasis": "纹理DM: 小依赖",
    "original_gldm_SmallDependenceHighGrayLevelEmphasis": "纹理DM: 小依赖高灰",
    "original_gldm_SmallDependenceLowGrayLevelEmphasis": "纹理DM: 小依赖低灰",
    # ── NGTDM (邻域灰度差矩阵) ──
    "original_ngtdm_Busyness": "纹理NGTDM: 繁忙度",
    "original_ngtdm_Coarseness": "纹理NGTDM: 粗糙度",
    "original_ngtdm_Complexity": "纹理NGTDM: 复杂度",
    "original_ngtdm_Contrast": "纹理NGTDM: 对比度",
    "original_ngtdm_Strength": "纹理NGTDM: 强度",
}

GROUP_PREFIX = {
    "shape2D": "SHAPE",
    "shape": "SHAPE",
    "firstorder": "INT",
    "glcm": "TEX(GLCM)",
    "glrlm": "TEX(RLM)",
    "glszm": "TEX(SZM)",
    "gldm": "TEX(DM)",
    "ngtdm": "TEX(NGTDM)",
}

TOKEN_REWRITE = {}


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


def _require_matplotlib_pyplot():
    global plt
    if plt is None:
        import matplotlib.pyplot as _plt
        plt = _plt
        _ensure_cjk_fonts()
    return plt


def _require_shap():
    global shap
    if shap is None:
        import shap as _shap
        shap = _shap
    return shap


def _normalize_optional_text(value: Optional[str], fallback: str) -> str:
    cleaned = str(value).strip() if value is not None else ""
    return cleaned if cleaned else fallback


def format_output_space_label(value: Optional[str]) -> str:
    if value is None:
        return ""

    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if not normalized:
        return ""
    if "prob" in normalized:
        return "probability"
    if "log odds" in normalized or "logit" in normalized or "margin" in normalized:
        return "raw score"
    if "raw" in normalized or "treeexplainer" in normalized:
        return "raw score"
    return str(value).strip()


def build_shap_axis_label(
    *,
    positive_class_name: Optional[str] = None,
    output_space: Optional[str] = None,
    base_label: str = "SHAP contribution",
) -> str:
    label = base_label
    positive_label = str(positive_class_name).strip() if positive_class_name is not None else ""
    if positive_label:
        label = f"{label} toward {_normalize_optional_text(positive_class_name, 'positive class')}"

    output_space_label = format_output_space_label(output_space)
    if output_space_label:
        label = f"{label} ({output_space_label})"
    return label


def _split_camel(text: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", text).strip()


def _rewrite_tokens(phrase: str) -> str:
    return " ".join(TOKEN_REWRITE.get(token, token) for token in phrase.split())


def _abbreviate_tail_words(phrase: str) -> str:
    words = phrase.split()
    if len(words) < 3:
        return phrase

    tail_initials = "".join(word[0].upper() for word in words[2:] if word)
    if not tail_initials:
        return " ".join(words[:2])

    return " ".join([words[0], words[1], tail_initials])


def paper_friendly_name(col: str) -> str:
    if col in MANUAL_MAP:
        return MANUAL_MAP[col]

    stripped = re.sub(r"^original_", "", col)
    match = re.match(r"^([A-Za-z0-9]+)_(.+)$", stripped)
    if match:
        group, name = match.group(1), match.group(2)
    else:
        group, name = "", stripped

    prefix = GROUP_PREFIX.get(group, group.upper() if group else "")
    display_name = _rewrite_tokens(_split_camel(name))
    display_name = re.sub(r"\s+", " ", display_name).strip()
    display_name = _abbreviate_tail_words(display_name)
    result = f"{prefix}: {display_name}" if prefix else display_name
    return result


def format_feature_name(col: str, lang: str = "en") -> str:
    normalized_lang = str(lang).strip().lower()
    if normalized_lang == "zh":
        return MANUAL_MAP_ZH.get(col, paper_friendly_name(col))
    return paper_friendly_name(col)


def short_feature_name(col: str) -> str:
    stripped = re.sub(r"^original_", "", col)
    match = re.match(r"^([A-Za-z0-9]+)_(.+)$", stripped)
    if not match:
        return stripped

    group, name = match.group(1), match.group(2)
    group_map = {
        "shape2D": "Shape2D",
        "shape": "Shape",
        "firstorder": "FirstOrder",
        "glcm": "GLCM",
        "glrlm": "GLRLM",
        "glszm": "GLSZM",
        "gldm": "GLDM",
        "ngtdm": "NGTDM",
    }
    group_display = group_map.get(group, group.upper() if group else "")
    return f"{group_display}: {name}" if group_display else name


def parse_training_csv_from_summary(summary_txt: str) -> str:
    with open(summary_txt, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip().startswith("Training CSV:"):
                return line.split("Training CSV:", 1)[1].strip()
    raise ValueError("Cannot find 'Training CSV:' in summary txt.")


def prepare_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    drop_cols = [col for col in DROP_IF_PRESENT if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df[df[label_col] != -1].copy()
    df[label_col] = df[label_col].astype(int)
    return df


def reconstruct_X_explain(
    train_csv: str,
    label_col: str,
    background_samples: int,
    explain_samples: Optional[int],
) -> pd.DataFrame:
    np_mod = _require_numpy()
    pd_mod = _require_pandas()

    df = pd_mod.read_csv(train_csv)
    df = prepare_df(df, label_col)
    x_train = df.drop(columns=[label_col]).copy()

    np_mod.random.seed(42)
    if len(x_train) > background_samples:
        _ = np_mod.random.choice(len(x_train), size=background_samples, replace=False)

    if explain_samples is None:
        explain_samples = min(500, len(x_train))

    n_explain = min(explain_samples, len(x_train))
    if n_explain < len(x_train):
        explain_idx = np_mod.random.choice(len(x_train), size=n_explain, replace=False)
        return x_train.iloc[explain_idx].copy()
    return x_train.copy()


def load_shap_values(shap_values_csv: str) -> pd.DataFrame:
    pd_mod = _require_pandas()
    shap_df = pd_mod.read_csv(shap_values_csv)
    if len(shap_df.columns) > 0 and shap_df.columns[0].lower().startswith("unnamed"):
        shap_df = shap_df.drop(columns=[shap_df.columns[0]])
    return shap_df


def save_current_figure(
    out_path: str,
    export_formats: Sequence[str] = ("png",),
    dpi: int = 300,
    bbox_inches: Optional[str] = None,
    pad_inches: Optional[float] = None,
) -> list[str]:
    plt_mod = _require_matplotlib_pyplot()
    base, ext = os.path.splitext(out_path)
    normalized = [fmt.lower().lstrip(".") for fmt in export_formats]
    if ext:
        default_format = ext.lower().lstrip(".")
        if default_format not in normalized:
            normalized.insert(0, default_format)
    saved_paths = []
    for fmt in normalized:
        target = f"{base}.{fmt}"
        savefig_kwargs = {"dpi": dpi, "bbox_inches": bbox_inches}
        if pad_inches is not None:
            savefig_kwargs["pad_inches"] = pad_inches
        plt_mod.savefig(target, **savefig_kwargs)
        saved_paths.append(target)
    return saved_paths


def hide_text_in_figure(fig) -> None:
    from matplotlib.text import Text

    for text in fig.findobj(match=Text):
        text.set_visible(False)


def save_beeswarm_plot(
    shap_values,
    x_features: pd.DataFrame,
    out_path: str,
    max_display: int,
    *,
    feature_name_formatter: Optional[Callable[[str], str]] = None,
    save_feature_name_map: bool = False,
    textless_svg_path: Optional[str] = None,
    export_formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 300,
    figsize: tuple[int, int] = (12, 9),
    plot_type: str = "dot",
    positive_class_name: Optional[str] = None,
    output_space: Optional[str] = None,
    x_label: Optional[str] = None,
) -> list[str]:
    np_mod = _require_numpy()
    pd_mod = _require_pandas()
    plt_mod = _require_matplotlib_pyplot()
    shap_mod = _require_shap()

    x_use = x_features.copy()
    if feature_name_formatter is not None:
        rename_map = {col: feature_name_formatter(col) for col in x_use.columns}
        x_use.rename(columns=rename_map, inplace=True)
        if save_feature_name_map:
            base, _ = os.path.splitext(out_path)
            (
                pd_mod.Series(rename_map, name="paper_name")
                .rename_axis("raw_name")
                .to_csv(f"{base}_feature_name_map.csv")
            )

    def _render(show_text: bool, target_path: str, formats: Sequence[str]) -> list[str]:
        with plt_mod.rc_context(
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
                "axes.unicode_minus": False,
                "font.size": 15,
                "axes.titlesize": 19,
                "axes.labelsize": 17,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 14,
            }
        ):
            plt_mod.figure(figsize=figsize)
            shap_mod.summary_plot(
                np_mod.asarray(shap_values),
                x_use,
                plot_type=plot_type,
                max_display=max_display,
                show=False,
            )
            ax = plt_mod.gca()
            xlim = ax.get_xlim()
            xticks = ax.get_xticks()
            xticks = np_mod.asarray([tick for tick in xticks if np_mod.isfinite(tick)])
            xticks = np_mod.unique(np_mod.sort(xticks))
            if xticks.size >= 2:
                steps = np_mod.diff(xticks)
                steps = steps[steps > 0]
                if steps.size > 0:
                    step = float(np_mod.min(steps))
                    left = np_mod.floor(xlim[0] / step) * step
                    right = np_mod.ceil(xlim[1] / step) * step
                    ax.set_xlim(left, right)
            final_x_label = x_label or build_shap_axis_label(
                positive_class_name=positive_class_name,
                output_space=output_space,
            )
            if final_x_label:
                ax.set_xlabel(final_x_label)
            if not show_text:
                hide_text_in_figure(plt_mod.gcf())
            plt_mod.tight_layout()
            saved_paths = save_current_figure(target_path, export_formats=formats, dpi=dpi)
            plt_mod.close()
            return saved_paths

    saved_paths: list[str] = []
    if textless_svg_path is not None:
        saved_paths.extend(_render(False, textless_svg_path, ("svg",)))
    saved_paths.extend(_render(True, out_path, export_formats))
    return saved_paths


def save_waterfall_plot(
    shap_values,
    feature_values,
    feature_names,
    base_value: float,
    out_path: str,
    max_display: int,
    *,
    textless_svg_path: Optional[str] = None,
    title: Optional[str] = None,
    title_fontsize: float = 18.0,
    export_formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 150,
    figsize: tuple[int, int] = (12, 8),
    bbox_inches: str = "tight",
    pad_inches: Optional[float] = None,
    positive_class_name: Optional[str] = None,
    output_space: Optional[str] = None,
    x_label: Optional[str] = None,
) -> list[str]:
    np_mod = _require_numpy()
    plt_mod = _require_matplotlib_pyplot()
    shap_mod = _require_shap()

    shap_values = np_mod.asarray(shap_values).reshape(1, -1)
    feature_values = np_mod.asarray(feature_values).reshape(1, -1)
    explanation = shap_mod.Explanation(
        values=shap_values,
        base_values=np_mod.array([base_value]),
        data=feature_values,
        feature_names=list(feature_names),
    )

    def _render(show_text: bool, target_path: str, formats: Sequence[str], plot_title: Optional[str]) -> list[str]:
        with plt_mod.rc_context(
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
                "axes.unicode_minus": False,
                "font.size": 20,
                "axes.titlesize": 24,
                "axes.labelsize": 20,
                "xtick.labelsize": 18,
                "ytick.labelsize": 18,
                "legend.fontsize": 18,
            }
        ):
            plt_mod.figure(figsize=figsize)
            shap_mod.plots.waterfall(explanation[0], show=False, max_display=max_display)
            if plot_title:
                plt_mod.gcf().suptitle(plot_title, fontsize=title_fontsize, y=0.98)
            final_x_label = x_label or build_shap_axis_label(
                positive_class_name=positive_class_name,
                output_space=output_space,
            )
            if final_x_label:
                ax = plt_mod.gca()
                ax.set_xlabel(final_x_label)
            if not show_text:
                hide_text_in_figure(plt_mod.gcf())
            plt_mod.tight_layout(rect=(0.0, 0.0, 1.0, 0.94 if plot_title else 0.98), pad=1.15)
            saved_paths = save_current_figure(
                target_path,
                export_formats=formats,
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
            )
            plt_mod.close()
            return saved_paths

    saved_paths: list[str] = []
    if textless_svg_path is not None:
        saved_paths.extend(_render(False, textless_svg_path, ("svg",), None))
    saved_paths.extend(_render(True, out_path, export_formats, title))
    return saved_paths
