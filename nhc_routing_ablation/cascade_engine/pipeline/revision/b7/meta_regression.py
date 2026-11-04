"""B7：中心/图像层异质性（I²、简单 meta 回归）。"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def heterogeneity_i2_tau2(
    values_by_group: dict[str, list[float]],
) -> dict[str, Optional[float]]:
    """
    基于组内方差与总方差近似 I²；τ² 为组间方差分量（简化随机效应）。

    各组样本量不等时仍可用，但小样本中心需谨慎解读。
    """
    groups = {k: [float(x) for x in v if x is not None and not np.isnan(x)] for k, v in values_by_group.items()}
    groups = {k: v for k, v in groups.items() if len(v) >= 1}
    if len(groups) < 2:
        return {"i2_percent": None, "tau2": None, "q": None, "df": None, "n_groups": len(groups)}

    all_vals = []
    group_means = []
    group_ns = []
    for vals in groups.values():
        all_vals.extend(vals)
        group_means.append(np.mean(vals))
        group_ns.append(len(vals))

    grand_mean = float(np.mean(all_vals))
    n_total = len(all_vals)
    k = len(groups)

    ss_between = sum(n * (m - grand_mean) ** 2 for n, m in zip(group_ns, group_means))
    ss_total = sum((x - grand_mean) ** 2 for x in all_vals)
    ss_within = ss_total - ss_between

    df_between = k - 1
    df_within = n_total - k
    if df_within <= 0 or ss_within <= 0:
        return {"i2_percent": None, "tau2": None, "q": float(ss_between), "df": df_between, "n_groups": k}

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    q = ss_between / ms_within if ms_within > 0 else 0.0
    i2 = max(0.0, (q - df_between) / q) * 100.0 if q > 0 else 0.0
    tau2 = max(0.0, ms_between - ms_within) if ms_between > ms_within else 0.0

    return {
        "i2_percent": float(i2),
        "tau2": float(tau2),
        "q": float(q),
        "df": int(df_between),
        "n_groups": k,
    }


def ols_univariate(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """单变量 OLS：y ~ x，返回 coef、intercept、r2。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < 3:
        return {"n": int(n), "coef": None, "intercept": None, "r2": None}
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx <= 0:
        return {"n": int(n), "coef": None, "intercept": None, "r2": None}
    coef = float(np.sum((x - x_mean) * (y - y_mean)) / ss_xx)
    intercept = float(y_mean - coef * x_mean)
    y_hat = intercept + coef * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"n": int(n), "coef": coef, "intercept": intercept, "r2": r2}


def meta_regress_image_dice(
    rows: list[dict],
    *,
    covariate_keys: list[str],
) -> dict[str, Any]:
    """
    图像级：dice ~ 每个协变量（单变量 OLS）。

    rows 需含 dice 与协变量字段（如 malignant_ratio_center、roi_ratio 等）。
    """
    y = np.array([r.get("dice") for r in rows], dtype=np.float64)
    out: dict[str, Any] = {"target": "dice", "covariates": {}}
    for key in covariate_keys:
        x = np.array([r.get(key) for r in rows], dtype=np.float64)
        out["covariates"][key] = ols_univariate(x, y)
    return out
