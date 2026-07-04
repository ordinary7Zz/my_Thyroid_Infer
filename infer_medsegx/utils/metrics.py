# -*- coding: utf-8 -*-
"""
Pure numpy/scipy implementation of Dice (DSC), HD95 and bootstrap CI95.
No dependency on monai.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_coeff(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient between two binary masks (same shape)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0  # both empty → perfect match
    return (2.0 * intersection) / denom


def _hd95_one_sided(x: np.ndarray, y: np.ndarray) -> float:
    """One-sided 95th-percentile distance: for each foreground pixel in x,
    the Euclidean distance to the nearest foreground pixel in y."""
    distances = distance_transform_edt(~y)
    indexes = np.nonzero(x)
    return float(np.percentile(distances[indexes], 95))


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile Hausdorff Distance (in pixels).

    Uses the symmetric distance-transform approach over all foreground
    pixels:
        d = max(
            percentile_95(d(pred_fg → gt_fg)),
            percentile_95(d(gt_fg  → pred_fg))
        )
    where d(A→B) is, for each foreground pixel in A, the Euclidean distance
    to the nearest foreground pixel in B (computed via EDT).

    Boundary cases:
        - pred & gt both non-empty: normal calculation
        - pred non-empty, gt empty (false positive): return 0.0
        - pred empty, gt non-empty (false negative): return 0.0
        - pred & gt both empty (true negative):     return 0.0
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    d_pred_to_gt = _hd95_one_sided(pred, gt)
    d_gt_to_pred = _hd95_one_sided(gt, pred)

    return float(max(d_pred_to_gt, d_gt_to_pred))


def bootstrap_ci(values, n_boot=2000, ci=95, seed=42):
    """Bootstrap confidence interval for the mean.

    Returns (mean, lower, upper).
    NaNs are removed before bootstrapping.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan'), float('nan'), float('nan')

    rng = np.random.default_rng(seed)
    n = values.size
    boot_means = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (100 - ci) / 2
    mean = values.mean()
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return mean, lower, upper
