"""
统一分割指标计算模块
====================
为所有分割推理脚本提供一致的 Dice / HD95 / ECE 及 Bootstrap CI95 计算。

统一规范:
  - Dice: 2|P∩G| / (|P|+|G|)，无 smooth
      TN (pred 和 gt 都为空): 1.0
      FP (pred 有, gt 空):    0.0
      FN (pred 空, gt 有):    0.0
  - HD95: scipy EDT 双向取 max(p95)
      任一侧为空: 0.0
  - CI95: Bootstrap percentile, n_boot=2000, seed=42
  - GT 二值化: > 0（任意非零像素视为前景）

API:
  函数式 (推荐): compute_dice, compute_hd95, bootstrap_ci, logits_to_binary
  类式 (兼容):   Dice, HD95, ECE (nn.Module 子类，供 dinov3_unet 使用)
  辅助:          mean_ci95 (正态近似，仅为兼容旧调用，推荐改用 bootstrap_ci)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt as _edt


# ============================================================================
# 工具函数
# ============================================================================

def _to_bool_np(mask):
    """将 tensor 或 ndarray 转为 2D bool numpy 数组。"""
    if isinstance(mask, torch.Tensor):
        if mask.dim() == 3:
            if mask.shape[0] == 1 or mask.shape[2] == 1:
                mask = mask.squeeze()
            else:
                mask = mask[0]
        if mask.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got: {mask.shape}")
        mask = mask.detach().cpu().numpy()
    else:
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask.squeeze()
    return mask.astype(bool)


# ============================================================================
# Dice (函数式)
# ============================================================================

def compute_dice(pred, gt):
    """计算单例 Dice 系数: 2|P∩G| / (|P|+|G|)，无 smooth。

    边界情况:
      - pred 和 gt 都为空 (TN): 返回 1.0
      - 单侧为空 (FP/FN):       返回 0.0

    Args:
        pred: 二值 mask (tensor / ndarray)，任意非零视为前景
        gt:   同上

    Returns:
        float
    """
    p = _to_bool_np(pred)
    g = _to_bool_np(gt)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / denom)


# ============================================================================
# HD95 (函数式)
# ============================================================================

def _hd95_one_sided(x_bool, y_bool):
    """单向 HD: x 前景像素到 y 前景表面的距离的 p95。"""
    distances = _edt(~y_bool)
    indexes = np.nonzero(x_bool)
    return float(np.percentile(distances[indexes], 95))


def compute_hd95(pred, gt):
    """计算单例 HD95 (像素单位): 双向 EDT 取 max(p95)。

    边界情况:
      - pred 或 gt 为空: 返回 0.0

    Args:
        pred: 二值 mask (tensor / ndarray)
        gt:   同上

    Returns:
        float
    """
    p = _to_bool_np(pred)
    g = _to_bool_np(gt)
    if not p.any() or not g.any():
        return 0.0
    d1 = _hd95_one_sided(p, g)
    d2 = _hd95_one_sided(g, p)
    return max(d1, d2)


# ============================================================================
# Bootstrap CI95 (函数式)
# ============================================================================

def bootstrap_ci(values, n_boot=2000, seed=42, ci=0.95):
    """Bootstrap 置信区间 (percentile method)。

    Args:
        values: 逐病例指标值列表
        n_boot: Bootstrap 采样次数
        seed:   随机种子
        ci:     置信水平

    Returns:
        (mean, lower, upper)
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lower, upper


def mean_ci95(values):
    """正态近似 CI95 (仅为兼容旧调用，推荐改用 bootstrap_ci)。

    Returns:
        (mean, lower, upper)
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if n == 1:
        return mean, mean, mean
    std = float(np.std(arr, ddof=1))
    margin = 1.96 * std / np.sqrt(n)
    return mean, mean - margin, mean + margin


# ============================================================================
# Logits → 二值 mask
# ============================================================================

def logits_to_binary(logits, threshold=0.5):
    """将模型输出的 logits 转为二值 numpy mask (0/1)。

    Args:
        logits: tensor [1, H, W] 或 [H, W]，或 numpy array
        threshold: sigmoid 后的二值化阈值

    Returns:
        np.ndarray (H, W) uint8
    """
    if isinstance(logits, torch.Tensor):
        if logits.dim() == 3 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        np_logits = logits.detach().cpu().numpy()
    else:
        np_logits = np.asarray(logits)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(np_logits, -50, 50)))
    return (sigmoid > threshold).astype(np.uint8)


# ============================================================================
# 类式 API (兼容 dinov3_unet 的 nn.Module 调用方式)
# ============================================================================

class Dice(nn.Module):
    """Dice coefficient (nn.Module 接口，供 dinov3_unet 使用)。

    forward(predict, target) → tensor 或 None (GT 为空时返回 None 以保持原行为)
    """

    def __init__(self):
        super().__init__()

    def forward(self, predict, target):
        pred_np = _to_bool_np(predict)
        target_np = _to_bool_np(target)

        # 保持 dinov3_unet 原有行为: GT 为空时返回 None (跳过该样本)
        if not np.any(target_np):
            return None

        denom = pred_np.sum() + target_np.sum()
        if denom == 0:
            return torch.tensor(1.0, dtype=torch.float32)
        dice = float(2.0 * np.logical_and(pred_np, target_np).sum() / denom)
        return torch.tensor(dice, dtype=torch.float32)


class HD95(nn.Module):
    """HD95 (nn.Module 接口，供 dinov3_unet 使用)。

    forward(predict, target) → tensor 或 None (GT 为空时返回 None)
    """

    def __init__(self):
        super().__init__()

    def forward(self, predict, target):
        pred_np = _to_bool_np(predict)
        target_np = _to_bool_np(target)

        if not np.any(target_np):
            return None

        if not np.any(pred_np):
            return torch.tensor(0.0, dtype=torch.float32)

        d1 = _hd95_one_sided(pred_np, target_np)
        d2 = _hd95_one_sided(target_np, pred_np)
        return torch.tensor(max(d1, d2), dtype=torch.float32)


class ECE(nn.Module):
    """Expected Calibration Error (nn.Module 接口，仅 dinov3_unet 使用)。"""

    def __init__(self, n_bins=15):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, probs, target):
        if probs.dim() == 3:
            probs = probs[0]

        probs_flat = probs.contiguous().view(-1).detach().cpu().numpy().astype(np.float32)
        target_flat = target.contiguous().view(-1).detach().cpu().numpy().astype(np.int32)

        probs_flat = np.clip(probs_flat, 1e-7, 1.0 - 1e-7)

        pred_labels = (probs_flat >= 0.5).astype(np.int32)
        confidences = np.where(pred_labels == 1, probs_flat, 1.0 - probs_flat)
        accuracies = (pred_labels == target_flat).astype(np.float32)

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float32)
        ece = 0.0
        n_samples = float(len(confidences))

        for i in range(self.n_bins):
            start = bin_edges[i]
            end = bin_edges[i + 1]
            if i == self.n_bins - 1:
                in_bin = (confidences >= start) & (confidences <= end)
            else:
                in_bin = (confidences >= start) & (confidences < end)
            if not np.any(in_bin):
                continue
            conf_bin = confidences[in_bin].mean()
            acc_bin = accuracies[in_bin].mean()
            weight = float(in_bin.sum()) / n_samples
            ece += weight * abs(acc_bin - conf_bin)

        return torch.tensor(float(ece), dtype=torch.float32, device=probs.device)
