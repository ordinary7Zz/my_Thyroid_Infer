"""
分割评估指标：Dice、HD95 及 95% 置信区间。

依赖：torch, numpy, scipy
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt


# ---------------------------------------------------------------------------
# 单样本指标
# ---------------------------------------------------------------------------

def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算单张图像的 Dice 系数。pred 和 gt 为二值 (0/1) numpy 数组。"""
    smooth = 1.0
    intersection = (pred * gt).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth))


def compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算单张图像的 95% Hausdorff 距离 (像素单位)。

    pred / gt: 二值 (0/1) numpy 数组, shape [H, W]
    """
    # 如果预测或 GT 为空，退化为单点以避免数值错误
    if pred.sum() == 0:
        pred = pred.copy()
        pred[0, 0] = 1
    if gt.sum() == 0:
        gt = gt.copy()
        gt[0, 0] = 1

    # pred -> gt 的距离
    dt_gt = edt(np.logical_not(gt))
    d_pred_to_gt = np.percentile(dt_gt[np.nonzero(pred)], 95)

    # gt -> pred 的距离
    dt_pred = edt(np.logical_not(pred))
    d_gt_to_pred = np.percentile(dt_pred[np.nonzero(gt)], 95)

    return float(max(d_pred_to_gt, d_gt_to_pred))


# ---------------------------------------------------------------------------
# 批量统计
# ---------------------------------------------------------------------------

def mean_ci95(values):
    """计算均值和 95% 置信区间 (基于正态近似)。

    返回: (mean, ci_low, ci_high)
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    std = float(np.std(arr, ddof=1))
    margin = 1.96 * std / np.sqrt(arr.size)
    return mean, mean - margin, mean + margin


# ---------------------------------------------------------------------------
# logits -> 二值 mask
# ---------------------------------------------------------------------------

def logits_to_binary(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """将模型输出的 logits 转为二值 numpy mask (0/1)。

    logits: shape [1, H, W] 或 [H, W]
    """
    if logits.dim() == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    np_logits = logits.detach().cpu().numpy()
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(np_logits, -50, 50)))
    return (sigmoid > threshold).astype(np.uint8)
