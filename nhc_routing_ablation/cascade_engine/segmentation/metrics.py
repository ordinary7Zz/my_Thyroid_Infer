"""
分割评估指标：Dice、IoU、HD95、pairwise IoU、average agreement、ECE。

重写自 Segmentation_Agent/utils/metrics.py，逻辑保持一致。
"""

from __future__ import annotations

import numpy as np


# HD95 评估分辨率（统一 resize 后再算，保证不同图像尺寸下结果可比）
HD95_EVAL_SIZE = (224, 224)


def _resize_mask(mask: np.ndarray, target_size: tuple[int, int] = HD95_EVAL_SIZE) -> np.ndarray:
    """将 mask resize 到目标尺寸（最近邻插值，保持二值性）。"""
    import cv2
    if mask.shape[:2] == target_size:
        return mask
    return cv2.resize(
        mask.astype(np.uint8),
        (target_size[1], target_size[0]),  # cv2 是 (W, H)
        interpolation=cv2.INTER_NEAREST,
    )


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice 系数 (0-1, 越高越好)。两个空 mask 返回 1.0。"""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float((2.0 * intersection) / union)


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """IoU (0-1, 越高越好)。两个空 mask 返回 1.0。"""
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray, percentile: int = 95) -> float:
    """
    95 百分位 Hausdorff 距离 (越低越好)。

    统一在 224x224 分辨率下计算，避免不同原图尺寸导致 HD95 不可比。
    使用 scipy EDT 双向计算表面距离，取 max(p95)。

    边界情况:
      - pred 或 gt 为空: 返回 inf
      - 两者都为空: 返回 0.0
    """
    from scipy.ndimage import distance_transform_edt as _edt

    # 统一 resize 到 224x224
    p = _resize_mask(np.asarray(pred_mask), HD95_EVAL_SIZE).astype(bool)
    g = _resize_mask(np.asarray(gt_mask), HD95_EVAL_SIZE).astype(bool)

    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float("inf")

    # 单向表面距离: x 前景像素到 y 最近表面像素的距离
    def _hd95_one_sided(x_bool, y_bool):
        distances = _edt(~y_bool)
        indexes = np.nonzero(x_bool)
        return float(np.percentile(distances[indexes], percentile))

    d1 = _hd95_one_sided(p, g)  # pred -> gt surface
    d2 = _hd95_one_sided(g, p)  # gt -> pred surface
    return max(d1, d2)


def compute_pairwise_iou(masks: list) -> np.ndarray:
    """计算 n 个 mask 之间的 pairwise IoU 矩阵 (n×n)。"""
    n = len(masks)
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        iou_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            iou = compute_iou(masks[i], masks[j])
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou
    return iou_matrix


def compute_average_agreement(masks: list) -> np.ndarray:
    """每个 mask 与其余所有 mask 的平均 IoU（排除自身）。"""
    n = len(masks)
    if n < 2:
        return np.ones(n)
    iou_matrix = compute_pairwise_iou(masks)
    avg = np.zeros(n)
    for i in range(n):
        avg[i] = (iou_matrix[i].sum() - 1.0) / (n - 1)
    return avg


def compute_ece(
    confidence_map: np.ndarray,
    gt_mask: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error（分割版）。

    将 confidence_map 视为前景概率，gt_mask 为 0/1 标签，
    按等宽分箱计算 |平均准确率 - 平均置信度| 的加权平均。
    """
    conf = np.asarray(confidence_map, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt_mask, dtype=np.float64).reshape(-1)

    if conf.shape[0] != gt.shape[0]:
        raise ValueError(
            f"shape mismatch: conf {conf.shape[0]} vs gt {gt.shape[0]}"
        )

    conf = np.clip(conf, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)

    total = conf.shape[0]
    if total == 0:
        return 0.0

    bin_ids = np.floor(conf * n_bins).astype(np.int32)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(gt[mask].mean())
        ece += abs(bin_acc - bin_conf) * (count / total)

    return float(ece)
