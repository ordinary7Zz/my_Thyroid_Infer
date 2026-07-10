"""
Segmentation metrics calculation
Includes Dice coefficient, Hausdorff Distance 95%, and IoU
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from typing import Union, Optional


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Dice coefficient between prediction and ground truth
    
    Args:
        pred_mask: Predicted binary mask (H, W) with values 0 or 1
        gt_mask: Ground truth binary mask (H, W) with values 0 or 1
        
    Returns:
        Dice coefficient (0-1, higher is better)
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    
    if union == 0:
        # Both masks are empty
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return float(dice)


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two masks
    
    Args:
        mask1: Binary mask (H, W) with values 0 or 1
        mask2: Binary mask (H, W) with values 0 or 1
        
    Returns:
        IoU score (0-1, higher is better)
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return float(iou)


def compute_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray, percentile: int = 95) -> float:
    """
    Calculate 95th percentile Hausdorff Distance
    
    Args:
        pred_mask: Predicted binary mask (H, W) with values 0 or 1
        gt_mask: Ground truth binary mask (H, W) with values 0 or 1
        percentile: Percentile to use (default: 95)
        
    Returns:
        HD95 distance (lower is better), returns inf if one mask is empty
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    # Get boundary points
    pred_points = np.argwhere(pred_mask)
    gt_points = np.argwhere(gt_mask)
    
    # Handle empty masks
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')
    
    # Compute directed Hausdorff distances
    distances_pred_to_gt = np.min(
        np.sqrt(((pred_points[:, None, :] - gt_points[None, :, :]) ** 2).sum(axis=2)),
        axis=1
    )
    distances_gt_to_pred = np.min(
        np.sqrt(((gt_points[:, None, :] - pred_points[None, :, :]) ** 2).sum(axis=2)),
        axis=1
    )
    
    # Get 95th percentile
    hd95_pred_to_gt = np.percentile(distances_pred_to_gt, percentile)
    hd95_gt_to_pred = np.percentile(distances_gt_to_pred, percentile)
    
    # Return maximum of both directions
    hd95 = max(hd95_pred_to_gt, hd95_gt_to_pred)
    
    return float(hd95)


def compute_pairwise_iou(masks: list) -> np.ndarray:
    """
    Compute pairwise IoU between all masks
    
    Args:
        masks: List of binary masks
        
    Returns:
        IoU matrix (n x n) where n is the number of masks
    """
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
    """
    Compute average agreement score for each mask with all others
    
    Args:
        masks: List of binary masks
        
    Returns:
        Array of average IoU scores for each mask
    """
    iou_matrix = compute_pairwise_iou(masks)
    
    # For each mask, compute average IoU with all other masks
    # Exclude self-comparison (diagonal)
    n = len(masks)
    avg_agreement = np.zeros(n)
    
    for i in range(n):
        # Sum all IoU scores except self (diagonal)
        avg_agreement[i] = (iou_matrix[i].sum() - 1.0) / (n - 1) if n > 1 else 1.0
    
    return avg_agreement


def compute_ece(
    confidence_map: np.ndarray,
    gt_mask: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Calculate Expected Calibration Error (ECE) for segmentation.

    This treats `confidence_map` as per-pixel probability for the foreground class (class=1),
    and `gt_mask` as per-pixel binary label (0/1). ECE is computed over all pixels.

    Args:
        confidence_map: (H, W) array of probabilities (ideally in [0, 1])
        gt_mask: (H, W) binary ground-truth mask (0/1)
        n_bins: number of equal-width bins in [0, 1]

    Returns:
        ECE value (lower is better)
    """
    if confidence_map is None:
        raise ValueError("confidence_map is required")
    if gt_mask is None:
        raise ValueError("gt_mask is required")

    conf = np.asarray(confidence_map, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt_mask, dtype=np.float64).reshape(-1)

    if conf.shape[0] != gt.shape[0]:
        raise ValueError(f"shape mismatch: conf has {conf.shape[0]} pixels, gt has {gt.shape[0]} pixels")

    # Clip to valid probability range to be robust to logits-like values
    conf = np.clip(conf, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)

    total = conf.shape[0]
    if total == 0:
        return 0.0

    # Bin assignment: [0,1] -> {0..n_bins-1}
    bin_ids = np.floor(conf * n_bins).astype(np.int32)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf_mean = float(conf[mask].mean())
        bin_acc_mean = float(gt[mask].mean())  # fraction of foreground pixels in this bin
        ece += abs(bin_acc_mean - bin_conf_mean) * (count / total)

    return float(ece)

