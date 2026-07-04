"""分割评估指标：Dice、HD95、ECE 及 Bootstrap CI95。"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt as edt


def _to_bool_2d(mask: torch.Tensor) -> np.ndarray:
    """将 tensor 转为 2D bool numpy 数组。"""
    if mask.dim() == 3:
        if mask.shape[0] == 1 or mask.shape[2] == 1:
            mask = mask.squeeze()
        else:
            mask = mask[0]
    if mask.dim() != 2:
        raise ValueError(f"Expected 2D tensors, got mask: {mask.shape}")
    return mask.detach().cpu().numpy().astype(bool)


# =========================
# Dice（逐病例）
# =========================
class Dice(nn.Module):
    """Dice coefficient calculator for binary segmentation tasks."""

    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, predict, target):
        pred_np = _to_bool_2d(predict)
        target_np = _to_bool_2d(target)

        if not np.any(target_np):
            return None

        smooth = 1.0
        intersection = np.logical_and(pred_np, target_np).sum()
        dice = (2.0 * intersection + smooth) / (
            pred_np.sum() + target_np.sum() + smooth
        )
        return torch.tensor(dice, dtype=torch.float32)


# =========================
# HD95（逐病例）
# =========================
class HD95(nn.Module):
    """HD95 calculator using distance transform."""

    def __init__(self):
        super(HD95, self).__init__()

    def forward(self, predict, target):
        return self.calculate_hd(predict, target)

    def calculate_hd(self, predict, target):
        pred_np = _to_bool_2d(predict)
        target_np = _to_bool_2d(target)

        pred_empty = not np.any(pred_np)
        gt_empty = not np.any(target_np)

        # 边界情况统一处理: pred 或 gt 为空时返回 0.0
        if pred_empty or gt_empty:
            return torch.tensor(0.0, dtype=torch.float32)

        hd1 = self.hd_distance(pred_np, target_np)
        hd2 = self.hd_distance(target_np, pred_np)

        return torch.tensor(max(hd1, hd2), dtype=torch.float32)

    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        indexes = np.nonzero(x)
        distances = edt(~y)
        return float(np.percentile(distances[indexes], 95))


# =========================
# ECE（逐病例，分割模型）
# =========================
class ECE(nn.Module):
    """Expected Calibration Error (ECE) for binary segmentation."""

    def __init__(self, n_bins: int = 15):
        super(ECE, self).__init__()
        self.n_bins = n_bins

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs:  概率预测，形状 [C, H, W] 或 [H, W]，值域 [0, 1]
            target: 二值标签，形状与 probs 兼容，取值 {0, 1}

        Returns:
            标量 Tensor，表示该样本的 ECE 值
        """
        if probs.dim() == 3:
            probs = probs[0]

        probs_flat = probs.contiguous().view(-1).detach().cpu().numpy().astype(
            np.float32
        )
        target_flat = target.contiguous().view(-1).detach().cpu().numpy().astype(
            np.int32
        )

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


# =========================
# Bootstrap CI95（通用）
# =========================
def bootstrap_ci(values, n_boot=5000, ci=0.95, seed=0):
    """
    基于逐病例指标计算 mean 和 CI95（bootstrap）

    Returns:
        (mean, (lower, upper))
    """
    values = np.asarray(values, dtype=np.float32)
    values = values[~np.isnan(values)]
    n = len(values)

    if n == 0:
        return float("nan"), (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    alpha = 1.0 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return values.mean(), (lower, upper)
