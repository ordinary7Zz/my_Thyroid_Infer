import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from tqdm import tqdm


def _to_bool_2d(mask: torch.Tensor) -> np.ndarray:
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
    """
    Dice coefficient calculator for binary segmentation tasks.
    """
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
    """
    HD95 calculator for binary segmentation tasks.
    使用距离变换方法计算 Hausdorff Distance (95%)
    """
    def __init__(self):
        super(HD95, self).__init__()

    def forward(self, predict, target):
        return self.calculate_hd(predict, target)

    def calculate_hd(self, predict, target):
        pred_np = _to_bool_2d(predict)
        target_np = _to_bool_2d(target)

        if not np.any(target_np):
            return None

        pred_empty = not np.any(pred_np)

        if pred_empty:
            h, w = pred_np.shape
            max_distance = float(np.sqrt(h * h + w * w))
            return torch.tensor(max_distance, dtype=torch.float32)

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
    """
    Expected Calibration Error (ECE) for binary segmentation.
    适用于基于概率的分割预测，用于衡量模型置信度与实际准确率的一致性。
    """

    def __init__(self, n_bins: int = 15):
        super(ECE, self).__init__()
        self.n_bins = n_bins

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs:  概率预测，形状 [C, H, W] 或 [H, W]，值域在 [0, 1]
            target: 二值标签，形状与 probs 兼容，取值 {0, 1}

        Returns:
            标量 Tensor，表示该样本的 ECE 值
        """
        # 只使用前景通道（假设为二分类分割）
        if probs.dim() == 3:
            probs = probs[0]

        probs_flat = probs.contiguous().view(-1).detach().cpu().numpy().astype(np.float32)
        target_flat = target.contiguous().view(-1).detach().cpu().numpy().astype(np.int32)

        # 裁剪到 (0, 1) 范围，避免极端值带来的数值问题
        probs_flat = np.clip(probs_flat, 1e-7, 1.0 - 1e-7)

        # 预测类别（阈值 0.5）与对应置信度（对预测类别的置信度）
        pred_labels = (probs_flat >= 0.5).astype(np.int32)
        confidences = np.where(pred_labels == 1, probs_flat, 1.0 - probs_flat)

        accuracies = (pred_labels == target_flat).astype(np.float32)

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float32)
        ece = 0.0
        n_samples = float(len(confidences))

        for i in range(self.n_bins):
            start = bin_edges[i]
            end = bin_edges[i + 1]

            # 左闭右开，最后一个 bin 包含 1.0
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


# =========================
# 模型评估（逐病例 + CI95）
# =========================
def evaluate_model(net, dataloader, device):
    """
    Evaluate segmentation model with:
    - Dice (per-case)
    - HD95 (per-case)
    - ECE (per-case, calibration)
    - CI95 for all metrics (bootstrap)

    Returns:
        dict with mean + CI95
    """
    net.eval()

    dice_calculator = Dice()
    hd_calculator = HD95()
    ece_calculator = ECE()

    all_dice_values = []
    all_hd_values = []
    all_ece_values = []

    for batch in tqdm(dataloader, desc="Evaluating Model", leave=False):
        # 兼容不同 batch 格式
        if isinstance(batch, dict):
            image = batch["image"]
            mask_true = batch["label"]
        else:
            image, mask_true = batch

        image = image.to(device)
        mask_true = mask_true.to(device)

        with torch.no_grad():
            mask_pred = net(image)
            if isinstance(mask_pred, (list, tuple)):
                mask_pred = mask_pred[0]

            mask_pred = torch.sigmoid(mask_pred)
            mask_pred_binary = (mask_pred > 0.5).float()

        batch_size = image.size(0)

        # =========================
        # 逐病例计算 Dice / HD95 / ECE
        # =========================
        for i in range(batch_size):
            prob_i = mask_pred[i]
            pred_i = mask_pred_binary[i]
            true_i = (mask_true[i] > 0.5).float()

            dice_i = dice_calculator(pred_i, true_i)
            if dice_i is not None:
                all_dice_values.append(dice_i.item())

            # HD95
            try:
                hd_i = hd_calculator(pred_i, true_i)
                if hd_i is not None:
                    all_hd_values.append(hd_i.item())
            except Exception as e:
                print(f"[Warning] HD95 failed on sample {i}: {e}")

            # ECE
            try:
                ece_i = ece_calculator(prob_i, true_i).item()
                all_ece_values.append(ece_i)
            except Exception as e:
                print(f"[Warning] ECE failed on sample {i}: {e}")

    net.train()

    # =========================
    # 计算 mean + CI95
    # =========================
    dice_mean, dice_ci95 = bootstrap_ci(all_dice_values)
    hd95_mean, hd95_ci95 = bootstrap_ci(all_hd_values)
    ece_mean, ece_ci95 = bootstrap_ci(all_ece_values)

    # 将逐病例值也保留到 4 位小数，便于后续输出一致性
    rounded_dice_values = [round(float(v), 4) for v in all_dice_values]
    rounded_hd_values = [round(float(v), 4) for v in all_hd_values]
    rounded_ece_values = [round(float(v), 4) for v in all_ece_values]

    results = {
        "Dice": {
            "mean": round(dice_mean, 4),
            "CI95": (round(dice_ci95[0], 4), round(dice_ci95[1], 4)),
            "values": rounded_dice_values,
        },
        "HD95": {
            "mean": round(hd95_mean, 4),
            "CI95": (round(hd95_ci95[0], 4), round(hd95_ci95[1], 4)),
            "values": rounded_hd_values,
        },
        "ECE": {
            "mean": round(ece_mean, 4),
            "CI95": (round(ece_ci95[0], 4), round(ece_ci95[1], 4)),
            "values": rounded_ece_values,
        },
    }

    return results
