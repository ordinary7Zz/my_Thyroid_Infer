"""
纯图像/掩码 IO 操作。不进行归一化、不进行二值化、不进行模型预处理。

设计决策：
  load_image 返回 uint8 [0,255]（不做 /255）。
  load_mask 返回 uint8 [0,255]（不二值化）。
  这已被证明安全——两个原始 DINO-UNet 模型的 preprocess 方法内部都有
  dtype 防御检查 (if float: ×255 → uint8)，最终进网络的数值完全一致。
  二值化是评估需求（算 Dice/HD95），不是 IO 职责，由消费方按需调用 binarize_mask。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

PathLike = Union[str, Path]


class ImageIO:
    """纯图像/掩码 IO。不归一化、不二值化、不做模型预处理。"""

    @staticmethod
    def load_image(path: PathLike) -> np.ndarray:
        """
        从文件加载 RGB 图像。

        Returns:
            (H, W, 3) uint8, RGB, [0, 255]
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"图像文件不存在: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法加载图像: {path}")

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def load_mask(path: PathLike) -> np.ndarray:
        """
        从文件加载灰度掩码（不二值化）。

        Returns:
            (H, W) uint8, [0, 255]。消费方按需调用 binarize_mask。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"掩码文件不存在: {path}")

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法加载掩码: {path}")

        return mask

    @staticmethod
    def save_mask(mask: np.ndarray, path: PathLike) -> None:
        """
        将掩码保存为 PNG。接受 0/1 二值或 0/255，统一存为 0/255。

        Args:
            mask: (H, W) 数组，值为 0/1 或 0/255。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mask = np.asarray(mask)
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        cv2.imwrite(str(path), mask)

    @staticmethod
    def save_image(image: np.ndarray, path: PathLike) -> None:
        """将 RGB 图像保存为文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        image = np.asarray(image).astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(path), image)

    @staticmethod
    def binarize_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
        """
        将掩码二值化为 0/1。

        Args:
            mask: 输入掩码，任意数值类型。
            threshold: 二值化阈值（基于 [0, 255] 范围）。

        Returns:
            (H, W) uint8, 值为 0 或 1。
        """
        return (np.asarray(mask) > threshold).astype(np.uint8)

    @staticmethod
    def resize_image(image: np.ndarray, target_hw: tuple) -> np.ndarray:
        """
        调整图像尺寸（双线性插值）。

        Args:
            image: (H, W, C) 或 (H, W) 数组。
            target_hw: (target_height, target_width)。

        Returns:
            调整后的图像，保持原 dtype。
        """
        h, w = target_hw
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def resize_mask(mask: np.ndarray, target_hw: tuple) -> np.ndarray:
        """
        调整掩码尺寸（最近邻插值，避免引入中间值）。

        Args:
            mask: (H, W) 数组。
            target_hw: (target_height, target_width)。

        Returns:
            调整后的掩码，保持原 dtype。
        """
        h, w = target_hw
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
