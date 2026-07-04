"""
数据预处理（推理专用）
验证/测试集预处理：缩放 + 黑边填充 + 归一化，无数据增强。
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_val_transforms(image_size: int, mean: list, std: list):
    """
    验证/推理集预处理流水线：
      1. LongestMaxSize 缩放到目标尺寸（保持宽高比）
      2. PadIfNeeded 黑边填充到正方形
      3. Normalize 归一化
      4. ToTensorV2 转 Tensor (C, H, W)
    """
    return A.Compose([
        A.LongestMaxSize(max_size=image_size, p=1.0),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,  # 黑边填充
            p=1.0,
        ),
        A.Normalize(mean=mean, std=std, p=1.0),
        ToTensorV2(),
    ])
