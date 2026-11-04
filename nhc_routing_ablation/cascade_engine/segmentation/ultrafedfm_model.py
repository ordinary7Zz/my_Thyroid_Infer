"""
UltraFedFM 分割模型。

架构：MAE ViT-Base 编码器 + SMP UNet 解码器 + Sigmoid 头。
输入 RGB，输出单通道 sigmoid 概率。

L1 模型：依赖 albumentations + segmentation_models_pytorch（本地修改版），
架构文件在 infer_ultrafedfm/ 下，通过 infer_root 引入 sys.path。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base_model import BaseSegmentationModel, SegModelOutput
from .model_factory import register_seg_model


@register_seg_model("ultrafedfm")
class UltraFedFMSegmentationModel(BaseSegmentationModel):
    """UltraFedFM (MAE ViT-B + SMP UNet) 分割模型。"""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        img_size: int = 224,
        threshold: float = 0.5,
        base_dataset_performance: Optional[dict] = None,
        dataset_info: Optional[dict] = None,
        device: str = "cuda",
        infer_root: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            input_size=(img_size, img_size),
            threshold=threshold,
            base_dataset_performance=base_dataset_performance,
            dataset_info=dataset_info,
            device=device,
            **kwargs,
        )
        self.img_size = img_size
        self.infer_root = infer_root
        self._transform = None

    def load_model(self) -> None:
        print(f"  加载 UltraFedFM {self.model_name} 从 {self.model_path}")

        # 引入本地修改的 segmentation_models_pytorch
        if self.infer_root:
            sys.path.insert(0, self.infer_root)
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                f"无法导入 segmentation_models_pytorch。请确认 infer_root 配置正确: {e}"
            )

        self.model = smp.Unet(
            encoder_name="mae",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )

        # 加载权重（直接 state_dict）
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

        # 构建 albumentations transform
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            self._transform = A.Compose([
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                A.Resize(self.img_size, self.img_size),
                ToTensorV2(),
            ])
        except ImportError:
            # 降级：手动预处理
            self._transform = None

        print(f"    ✓ UltraFedFM 加载成功")

    def predict(self, image: np.ndarray) -> SegModelOutput:
        """
        分割推理。

        Args:
            image: (H, W, 3) RGB uint8 [0,255].
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        original_h, original_w = image.shape[:2]

        # 预处理
        if self._transform is not None:
            transformed = self._transform(image=image)
            input_tensor = transformed["image"].unsqueeze(0).to(self.device)
        else:
            # 降级手动预处理
            img = cv2.resize(image, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            input_tensor = (
                torch.from_numpy(img.transpose(2, 0, 1))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )

        with torch.no_grad():
            output = self.model(input_tensor)  # (1, 1, H, W), sigmoid 已应用

        prob = output.squeeze().cpu().numpy().astype(np.float32)

        del output, input_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # resize 回原图
        if prob.shape != (original_h, original_w):
            prob_tensor = torch.from_numpy(prob).unsqueeze(0).unsqueeze(0)
            prob_resized = (
                F.interpolate(
                    prob_tensor,
                    size=(original_h, original_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
                .astype(np.float32)
            )
        else:
            prob_resized = prob

        mask = (prob_resized > self.threshold).astype(np.uint8)

        return SegModelOutput(
            model_name=self.model_name,
            mask=mask,
            confidence_map=prob_resized,
            metadata=self.get_metadata(),
        )
