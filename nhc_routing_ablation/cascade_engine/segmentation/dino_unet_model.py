"""
DINO-UNet 分割模型。

重写自 Segmentation_Agent/models/dino_unet_model.py。
关键改动：
  - predict 接收 uint8 [0,255]（与 shared/image_io 一致）
  - preprocess 的 dtype 检查保留（兼容性），但新流程直接传 uint8
  - 架构来自 shared/model_architectures/dino_unet.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from shared.model_architectures import DINOv3_S_UNet

from .base_model import BaseSegmentationModel, SegModelOutput
from .model_factory import register_seg_model


@register_seg_model("dinov3_unet")
class DINOUNetSegmentationModel(BaseSegmentationModel):
    """DINOv3 ViT-S + UNet 分割模型。"""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        input_size: tuple[int, int] = (224, 224),
        threshold: float = 0.5,
        base_dataset_performance: Optional[dict] = None,
        dataset_info: Optional[dict] = None,
        device: str = "cuda",
        use_dilation: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            input_size=input_size,
            threshold=threshold,
            base_dataset_performance=base_dataset_performance,
            dataset_info=dataset_info,
            device=device,
            **kwargs,
        )
        self.use_dilation = use_dilation
        self.transform = transforms.Compose([
            transforms.Resize(list(input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load_model(self) -> None:
        """加载 DINOv3_S_UNet 权重。"""
        print(f"  加载 {self.model_name} 从 {self.model_path}")

        if not self.model_path:
            raise ValueError("model_path is required")

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"权重文件不存在: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint

        # 自动检测 use_dilation
        if self.use_dilation is None:
            dilation_keys = [k for k in state_dict.keys() if "dilate" in k]
            self.use_dilation = len(dilation_keys) > 0
            print(f"    use_dilation={'True' if self.use_dilation else 'False'}")

        self.model = DINOv3_S_UNet(pretrained=False, use_dilation=self.use_dilation)

        try:
            self.model.load_state_dict(state_dict, strict=True)
            print(f"    ✓ 严格模式加载成功")
        except RuntimeError:
            model_keys = set(self.model.state_dict().keys())
            filtered = {k: v for k, v in state_dict.items() if k in model_keys}
            self.model.load_state_dict(filtered, strict=False)
            print(f"    ✓ 非严格模式加载（匹配 {len(filtered)}/{len(model_keys)} 键）")

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像。

        Args:
            image: (H, W, 3) RGB，uint8 [0,255] 或 float [0,1]（兼容）。

        Returns:
            (1, 3, H, W) tensor on device.
        """
        image = np.asarray(image)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image)
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image: np.ndarray) -> SegModelOutput:
        """
        分割推理。

        Args:
            image: (H, W, 3) RGB uint8 [0,255].
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        original_shape = image.shape[:2]
        input_tensor = self.preprocess(image)

        with torch.no_grad():
            output = self.model(input_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            del output, input_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mask = self._postprocess(prob_map, original_shape, self.threshold)

        if prob_map.shape != original_shape:
            prob_map_resized = cv2.resize(
                prob_map,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            prob_map_resized = prob_map

        return SegModelOutput(
            model_name=self.model_name,
            mask=mask,
            confidence_map=prob_map_resized,
            metadata=self.get_metadata(),
        )

    def _postprocess(
        self,
        prob_map: np.ndarray,
        original_shape: tuple,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """概率图 → 二值 mask + 形态学清理。"""
        while prob_map.ndim > 2:
            prob_map = prob_map.squeeze(0)

        if prob_map.shape != original_shape:
            prob_map = cv2.resize(
                prob_map,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        mask = (prob_map > threshold).astype(np.uint8)

        # 去除小连通区域 + 闭运算
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 50:
                cleaned[labels == i] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        return cleaned
