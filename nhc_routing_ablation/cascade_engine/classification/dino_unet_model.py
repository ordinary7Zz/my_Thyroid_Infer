"""
DINO-UNet 多任务分类模型。

重写自 Classification_Agent/models/dino_unet_model.py。
共享 DINOv3 backbone，分类头输出良恶性（sigmoid）或 TI-RADS（softmax）。
requires_mask = False（多任务模型内部产生 mask，不依赖外部 mask）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from shared.model_architectures import DINOv3_S_UNet_MULTITASK

from .base_model import BaseClassificationModel, ClsModelOutput
from .model_factory import register_cls_model


TIRADS_CLASS_NAMES = ["TR1", "TR2", "TR3", "TR4", "TR5"]
BINARY_CLASS_NAMES = ["良性", "恶性"]


@register_cls_model("dinov3_unet_multitask")
class DINOUNetModel(BaseClassificationModel):
    """DINOv3 ViT-S + UNet 多任务分类模型。"""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        use_tirads: bool = False,
        base_dataset_performance: Optional[dict] = None,
        dataset_info: Optional[dict] = None,
        device: str = "cuda",
        input_size: tuple[int, int] = (224, 224),
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            use_tirads=use_tirads,
            base_dataset_performance=base_dataset_performance,
            dataset_info=dataset_info,
            device=device,
            **kwargs,
        )
        self.input_size = input_size
        self.class_names = TIRADS_CLASS_NAMES if use_tirads else BINARY_CLASS_NAMES
        self.transform = transforms.Compose([
            transforms.Resize(list(input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @property
    def requires_mask(self) -> bool:
        return False

    def load_model(self) -> None:
        """加载 DINOv3_S_UNet_MULTITASK 权重。"""
        print(f"  加载 {self.model_name} 从 {self.model_path}")

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"权重文件不存在: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint

        # 自动检测 use_dilation
        dilation_keys = [k for k in state_dict.keys() if "dilate" in k]
        use_dilation = len(dilation_keys) > 0

        self.model = DINOv3_S_UNet_MULTITASK(pretrained=False, use_dilation=use_dilation)

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

    def validate_inputs(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        if image is None:
            raise ValueError("image 不能为 None")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """uint8 [0,255] → PIL → transform → tensor。"""
        image = np.asarray(image)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image)
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> ClsModelOutput:
        """分类推理。mask 不使用（多任务模型自行产生）。"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        self.validate_inputs(image, mask)
        input_tensor = self.preprocess(image)

        with torch.no_grad():
            seg_out, benign_malignant, tirads = self.model(input_tensor)

        if self.use_tirads:
            probs = torch.softmax(tirads, dim=1)[0].cpu().numpy()
            predictions = {name: float(p) for name, p in zip(self.class_names, probs)}
        else:
            prob_mal = float(torch.sigmoid(benign_malignant)[0, 0].cpu().item())
            predictions = {"良性": 1.0 - prob_mal, "恶性": prob_mal}

        top_class = max(predictions, key=predictions.get)
        top_confidence = predictions[top_class]

        del seg_out, benign_malignant, tirads, input_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ClsModelOutput(
            model_name=self.model_name,
            predictions=predictions,
            top_class=top_class,
            top_confidence=top_confidence,
            requires_mask=self.requires_mask,
            metadata=self.get_info(),
        )
