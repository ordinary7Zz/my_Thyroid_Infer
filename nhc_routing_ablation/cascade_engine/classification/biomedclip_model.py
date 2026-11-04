"""
BiomedCLIP 分类模型。

架构：open_clip visual tower (ViT-B/16) + MLP 分类头。
RGB 224x224 输入，CLIP 标准归一化。

L2 模型：依赖 open_clip_torch + safetensors，
预训练骨干目录含 open_clip_config.json + .bin/.safetensors。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .base_model import BaseClassificationModel, ClsModelOutput
from .model_factory import register_cls_model


def _load_biomedclip_backbone(model_dir: str):
    """从本地目录加载 BiomedCLIP 视觉编码器。"""
    from open_clip.model import _build_vision_tower

    config_path = os.path.join(model_dir, "open_clip_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_cfg = cfg["model_cfg"]

    # 查找权重文件
    local_weights = None
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".safetensors"):
            local_weights = os.path.join(model_dir, fname)
            break
        if fname.endswith((".bin", ".pt", ".pth")) and local_weights is None:
            local_weights = os.path.join(model_dir, fname)

    if local_weights is None:
        raise FileNotFoundError(f"在 {model_dir} 中未找到权重文件")

    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    visual = _build_vision_tower(embed_dim, vision_cfg)

    # 加载 visual 参数
    if local_weights.endswith(".safetensors"):
        from safetensors.torch import load_file
        full_state = load_file(local_weights)
    else:
        full_state = torch.load(local_weights, map_location="cpu")

    visual_state = {}
    for k, v in full_state.items():
        if k.startswith("visual."):
            visual_state[k[len("visual."):]] = v
    visual.load_state_dict(visual_state, strict=False)
    return visual


class _BiomedCLIPClassifier(nn.Module):
    """BiomedCLIP 视觉编码器 + 分类头（内联定义，避免外部依赖）。"""

    def __init__(self, num_classes: int, dropout: float, model_dir: str):
        super().__init__()
        self.visual = _load_biomedclip_backbone(model_dir)
        self.embed_dim = self._get_embed_dim(self.visual)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

    @staticmethod
    def _get_embed_dim(visual) -> int:
        try:
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                out = visual(dummy)
            return out.shape[-1]
        except Exception:
            pass
        return 768

    def forward(self, x):
        features = self.visual(x)
        return self.classifier(features)


@register_cls_model("biomedclip")
class BiomedCLIPClassificationModel(BaseClassificationModel):
    """BiomedCLIP (open_clip ViT) 分类模型。"""

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        model_name: str,
        model_path: str,
        model_dir: str,
        num_classes: int = 2,
        img_size: int = 224,
        dropout: float = 0.3,
        class_names: Optional[list[str]] = None,
        base_dataset_performance: Optional[dict] = None,
        dataset_info: Optional[dict] = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            use_tirads=(num_classes > 2),
            base_dataset_performance=base_dataset_performance,
            dataset_info=dataset_info,
            device=device,
            **kwargs,
        )
        self.model_dir = model_dir
        self.num_classes = num_classes
        self.img_size = img_size
        self.dropout = dropout
        self.class_names = class_names or (
            ["良性", "恶性"] if num_classes == 2
            else [f"TR{i+1}" for i in range(num_classes)]
        )
        self._transform = None

    @property
    def requires_mask(self) -> bool:
        return False

    def load_model(self) -> None:
        print(f"  加载 BiomedCLIP {self.model_name} 从 {self.model_path}")

        self.model = _BiomedCLIPClassifier(
            num_classes=self.num_classes,
            dropout=self.dropout,
            model_dir=self.model_dir,
        )

        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

        self._transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD),
        ])

        print(f"    ✓ BiomedCLIP 加载成功 (classes={self.class_names})")

    def validate_inputs(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        if image is None:
            raise ValueError("image 不能为 None")

    def predict(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> ClsModelOutput:
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        self.validate_inputs(image, mask)

        # uint8 → PIL
        image = np.asarray(image)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        input_tensor = self._transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)  # (1, num_classes)

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        predictions = {
            self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
        }
        top_class = max(predictions, key=predictions.get)

        del logits, input_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ClsModelOutput(
            model_name=self.model_name,
            predictions=predictions,
            top_class=top_class,
            top_confidence=predictions[top_class],
            requires_mask=False,
            metadata=self.get_info(),
        )
