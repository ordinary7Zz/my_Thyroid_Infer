"""
MedSigLIP 分类模型。

架构：SigLIP ViT 视觉编码器 + 线性分类头。
灰度图复制为三通道，448x448 输入，[0.5,0.5,0.5] 归一化。

L2 模型：依赖 transformers + albumentations + timm，
checkpoint 含 model_state_dict + config + class_names。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base_model import BaseClassificationModel, ClsModelOutput
from .model_factory import register_cls_model


@register_cls_model("medsiglip")
class MedSigLIPClassificationModel(BaseClassificationModel):
    """MedSigLIP (SigLIP ViT) 分类模型。"""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        pretrained_model_path: Optional[str] = None,
        num_classes: int = 2,
        img_size: int = 448,
        dropout: float = 0.1,
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
        self.pretrained_model_path = pretrained_model_path
        self.num_classes = num_classes
        self.img_size = img_size
        self.dropout = dropout
        self.class_names: list[str] = []
        self._transform = None

    @property
    def requires_mask(self) -> bool:
        return False

    def load_model(self) -> None:
        print(f"  加载 MedSigLIP {self.model_name} 从 {self.model_path}")

        # 读取 checkpoint 中的 config 和 class_names
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model_cfg = checkpoint.get("config", {}).get("model", {})
        data_cfg = checkpoint.get("config", {}).get("data", {})

        actual_num_classes = model_cfg.get("num_classes", self.num_classes)
        actual_dropout = model_cfg.get("classifier_dropout", self.dropout)
        actual_img_size = data_cfg.get("image_size", self.img_size)
        local_files_only = model_cfg.get("local_files_only", True)
        model_name_or_path = self.pretrained_model_path or model_cfg.get(
            "model_name", "google/medsiglip-448"
        )

        self.num_classes = actual_num_classes
        self.img_size = actual_img_size
        self.class_names = checkpoint.get("class_names") or [
            str(i) for i in range(actual_num_classes)
        ]
        # 统一为中文标签（与其他分类模型一致，方便 LLM Agent 理解）
        if self.num_classes == 2:
            self.class_names = ["良性", "恶性"]

        # 构建 MedSigLIPClassifier
        import sys
        infer_root = self.extra.get("infer_root")
        if infer_root:
            sys.path.insert(0, infer_root)
        try:
            from model import MedSigLIPClassifier
        except ImportError:
            # 内联定义（避免强依赖外部文件）
            from transformers import AutoModel, AutoConfig
            class MedSigLIPClassifier(torch.nn.Module):
                def __init__(self, model_name, num_classes, dropout, local_files_only):
                    super().__init__()
                    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                    self.full_model = AutoModel.from_pretrained(
                        model_name, trust_remote_code=True, local_files_only=local_files_only
                    )
                    self.vision_encoder = self.full_model.vision_model
                    emb_dim = config.vision_config.hidden_size
                    self.dropout = torch.nn.Dropout(dropout)
                    self.classifier = torch.nn.Linear(emb_dim, num_classes)

                def forward(self, pixel_values):
                    outputs = self.vision_encoder(pixel_values=pixel_values)
                    pooled = outputs.pooler_output
                    return self.classifier(self.dropout(pooled))

        self.model = MedSigLIPClassifier(
            model_name=model_name_or_path,
            num_classes=actual_num_classes,
            dropout=actual_dropout,
            local_files_only=local_files_only,
        )

        # 加载微调权重
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # 过滤掉文本编码器
        filtered = {
            k: v for k, v in state_dict.items()
            if not any(k.startswith(p) for p in [
                "full_model.text_model.", "full_model.text.", "full_model.text_projection."
            ])
        }
        self.model.load_state_dict(filtered, strict=False)

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

        # 构建 transform
        mean = data_cfg.get("mean", [0.5, 0.5, 0.5])
        std = data_cfg.get("std", [0.5, 0.5, 0.5])
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            self._transform = A.Compose([
                A.LongestMaxSize(max_size=self.img_size, p=1.0),
                A.PadIfNeeded(
                    min_height=self.img_size, min_width=self.img_size,
                    border_mode=0, p=1.0,
                ),
                A.Normalize(mean=mean, std=std, p=1.0),
                ToTensorV2(),
            ])
        except ImportError:
            self._transform = None

        print(f"    ✓ MedSigLIP 加载成功 (classes={self.class_names})")

    def validate_inputs(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        if image is None:
            raise ValueError("image 不能为 None")

    def predict(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> ClsModelOutput:
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        self.validate_inputs(image, mask)

        # RGB → 灰度 → 三通道复制
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        img_3ch = np.stack([gray] * 3, axis=-1)  # (H, W, 3)

        if self._transform is not None:
            transformed = self._transform(image=img_3ch)
            input_tensor = transformed["image"].unsqueeze(0).to(self.device)
        else:
            # 降级手动预处理
            from torchvision import transforms as T
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            input_tensor = transform(img_3ch).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)  # (1, num_classes)

        logits_np = logits.squeeze(0).cpu().numpy()

        if self.num_classes <= 2:
            # 二分类：sigmoid
            if logits_np.shape[0] == 1:
                p_mal = float(torch.sigmoid(logits).squeeze().item())
                predictions = {"良性": 1.0 - p_mal, "恶性": p_mal}
            else:
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                predictions = {
                    self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
                }
        else:
            # 多分类：softmax
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
