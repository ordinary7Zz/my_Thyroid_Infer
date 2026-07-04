"""
MedSigLIP 分类模型（推理专用）
从训练检查点加载模型架构和权重，结构与训练时完全一致以保证 state_dict 可正确加载。
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MedSigLIPClassifier(nn.Module):
    """
    基于 MedSigLIP 视觉编码器的医学图像分类器（推理用）。

    架构:
        [图像 448x448] -> [MedSigLIP ViT Encoder] -> [CLS Token / Pooled Output]
            -> [Dropout] -> [Linear(num_classes)] -> [Logits]

    Args:
        model_name: HuggingFace 模型名或本地路径
        num_classes: 分类类别数
        dropout: 分类头 dropout（推理时 eval() 会禁用，但模块需存在以匹配 state_dict）
        local_files_only: 是否仅从本地加载
    """

    def __init__(
        self,
        model_name: str = "google/medsiglip-448",
        num_classes: int = 2,
        dropout: float = 0.1,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        # 加载完整 MedSigLIP 模型（与训练时保持一致）
        print(f"[Model] Loading from: {model_name} (local_files_only={local_files_only})")
        self.full_model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )

        # 提取视觉编码器（ViT），与训练时保持相同的模块注册名
        self.vision_encoder = self.full_model.vision_model

        config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        self.emb_dim = config.vision_config.hidden_size
        print(f"[Model] Vision encoder embedding dim: {self.emb_dim}")

        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.emb_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Args:
            pixel_values: (B, C, H, W) 归一化后的图像张量

        Returns:
            dict with:
                logits: (B, num_classes) 分类 logits
                embeddings: (B, emb_dim) 视觉嵌入
        """
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)

        if vision_outputs.pooler_output is not None:
            embeddings = vision_outputs.pooler_output
        else:
            embeddings = vision_outputs.last_hidden_state[:, 0, :]

        embeddings = self.dropout(embeddings)
        logits = self.classifier(embeddings)

        return {"logits": logits, "embeddings": embeddings}
