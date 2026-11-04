"""
分割模型注册表。

重写自 Segmentation_Agent/models/model_registry.py。
predict_all 对每个模型 try-except，失败跳过不中断。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_model import BaseSegmentationModel, SegModelOutput


class SegModelRegistry:
    """管理多个分割模型，批量预测。"""

    def __init__(self):
        self.models: list[BaseSegmentationModel] = []
        self.model_names: list[str] = []

    def register_model(self, model: BaseSegmentationModel) -> None:
        if not isinstance(model, BaseSegmentationModel):
            raise TypeError(f"期望 BaseSegmentationModel，得到 {type(model)}")
        if model.model_name in self.model_names:
            raise ValueError(f"模型 '{model.model_name}' 已注册")
        self.models.append(model)
        self.model_names.append(model.model_name)
        print(f"  ✓ 注册分割模型: {model.model_name}")

    def unregister_model(self, model_name: str) -> None:
        if model_name not in self.model_names:
            raise ValueError(f"模型 '{model_name}' 不存在")
        idx = self.model_names.index(model_name)
        self.models.pop(idx)
        self.model_names.pop(idx)

    def get_model(self, model_name: str) -> Optional[BaseSegmentationModel]:
        if model_name not in self.model_names:
            return None
        return self.models[self.model_names.index(model_name)]

    def predict_all(self, image: np.ndarray) -> list[SegModelOutput]:
        """
        对输入图像运行所有模型。

        Args:
            image: (H, W, 3) RGB uint8 [0,255]

        Returns:
            成功的 ModelOutput 列表（失败模型跳过）。
        """
        if not self.models:
            raise RuntimeError("注册表中没有模型")

        predictions: list[SegModelOutput] = []
        for model in self.models:
            try:
                output = model.predict(image)
                predictions.append(output)
            except Exception as e:
                print(f"  ✗ 模型 {model.model_name} 推理失败: {e}")
        return predictions

    def list_models(self) -> list[str]:
        return self.model_names.copy()

    def get_model_info(self) -> list[dict]:
        return [m.get_metadata() for m in self.models]

    def __len__(self) -> int:
        return len(self.models)

    def __repr__(self) -> str:
        return f"SegModelRegistry(num={len(self.models)}, models={self.model_names})"
