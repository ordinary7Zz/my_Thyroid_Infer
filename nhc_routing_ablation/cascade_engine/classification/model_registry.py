"""
分类模型注册表。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_model import BaseClassificationModel, ClsModelOutput


class ClsModelRegistry:
    """管理多个分类模型，批量预测。"""

    def __init__(self):
        self.models: dict[str, BaseClassificationModel] = {}

    def register_model(self, model: BaseClassificationModel) -> None:
        if not isinstance(model, BaseClassificationModel):
            raise TypeError(f"期望 BaseClassificationModel，得到 {type(model)}")
        if model.model_name in self.models:
            raise ValueError(f"模型 '{model.model_name}' 已注册")
        self.models[model.model_name] = model
        print(f"  ✓ 注册分类模型: {model.model_name}")

    def load_all_models(self) -> None:
        for model in self.models.values():
            if not model.is_loaded:
                model.load_model()

    def predict_all(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> list[ClsModelOutput]:
        """
        对输入图像运行所有模型。

        Args:
            image: (H, W, 3) RGB uint8 [0,255]
            mask: (H, W) uint8 [0,255]（requires_mask 模型需要）

        Returns:
            成功的 ClsModelOutput 列表。
        """
        if not self.models:
            raise RuntimeError("注册表中没有模型")

        results: list[ClsModelOutput] = []
        for model in self.models.values():
            try:
                model.validate_inputs(image, mask)
                output = model.predict(image, mask)
                results.append(output)
            except Exception as e:
                print(f"  ✗ 模型 {model.model_name} 推理失败: {e}")
        return results

    def list_models(self) -> list[str]:
        return list(self.models.keys())

    def get_model_info(self) -> list[dict]:
        return [m.get_info() for m in self.models.values()]

    def get_independent_models(self) -> list[tuple[str, "BaseClassificationModel"]]:
        """返回 requires_mask=False 的独立分类模型（用于 Step 1 共识判断）。"""
        return [(name, m) for name, m in self.models.items() if not m.requires_mask]

    def get_mask_dependent_models(self) -> list[tuple[str, "BaseClassificationModel"]]:
        """返回 requires_mask=True 的模型（AutoGluon，用于 Step 4）。"""
        return [(name, m) for name, m in self.models.items() if m.requires_mask]

    def __len__(self) -> int:
        return len(self.models)

    def __repr__(self) -> str:
        return f"ClsModelRegistry(num={len(self.models)}, models={list(self.models.keys())})"
