"""
分类模型基类与输出数据结构。

重写自 Classification_Agent/models/base_model.py。
关键改动：
  - predict 接收 uint8 [0,255] image（与 shared/image_io 一致）
  - ModelOutput 改名 ClsModelOutput，避免与分割侧混淆
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ClsModelOutput:
    """分类模型输出。"""

    model_name: str
    predictions: dict[str, float]   # {class_name: probability}
    top_class: str
    top_confidence: float
    requires_mask: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "predictions": dict(self.predictions),
            "top_class": self.top_class,
            "top_confidence": float(self.top_confidence),
            "requires_mask": self.requires_mask,
            "metadata": self.metadata or {},
        }


class BaseClassificationModel(ABC):
    """分类模型抽象基类。"""

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        use_tirads: bool = False,
        base_dataset_performance: Optional[dict] = None,
        dataset_info: Optional[dict] = None,
        device: str = "cuda",
        **kwargs,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.use_tirads = use_tirads
        self.base_dataset_performance = base_dataset_performance or {}
        self.dataset_info = dataset_info or {}
        self.device = device
        self.model = None
        self.is_loaded = False
        self.extra = kwargs

    @property
    @abstractmethod
    def requires_mask(self) -> bool:
        """该模型是否需要 mask 输入。"""

    @abstractmethod
    def load_model(self) -> None:
        """加载模型权重。"""

    def unload_model(self) -> None:
        """卸载模型，释放显存。"""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abstractmethod
    def predict(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> ClsModelOutput:
        """
        分类推理。

        Args:
            image: (H, W, 3) RGB uint8 [0,255]
            mask: (H, W) uint8 [0,255]，requires_mask=True 时必须提供
        """

    @abstractmethod
    def validate_inputs(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """验证输入合法性。"""

    def get_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "use_tirads": self.use_tirads,
            "requires_mask": self.requires_mask,
            "device": self.device,
            "base_dataset_performance": self.base_dataset_performance,
            "dataset_info": self.dataset_info,
            "is_loaded": self.is_loaded,
            **self.extra,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name}, loaded={self.is_loaded})"
