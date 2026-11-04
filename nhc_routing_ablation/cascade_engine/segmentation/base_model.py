"""
分割模型基类与输出数据结构。

重写自 Segmentation_Agent/models/base_model.py。
关键改动：
  - predict 接收 uint8 [0,255]（而非 [0,1] float），与 shared/image_io 一致
  - base_dataset_performance / dataset_info 作为显式参数（而非塞入 kwargs）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class SegModelOutput:
    """分割模型输出。"""

    model_name: str
    mask: np.ndarray               # (H, W) uint8, 0/1
    confidence_map: Optional[np.ndarray] = None  # (H, W) float32, [0,1]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "model_name": self.model_name,
            "mask_shape": self.mask.shape,
            "mask_area": int(np.sum(self.mask)),
            "metadata": self.metadata or {},
        }
        if self.confidence_map is not None and np.sum(self.mask) > 0:
            result["has_confidence_map"] = True
            result["mean_confidence"] = float(np.mean(self.confidence_map[self.mask > 0]))
        else:
            result["has_confidence_map"] = False
            result["mean_confidence"] = 0.0
        return result


class BaseSegmentationModel(ABC):
    """分割模型抽象基类。"""

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        input_size: tuple[int, int] = (224, 224),
        threshold: float = 0.5,
        base_dataset_performance: Optional[dict] = None,
        dataset_info: Optional[dict] = None,
        device: str = "cuda",
        **kwargs,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.input_size = input_size
        self.threshold = threshold
        self.base_dataset_performance = base_dataset_performance or {}
        self.dataset_info = dataset_info or {}
        self.device = device
        self.model = None
        self.is_loaded = False
        self.extra = kwargs

    @abstractmethod
    def load_model(self) -> None:
        """加载模型权重。必须设置 self.model 和 self.is_loaded=True。"""

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
    def predict(self, image: np.ndarray) -> SegModelOutput:
        """
        运行分割推理。

        Args:
            image: (H, W, 3) RGB uint8 [0,255]
        """

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "input_size": list(self.input_size),
            "threshold": self.threshold,
            "device": self.device,
            "base_dataset_performance": self.base_dataset_performance,
            "dataset_info": self.dataset_info,
            "is_loaded": self.is_loaded,
            **self.extra,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name}, loaded={self.is_loaded})"
