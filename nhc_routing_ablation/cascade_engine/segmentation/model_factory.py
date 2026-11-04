"""
分割模型工厂：type → 子类映射。

新模型子类用 @register_seg_model("type_name") 装饰器注册，
入口脚本通过 build_seg_model(cfg) 根据 config 的 type 字段自动选择子类。
"""

from __future__ import annotations

from typing import Type

from .base_model import BaseSegmentationModel


SEG_MODEL_REGISTRY: dict[str, Type[BaseSegmentationModel]] = {}


def register_seg_model(type_name: str):
    """装饰器：注册分割模型子类到全局 registry。"""
    def decorator(cls: Type[BaseSegmentationModel]):
        SEG_MODEL_REGISTRY[type_name] = cls
        return cls
    return decorator


def build_seg_model(cfg: dict) -> BaseSegmentationModel:
    """
    根据 config 条目的 type 字段构建分割模型实例。

    Args:
        cfg: config 中单个模型条目，必须含 'type' 字段。

    Returns:
        对应的 BaseSegmentationModel 子类实例（未调用 load_model）。

    Raises:
        ValueError: type 未注册。
    """
    type_name = cfg.get("type", "dinov3_unet")
    if type_name not in SEG_MODEL_REGISTRY:
        available = list(SEG_MODEL_REGISTRY.keys())
        raise ValueError(f"未知的分割模型类型: '{type_name}'. 可用: {available}")
    cls = SEG_MODEL_REGISTRY[type_name]
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    # config 用 name，构造函数用 model_name
    if "name" in kwargs and "model_name" not in kwargs:
        kwargs["model_name"] = kwargs.pop("name")
    return cls(**kwargs)


def list_seg_model_types() -> list[str]:
    """列出所有已注册的分割模型类型。"""
    return list(SEG_MODEL_REGISTRY.keys())
