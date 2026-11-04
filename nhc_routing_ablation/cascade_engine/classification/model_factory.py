"""
分类模型工厂：type → 子类映射。

新模型子类用 @register_cls_model("type_name") 装饰器注册，
入口脚本通过 build_cls_model(cfg) 根据 config 的 type 字段自动选择子类。
"""

from __future__ import annotations

from typing import Type

from .base_model import BaseClassificationModel


CLS_MODEL_REGISTRY: dict[str, Type[BaseClassificationModel]] = {}


def register_cls_model(type_name: str):
    """装饰器：注册分类模型子类到全局 registry。"""
    def decorator(cls: Type[BaseClassificationModel]):
        CLS_MODEL_REGISTRY[type_name] = cls
        return cls
    return decorator


def build_cls_model(cfg: dict) -> BaseClassificationModel:
    """
    根据 config 条目的 type 字段构建分类模型实例。

    Args:
        cfg: config 中单个模型条目，必须含 'type' 字段。

    Returns:
        对应的 BaseClassificationModel 子类实例（未调用 load_model）。

    Raises:
        ValueError: type 未注册。
    """
    type_name = cfg.get("type", "dinov3_unet_multitask")
    if type_name not in CLS_MODEL_REGISTRY:
        available = list(CLS_MODEL_REGISTRY.keys())
        raise ValueError(f"未知的分类模型类型: '{type_name}'. 可用: {available}")
    cls = CLS_MODEL_REGISTRY[type_name]
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    # config 用 name，构造函数用 model_name
    if "name" in kwargs and "model_name" not in kwargs:
        kwargs["model_name"] = kwargs.pop("name")
    return cls(**kwargs)


def list_cls_model_types() -> list[str]:
    """列出所有已注册的分类模型类型。"""
    return list(CLS_MODEL_REGISTRY.keys())
