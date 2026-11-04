"""Pipeline 包：延迟导入，避免 cascade ↔ routing 循环依赖。"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "CascadePipeline":
        from .cascade import CascadePipeline
        return CascadePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CascadePipeline"]
