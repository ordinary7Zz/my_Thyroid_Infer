"""B3 路由策略枚举与 config 解析。"""

from __future__ import annotations

from typing import Any, Optional

# 可部署策略 + 评测专用 oracle
ROUTING_MODES = frozenset({
    "best_single",
    "soft_voting",
    "average",  # alias → soft_voting
    "majority",
    "static_rule",
    "max_confidence",
    "learned",
    "llm",
    "oracle",
    # B4 Path B 附录（不进 B3 八种策略主表）
    "path_b_autogluon",
})

B4_ONLY_CLS_MODES = frozenset({"path_b_autogluon"})

PATH_B_INDIE_ONLY_MODES = frozenset({
    "best_single",
    "soft_voting",
    "average",
    "majority",
    "static_rule",
    "max_confidence",
    "learned",
    "oracle",
})


def normalize_routing_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    m = str(mode).strip().lower()
    if m == "average":
        return "soft_voting"
    return m


def validate_routing_mode(mode: str, *, allow_oracle: bool = True, allow_b4: bool = True) -> str:
    m = normalize_routing_mode(mode)
    if m is None or m not in ROUTING_MODES:
        raise ValueError(f"未知 routing_mode: {mode}")
    if m == "oracle" and not allow_oracle:
        raise ValueError("oracle 仅允许评测 / ablation 脚本")
    if m in B4_ONLY_CLS_MODES and not allow_b4:
        raise ValueError(f"{mode} 仅用于 B4 Path B 附录，不参与 B3 主表")
    return m


def is_b4_only_cls_mode(mode: str) -> bool:
    m = normalize_routing_mode(mode)
    return m in B4_ONLY_CLS_MODES


def get_seg_routing_mode(config: dict) -> str:
    agent = config.get("segmentation", {}).get("agent", {}) or {}
    explicit = normalize_routing_mode(agent.get("routing_mode"))
    if explicit:
        return validate_routing_mode(explicit, allow_oracle=True)
    if agent.get("enable_agent", True):
        return "llm"
    return "static_rule"


def get_cls_routing_mode(config: dict) -> str:
    agent = config.get("classification", {}).get("agent", {}) or {}
    explicit = normalize_routing_mode(agent.get("routing_mode"))
    if explicit:
        return validate_routing_mode(explicit, allow_oracle=True)
    if agent.get("enable_agent", True):
        return "llm"
    return "soft_voting"


def path_b_includes_autogluon(cls_mode: str) -> bool:
    """B3 策略 7（llm）与 B4 附录 B-b（path_b_autogluon）在 Path B 使用 AutoGluon。"""
    m = normalize_routing_mode(cls_mode)
    return m == "llm" or m == "path_b_autogluon"


def seg_uses_llm_api(seg_mode: str) -> bool:
    return normalize_routing_mode(seg_mode) == "llm"


def cls_uses_llm_api(cls_mode: str) -> bool:
    return normalize_routing_mode(cls_mode) == "llm"


def method_name_for_mode(task: str, mode: str) -> str:
    """results.json / routing_log 中的 method 字段。"""
    m = normalize_routing_mode(mode) or mode
    if task == "seg":
        return f"routing_{m}_seg"
    return f"routing_{m}_path_b"
