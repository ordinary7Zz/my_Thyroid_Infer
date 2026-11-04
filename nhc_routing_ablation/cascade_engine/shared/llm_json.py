"""LLM 路由响应 JSON 提取与截断补救。"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def salvage_routing_decision(text: str) -> Optional[dict[str, Any]]:
    """
    当回复在 reasoning 中途被截断、无法配平括号时，尽量救出 selected_model / confidence。
    仅用于避免整图降级；reasoning 可能缺失。
    """
    if not text or "selected_model" not in text and "selected_models" not in text:
        return None

    model: Optional[str] = None
    m_single = re.search(r'"selected_model"\s*:\s*"([^"\\]+)"', text)
    if m_single:
        model = m_single.group(1).strip()
    else:
        m_multi = re.search(r'"selected_models"\s*:\s*\[\s*"([^"\\]+)"', text)
        if m_multi:
            model = m_multi.group(1).strip()

    if not model:
        return None

    conf = 0.5
    m_conf = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    if m_conf:
        try:
            conf = max(0.0, min(1.0, float(m_conf.group(1))))
        except ValueError:
            pass

    reasoning = ""
    m_reason = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    if m_reason:
        reasoning = m_reason.group(1).replace("\\n", " ")[:500]
    if not reasoning:
        reasoning = "（模型输出截断，已 salvage selected_model）"

    out: dict[str, Any] = {
        "selected_model": model,
        "confidence": conf,
        "reasoning": reasoning,
        "_salvaged": True,
    }
    return out


def loads_routing_json(extract_fn, text: str) -> dict[str, Any]:
    """先完整提取 JSON；失败则 salvage。"""
    try:
        raw = extract_fn(text)
        return json.loads(raw)
    except Exception:
        salvaged = salvage_routing_decision(text)
        if salvaged:
            return salvaged
        raise
