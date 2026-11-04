"""Phase 3 性能与断点续跑配置（无重型依赖）。"""

from __future__ import annotations


def phase3_expert_judge_enabled(config: dict) -> bool:
    pipe = config.get("pipeline", {}) or {}
    rev = pipe.get("revision") or {}
    if "phase3_expert_judge" in rev:
        return bool(rev["phase3_expert_judge"])
    if "phase3_expert_judge" in pipe:
        return bool(pipe["phase3_expert_judge"])
    return True


def phase3_checkpoint_every(config: dict) -> int:
    pipe = config.get("pipeline", {}) or {}
    rev = pipe.get("revision") or {}
    raw = pipe.get("phase3_checkpoint_every")
    if raw is None:
        raw = rev.get("phase3_checkpoint_every")
    if raw is None:
        return 0
    return max(0, int(raw))


def validate_phase3_checkpoint(img_names: list[str], done_names: list[str]) -> bool:
    if len(done_names) > len(img_names):
        return False
    for i, name in enumerate(done_names):
        if img_names[i] != name:
            return False
    return True
