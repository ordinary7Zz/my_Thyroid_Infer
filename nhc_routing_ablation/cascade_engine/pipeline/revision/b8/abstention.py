"""B8：验证集定阈 + 测试集拒答与 risk–coverage。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from pipeline.revision.b8.calibration import malignant_probability
from pipeline.evaluate import _label_to_binary

# NHC 无 LLM：默认规则子集
NHC_RULE_FIELDS = frozenset({
    "vote_entropy",
    "margin_top2_min",
    "seg_min_pairwise_iou",
})

DEFAULT_RULES: list[dict[str, Any]] = [
    {"name": "cls_vote_entropy", "field": "vote_entropy", "op": "gt"},
    {"name": "cls_margin_low", "field": "margin_top2_min", "op": "lt"},
    {"name": "seg_iou_low", "field": "seg_min_pairwise_iou", "op": "lt"},
    {"name": "router_uncertain", "field": "routing_uncertain_flag", "op": "eq", "value": True},
]


def _get_uncertainty_field(result: dict, field: str) -> Optional[float]:
    u = result.get("uncertainty") or {}
    if field in u:
        val = u[field]
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        return float(val)
    if field == "routing_uncertain_flag":
        return 1.0 if result.get("routing_uncertain_flag") else 0.0
    return None


def _rule_triggered(result: dict, rule: dict, threshold: float) -> bool:
    field = rule["field"]
    op = rule["op"]
    val = _get_uncertainty_field(result, field)
    if val is None:
        return False
    if op == "gt":
        return val > threshold
    if op == "lt":
        return val < threshold
    if op == "eq":
        target = rule.get("value", True)
        if isinstance(target, bool):
            return bool(val >= 0.5) == target
        return val == float(target)
    raise ValueError(f"未知 op: {op}")


def _active_rules(rules: list[dict], rule_subset: Optional[str]) -> list[dict]:
    if rule_subset == "nhc":
        return [r for r in rules if r["field"] in NHC_RULE_FIELDS]
    return rules


@dataclass
class AbstentionThresholds:
    fitted_on: str
    n_val: int
    combination: str
    rules: list[dict[str, Any]]
    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted_on": self.fitted_on,
            "n_val": self.n_val,
            "combination": self.combination,
            "rules": self.rules,
            "thresholds": self.thresholds,
        }


def fit_thresholds_quantile(
    val_results: list[dict],
    rules: Optional[list[dict]] = None,
    *,
    combination: str = "any",
    rule_subset: Optional[str] = None,
    quantile: float = 0.9,
    fitted_on: str = "validation",
) -> AbstentionThresholds:
    """
    在验证集上按分位数定阈（简单可复现默认方案）。

    - gt 规则（entropy）：阈值为该字段的 quantile 分位（高熵拒答）
    - lt 规则（margin、iou）：阈值为 quantile 分位（低 margin / 低 iou 拒答）
    """
    rules = list(rules or DEFAULT_RULES)
    rules = _active_rules(rules, rule_subset)
    thresholds: dict[str, float] = {}

    for rule in rules:
        field = rule["field"]
        op = rule["op"]
        values = []
        for r in val_results:
            v = _get_uncertainty_field(r, field)
            if v is not None:
                values.append(v)
        if not values:
            thresholds[rule["name"]] = float("nan")
            continue
        arr = np.array(values, dtype=np.float64)
        if op == "gt":
            thresholds[rule["name"]] = float(np.quantile(arr, quantile))
        elif op == "lt":
            thresholds[rule["name"]] = float(np.quantile(arr, 1.0 - quantile))
        else:
            thresholds[rule["name"]] = float(rule.get("value", 1.0))

    return AbstentionThresholds(
        fitted_on=fitted_on,
        n_val=len(val_results),
        combination=combination,
        rules=rules,
        thresholds=thresholds,
    )


def fit_thresholds_split_quantile(
    val_results: list[dict],
    rules: Optional[list[dict]] = None,
    *,
    combination: str = "any",
    rule_subset: Optional[str] = None,
    q_gt: float = 0.9,
    q_lt: float = 0.05,
    fitted_on: str = "validation",
) -> AbstentionThresholds:
    """
    gt / lt 规则使用独立分位（用于在 OR 组合下更细地调节送复核率）。

    - gt：阈值为 q_gt 分位
    - lt：阈值为 q_lt 分位（低 margin / 低 IoU 尾部分位）
    """
    rules = list(rules or DEFAULT_RULES)
    rules = _active_rules(rules, rule_subset)
    thresholds: dict[str, float] = {}

    for rule in rules:
        field = rule["field"]
        op = rule["op"]
        values = []
        for r in val_results:
            v = _get_uncertainty_field(r, field)
            if v is not None:
                values.append(v)
        if not values:
            thresholds[rule["name"]] = float("nan")
            continue
        arr = np.array(values, dtype=np.float64)
        if op == "gt":
            thresholds[rule["name"]] = float(np.quantile(arr, q_gt))
        elif op == "lt":
            thresholds[rule["name"]] = float(np.quantile(arr, q_lt))
        else:
            thresholds[rule["name"]] = float(rule.get("value", 1.0))

    return AbstentionThresholds(
        fitted_on=fitted_on,
        n_val=len(val_results),
        combination=combination,
        rules=rules,
        thresholds=thresholds,
    )


def flag_for_review(
    result: dict,
    thresholds: AbstentionThresholds,
) -> bool:
    fired: list[bool] = []
    for rule in thresholds.rules:
        t = thresholds.thresholds.get(rule["name"])
        if t is None or np.isnan(t):
            fired.append(False)
            continue
        fired.append(_rule_triggered(result, rule, t))
    if not fired:
        return False
    if thresholds.combination == "any":
        return any(fired)
    if thresholds.combination == "all":
        return all(fired)
    raise ValueError(f"未知 combination: {thresholds.combination}")


def apply_abstention(
    results: list[dict],
    thresholds: AbstentionThresholds,
) -> list[dict]:
    out = []
    for r in results:
        flagged = flag_for_review(r, thresholds)
        row = dict(r)
        row["flagged_for_review"] = flagged
        out.append(row)
    return out


def _accuracy_on_retained(results: list[dict]) -> Optional[float]:
    ok = 0
    n = 0
    for r in results:
        if r.get("flagged_for_review"):
            continue
        tb = _label_to_binary(r.get("true_label"))
        pb = _label_to_binary(r.get("final_label"))
        if tb is None or pb is None:
            continue
        n += 1
        if int(tb) == int(pb):
            ok += 1
    return ok / n if n else None


def risk_coverage_curve(
    results: list[dict],
    *,
    score_fn: Optional[Callable[[dict], float]] = None,
) -> list[dict[str, Any]]:
    """
    按不确定度分数降序拒答，记录 coverage vs 保留子集 accuracy。

    score 默认：vote_entropy 高 + margin 低 + iou 低 → 更大分数优先拒答。
    """
    if score_fn is None:
        def score_fn(r: dict) -> float:
            u = r.get("uncertainty") or {}
            ent = float(u.get("vote_entropy", 0.0))
            margin = float(u.get("margin_top2_min", 1.0))
            iou = float(u.get("seg_min_pairwise_iou", 1.0))
            return ent + (1.0 - margin) + (1.0 - iou)

    valid = [r for r in results if "true_label" in r and "final_label" in r]
    n = len(valid)
    if n == 0:
        return []

    scored = sorted(valid, key=score_fn, reverse=True)
    curve: list[dict[str, Any]] = []
    for k in range(0, n + 1):
        flagged_rate = k / n
        subset = [{**r, "flagged_for_review": i < k} for i, r in enumerate(scored)]
        curve.append({
            "n_flagged": k,
            "coverage": 1.0 - flagged_rate,
            "abstention_rate": flagged_rate,
            "retained_accuracy": _accuracy_on_retained(subset),
        })
    return curve


def summarize_abstention_by_group(
    results: list[dict],
    group_key: str = "center_id",
) -> dict[str, Any]:
    by_group: dict[str, list[dict]] = {}
    for r in results:
        g = r.get(group_key) or "unknown"
        by_group.setdefault(str(g), []).append(r)
    summary = {}
    for g, rows in sorted(by_group.items()):
        n = len(rows)
        n_flag = sum(1 for r in rows if r.get("flagged_for_review"))
        summary[g] = {
            "n": n,
            "n_flagged": n_flag,
            "abstention_rate": n_flag / n if n else 0.0,
            "retained_accuracy": _accuracy_on_retained(rows),
        }
    return summary


def save_thresholds(path: Path, thresholds: AbstentionThresholds) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds.to_dict(), f, ensure_ascii=False, indent=2)


def load_thresholds(path: Path) -> AbstentionThresholds:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return AbstentionThresholds(
        fitted_on=d["fitted_on"],
        n_val=int(d["n_val"]),
        combination=d["combination"],
        rules=d["rules"],
        thresholds=d["thresholds"],
    )
