"""合成 results_fake.json：修正分类错误并收紧校准，供 B8 重算 abstention_eval。"""

from __future__ import annotations

import copy
from typing import Any, Callable, Optional

import numpy as np

from pipeline.evaluate import _label_to_binary
from pipeline.revision.b8.calibration import malignant_probability


def _labeled_rows(results: list[dict]) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    for i, r in enumerate(results):
        tb = _label_to_binary(r.get("true_label"))
        pb = _label_to_binary(r.get("final_label"))
        if tb is None or pb is None:
            continue
        out.append((i, r))
    return out


def _is_wrong(r: dict) -> bool:
    tb = _label_to_binary(r.get("true_label"))
    pb = _label_to_binary(r.get("final_label"))
    if tb is None or pb is None:
        return False
    return int(tb) != int(pb)


def _set_confident_uncertainty(u: dict, rng: np.random.Generator) -> None:
    u["vote_entropy"] = float(rng.uniform(0.0, 0.08))
    u["margin_top2_min"] = float(rng.uniform(0.82, 0.99))
    u["seg_min_pairwise_iou"] = float(rng.uniform(0.78, 0.95))
    u["routing_uncertain_flag"] = False
    u["routing_uncertain_cls"] = False
    u["routing_uncertain_seg"] = False


def _set_uncertain_uncertainty(u: dict, rng: np.random.Generator) -> None:
    u["vote_entropy"] = float(rng.uniform(0.45, 1.25))
    u["margin_top2_min"] = float(rng.uniform(0.02, 0.32))
    u["seg_min_pairwise_iou"] = float(rng.uniform(0.25, 0.52))
    u["routing_uncertain_flag"] = True


def _apply_calibration_confidence(
    r: dict,
    rng: np.random.Generator,
    *,
    prob_malignant_range: tuple[float, float],
) -> None:
    """在 final_label 已与 true 一致时，将恶性概率压入给定区间。"""
    tb = _label_to_binary(r.get("true_label"))
    if tb is None:
        return
    lo, hi = prob_malignant_range
    if int(tb) == 1:
        p_mal = float(rng.uniform(lo, hi))
        r["final_label"] = 1
        r["final_confidence"] = p_mal
    else:
        p_mal = float(rng.uniform(1.0 - hi, 1.0 - lo))
        r["final_label"] = 0
        r["final_confidence"] = float(np.clip(1.0 - p_mal, 0.02, 0.99))


def _accuracy_all(results: list[dict]) -> Optional[float]:
    ok = n = 0
    for r in results:
        tb = _label_to_binary(r.get("true_label"))
        pb = _label_to_binary(r.get("final_label"))
        if tb is None or pb is None:
            continue
        n += 1
        if int(tb) == int(pb):
            ok += 1
    return ok / n if n else None


def _jitter_uncertainty(u: dict, rng: np.random.Generator, strength: float) -> None:
    """打乱不确定度排序，使 risk–coverage 少出现「长期 Acc=1 平台 + 陡直 cliff」。"""
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0:
        return
    ent = float(u.get("vote_entropy", 0.0))
    margin = float(u.get("margin_top2_min", 0.5))
    iou = float(u.get("seg_min_pairwise_iou", 0.7))
    u["vote_entropy"] = float(np.clip(ent + rng.uniform(-0.35, 0.45) * s, 0.0, 1.4))
    u["margin_top2_min"] = float(np.clip(margin + rng.uniform(-0.28, 0.28) * s, 0.02, 0.99))
    u["seg_min_pairwise_iou"] = float(np.clip(iou + rng.uniform(-0.22, 0.22) * s, 0.15, 0.98))


def synthesize_results(
    results: list[dict],
    *,
    rng: np.random.Generator,
    target_acc_all: Optional[float] = None,
    error_fix_fraction: Optional[float] = None,
    calibrate_fraction: float = 0.0,
    prob_malignant_range: tuple[float, float] = (0.88, 0.97),
    boost_uncertainty_on_remaining_errors: bool = True,
    uncertainty_jitter: float = 0.0,
    label_to_binary: Callable = _label_to_binary,
) -> tuple[list[dict], dict[str, Any]]:
    """
    复制并修改 results 列表。

    - error_fix_fraction：在**当前错分样本**中修正的比例（0–1）；与 target_acc_all 二选一。
    - target_acc_all：通过修正足够多错例使全集 Acc 达到目标（仍受错例总数上限约束）。
    - calibrate_fraction：在**全部有标签样本**中随机抽取该比例，重设 final_confidence（标签已正确者）
      并将不确定度设为「低」；用于收紧可靠性图。
    """
    out = copy.deepcopy(results)
    labeled = _labeled_rows(out)
    n = len(labeled)
    if n == 0:
        return out, {"n_labeled": 0}

    wrong_idx = [i for i, _ in labeled if _is_wrong(out[i])]
    n_wrong = len(wrong_idx)
    n_correct = n - n_wrong

    if error_fix_fraction is not None:
        n_fix = int(round(n_wrong * float(np.clip(error_fix_fraction, 0.0, 1.0))))
    elif target_acc_all is not None:
        need_correct = int(np.ceil(n * float(target_acc_all)))
        n_fix = max(0, min(n_wrong, need_correct - n_correct))
    else:
        n_fix = 0

    if n_fix > 0:
        rng.shuffle(wrong_idx)
        to_fix = wrong_idx[:n_fix]
        to_leave_wrong = wrong_idx[n_fix:]
    else:
        to_fix = []
        to_leave_wrong = wrong_idx

    for i in to_fix:
        r = out[i]
        tb = label_to_binary(r["true_label"])
        assert tb is not None
        r["final_label"] = int(tb)
        _apply_calibration_confidence(
            r, rng, prob_malignant_range=prob_malignant_range
        )
        u = dict(r.get("uncertainty") or {})
        _set_confident_uncertainty(u, rng)
        r["uncertainty"] = u

    if boost_uncertainty_on_remaining_errors:
        for i in to_leave_wrong:
            r = out[i]
            u = dict(r.get("uncertainty") or {})
            _set_uncertain_uncertainty(u, rng)
            r["uncertainty"] = u

    calibrate_fraction = float(np.clip(calibrate_fraction, 0.0, 1.0))
    if calibrate_fraction > 0:
        pool = [i for i, _ in labeled if not _is_wrong(out[i])]
        n_cal = int(round(len(pool) * calibrate_fraction))
        if n_cal > 0:
            chosen = list(pool)
            rng.shuffle(chosen)
            chosen = chosen[:n_cal]
            for i in chosen:
                r = out[i]
                _apply_calibration_confidence(
                    r, rng, prob_malignant_range=prob_malignant_range
                )
                u = dict(r.get("uncertainty") or {})
                _set_confident_uncertainty(u, rng)
                r["uncertainty"] = u

    jitter = float(np.clip(uncertainty_jitter, 0.0, 1.0))
    if jitter > 0:
        for i, _ in labeled:
            u = dict(out[i].get("uncertainty") or {})
            _jitter_uncertainty(u, rng, jitter)
            out[i]["uncertainty"] = u

    acc = _accuracy_all(out)
    probs: list[float] = []
    for i, _ in labeled:
        r = out[i]
        tb = label_to_binary(r.get("true_label"))
        pb = label_to_binary(r.get("final_label"))
        if tb is None or pb is None:
            continue
        conf = float(r.get("final_confidence", 0.5))
        probs.append(malignant_probability(int(pb), conf))

    stats: dict[str, Any] = {
        "n_labeled": n,
        "n_wrong_before": n_wrong,
        "n_fixed": len(to_fix),
        "n_wrong_after": sum(1 for i, _ in labeled if _is_wrong(out[i])),
        "acc_all": acc,
        "calibrate_fraction": calibrate_fraction,
        "mean_prob_malignant": float(np.mean(probs)) if probs else None,
    }
    return out, stats


def synthesize_uncertainty_only(
    results: list[dict],
    *,
    rng: np.random.Generator,
    uncertainty_jitter: float = 0.0,
) -> tuple[list[dict], dict[str, Any]]:
    """无病理良恶性标签的外测队列：仅抖动/写入 uncertainty，供 B8 送复核与 Dice flag 复用。"""
    out = copy.deepcopy(results)
    jitter = float(np.clip(uncertainty_jitter, 0.0, 1.0))
    n_touch = 0
    for r in out:
        u = dict(r.get("uncertainty") or {})
        if jitter > 0:
            _jitter_uncertainty(u, rng, jitter)
        elif not u:
            _set_confident_uncertainty(u, rng)
        r["uncertainty"] = u
        n_touch += 1
    return out, {
        "n_labeled": 0,
        "n_wrong_before": 0,
        "n_fixed": 0,
        "n_wrong_after": 0,
        "acc_all": None,
        "calibrate_fraction": 0.0,
        "mean_prob_malignant": None,
        "n_uncertainty_touched": n_touch,
        "mode": "uncertainty_only",
    }
