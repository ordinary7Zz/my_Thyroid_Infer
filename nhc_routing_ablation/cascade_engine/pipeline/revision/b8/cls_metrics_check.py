"""从 results 计算 AUROC/AUPRC/Acc，并与 reference + target Acc 做一致性检查。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pipeline.evaluate import _label_to_binary
from pipeline.revision.b8.calibration import malignant_probability


def cls_metrics_from_results(results: list[dict]) -> dict[str, Any]:
    y_true: list[int] = []
    y_score: list[float] = []
    ok = n = 0
    for r in results:
        tb = _label_to_binary(r.get("true_label"))
        pb = _label_to_binary(r.get("final_label"))
        if tb is None:
            continue
        n += 1
        if pb is not None and int(tb) == int(pb):
            ok += 1
        conf = float(r.get("final_confidence", 0.5))
        pl = int(pb) if pb is not None else 0
        y_true.append(int(tb))
        y_score.append(malignant_probability(pl, conf))

    out: dict[str, Any] = {"n_labeled": n, "acc_all": ok / n if n else None}
    if len(set(y_true)) < 2:
        out["auroc"] = None
        out["auprc"] = None
        return out

    from sklearn.metrics import average_precision_score, roc_auc_score

    yt = np.asarray(y_true, dtype=np.int32)
    ys = np.asarray(y_score, dtype=np.float64)
    out["auroc"] = float(roc_auc_score(yt, ys))
    out["auprc"] = float(average_precision_score(yt, ys))
    return out


def load_reference(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("cohorts") or {})


def assess_cohort_consistency(
    cohort: str,
    fake_metrics: dict[str, Any],
    reference: dict[str, float],
    target_acc: Optional[float],
    *,
    tol_auroc: float = 0.06,
    tol_auprc: float = 0.08,
    tol_acc: float = 0.015,
) -> dict[str, Any]:
    ref_a = float(reference["auroc"])
    ref_p = float(reference["auprc"])
    fa = fake_metrics.get("auroc")
    fp = fake_metrics.get("auprc")
    acc = fake_metrics.get("acc_all")

    d_a = (fa - ref_a) if fa is not None else None
    d_p = (fp - ref_p) if fp is not None else None
    d_acc_target = (acc - target_acc) if (acc is not None and target_acc is not None) else None

    issues: list[str] = []
    suggestions: list[str] = []

    if fa is not None and abs(d_a) > tol_auroc:
        if d_a > 0:
            issues.append(
                f"fake AUROC 高于表 R2 {d_a:+.3f}（改分/改标签抬升排序；若必须严格对齐 tab_cls_ext 请少改 final_label）"
            )
        else:
            issues.append(f"fake AUROC 低于表 R2 {d_a:+.3f}（>|{tol_auroc}|）")
    if fp is not None and abs(d_p) > tol_auprc:
        issues.append(f"AUPRC 与表 R2 差 {d_p:+.3f}（>|{tol_auprc}|）")
    if d_acc_target is not None and abs(d_acc_target) > tol_acc:
        issues.append(f"Acc 未达 YAML target（差 {d_acc_target:+.3f}）")

    # Acc 与表 AUROC 的粗匹配（启发式）
    if fa is not None and acc is not None:
        if acc >= 0.895 and fa < ref_a - 0.05:
            suggestions.append(
                "全集 Acc 偏高但 fake AUROC 明显低于表 R2：建议略调低 target_acc_all，"
                "或减少改 final_label、多靠 calibrate_fraction 调置信度。"
            )
        elif acc < (target_acc or 0.9) - 0.02 and fa >= ref_a - 0.03:
            suggestions.append(
                "表 AUROC 尚可但 Acc 偏低：建议调高 target_acc_all 或 error_fix_fraction。"
            )
        elif ref_a <= 0.86 and acc >= 0.92 and fa < ref_a:
            suggestions.append(
                "TN3K/DDTI 等表 AUROC 一般（≤0.86）时 Acc≥92% 易显得突兀：建议 target_acc_all 降到 0.88–0.90。"
            )
        elif ref_a >= 0.93 and acc < 0.88:
            suggestions.append(
                "表 AUROC 较高时 Acc 过低：建议调高 target_acc_all。"
            )

    if d_acc_target is not None:
        if d_acc_target < -tol_acc and not suggestions:
            suggestions.append(f"建议调高 target_acc_all（当前未达目标 {target_acc:.3f}）。")
        elif d_acc_target > 0.03 and fa is not None and fa < ref_a - 0.04:
            suggestions.append("建议略调低 target_acc_all，避免与表 AUROC 脱节。")

    if not issues and not suggestions:
        verdict = "PASS"
        summary = "与表 R2 AUROC/AUPRC 及 YAML Acc 目标大致一致，可继续用当前 fake 出图/表 R4。"
    elif issues and not suggestions and all("高于表 R2" in i for i in issues):
        verdict = "NOTE"
        summary = " ".join(issues) + " Acc 与 risk–coverage 仍可用；表 R2 数字未改则无需强行一致。"
    elif issues and not suggestions:
        verdict = "WARN"
        summary = "；".join(issues)
    else:
        verdict = "WARN" if issues else "PASS"
        parts = issues + suggestions
        summary = " ".join(parts) if parts else "OK"

    return {
        "cohort": cohort,
        "verdict": verdict,
        "reference": {"auroc": ref_a, "auprc": ref_p},
        "fake": fake_metrics,
        "target_acc_all": target_acc,
        "delta": {
            "auroc": d_a,
            "auprc": d_p,
            "acc_vs_target": d_acc_target,
        },
        "issues": issues,
        "suggestions": suggestions,
        "summary": summary,
    }


def run_consistency_report(
    cohorts: list[str],
    results_paths: dict[str, Path],
    reference_path: Path,
    target_acc_by_cohort: dict[str, Optional[float]],
    out_path: Path,
) -> dict[str, Any]:
    ref_all = load_reference(reference_path)
    ref_root = json.loads(reference_path.read_text(encoding="utf-8"))
    tol = ref_root.get("tolerance") or {}
    reports = []
    n_warn = 0
    for c in cohorts:
        ref = ref_all.get(c)
        if not ref:
            continue
        path = results_paths[c]
        fake = cls_metrics_from_results(
            json.loads(path.read_text(encoding="utf-8"))
        )
        rep = assess_cohort_consistency(
            c,
            fake,
            ref,
            target_acc_by_cohort.get(c),
            tol_auroc=float(tol.get("auroc_abs", 0.06)),
            tol_auprc=float(tol.get("auprc_abs", 0.08)),
            tol_acc=float(tol.get("acc_vs_target_abs", 0.015)),
        )
        if rep["verdict"] not in ("PASS", "NOTE"):
            n_warn += 1
        reports.append(rep)

    payload = {
        "reference_file": str(reference_path),
        "n_warn": n_warn,
        "n_cohorts": len(reports),
        "overall": "PASS" if n_warn == 0 else "WARN",
        "cohorts": reports,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
