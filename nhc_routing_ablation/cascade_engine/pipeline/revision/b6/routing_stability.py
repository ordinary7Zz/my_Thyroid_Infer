"""
B6：在复用 intermediate 缓存的前提下，对 LLM 路由重复 N 次并汇总一致率（按 disagreement_tier 分层）。
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from shared.routing_log import seg_disagreement_tier
from segmentation.metrics import compute_dice


def _agreement_rate(choices: list[str]) -> float:
    if not choices:
        return float("nan")
    counts = Counter(choices)
    return float(max(counts.values())) / len(choices)


def _summarize_repeats(
    per_case: list[dict[str, Any]],
) -> dict[str, Any]:
    by_tier: dict[str, list[float]] = defaultdict(list)
    all_agree = []
    for row in per_case:
        tier = row.get("disagreement_tier", "unknown")
        ar = row.get("selection_agreement")
        if ar is not None and not np.isnan(ar):
            by_tier[tier].append(ar)
            all_agree.append(ar)
    stratified = {
        tier: {
            "n_cases": len(vals),
            "mean_selection_agreement": float(np.mean(vals)) if vals else None,
        }
        for tier, vals in by_tier.items()
    }
    return {
        "n_cases": len(per_case),
        "mean_selection_agreement": float(np.mean(all_agree)) if all_agree else None,
        "by_disagreement_tier": stratified,
        "per_case": per_case,
    }


def run_stability_from_caches(
    *,
    image_io,
    seg_agent,
    cls_agent,
    image_paths: list[Path],
    img_names: list[str],
    seg_model_configs: list[dict],
    cls_model_configs: list[dict],
    seg_cache_dir: Path,
    cls_cache_dir: Path,
    load_seg_npz,
    load_cls_json,
    check_classification_consensus,
    consensus_threshold: float,
    enable_seg_agent: bool,
    enable_cls_agent: bool,
    device_info: Optional[list[str]],
    gt_mask_dir: Optional[str],
    repeats: int = 5,
    max_cases: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    对每张图：若分割 Agent 开启则 5× select_best_mask；
    若分类无共识且 cls agent 开启则 5× resolve_path_b（需先有 seg 决策，用静态 agreement 选 mask 作 stub）。
    """
    from classification.base_model import ClsModelOutput
    from segmentation.agent import SegAgentDecision

    if max_cases is not None:
        image_paths = image_paths[:max_cases]
        img_names = img_names[:max_cases]

    from pipeline.revision.b6.cache_subset import load_cls_json_map_subset, load_seg_npz_subset

    wanted = list(img_names)
    seg_cache_by_model: dict[str, dict[str, Any]] = {}
    for model_cfg in seg_model_configs:
        model_name = model_cfg.get("name", "")
        cache_path = seg_cache_dir / f"{model_name}.npz"
        if cache_path.exists():
            seg_cache_by_model[model_name] = load_seg_npz_subset(
                cache_path, wanted, load_confidence=enable_seg_agent
            )
            hit = len(seg_cache_by_model[model_name])
            print(
                f"  ✓ seg 缓存: {cache_path.name}（子集 {hit}/{len(wanted)} 张）",
                flush=True,
            )
        else:
            print(f"  ✗ seg 缓存缺失: {cache_path}", flush=True)

    cls_cache_by_model: dict[str, dict[str, Any]] = {}
    for model_cfg in cls_model_configs:
        model_name = model_cfg.get("name", "")
        cache_path = cls_cache_dir / f"{model_name}.json"
        if cache_path.exists():
            cls_cache_by_model[model_name] = load_cls_json_map_subset(cache_path, wanted)
            hit = len(cls_cache_by_model[model_name])
            print(
                f"  ✓ cls 缓存: {cache_path.name}（子集 {hit}/{len(wanted)} 张）",
                flush=True,
            )
        else:
            print(f"  ✗ cls 缓存缺失: {cache_path}", flush=True)

    n_cases = len(img_names)
    if n_cases == 0:
        print("⚠️ 无图像可处理", flush=True)
        return _summarize_repeats([])

    llm_calls_planned = 0
    if enable_seg_agent:
        llm_calls_planned += n_cases * repeats
    if enable_cls_agent:
        # 实际 Path B 子集更小，此处为上界估计
        llm_calls_planned += n_cases * repeats
    print(
        f"\n处理 {n_cases} 张图，每张最多 seg×{repeats}"
        + (f" + cls PathB×{repeats}" if enable_cls_agent else "")
        + f" 次 LLM（约 ≤{llm_calls_planned} 次）\n",
        flush=True,
    )

    per_case: list[dict[str, Any]] = []
    t_batch_start = time.perf_counter()

    for idx, img_name in enumerate(img_names):
        t_case = time.perf_counter()
        img_path = image_paths[idx]
        image = image_io.load_image(img_path)

        seg_preds = []
        for model_cfg in seg_model_configs:
            model_name = model_cfg.get("name", "")
            by_name = seg_cache_by_model.get(model_name)
            if by_name is not None:
                pred = by_name.get(img_name)
                if pred is not None:
                    seg_preds.append(pred)

        indie_preds = []
        for model_cfg in cls_model_configs:
            model_name = model_cfg.get("name", "")
            cls_map = cls_cache_by_model.get(model_name)
            if cls_map is not None:
                pred = cls_map.get(img_name)
                if pred is not None:
                    indie_preds.append(pred)

        consensus, anchor_class = check_classification_consensus(
            indie_preds, min_confidence=consensus_threshold
        )

        gt_mask = None
        if gt_mask_dir:
            gt_path = Path(gt_mask_dir) / img_name
            if gt_path.exists():
                gt_mask = image_io.binarize_mask(image_io.load_mask(gt_path))

        judge_results = None
        judge_obj = getattr(seg_agent, "radiomics_judge", None)
        if judge_obj is not None and seg_preds:
            try:
                judge_results = judge_obj.judge_batch(image, [p.mask for p in seg_preds])
            except Exception as e:
                print(f"    ⚠️ radiomics judge 失败: {e}", flush=True)
                judge_results = None

        quality_results = None
        if seg_preds:
            masks = [p.mask for p in seg_preds]
            names = [p.model_name for p in seg_preds]
            quality_results = seg_agent.quality_evaluator.evaluate_batch(masks, names)
        tier = seg_disagreement_tier(quality_results)

        row: dict[str, Any] = {
            "image_name": img_name,
            "classification_consensus": consensus,
            "disagreement_tier": tier,
        }

        path_label = "A" if consensus else "B"
        n_seg = len(seg_preds)
        print(
            f"[{idx + 1}/{n_cases}] {img_name} | path={path_label} "
            f"consensus={consensus} tier={tier} seg_cands={n_seg}",
            flush=True,
        )

        if enable_seg_agent and seg_preds:
            seg_choices = []
            dice_vals = []
            for r in range(repeats):
                print(f"    seg LLM {r + 1}/{repeats} ...", flush=True)
                dec = seg_agent.select_best_mask(
                    image,
                    seg_preds,
                    classification_anchor=anchor_class,
                    gt_mask=gt_mask,
                    input_device_info=device_info,
                    judge_results=judge_results,
                    image_name=img_name,
                )
                seg_choices.append(dec.selected_model)
                print(f"      → {dec.selected_model}", flush=True)
                if gt_mask is not None:
                    dice_vals.append(float(compute_dice(dec.selected_mask, gt_mask)))
            row["seg_selection_agreement"] = _agreement_rate(seg_choices)
            row["seg_selected_models"] = seg_choices
            if dice_vals:
                row["seg_dice_std"] = float(np.std(dice_vals))
                row["seg_dice_range"] = float(max(dice_vals) - min(dice_vals))
        elif enable_seg_agent and not seg_preds:
            print("    ⚠️ 无分割缓存，跳过 seg LLM", flush=True)

        if enable_cls_agent and indie_preds and not consensus:
            static_dec = seg_agent._fallback_selection(
                seg_preds,
                gt_mask,
                quality_results,
                judge_results,
                anchor_class,
            ) if seg_preds and quality_results else None
            if static_dec is None and seg_preds:
                static_dec = SegAgentDecision(
                    selected_model=seg_preds[0].model_name,
                    selected_mask=seg_preds[0].mask,
                    confidence=0.5,
                    reasoning="stability stub",
                    all_predictions=[p.to_dict() for p in seg_preds],
                    path="B",
                )
            cls_choices = []
            for r in range(repeats):
                print(f"    cls PathB LLM {r + 1}/{repeats} ...", flush=True)
                dec = cls_agent.resolve_path_b(
                    indie_preds,
                    None,
                    static_dec,
                    voting_weights=None,
                    image_name=img_name,
                )
                cls_choices.append(dec.selected_class)
                print(f"      → {dec.selected_class}", flush=True)
            row["cls_selection_agreement"] = _agreement_rate(cls_choices)
            row["cls_selected_classes"] = cls_choices

        seg_ag = row.get("seg_selection_agreement")
        cls_ag = row.get("cls_selection_agreement")
        if seg_ag is not None:
            row["selection_agreement"] = seg_ag
        elif cls_ag is not None:
            row["selection_agreement"] = cls_ag
        else:
            row["selection_agreement"] = None
        per_case.append(row)

        agree = row.get("selection_agreement")
        elapsed = time.perf_counter() - t_case
        print(
            f"    完成 ({elapsed:.1f}s) selection_agreement={agree}",
            flush=True,
        )

    summary = _summarize_repeats(per_case)
    summary["repeats"] = repeats
    summary["enable_seg_agent"] = enable_seg_agent
    summary["enable_cls_agent"] = enable_cls_agent

    total_elapsed = time.perf_counter() - t_batch_start
    print(f"\n全部 {n_cases} 张处理完毕，耗时 {total_elapsed:.1f}s", flush=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"✓ stability_summary → {output_path}", flush=True)

    return summary
