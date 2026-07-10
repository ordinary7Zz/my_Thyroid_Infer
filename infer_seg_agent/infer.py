#!/usr/bin/env python3
"""
Segmentation Agent 推理脚本
============================
从 run_all.py 产出的多模型分割掩码中，使用 LLM Agent（默认 local_gpt_oss）
选择或融合最佳掩码。

工作流程:
  1. 读取 results/<task>/<model>/masks/*.png（多个模型的二值掩码）
  2. 构造 ModelOutput 对象（含掩码，可选概率图）
  3. 调用 SegmentationAgent.select_best_mask() 进行 LLM 决策
  4. 保存最终掩码 + 统一 metrics.log（Dice/HD95 + CI95）

用法:
  python infer_seg_agent/infer.py \\
      --task_dir results/nodule \\
      --models dinov3_unet medsam2 medsegx transunet ultrafedfm \\
      --gt_dir /path/to/gt_masks \\
      --config infer_seg_agent/config_seg_agent.yaml \\
      --output_dir results/nodule/seg_agent

  # 使用 local_gpt_oss（默认）
  python infer_seg_agent/infer.py --task_dir results/nodule --agent_type local_gpt_oss

  # 使用云端 LLM
  python infer_seg_agent/infer.py --task_dir results/nodule --agent_type llm
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import cv2
import yaml

# 确保能 import 同级模块
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 项目根目录
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base_model import ModelOutput
from agent.segmentation_agent import SegmentationAgent, AgentDecision
from utils.quality_evaluator import SegmentationQualityEvaluator
from utils.metrics import compute_dice, compute_hd95

# 统一指标模块
from seg_metrics import compute_dice as seg_compute_dice
from seg_metrics import compute_hd95 as seg_compute_hd95
from seg_metrics import bootstrap_ci


# ============================================================================
# 配置加载
# ============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_agent_type(agent_type: str) -> str:
    normalized = str(agent_type or "local_gpt_oss").strip().lower()
    if normalized in {"llm", "glm"}:
        return "llm"
    if normalized in {"local_gpt_oss", "local-gpt-oss", "gpt_oss_local"}:
        return "local_gpt_oss"
    return "local_gpt_oss"


# ============================================================================
# 读取多模型掩码
# ============================================================================

def find_model_mask_dirs(task_dir: Path, model_names: List[str]) -> Dict[str, Path]:
    """在 task_dir 下查找各模型的 masks 子目录。"""
    result = {}
    for name in model_names:
        mask_dir = task_dir / name / "masks"
        if mask_dir.is_dir():
            result[name] = mask_dir
        else:
            print(f"  ⚠️  模型 {name}: 掩码目录不存在: {mask_dir}")
    return result


def load_masks_from_dir(mask_dir: Path) -> Dict[str, np.ndarray]:
    """加载目录下所有 .png 掩码，返回 {stem: mask_array}。"""
    masks = {}
    for p in sorted(mask_dir.glob("*.png")):
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks[p.stem] = (mask > 127).astype(np.uint8)
    return masks


def collect_model_outputs(
    model_mask_dirs: Dict[str, Path],
    gt_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, List[ModelOutput]]:
    """
    收集所有模型的掩码，按图像名分组。
    返回: {image_stem: [ModelOutput, ...]}

    注意: 由于现有推理脚本不保存概率图，confidence_map 设为 None。
    """
    # 收集所有图像名
    all_stems = set()
    model_masks_by_name: Dict[str, Dict[str, np.ndarray]] = {}
    for model_name, mask_dir in model_mask_dirs.items():
        masks = load_masks_from_dir(mask_dir)
        model_masks_by_name[model_name] = masks
        all_stems.update(masks.keys())

    # 按图像分组构造 ModelOutput
    result: Dict[str, List[ModelOutput]] = {}
    for stem in sorted(all_stems):
        outputs = []
        for model_name in model_mask_dirs:
            masks = model_masks_by_name[model_name]
            if stem not in masks:
                continue
            mask = masks[stem]
            outputs.append(ModelOutput(
                model_name=model_name,
                mask=mask,
                confidence_map=None,  # 现有脚本不保存概率图
                metadata={
                    "training_data_devices": [],
                    "base_dataset_performance": {},
                    "dataset_info": {},
                },
            ))
        if outputs:
            result[stem] = outputs
    return result


# ============================================================================
# 指标计算与输出
# ============================================================================

def compute_and_save_metrics(
    decisions: Dict[str, AgentDecision],
    gt_masks: Optional[Dict[str, np.ndarray]],
    output_dir: Path,
    n_bootstrap: int = 2000,
) -> dict:
    """计算 Dice/HD95 并保存 metrics.log。"""
    dice_scores = []
    hd95_scores = []

    for stem, decision in decisions.items():
        if gt_masks and stem in gt_masks:
            gt = gt_masks[stem]
            pred_mask = decision.selected_mask
            # 尺寸对齐
            if pred_mask.shape != gt.shape:
                gt = cv2.resize(gt, (pred_mask.shape[1], pred_mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            dice = seg_compute_dice(pred_mask, gt)
            hd95 = seg_compute_hd95(pred_mask, gt)
            dice_scores.append(dice)
            hd95_scores.append(hd95)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.log"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Segmentation Agent Metrics\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples: {len(decisions)}\n")
        f.write(f"Samples with GT: {len(dice_scores)}\n")
        f.write("=" * 60 + "\n")

        if dice_scores:
            dice_mean, dice_lo, dice_hi = bootstrap_ci(dice_scores, n_boot=n_bootstrap)
            hd95_scores_finite = [v for v in hd95_scores if np.isfinite(v)]
            if hd95_scores_finite:
                hd95_mean, hd95_lo, hd95_hi = bootstrap_ci(hd95_scores_finite, n_boot=n_bootstrap)
            else:
                hd95_mean, hd95_lo, hd95_hi = float("nan"), float("nan"), float("nan")

            f.write(f"Dice:  {dice_mean:.4f}  (95% CI: [{dice_lo:.4f}, {dice_hi:.4f}])\n")
            f.write(f"HD95:  {hd95_mean:.4f}  (95% CI: [{hd95_lo:.4f}, {hd95_hi:.4f}])\n")
            f.write("=" * 60 + "\n")
            f.write("\nPer-sample:\n")
            f.write(f"{'filename':<40s} {'Dice':>8s} {'HD95':>8s}\n")
            f.write("-" * 60 + "\n")

            for stem, decision in decisions.items():
                if gt_masks and stem in gt_masks:
                    gt = gt_masks[stem]
                    pred_mask = decision.selected_mask
                    if pred_mask.shape != gt.shape:
                        gt = cv2.resize(gt, (pred_mask.shape[1], pred_mask.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                    d = seg_compute_dice(pred_mask, gt)
                    h = seg_compute_hd95(pred_mask, gt)
                    f.write(f"{stem:<40s} {d:>8.4f} {h:>8.4f}\n")

            print(f"\n  Dice: {dice_mean:.4f} (95% CI: [{dice_lo:.4f}, {dice_hi:.4f}])")
            print(f"  HD95: {hd95_mean:.4f} (95% CI: [{hd95_lo:.4f}, {hd95_hi:.4f}])")
        else:
            f.write("No GT masks available for evaluation.\n")
            print("\n  No GT masks available for evaluation.")

    print(f"  Metrics saved to: {log_path}")
    return {"dice_scores": dice_scores, "hd95_scores": hd95_scores}


def save_results_json(
    decisions: Dict[str, AgentDecision],
    output_dir: Path,
):
    """保存 Agent 决策结果为 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"results_{timestamp}.json"

    results = []
    for stem, decision in sorted(decisions.items()):
        result = decision.to_simplified_dict()
        result["image_name"] = stem
        results.append(result)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "num_images": len(results),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"  Results saved to: {json_path}")


def save_selected_masks(
    decisions: Dict[str, AgentDecision],
    output_dir: Path,
):
    """保存 Agent 选中的掩码。"""
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for stem, decision in decisions.items():
        mask_path = mask_dir / f"{stem}.png"
        cv2.imwrite(str(mask_path), (decision.selected_mask * 255).astype(np.uint8))

    print(f"  Masks saved to: {mask_dir}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Segmentation Agent — 多模型分割掩码选择/融合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task_dir", type=str, required=True,
        help="run_all.py 输出的任务目录（如 results/nodule），下含各模型的 masks/ 子目录",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="参与决策的模型列表（默认: 自动扫描 task_dir 下所有含 masks/ 的子目录）",
    )
    parser.add_argument(
        "--gt_dir", type=str, default=None,
        help="GT 掩码目录（用于计算 Dice/HD95）",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（默认: <task_dir>/seg_agent）",
    )
    parser.add_argument(
        "--config", type=str, default=str(SCRIPT_DIR / "config_seg_agent.yaml"),
        help="Agent 配置文件路径",
    )
    parser.add_argument(
        "--agent_type", type=str, default=None,
        choices=["llm", "local_gpt_oss"],
        help="覆盖配置中的 agent type（默认: local_gpt_oss）",
    )
    parser.add_argument(
        "--use_agent", action="store_true", default=None,
        help="启用 LLM Agent 决策（默认启用）",
    )
    parser.add_argument(
        "--no_agent", action="store_true",
        help="禁用 LLM Agent，使用投票/置信度选择",
    )
    parser.add_argument(
        "--save_masks", action="store_true", default=True,
        help="保存 Agent 选中的掩码（默认开启）",
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=2000,
        help="Bootstrap CI95 迭代次数",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 确定 agent type
    agent_cfg = config.get("agent", {})
    agent_type = args.agent_type or agent_cfg.get("type", "local_gpt_oss")
    agent_type = _normalize_agent_type(agent_type)

    # 是否启用 agent
    use_agent = True
    if args.no_agent:
        use_agent = False
    elif args.use_agent is not None:
        use_agent = True

    print("=" * 70)
    print("  Segmentation Agent 推理")
    print("=" * 70)
    print(f"  任务目录: {args.task_dir}")
    print(f"  Agent 类型: {agent_type}")
    print(f"  使用 Agent: {use_agent}")
    if agent_type == "local_gpt_oss":
        local_cfg = config.get("agent_local_llm", {})
        print(f"  本地模型: {local_cfg.get('model_path', '(未配置)')}")
    print("=" * 70)

    # 查找模型掩码目录
    task_dir = Path(args.task_dir)
    if not task_dir.is_absolute():
        task_dir = PROJECT_ROOT / task_dir

    if args.models:
        model_names = args.models
    else:
        # 自动扫描
        model_names = []
        for d in sorted(task_dir.iterdir()):
            if d.is_dir() and (d / "masks").is_dir():
                model_names.append(d.name)

    if not model_names:
        print(f"\n  ✗ 未找到任何模型掩码目录。请确认 task_dir 下有 <model>/masks/ 结构。")
        sys.exit(1)

    print(f"\n  参与决策的模型 ({len(model_names)}): {', '.join(model_names)}")

    model_mask_dirs = find_model_mask_dirs(task_dir, model_names)
    if not model_mask_dirs:
        print(f"\n  ✗ 未找到任何有效的掩码目录")
        sys.exit(1)

    # 加载 GT 掩码（可选）
    gt_masks = None
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
        if not gt_dir.is_absolute():
            gt_dir = PROJECT_ROOT / gt_dir
        if gt_dir.is_dir():
            gt_masks = load_masks_from_dir(gt_dir)
            print(f"  GT 掩码: {len(gt_masks)} 张")

    # 收集多模型输出
    print(f"\n  加载多模型掩码...")
    image_outputs = collect_model_outputs(model_mask_dirs, gt_masks)
    print(f"  共 {len(image_outputs)} 张图像有多模型掩码\n")

    if not image_outputs:
        print("  ✗ 没有有效的多模型掩码")
        sys.exit(1)

    # 输出目录
    output_dir = Path(args.output_dir) if args.output_dir else task_dir / "seg_agent"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 Agent
    llm_cfg = config.get("agent_llm", {}) or {}
    local_llm_cfg = config.get("agent_local_llm", {}) or {}
    base_datasets_info = config.get("base_datasets_info", {}) or {}
    ensemble_cfg = agent_cfg.get("ensemble", {}) or {}

    seg_agent = SegmentationAgent(
        backend_type=agent_type,
        api_key=llm_cfg.get("api_key"),
        model_name=llm_cfg.get("model_name", "qwen2.5-32b-instruct"),
        temperature=llm_cfg.get("temperature", 0.3),
        max_tokens=llm_cfg.get("max_tokens", 1024),
        base_url=llm_cfg.get("base_url"),
        base_datasets_info=base_datasets_info,
        ensemble_enabled=ensemble_cfg.get("enabled", False),
        ensemble_top_k=ensemble_cfg.get("top_k", 1),
        ensemble_method=ensemble_cfg.get("method", "weighted_average"),
        ensemble_threshold=ensemble_cfg.get("threshold", 0.5),
        max_retries=agent_cfg.get("max_retries", 3),
        include_disagreement_metrics_in_prompt=agent_cfg.get(
            "include_disagreement_metrics_in_prompt", True
        ),
        local_model_path=local_llm_cfg.get("model_path"),
        local_device_map=local_llm_cfg.get("device_map", "auto"),
        local_torch_dtype=local_llm_cfg.get("torch_dtype", "bfloat16"),
        local_trust_remote_code=local_llm_cfg.get("trust_remote_code", True),
        local_max_new_tokens=local_llm_cfg.get("max_new_tokens", 1024),
    )

    # 逐图决策
    decisions: Dict[str, AgentDecision] = {}
    total = len(image_outputs)
    t0 = time.time()

    for idx, (stem, predictions) in enumerate(image_outputs.items(), 1):
        print(f"\n  [{idx}/{total}] {stem}")
        gt_mask = gt_masks.get(stem) if gt_masks else None

        try:
            if use_agent:
                decision = seg_agent.select_best_mask(
                    predictions,
                    gt_mask=gt_mask,
                )
            else:
                # 非 agent 模式: 使用质量评估器选择一致性最高的掩码
                quality_evaluator = SegmentationQualityEvaluator()
                masks = [p.mask for p in predictions]
                model_names_local = [p.model_name for p in predictions]
                quality_results = quality_evaluator.evaluate_batch(masks, model_names_local)
                decision = seg_agent._fallback_selection(predictions, gt_mask, quality_results)

            decisions[stem] = decision
            print(f"    ✓ 选中: {decision.selected_model} (confidence={decision.confidence:.4f})")
            if decision.dice_score is not None:
                print(f"    Dice={decision.dice_score:.4f}")

        except Exception as e:
            print(f"    ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  完成: {len(decisions)}/{total} 张图像，耗时 {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # 保存结果
    save_results_json(decisions, output_dir)
    if args.save_masks:
        save_selected_masks(decisions, output_dir)
    compute_and_save_metrics(decisions, gt_masks, output_dir, args.n_bootstrap)


if __name__ == "__main__":
    main()
