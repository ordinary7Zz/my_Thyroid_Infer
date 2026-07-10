#!/usr/bin/env python3
"""
Classification Agent 推理脚本
=============================
从 run_all.py 产出的多模型分类 CSV 中，使用 LLM Agent（默认 local_gpt_oss）
选择最佳模型预测或进行 soft voting 融合。

工作流程:
  1. 读取 results/<task>/<model>/predictions*.csv（多个模型的预测结果）
  2. 构造 ModelOutput 对象（含概率分布、置信度、元数据）
  3. 调用 LLMClassificationAgent.select_best_model_batch() 进行 LLM 决策
  4. 保存最终 predictions.csv + 统一 metrics.log（AUROC/AUPRC/Acc/F1 等）

用法:
  python infer_cls_agent/infer.py \\
      --task_dir results/binary \\
      --models biomedclip medsiglip dinov3_unet_multitask ultrafedfm autogluon \\
      --label_json /path/to/labels.json \\
      --label_field malignancy \\
      --config infer_cls_agent/config_cls_agent.yaml \\
      --output_dir results/binary/cls_agent

  # 使用 local_gpt_oss（默认）
  python infer_cls_agent/infer.py --task_dir results/binary --agent_type local_gpt_oss

  # 使用云端 LLM
  python infer_cls_agent/infer.py --task_dir results/binary --agent_type llm
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
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
from agent.classification_agent import (
    LLMClassificationAgent,
    AgentDecision,
    _average_class_probabilities,
    _winning_class_from_avg_probs,
)

# 统一指标模块
from cls_metrics import compute_all_metrics, format_metrics_report


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
# 读取多模型 CSV
# ============================================================================

def find_model_csv_files(task_dir: Path, model_names: List[str]) -> Dict[str, Path]:
    """在 task_dir 下查找各模型的 predictions*.csv 文件。"""
    result = {}
    for name in model_names:
        model_dir = task_dir / name
        if not model_dir.is_dir():
            print(f"  ⚠️  模型 {name}: 目录不存在: {model_dir}")
            continue
        # 查找 predictions*.csv
        csvs = sorted(model_dir.glob("predictions*.csv"))
        if not csvs:
            # 也尝试 results*.csv
            csvs = sorted(model_dir.glob("results*.csv"))
        if csvs:
            result[name] = csvs[-1]  # 取最新的
        else:
            print(f"  ⚠️  模型 {name}: 未找到 predictions*.csv")
    return result


def parse_csv_predictions(csv_path: Path) -> List[dict]:
    """解析 CSV 预测文件，返回行列表。"""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def collect_model_outputs(
    model_csv_files: Dict[str, Path],
) -> Tuple[Dict[str, List[ModelOutput]], Dict[str, str]]:
    """
    收集所有模型的 CSV 预测，按图像名分组。
    返回: ({image_name: [ModelOutput, ...]}, {image_name: filename})
    """
    # 解析所有 CSV
    model_rows: Dict[str, Dict[str, dict]] = {}  # {model_name: {filename: row}}
    all_filenames = set()

    for model_name, csv_path in model_csv_files.items():
        rows = parse_csv_predictions(csv_path)
        file_to_row = {}
        for row in rows:
            fn = row.get("filename", "")
            if fn:
                file_to_row[fn] = row
                all_filenames.add(fn)
        model_rows[model_name] = file_to_row

    # 按图像分组构造 ModelOutput
    result: Dict[str, List[ModelOutput]] = {}
    filename_map: Dict[str, str] = {}

    for fn in sorted(all_filenames):
        outputs = []
        for model_name in model_csv_files:
            file_to_row = model_rows.get(model_name, {})
            row = file_to_row.get(fn)
            if row is None:
                continue

            # 解析预测
            predictions = {}
            top_class = row.get("predicted_class", "")
            top_confidence = float(row.get("confidence", 0.5))

            # 解析概率列
            for key, value in row.items():
                if key.startswith("prob_"):
                    class_name = key[5:]  # 去掉 "prob_" 前缀
                    try:
                        predictions[class_name] = float(value)
                    except (ValueError, TypeError):
                        pass

            # 如果没有概率列，用 confidence 构造
            if not predictions and top_class:
                predictions[top_class] = top_confidence
                # 对于二分类，补充另一类
                if len(predictions) == 1:
                    other = "other"
                    predictions[other] = 1.0 - top_confidence

            # 确保 top_class 在 predictions 中
            if top_class and top_class not in predictions:
                predictions[top_class] = top_confidence

            # 重新计算 top_confidence（取最大概率）
            if predictions:
                best_cls = max(predictions.items(), key=lambda x: x[1])
                top_class = best_cls[0]
                top_confidence = best_cls[1]

            outputs.append(ModelOutput(
                model_name=model_name,
                predictions=predictions,
                top_class=str(top_class),
                top_confidence=float(top_confidence),
                requires_mask=False,
                metadata={
                    "training_data_devices": [],
                    "dataset_info": {},
                    "validation_metrics": {},
                    "base_dataset_performance": {},
                },
            ))

        if outputs:
            result[fn] = outputs
            filename_map[fn] = fn

    return result, filename_map


# ============================================================================
# 标签加载
# ============================================================================

def load_labels(label_json: str, label_field: str = "malignancy") -> Dict[str, int]:
    """加载标签文件。支持 [{filename, malignancy}, ...] 或 {filename: label} 格式。"""
    with open(label_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            fn = item.get("filename") or item.get("image_name", "")
            label = item.get(label_field, item.get("label", -1))
            if fn and label is not None:
                labels[fn] = int(label)
    elif isinstance(data, dict):
        for k, v in data.items():
            labels[k] = int(v)

    return labels


def label_lookup(labels: Dict[str, int], image_name: str) -> Optional[int]:
    """查找标签，支持模糊匹配。"""
    if image_name in labels:
        return labels[image_name]

    # 尝试去掉前缀匹配（如 "TN3K_test_0001.jpg" -> "0001.jpg"）
    stem = Path(image_name).stem
    ext = Path(image_name).suffix
    tokens = stem.split("_")
    if len(tokens) >= 2:
        candidate = tokens[-1] + ext
        if candidate in labels:
            return labels[candidate]

    return None


# ============================================================================
# 指标计算与输出
# ============================================================================

def compute_and_save_metrics(
    decisions: Dict[str, AgentDecision],
    labels: Dict[str, int],
    output_dir: Path,
    num_classes: int,
    n_bootstrap: int = 2000,
):
    """计算分类指标并保存 metrics.log。"""
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    for fn, decision in decisions.items():
        gt = label_lookup(labels, fn)
        if gt is None or gt < 0:
            continue

        # 获取概率分布
        probs = decision.all_predictions
        # 从 all_predictions 中取选中模型的概率
        # 由于 decision.all_predictions 是 list of dict，我们直接用 decision 的结果
        pred_class = decision.selected_class
        confidence = decision.confidence

        # 构造概率向量
        # 尝试从 all_predictions 获取选中模型的完整概率
        selected_probs = None
        for pred in probs:
            if pred.get("model_name") == decision.selected_model:
                selected_probs = pred.get("predictions", {})
                break

        if selected_probs is None:
            # fallback: 用 confidence 构造
            if num_classes == 2:
                selected_probs = {pred_class: confidence, "other": 1.0 - confidence}
            else:
                selected_probs = {pred_class: confidence}

        # 构造统一的概率向量
        if num_classes == 2:
            # 二分类: P(正类=1)
            # 尝试匹配 "恶性"/"1"/"malignant"
            p_positive = None
            for key in ["恶性", "1", "malignant", "1.0"]:
                if key in selected_probs:
                    p_positive = selected_probs[key]
                    break
            if p_positive is None:
                # 取第二个类别的概率
                probs_list = list(selected_probs.values())
                p_positive = probs_list[-1] if len(probs_list) >= 2 else confidence

            y_true_list.append(gt)
            y_pred_list.append(1 if p_positive >= 0.5 else 0)
            y_prob_list.append([1.0 - p_positive, p_positive])
        else:
            # 多分类
            class_indices = sorted(set(labels.values()) | {i for i in range(num_classes)})
            prob_vec = []
            for c in range(num_classes):
                prob_vec.append(selected_probs.get(str(c), selected_probs.get(str(c + 1), 0.0)))
            # 归一化
            total = sum(prob_vec)
            if total > 0:
                prob_vec = [p / total for p in prob_vec]
            else:
                prob_vec = [1.0 / num_classes] * num_classes

            y_true_list.append(gt)
            y_pred_list.append(np.argmax(prob_vec))
            y_prob_list.append(prob_vec)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.log"

    if not y_true_list:
        print("  无可用标签样本，跳过指标计算")
        with open(log_path, "w") as f:
            f.write("No labeled samples available for evaluation.\n")
        return

    y_true = np.array(y_true_list, dtype=np.int32)
    y_pred = np.array(y_pred_list, dtype=np.int32)
    y_prob = np.array(y_prob_list, dtype=np.float64)

    metrics = compute_all_metrics(y_true, y_pred, y_prob, num_classes, n_bootstrap=n_bootstrap)

    report = format_metrics_report(
        metrics, is_binary=(num_classes == 2),
        labels=y_true, preds=y_pred, n_bootstrap=n_bootstrap,
    )

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\nSamples: {len(y_true)}\n")
        f.write(f"Classes: {num_classes}\n")

    print(report)
    print(f"\n  Metrics saved to: {log_path}")


def save_predictions_csv(
    decisions: Dict[str, AgentDecision],
    output_dir: Path,
):
    """保存 Agent 决策结果为 CSV。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"predictions_{timestamp}.csv"

    fieldnames = ["filename", "selected_model", "predicted_class", "confidence", "reasoning"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fn, decision in sorted(decisions.items()):
            writer.writerow({
                "filename": fn,
                "selected_model": decision.selected_model,
                "predicted_class": decision.selected_class,
                "confidence": f"{decision.confidence:.6f}",
                "reasoning": decision.reasoning[:200],
            })

    print(f"  Predictions saved to: {csv_path}")

    # 同时保存 JSON 格式
    json_path = output_dir / f"results_{timestamp}.json"
    results = []
    for fn, decision in sorted(decisions.items()):
        results.append({
            "image_name": fn,
            "selected_model": decision.selected_model,
            "predicted_class": decision.selected_class,
            "confidence": float(decision.confidence),
            "reasoning": decision.reasoning,
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "num_images": len(results),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"  Results JSON saved to: {json_path}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classification Agent — 多模型分类预测选择/融合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task_dir", type=str, required=True,
        help="run_all.py 输出的任务目录（如 results/binary），下含各模型的 predictions*.csv",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="参与决策的模型列表（默认: 自动扫描 task_dir 下所有含 CSV 的子目录）",
    )
    parser.add_argument(
        "--label_json", type=str, default=None,
        help="标签 JSON 文件（用于计算分类指标）",
    )
    parser.add_argument(
        "--label_field", type=str, default="malignancy",
        help="标签字段名（默认: malignancy）",
    )
    parser.add_argument(
        "--num_classes", type=int, default=2,
        help="类别数（默认: 2，二分类）",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（默认: <task_dir>/cls_agent）",
    )
    parser.add_argument(
        "--config", type=str, default=str(SCRIPT_DIR / "config_cls_agent.yaml"),
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
        help="禁用 LLM Agent，使用 soft voting 融合",
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
    print("  Classification Agent 推理")
    print("=" * 70)
    print(f"  任务目录: {args.task_dir}")
    print(f"  Agent 类型: {agent_type}")
    print(f"  使用 Agent: {use_agent}")
    if agent_type == "local_gpt_oss":
        local_cfg = config.get("agent_local_llm", {})
        print(f"  本地模型: {local_cfg.get('model_path', '(未配置)')}")
    print("=" * 70)

    # 查找模型 CSV 文件
    task_dir = Path(args.task_dir)
    if not task_dir.is_absolute():
        task_dir = PROJECT_ROOT / task_dir

    if args.models:
        model_names = args.models
    else:
        # 自动扫描
        model_names = []
        for d in sorted(task_dir.iterdir()):
            if d.is_dir():
                csvs = list(d.glob("predictions*.csv")) + list(d.glob("results*.csv"))
                if csvs:
                    model_names.append(d.name)

    if not model_names:
        print(f"\n  ✗ 未找到任何模型 CSV 文件。请确认 task_dir 下有 <model>/predictions*.csv 结构。")
        sys.exit(1)

    print(f"\n  参与决策的模型 ({len(model_names)}): {', '.join(model_names)}")

    model_csv_files = find_model_csv_files(task_dir, model_names)
    if not model_csv_files:
        print(f"\n  ✗ 未找到任何有效的 CSV 文件")
        sys.exit(1)

    # 收集多模型输出
    print(f"\n  加载多模型预测...")
    image_outputs, filename_map = collect_model_outputs(model_csv_files)
    print(f"  共 {len(image_outputs)} 张图像有多模型预测\n")

    if not image_outputs:
        print("  ✗ 没有有效的多模型预测")
        sys.exit(1)

    # 加载标签
    labels = {}
    if args.label_json:
        label_path = Path(args.label_json)
        if not label_path.is_absolute():
            label_path = PROJECT_ROOT / label_path
        if label_path.exists():
            labels = load_labels(str(label_path), args.label_field)
            print(f"  标签: {len(labels)} 条")

    # 输出目录
    output_dir = Path(args.output_dir) if args.output_dir else task_dir / "cls_agent"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 Agent（仅在 use_agent 时）
    agent = None
    if use_agent:
        llm_cfg = config.get("agent_llm", {}) or {}
        local_llm_cfg = config.get("agent_local_llm", {}) or {}
        base_datasets_info = config.get("base_datasets_info", {}) or {}

        agent = LLMClassificationAgent(
            api_key=llm_cfg.get("api_key"),
            model_name=llm_cfg.get("model_name", "qwen3.5-flash"),
            temperature=llm_cfg.get("temperature", 0.3),
            max_tokens=llm_cfg.get("max_tokens", 1024),
            base_datasets_info=base_datasets_info,
            max_batch_size=agent_cfg.get("max_batch_size", 10),
            selection_mode=agent_cfg.get("selection_mode", "deterministic"),
            top_k=agent_cfg.get("top_k", 1),
            backend_type=agent_type,
            base_url=llm_cfg.get("base_url"),
            local_model_path=local_llm_cfg.get("model_path"),
            local_device_map=local_llm_cfg.get("device_map", "auto"),
            local_torch_dtype=local_llm_cfg.get("torch_dtype", "bfloat16"),
            local_trust_remote_code=local_llm_cfg.get("trust_remote_code", True),
            local_max_new_tokens=local_llm_cfg.get("max_new_tokens", 1024),
        )

    # 逐图/批量决策
    decisions: Dict[str, AgentDecision] = {}
    total = len(image_outputs)
    t0 = time.time()

    if use_agent and agent is not None:
        # 使用批量决策
        batch_data = []
        for fn, preds in image_outputs.items():
            batch_data.append({
                "image_file": fn,
                "image_name": fn,
                "predictions": preds,
            })

        try:
            print(f"  使用 Agent 批量决策 ({len(batch_data)} 张)...")
            batch_decisions = agent.select_best_model_batch(
                batch_data,
                incremental_save_path=str(output_dir / "incremental_results.json"),
            )
            for i, fn in enumerate(image_outputs.keys()):
                if i < len(batch_decisions):
                    decisions[fn] = batch_decisions[i]
        except Exception as e:
            print(f"  ✗ 批量决策失败: {e}，回退到逐图决策")
            for fn, preds in image_outputs.items():
                try:
                    decision = agent.select_best_model(preds)
                    decisions[fn] = decision
                except Exception as e2:
                    print(f"    ✗ {fn}: {e2}")
                    # 降级选择
                    best = max(preds, key=lambda p: p.top_confidence)
                    decisions[fn] = AgentDecision(
                        selected_model=best.model_name,
                        selected_class=best.top_class,
                        confidence=best.top_confidence,
                        reasoning=f"降级选择: 最高置信度模型 {best.model_name}",
                        all_predictions=[p.to_dict() for p in preds],
                    )
    else:
        # 非 agent 模式: soft voting
        print(f"  使用 soft voting 融合 (top_k={agent_cfg.get('top_k', 1)})...")
        top_k = agent_cfg.get("top_k", 1)

        for fn, preds in image_outputs.items():
            if len(preds) == 0:
                continue

            # 检查是否所有模型 top_class 一致
            first_class = preds[0].top_class
            all_same = all(p.top_class == first_class for p in preds)

            if all_same:
                # 一致: 选最高置信度
                best = max(preds, key=lambda p: p.top_confidence)
                decisions[fn] = AgentDecision(
                    selected_model=best.model_name,
                    selected_class=best.top_class,
                    confidence=best.top_confidence,
                    reasoning=f"所有模型一致预测为 {first_class}，选最高置信度 {best.model_name}",
                    all_predictions=[p.to_dict() for p in preds],
                )
            else:
                # 不一致: soft voting
                sorted_preds = sorted(preds, key=lambda p: p.top_confidence, reverse=True)
                k = min(top_k, len(sorted_preds))
                subset = sorted_preds[:k]
                avg_probs = _average_class_probabilities(subset)
                cls, conf = _winning_class_from_avg_probs(avg_probs)
                decisions[fn] = AgentDecision(
                    selected_model="soft_voting_topk",
                    selected_class=cls,
                    confidence=conf,
                    reasoning=f"Soft voting (top-{k}): {cls} ({conf:.4f})",
                    all_predictions=[p.to_dict() for p in preds],
                )

            print(f"    {fn}: {decisions[fn].selected_class} ({decisions[fn].confidence:.4f})")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  完成: {len(decisions)}/{total} 张图像，耗时 {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # 保存结果
    save_predictions_csv(decisions, output_dir)
    compute_and_save_metrics(decisions, labels, output_dir, args.num_classes, args.n_bootstrap)


if __name__ == "__main__":
    main()
