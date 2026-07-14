#!/usr/bin/env python3
"""补算各中心 AUPRC 的 bootstrap 95% CI，更新 per_by_center_ci.csv。

原因: analyze_center_performance.py 的 compute_cls_metrics_ci() 只计算了
AUROC/Acc/F1，未计算 AUPRC。全局 AUPRC 从原日志解析已有，但每中心 AUPRC 为空。

本脚本:
  1. 读取 binary/tirads 的 predictions CSV
  2. 按中心分组
  3. 对每中心做 1000 次 bootstrap 计算 AUPRC 95% CI
  4. 更新 per_by_center_ci.csv 中空的 AUPRC 字段

用法:
  cd /Users/wangbd/sysu/my_Thyroid_infer
  python3 compute_auprc_per_center.py
"""

import sys
import csv
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np

# 复用现有模块
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from identify_center import KNOWN_CENTERS, center_hash, scan_centers_from_dir
from analyze_center_performance import (
    parse_cls_csv,
    get_center_from_filename,
    build_hash2center,
    find_cls_csv,
    TASKS,
    DEFAULT_RESULTS,
    N_BOOTSTRAP,
    CI_LEVEL,
    LARGE_SAMPLE_THRESHOLD,
    LARGE_SAMPLE_BOOTSTRAP,
    fmt_ci,
)

# 输出 CSV 路径（与 analyze_center_performance.py 一致）
OUTPUT_CSV = Path("/Users/wangbd/sysu/my_papers/ThyroidAgent/nature/data/per_by_center_ci.csv")


# ============================================================
# AUPRC 计算
# ============================================================

def _auprc_binary(y_true, scores):
    """二分类 AUPRC（Average Precision）。

    算法: 按 score 降序排列，逐步计算 precision-recall，用 recall 增量乘以 precision 求和。

    Args:
        y_true: 0/1 标签数组
        scores: 正类概率数组

    Returns:
        AUPRC 值 (float)
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    pos = np.sum(y_true == 1)
    if pos == 0:
        return 0.0
    neg = len(y_true) - pos
    if neg == 0:
        return 1.0

    # 按 score 降序排列
    order = np.argsort(-scores)
    y_sorted = y_true[order]

    # 累积 TP 和 FP
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)

    # 每个位置的 precision 和 recall
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / pos

    # Average Precision = sum of (recall[i] - recall[i-1]) * precision[i]
    # recall[0] - 0 = recall[0]
    recall_diff = np.diff(recall, prepend=0.0)
    auprc = np.sum(recall_diff * precision)
    return float(auprc)


def _auprc_macro(y_true, probs, num_classes):
    """多分类 Macro AUPRC（One-vs-Rest 平均）。

    Args:
        y_true: 真实标签数组
        probs: 概率字典列表 [{cls: prob, ...}, ...] 或概率矩阵
        num_classes: 类别数

    Returns:
        Macro AUPRC (float)
    """
    y_true = np.asarray(y_true)

    # 将 probs 转为矩阵
    n = len(y_true)
    if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], dict):
        probs_matrix = np.zeros((n, num_classes))
        for i, p in enumerate(probs):
            for c, v in p.items():
                if 0 <= c < num_classes:
                    probs_matrix[i, c] = v
    else:
        probs_matrix = np.asarray(probs)

    auprcs = []
    for c in range(num_classes):
        pos = np.sum(y_true == c)
        if pos == 0:
            continue  # 跳过无正样本的类
        labels_bin = (y_true == c).astype(int)
        scores_c = probs_matrix[:, c]
        auprcs.append(_auprc_binary(labels_bin, scores_c))

    return float(np.mean(auprcs)) if auprcs else 0.0


def compute_auprc_ci(samples, num_classes):
    """计算 AUPRC 的点估计和 bootstrap 95% CI。

    Args:
        samples: [(filename, predicted_class, probs_dict, true_label), ...]
        num_classes: 类别数 (binary=2, tirads=5)

    Returns:
        (point_estimate, ci_lo, ci_hi) 或 None
    """
    valid = [s for s in samples if s[3] is not None]
    n = len(valid)
    if n == 0:
        return None

    y_true = np.array([s[3] for s in valid])

    # 构建概率矩阵
    probs_matrix = np.zeros((n, num_classes))
    for i, s in enumerate(valid):
        for c, p in s[2].items():
            if 0 <= c < num_classes:
                probs_matrix[i, c] = p

    # 点估计
    if num_classes == 2:
        # 二分类: 用 class 1 作为正类
        point = _auprc_binary(y_true, probs_matrix[:, 1])
    else:
        # 多分类: macro AUPRC
        point = _auprc_macro(y_true, probs_matrix, num_classes)

    # Bootstrap
    n_boot = N_BOOTSTRAP if n < LARGE_SAMPLE_THRESHOLD else LARGE_SAMPLE_BOOTSTRAP
    rng = np.random.RandomState(42)
    boot_vals = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        pm = probs_matrix[idx]

        if num_classes == 2:
            boot_vals[b] = _auprc_binary(yt, pm[:, 1])
        else:
            auprcs = []
            for c in range(num_classes):
                pos = np.sum(yt == c)
                if pos == 0:
                    continue
                labels_bin = (yt == c).astype(int)
                scores_c = pm[:, c]
                auprcs.append(_auprc_binary(labels_bin, scores_c))
            boot_vals[b] = np.mean(auprcs) if auprcs else 0.0

    alpha = (100 - CI_LEVEL) / 2.0
    lo = float(np.percentile(boot_vals, alpha))
    hi = float(np.percentile(boot_vals, 100 - alpha))
    return (float(point), lo, hi)


# ============================================================
# 主流程
# ============================================================

def main():
    results_dir = DEFAULT_RESULTS
    if not results_dir.is_dir():
        print(f"错误: 结果目录不存在: {results_dir}", file=sys.stderr)
        sys.exit(1)

    h2c = build_hash2center([])

    # 需要计算 AUPRC 的任务
    cls_tasks = ["binary", "tirads"]

    # auprc_data[task][model][center] = (point, lo, hi) or None
    auprc_data = {}

    for task in cls_tasks:
        task_info = TASKS[task]
        num_classes = 2 if task == "binary" else 5
        models = task_info["models"]
        auprc_data[task] = {}

        for model in models:
            model_dir = results_dir / task / model
            if not model_dir.is_dir():
                print(f"  跳过 {task}/{model}: 目录不存在")
                continue

            csv_path = find_cls_csv(model_dir)
            if not csv_path:
                print(f"  跳过 {task}/{model}: 无 predictions CSV")
                continue

            print(f"处理 {task}/{model} ...", end=" ", flush=True)
            samples = parse_cls_csv(csv_path, num_classes)

            # 按中心分组
            by_center = defaultdict(list)
            for s in samples:
                center = get_center_from_filename(s[0], h2c)
                by_center[center].append(s)

            # 计算每中心 AUPRC
            auprc_data[task][model] = {}
            n_centers = 0
            for center, cent_samples in by_center.items():
                ci = compute_auprc_ci(cent_samples, num_classes)
                if ci is not None:
                    auprc_data[task][model][center] = ci
                    n_centers += 1
            print(f"{n_centers} 个中心")

    # ============================================================
    # 更新 CSV
    # ============================================================
    print(f"\n更新 CSV: {OUTPUT_CSV}")

    # 读取现有 CSV
    lines = OUTPUT_CSV.read_text(encoding="utf-8").splitlines()

    # 解析为行列表，保留分段结构
    updated_lines = []
    in_cls_section = False
    cls_header_written = False
    updated_count = 0

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("=== 分割任务"):
            in_cls_section = False
            updated_lines.append(line)
            continue

        if stripped.startswith("=== 分类任务"):
            in_cls_section = True
            cls_header_written = False
            updated_lines.append(line)
            continue

        if in_cls_section and not cls_header_written:
            # 这是表头行: task,model,center,n,AUROC [CI95],AUPRC [CI95],...
            cls_header_written = True
            updated_lines.append(line)
            continue

        if in_cls_section and stripped and not stripped.startswith("==="):
            # 数据行
            parts = list(csv.reader([line]))[0]
            if len(parts) >= 7:
                task, model, center = parts[0], parts[1], parts[2]
                n = parts[3]
                auroc = parts[4]
                # parts[5] = AUPRC (可能为空)
                accuracy = parts[6] if len(parts) > 6 else ""
                f1 = parts[7] if len(parts) > 7 else ""

                # 填充 AUPRC（全局行保留原值，仅填充每中心行）
                if center == "全局":
                    # 全局 AUPRC 已从日志解析，保留原值
                    auprc_str = parts[5] if len(parts) > 5 else ""
                elif task in auprc_data and model in auprc_data[task]:
                    ci = auprc_data[task][model].get(center)
                    if ci is not None:
                        auprc_str = fmt_ci(ci)
                        updated_count += 1
                    else:
                        auprc_str = ""
                else:
                    auprc_str = parts[5] if len(parts) > 5 else ""

                # 重新组装行
                new_row = [task, model, center, n, auroc, auprc_str, accuracy, f1]
                updated_lines.append(",".join([
                    f'"{f}"' if "[" in str(f) else str(f) for f in new_row
                ]) if any("[" in str(f) for f in new_row) else ",".join(new_row))
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    # 写回
    OUTPUT_CSV.write_text("\n".join(updated_lines), encoding="utf-8")
    print(f"已更新 {updated_count} 个 AUPRC 值")

    # 打印摘要
    print("\n=== AUPRC 摘要 ===")
    for task in cls_tasks:
        if task not in auprc_data:
            continue
        print(f"\n{task}:")
        for model in TASKS[task]["models"]:
            if model not in auprc_data[task]:
                continue
            centers = auprc_data[task][model]
            if not centers:
                continue
            # 打印前5个中心
            sorted_centers = sorted(centers.items(), key=lambda x: -len(x[0]))
            print(f"  {model}: {len(centers)} centers")
            for center, (val, lo, hi) in sorted_centers[:5]:
                print(f"    {center}: {val:.4f} [{lo:.4f}, {hi:.4f}]")
            if len(centers) > 5:
                print(f"    ... ({len(centers) - 5} more)")


if __name__ == "__main__":
    main()
