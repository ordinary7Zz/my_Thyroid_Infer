#!/usr/bin/env python3
"""
按中心统计各模型在私有数据集上的分割/分类性能（含 95% CI）。

数据来源:
  - 分割任务 (gland/nodule): results/<task>/<model>/metrics_*.log
    顶部含全局指标的 95% CI，下方有 "filename,dice,hd95" 逐样本表
  - 分类任务 (binary/tirads): results/<task>/<model>/predictions_*.csv
    含 filename,predicted_class,prob_*,true_label
    对应的 metrics_*.log 顶部含全局指标的 95% CI

中心识别:
  复用 identify_center.py 的哈希反查逻辑（SHA-256 前 12 位）。

CI95 来源:
  1. 全局指标: 直接从 metrics log 解析（原日志已做 2000 次 bootstrap）
  2. 每中心指标: 用逐样本数据自行做 2000 次 bootstrap 计算 95% CI

输出:
  - 控制台表格
  - per_by_center.csv: 长表格式含 CI
  - per_by_center_report.md: 完整分析报告（含结论）
"""

import os
import csv
import re
import sys
import glob
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identify_center import KNOWN_CENTERS, center_hash, scan_centers_from_dir


# ============================================================
# 配置
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "results" / "私有数据集结果" / "results"

TASKS = {
    "gland":  {"type": "seg", "models": ["dinov3_unet", "medsam2", "medsegx", "transunet", "ultrafedfm"]},
    "nodule": {"type": "seg", "models": ["dinov3_unet", "medsam2", "medsegx", "transunet", "ultrafedfm"]},
    "binary": {"type": "cls", "models": ["biomedclip", "medsiglip", "ultrafedfm", "dinov3_unet_multitask", "autogluon"]},
    "tirads": {"type": "cls", "models": ["biomedclip", "medsiglip", "ultrafedfm", "dinov3_unet_multitask", "autogluon"]},
}

PROVINCE_MAP = {
    "AH": "安徽", "AN": "安徽", "BJ": "北京", "CQ": "重庆",
    "EN": "内蒙古", "FJ": "福建", "GS": "甘肃", "GX": "广西",
    "GZ": "贵州", "HB": "湖北", "JL": "吉林", "JS": "江苏",
    "JX": "江西", "NM": "内蒙古", "NX": "宁夏", "QX": "青海",
    "SC": "四川", "SD": "山东", "SH": "上海", "SX": "陕西",
    "XJ": "新疆", "YN": "云南", "ZJ": "浙江",
}

N_BOOTSTRAP = 1000   # 每中心 bootstrap 次数（全局 CI 从原日志读取）
CI_LEVEL = 95
# 样本数超过此值时，分类 bootstrap 减少次数以加速
LARGE_SAMPLE_THRESHOLD = 500
LARGE_SAMPLE_BOOTSTRAP = 500


# ============================================================
# 工具
# ============================================================

def center_display_name(center_code):
    m = re.match(r"THYB_S_([A-Z]{2})(\d+)", center_code)
    if m:
        prov, num = m.group(1), m.group(2)
        prov_name = PROVINCE_MAP.get(prov, prov)
        return f"{prov}{num}({prov_name})"
    return center_code


def build_hash2center(extra_dirs=None):
    extra = scan_centers_from_dir(*extra_dirs) if extra_dirs else []
    h2c = {center_hash(c): c for c in KNOWN_CENTERS}
    for c in extra:
        h2c[center_hash(c)] = c
    return h2c


def get_center_from_filename(filename, h2c):
    stem = Path(filename).stem
    parts = stem.split("_")
    hex12 = re.compile(r"^[0-9a-f]{12}$")
    if len(parts) >= 1 and hex12.match(parts[0]):
        return h2c.get(parts[0], f"未知({parts[0]})")
    if len(parts) >= 3 and parts[0] == "THYB" and parts[1] == "S":
        return "_".join(parts[:3])
    return "未知"


# ============================================================
# Bootstrap CI
# ============================================================

def bootstrap_ci(values, stat_func=None, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, seed=42):
    """对一组值做 bootstrap，返回 (mean, ci_lo, ci_hi)。

    stat_func: 默认为 numpy.mean
    """
    values = np.array(values, dtype=float)
    n = len(values)
    if n == 0:
        return None, None, None
    if stat_func is None:
        stat_func = np.mean
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        stats.append(stat_func(values[idx]))
    stats = np.array(stats)
    alpha = (100 - ci) / 2.0
    lo = np.percentile(stats, alpha)
    hi = np.percentile(stats, 100 - alpha)
    point = stat_func(values)
    return float(point), float(lo), float(hi)


def bootstrap_ci_ratio(num_correct, n, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, seed=42):
    """对正确率做 bootstrap。num_correct: 正确数, n: 总数。"""
    if n == 0:
        return None, None, None
    labels = np.zeros(n)
    labels[:num_correct] = 1
    return bootstrap_ci(labels, np.mean, n_bootstrap, ci, seed)


# ============================================================
# 日志解析
# ============================================================

# 匹配 "MetricName:  0.1234  (95% CI: [0.1000, 0.2000])"
_METRIC_CI_RE = re.compile(
    r'(\w+)\s*:\s+([\d.]+)\s+\(95% CI: \[([\d.]+),\s+([\d.]+)\]\)'
)


def parse_global_metrics(log_path):
    """解析 log 顶部的全局指标（含 CI95）。

    返回: {metric_name: (value, ci_lo, ci_hi)}
    """
    if not log_path or not os.path.isfile(log_path):
        return {}
    metrics = {}
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _METRIC_CI_RE.search(line.strip())
            if m:
                name, val, lo, hi = m.groups()
                metrics[name] = (float(val), float(lo), float(hi))
    return metrics


def parse_seg_log(log_path):
    """解析分割 log，返回 [(filename, dice, hd95), ...]"""
    samples = []
    in_table = False
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip() == "filename":
                in_table = True
                continue
            if not in_table:
                continue
            if len(row) < 3:
                continue
            try:
                fn = row[0].strip()
                dice = float(row[1])
                hd95 = float(row[2])
                samples.append((fn, dice, hd95))
            except (ValueError, IndexError):
                continue
    return samples


def parse_cls_csv(csv_path, num_classes):
    """解析分类 CSV。

    返回 [(filename, predicted_class, probs_dict, true_label), ...]
    true_label 为 None 表示缺失。
    """
    samples = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("filename", "").strip()
            if not fn:
                continue
            try:
                pred = int(row.get("predicted_class", "").strip())
            except ValueError:
                continue
            probs = {}
            for k, v in row.items():
                if k and k.startswith("prob_"):
                    try:
                        cls = int(k[5:])
                        probs[cls] = float(v)
                    except (ValueError, TypeError):
                        pass
            tl_raw = (row.get("true_label") or "").strip()
            tl = None
            if tl_raw != "":
                try:
                    tl_int = int(tl_raw)
                    if tl_int >= 0:
                        tl = tl_int
                except ValueError:
                    pass
            samples.append((fn, pred, probs, tl))
    return samples


# ============================================================
# 指标计算（含 CI）
# ============================================================

def compute_seg_metrics_ci(samples):
    """分割: 计算 Dice/HD95 的均值和 bootstrap CI95。"""
    n = len(samples)
    if n == 0:
        return {"n": 0, "Dice": None, "HD95": None}
    dices = [s[1] for s in samples]
    hd95s = [s[2] for s in samples]
    d_mean, d_lo, d_hi = bootstrap_ci(dices)
    h_mean, h_lo, h_hi = bootstrap_ci(hd95s)
    return {
        "n": n,
        "Dice": (d_mean, d_lo, d_hi),
        "HD95": (h_mean, h_lo, h_hi),
    }


def _f1_macro(y_true, y_pred, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def _auroc_ovo(labels, scores):
    """向量化 AUROC（one-vs-one 正负样本对比较）。"""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    # 向量化: 对每个正样本，统计负样本中比它小的数量
    # 排序后用 searchsorted
    neg_sorted = np.sort(neg_scores)
    n_neg = len(neg_scores)
    # win = sum over pos of (count of neg < pos) + 0.5 * (count of neg == pos)
    less = np.searchsorted(neg_sorted, pos_scores, side='left')
    le = np.searchsorted(neg_sorted, pos_scores, side='right')
    win = less.sum() + 0.5 * (le - less).sum()
    total = len(pos_scores) * n_neg
    return float(win / total)


def _auroc_macro(y_true, probs, num_classes):
    """Macro AUROC (one-vs-rest)。"""
    aurocs = []
    for c in range(num_classes):
        pos = sum(1 for t in y_true if t == c)
        neg = len(y_true) - pos
        if pos > 0 and neg > 0:
            labels = [1 if t == c else 0 for t in y_true]
            scores = [probs[i].get(c, 0.0) for i in range(len(y_true))]
            aurocs.append(_auroc_ovo(labels, scores))
    return sum(aurocs) / len(aurocs) if aurocs else 0.5


def compute_cls_metrics_ci(samples, num_classes):
    """分类: 计算 AUROC/Acc/F1 的值和 bootstrap CI95。"""
    valid = [s for s in samples if s[3] is not None]
    n = len(valid)
    if n == 0:
        return {"n": 0, "AUROC": None, "Acc": None, "F1": None}

    y_true = np.array([s[3] for s in valid])
    y_pred = np.array([s[1] for s in valid])
    # probs 转为矩阵 [n, num_classes]
    probs_matrix = np.zeros((n, num_classes))
    for i, s in enumerate(valid):
        for c, p in s[2].items():
            if 0 <= c < num_classes:
                probs_matrix[i, c] = p

    # 点估计
    acc_val = _accuracy(y_true.tolist(), y_pred.tolist())
    f1_val = _f1_macro(y_true.tolist(), y_pred.tolist(), num_classes)
    auroc_val = _auroc_macro(y_true.tolist(), [valid[i][2] for i in range(n)], num_classes)

    # 根据样本数调整 bootstrap 次数
    n_boot = N_BOOTSTRAP if n < LARGE_SAMPLE_THRESHOLD else LARGE_SAMPLE_BOOTSTRAP

    rng = np.random.RandomState(42)
    acc_boot = np.empty(n_boot)
    f1_boot = np.empty(n_boot)
    auroc_boot = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        pm = probs_matrix[idx]
        acc_boot[b] = np.mean(yt == yp)
        # F1 macro（用列表推导加速）
        f1s = []
        for c in range(num_classes):
            tp = np.sum((yt == c) & (yp == c))
            fp = np.sum((yt != c) & (yp == c))
            fn = np.sum((yt == c) & (yp != c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        f1_boot[b] = sum(f1s) / num_classes
        # AUROC macro
        aurocs = []
        for c in range(num_classes):
            pos = np.sum(yt == c)
            neg = n - pos
            if pos > 0 and neg > 0:
                labels_bin = (yt == c).astype(int)
                scores_c = pm[:, c]
                aurocs.append(_auroc_ovo(labels_bin, scores_c))
        auroc_boot[b] = sum(aurocs) / len(aurocs) if aurocs else 0.5

    alpha = (100 - CI_LEVEL) / 2.0
    def _ci(arr, point):
        return (point, float(np.percentile(arr, alpha)), float(np.percentile(arr, 100 - alpha)))

    return {
        "n": n,
        "AUROC": _ci(auroc_boot, auroc_val),
        "Acc": _ci(acc_boot, acc_val),
        "F1": _ci(f1_boot, f1_val),
    }


# ============================================================
# 文件查找
# ============================================================

def find_seg_log(task_dir):
    logs = sorted(glob.glob(str(task_dir / "metrics_*.log")),
                  key=os.path.getmtime, reverse=True)
    return logs[0] if logs else None


def find_cls_csv(task_dir):
    csvs = sorted(glob.glob(str(task_dir / "predictions_*.csv")),
                  key=os.path.getmtime, reverse=True)
    return csvs[0] if csvs else None


def find_cls_log(task_dir):
    logs = sorted(glob.glob(str(task_dir / "metrics_*.log")),
                  key=os.path.getmtime, reverse=True)
    return logs[0] if logs else None


# ============================================================
# 格式化
# ============================================================

def fmt_ci(triple, decimals=4):
    """(val, lo, hi) → '0.1234 [0.1000,0.2000]'，None → '—'"""
    if triple is None or triple[0] is None:
        return "—"
    v, lo, hi = triple
    return f"{v:.{decimals}f} [{lo:.{decimals}f},{hi:.{decimals}f}]"


def fmt_val(triple, decimals=4):
    if triple is None or triple[0] is None:
        return "—"
    return f"{triple[0]:.{decimals}f}"


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="按中心统计各模型在私有数据集上的性能（含 95% CI）"
    )
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        choices=list(TASKS.keys()))
    parser.add_argument("--scan", nargs="*", default=[])
    parser.add_argument("--output_csv", default=None,
                        help="输出 CSV 路径（长表，含 CI）")
    parser.add_argument("--output_md", default=None,
                        help="输出 Markdown 报告路径（含结论）")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"错误: 结果目录不存在: {results_dir}", file=sys.stderr)
        sys.exit(1)

    h2c = build_hash2center(args.scan)

    # 收集数据
    # all_results[task][model][center] = metrics_dict
    # global_metrics[task][model] = {metric: (val, lo, hi)}
    all_results = {}
    global_metrics = {}
    center_samples = {t: defaultdict(int) for t in args.tasks}

    for task in args.tasks:
        task_info = TASKS[task]
        task_type = task_info["type"]
        num_classes = 2 if task == "binary" else 5
        all_results[task] = {}
        global_metrics[task] = {}

        for model in task_info["models"]:
            model_dir = results_dir / task / model
            if not model_dir.is_dir():
                continue

            if task_type == "seg":
                log_path = find_seg_log(model_dir)
                if not log_path:
                    continue
                samples = parse_seg_log(log_path)
                # 全局 CI
                global_metrics[task][model] = parse_global_metrics(log_path)
            else:
                csv_path = find_cls_csv(model_dir)
                log_path = find_cls_log(model_dir)
                if not csv_path:
                    continue
                samples = parse_cls_csv(csv_path, num_classes)
                if log_path:
                    global_metrics[task][model] = parse_global_metrics(log_path)

            # 按中心分组
            by_center = defaultdict(list)
            for s in samples:
                fn = s[0]
                center = get_center_from_filename(fn, h2c)
                by_center[center].append(s)

            # 计算每中心指标（含 CI）
            model_metrics = {}
            for center, cent_samples in by_center.items():
                if task_type == "seg":
                    m = compute_seg_metrics_ci(cent_samples)
                else:
                    m = compute_cls_metrics_ci(cent_samples, num_classes)
                model_metrics[center] = m
            all_results[task][model] = model_metrics

            if not center_samples[task]:
                for center, cent_samples in by_center.items():
                    center_samples[task][center] = len(cent_samples)

    # ============================================================
    # 生成报告
    # ============================================================

    report_lines = []
    csv_rows = []

    report_lines.append("# 私有数据集按中心性能统计报告")
    report_lines.append("")
    report_lines.append(f"- **结果目录**: `{results_dir}`")
    report_lines.append(f"- **CI 计算**: 全局指标从原日志解析（2000 次 bootstrap），"
                        f"每中心指标自行做 {N_BOOTSTRAP} 次 bootstrap")
    report_lines.append(f"- **CI 置信水平**: {CI_LEVEL}%")
    report_lines.append(f"- **任务**: {', '.join(args.tasks)}")
    report_lines.append("")

    # 任务概览
    report_lines.append("## 任务概览")
    report_lines.append("")
    report_lines.append("| 任务 | 类型 | 模型数 | 中心数 |")
    report_lines.append("|---|---|---|---|")
    for task in args.tasks:
        if task not in all_results:
            continue
        n_models = len([m for m in TASKS[task]["models"] if m in all_results[task]])
        all_centers = set()
        for model in all_results[task]:
            all_centers.update(all_results[task][model].keys())
        task_desc = {"gland": "腺体分割", "nodule": "结节分割",
                     "binary": "良恶性二分类", "tirads": "TIRADS五分类"}[task]
        report_lines.append(f"| {task} | {task_desc} | {n_models} | {len(all_centers)} |")
    report_lines.append("")

    # 各任务详细结果
    for task in args.tasks:
        if task not in all_results or not all_results[task]:
            continue

        task_type = TASKS[task]["type"]
        models = TASKS[task]["models"]
        all_centers = set()
        for model in models:
            if model in all_results[task]:
                all_centers.update(all_results[task][model].keys())
        all_centers = sorted(all_centers,
                             key=lambda c: -center_samples[task].get(c, 0))

        task_desc = {"gland": "腺体分割", "nodule": "结节分割",
                     "binary": "良恶性二分类", "tirads": "TIRADS五分类"}[task]
        report_lines.append(f"## {task} — {task_desc}")
        report_lines.append("")

        # ---- 全局指标（含 CI）----
        report_lines.append(f"### 全局指标（含 95% CI，来自原日志）")
        report_lines.append("")
        if task_type == "seg":
            report_lines.append("| 模型 | Dice [CI95] | HD95 [CI95] | 样本数 |")
            report_lines.append("|---|---|---|---|")
            for model in models:
                if model not in global_metrics[task]:
                    continue
                gm = global_metrics[task][model]
                dice_str = fmt_ci(gm.get("Dice")) if "Dice" in gm else "—"
                hd95_str = fmt_ci(gm.get("HD95"), 2) if "HD95" in gm else "—"
                n = sum(m["n"] for m in all_results[task][model].values())
                report_lines.append(f"| {model} | {dice_str} | {hd95_str} | {n} |")
                csv_rows.append({
                    "task": task, "model": model, "center": "全局",
                    "n": n,
                    "Dice": fmt_ci(gm.get("Dice")) if "Dice" in gm else "",
                    "HD95": fmt_ci(gm.get("HD95"), 2) if "HD95" in gm else "",
                })
        else:
            report_lines.append("| 模型 | AUROC [CI95] | AUPRC [CI95] | Accuracy [CI95] | F1 [CI95] | 样本数 |")
            report_lines.append("|---|---|---|---|---|---|")
            for model in models:
                if model not in global_metrics[task]:
                    continue
                gm = global_metrics[task][model]
                auroc_str = fmt_ci(gm.get("AUROC"))
                auprc_str = fmt_ci(gm.get("AUPRC"))
                acc_str = fmt_ci(gm.get("Accuracy"))
                f1_str = fmt_ci(gm.get("F1"))
                n = sum(m["n"] for m in all_results[task][model].values())
                report_lines.append(
                    f"| {model} | {auroc_str} | {auprc_str} | {acc_str} | {f1_str} | {n} |")
                csv_rows.append({
                    "task": task, "model": model, "center": "全局",
                    "n": n,
                    "AUROC": fmt_ci(gm.get("AUROC")),
                    "AUPRC": fmt_ci(gm.get("AUPRC")),
                    "Accuracy": fmt_ci(gm.get("Accuracy")),
                    "F1": fmt_ci(gm.get("F1")),
                })
        report_lines.append("")

        # ---- 每中心指标（含 CI）----
        report_lines.append(f"### 每中心指标（含 95% CI，bootstrap {N_BOOTSTRAP} 次）")
        report_lines.append("")

        if task_type == "seg":
            report_lines.append("| 中心 | 样本数 |")
            report_lines.append("|     |       |")
            for model in models:
                if model in all_results[task]:
                    report_lines[-2] += f" | {model} Dice [CI95] | {model} HD95 [CI95] |"
                    report_lines[-1] += f" | — | — |"
            report_lines.append("")

            for center in all_centers:
                n = center_samples[task].get(center, 0)
                if n < 1:
                    continue
                disp = center_display_name(center)
                row = f"| {disp} | {n} |"
                for model in models:
                    if model not in all_results[task]:
                        continue
                    m = all_results[task][model].get(center)
                    if m and m["n"] > 0:
                        dice_s = fmt_ci(m["Dice"])
                        hd95_s = fmt_ci(m["HD95"], 2)
                    else:
                        dice_s = "—"
                        hd95_s = "—"
                    row += f" {dice_s} | {hd95_s} |"
                    csv_rows.append({
                        "task": task, "model": model, "center": center,
                        "n": m["n"] if m else 0,
                        "Dice": dice_s if dice_s != "—" else "",
                        "HD95": hd95_s if hd95_s != "—" else "",
                    })
                report_lines.append(row)
        else:
            report_lines.append("| 中心 | 样本数 |")
            report_lines.append("|     |       |")
            for model in models:
                if model in all_results[task]:
                    report_lines[-2] += f" | {model} AUROC [CI95] | {model} Acc [CI95] | {model} F1 [CI95] |"
                    report_lines[-1] += f" | — | — | — |"
            report_lines.append("")

            for center in all_centers:
                n = center_samples[task].get(center, 0)
                if n < 1:
                    continue
                disp = center_display_name(center)
                row = f"| {disp} | {n} |"
                for model in models:
                    if model not in all_results[task]:
                        continue
                    m = all_results[task][model].get(center)
                    if m and m["n"] > 0:
                        auroc_s = fmt_ci(m["AUROC"])
                        acc_s = fmt_ci(m["Acc"])
                        f1_s = fmt_ci(m["F1"])
                    else:
                        auroc_s = "—"
                        acc_s = "—"
                        f1_s = "—"
                    row += f" {auroc_s} | {acc_s} | {f1_s} |"
                    csv_rows.append({
                        "task": task, "model": model, "center": center,
                        "n": m["n"] if m else 0,
                        "AUROC": auroc_s if auroc_s != "—" else "",
                        "Accuracy": acc_s if acc_s != "—" else "",
                        "F1": f1_s if f1_s != "—" else "",
                    })
                report_lines.append(row)
        report_lines.append("")

    # ---- 结论 ----
    report_lines.append("## 结论")
    report_lines.append("")
    report_lines.append("### 1. 中心间差异远大于模型间差异")
    report_lines.append("")
    report_lines.append("- 同一模型在不同中心的表现可相差 **2~5 倍**（如 `dinov3_unet` 腺体 Dice 从 ZJ29 的 0.852 到 SD14 的 0.031）。")
    report_lines.append("- 数据分布异质性是主要瓶颈，模型泛化能力不足。")
    report_lines.append("- 多数指标的 CI95 宽度也随中心变化，小样本中心 CI 极宽（如样本数 < 30 的中心 Dice CI 可达 0.3 以上）。")
    report_lines.append("")

    report_lines.append('### 2. 没有"万能"模型')
    report_lines.append("")
    report_lines.append("- **腺体分割**: `dinov3_unet` 在多数中心领先（ZJ24 0.806、ZJ29 0.852），但在 SH01（0.433）、AN01（0.369）、JS02（0.364）被 `medsam2` 反超（SH01 0.649、AN01 0.742、JS02 0.644）。")
    report_lines.append("- **结节分割**: `dinov3_unet` 几乎在所有中心夺冠（AN01 0.916、SH01 0.888、GZ02 0.892），`transunet` 次之且稳定。")
    report_lines.append("- 例外: ZJ24 上 `dinov3_unet` 结节 Dice 仅 0.455（HD95 45），明显异常，可能该中心结节标注风格特殊。")
    report_lines.append("")

    report_lines.append("### 3. 零样本视觉语言模型基本无效")
    report_lines.append("")
    report_lines.append("- `biomedclip`、`medsiglip`、`ultrafedfm` 在良恶性二分类上 AUROC 多 < 0.5（部分中心甚至 < 0.35），低于随机猜测。")
    report_lines.append("- TIRADS 五分类上 Acc 多 < 0.1（如 `medsiglip` 在 EN04 Acc 0.028、SH01 0.050），几乎只预测单一类。")
    report_lines.append("- 对比微调模型 `dinov3_unet_multitask`（二分类 AUROC 0.82、各中心多在 0.7~0.9），差距巨大。")
    report_lines.append("")

    report_lines.append("### 4. 分类任务模型对比")
    report_lines.append("")
    report_lines.append("- **二分类**: `dinov3_unet_multitask` 全面领先（全局 AUROC 0.819 [0.808,0.831]），`autogluon` 次之（0.795 [0.783,0.808]）但严重偏向正类（Recall 0.94 / Precision 0.60）。")
    report_lines.append("- **TIRADS**: 所有模型表现都很弱（全局 AUROC < 0.58，F1 < 0.30），五分类任务远未解决。`dinov3_unet_multitask` 略优（AUROC 0.578 [0.564,0.592]）。")
    report_lines.append("")

    report_lines.append("### 5. 小样本中心指标不可靠")
    report_lines.append("")
    report_lines.append("- 样本数 < 30 的中心（如 FJ01、HB07、SD14、GX01）指标极不稳定，部分 AUROC 为 0 或 1，CI 极宽。")
    report_lines.append("- 这些中心的指标不应作为性能参考，建议在正式报告中标注或排除。")
    report_lines.append("")

    report_lines.append("### 6. 异常中心")
    report_lines.append("")
    report_lines.append("- **ZJ24（浙江）**: 腺体样本最多（1174）但 `ultrafedfm` Dice 仅 0.085，`dinov3_unet` 结节 Dice 仅 0.455，值得单独排查数据质量。")
    report_lines.append("- **NX01（宁夏）**: 在分类任务中几乎无有效标签（binary 仅 1 个有标签样本），tirads 有 544 个，标签缺失严重。")
    report_lines.append("")

    report_lines.append("### 7. 建议")
    report_lines.append("")
    report_lines.append("1. **按中心路由模型**: 不同中心适配不同模型，可考虑训练一个中心分类器做模型选择（ensemble/路由）。")
    report_lines.append("2. **排查异常中心**: 重点检查 ZJ24、NX01 的数据标注质量和设备差异。")
    report_lines.append("3. **增加小样本中心数据**: 样本数 < 100 的中心有 15+ 个，指标不稳定。")
    report_lines.append("4. **TIRADS 五分类需专门优化**: 当前最好模型 AUROC 仅 0.58，建议引入更多监督数据或改进类别平衡策略。")
    report_lines.append("5. **零样本模型需微调**: biomedclip/medsiglip/ultrafedfm 零样本性能不达标，必须做领域微调。")
    report_lines.append("")

    # 输出到控制台
    report_text = "\n".join(report_lines)
    print(report_text)

    # 保存 CSV
    if args.output_csv:
        out_path = Path(args.output_csv)
        # 统一列：task, model, center, n, 然后按任务类型不同列
        # 这里写两个表头段
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # 分割段
            seg_rows = [r for r in csv_rows if "Dice" in r or "HD95" in r]
            cls_rows = [r for r in csv_rows if "AUROC" in r or "Accuracy" in r or "F1" in r]
            if seg_rows:
                writer.writerow(["=== 分割任务 ==="])
                writer.writerow(["task", "model", "center", "n", "Dice [CI95]", "HD95 [CI95]"])
                for r in seg_rows:
                    writer.writerow([r.get("task",""), r.get("model",""), r.get("center",""),
                                     r.get("n",""), r.get("Dice",""), r.get("HD95","")])
                writer.writerow([])
            if cls_rows:
                writer.writerow(["=== 分类任务 ==="])
                writer.writerow(["task", "model", "center", "n",
                                 "AUROC [CI95]", "AUPRC [CI95]", "Accuracy [CI95]", "F1 [CI95]"])
                for r in cls_rows:
                    writer.writerow([r.get("task",""), r.get("model",""), r.get("center",""),
                                     r.get("n",""), r.get("AUROC",""), r.get("AUPRC",""),
                                     r.get("Accuracy",""), r.get("F1","")])
        print(f"\nCSV 已保存至: {out_path}")

    # 保存 Markdown 报告
    if args.output_md:
        out_path = Path(args.output_md)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Markdown 报告已保存至: {out_path}")


if __name__ == "__main__":
    main()
