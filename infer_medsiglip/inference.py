#!/usr/bin/env python3
"""
MedSigLIP 分类推理脚本（独立可运行版本）

本目录为最小可运行版本，无需依赖外部代码文件，只需安装 requirements.txt 中的依赖。

用法:
    # 基本推理（仅输出 CSV）
    python inference.py \\
        --checkpoint /path/to/best_model.pt \\
        --model_path /path/to/medsiglip-448 \\
        --input /path/to/images/ \\
        --output predictions.csv

    # 带标签文件的推理（输出 CSV + 性能指标 .log）
    python inference.py \\
        --checkpoint /path/to/best_model.pt \\
        --model_path /path/to/medsiglip-448 \\
        --input /path/to/images/ \\
        --output predictions.csv \\
        --label_file /path/to/labels.json \\
        --label_field malignancy

    # 指定设备和批量大小
    python inference.py \\
        --checkpoint /path/to/best_model.pt \\
        --model_path /path/to/medsiglip-448 \\
        --input /path/to/images/ \\
        --output predictions.csv \\
        --device cuda:0 \\
        --batch_size 32
"""

import os
import sys
import json
import argparse

# 确保可从任意目录运行：将脚本所在目录加入 sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import MedSigLIPClassifier
from transforms import get_val_transforms
from metrics import compute_all_metrics, format_metrics_report


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="MedSigLIP classification inference (standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to image file or directory of images")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path")
    parser.add_argument("--label_file", type=str, default=None,
                        help="Path to label JSON file (optional). "
                             "Format: [{\"filename\": \"a.jpg\", \"<field>\": 0}, ...]")
    parser.add_argument("--label_field", type=str, default=None,
                        help="Field name in label JSON for current task "
                             "(required when --label_file is provided)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained MedSigLIP weights directory "
                             "(e.g., /path/to/medsiglip-448)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, cuda:0, cpu (default: auto)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--n_bootstrap", type=int, default=2000,
                        help="Number of bootstrap iterations for CI95")
    parser.add_argument("--metrics_output", type=str, default=None,
                        help="Output path for metrics .log file "
                             "(default: auto-derive from --output)")
    parser.add_argument("--label_offset", type=int, default=None,
                        help="Subtract this value from labels to convert to 0-indexed. "
                             "E.g., --label_offset 1 for 1-indexed labels (1~5 -> 0~4). "
                             "Default: auto-detect (auto-shift when min label = 1 and "
                             "max label = num_classes)")
    return parser.parse_args()


def load_labels(label_file, label_field):
    """从 JSON 文件加载标签。

    Args:
        label_file: JSON 文件路径
        label_field: 标签字段名

    Returns:
        dict: {filename: int_label}
    """
    with open(label_file, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Label file must be a JSON array, got {type(data)}")

    label_map = {}
    for item in data:
        filename = item["filename"]
        if label_field not in item:
            raise KeyError(
                f"Field '{label_field}' not found in entry for '{filename}'. "
                f"Available fields: {list(item.keys())}"
            )
        label_map[filename] = int(item[label_field])

    return label_map


def collect_images(input_path):
    """收集输入路径下的所有图像。"""
    from pathlib import Path

    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        images = sorted([
            p for p in input_path.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])
        return images
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


def softmax(logits):
    """数值稳定的 softmax。"""
    x = np.asarray(logits, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    x_max = x.max(axis=-1, keepdims=True)
    e_x = np.exp(np.clip(x - x_max, -50, 50))
    probs = e_x / e_x.sum(axis=-1, keepdims=True)
    probs = np.clip(probs, 1e-15, 1.0)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs


def process_batch(model, device, tensors, paths, num_classes,
                  class_names, label_map):
    """处理一批预处理后的图像张量。

    Returns:
        list of dict, 每个元素包含:
            - row: CSV 行 (filename, predicted_class, confidence, [true_label])
            - probs: (C,) 概率数组
            - true_idx: int 或 None
    """
    pixel_values = torch.stack(tensors).to(device)

    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs["logits"].cpu().numpy()  # (B, num_classes)

    # 计算概率
    if num_classes <= 1:
        # 单 logit -> sigmoid，扩展为 2 类概率
        probs_pos = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        probs_all = np.column_stack([1.0 - probs_pos, probs_pos])
    else:
        probs_all = softmax(logits)

    results = []
    for i, img_path in enumerate(paths):
        pred_idx = int(probs_all[i].argmax())
        confidence = float(probs_all[i][pred_idx])
        pred_name = (
            class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        )
        filename = os.path.basename(str(img_path))

        row = {
            "filename": filename,
            "predicted_class": pred_name,
            "confidence": round(confidence, 6),
        }

        true_idx = None
        if label_map is not None:
            true_idx = label_map.get(filename, None)
            if true_idx is not None:
                row["true_label"] = (
                    class_names[true_idx]
                    if true_idx < len(class_names)
                    else str(true_idx)
                )
            else:
                row["true_label"] = ""

        results.append({
            "row": row,
            "probs": probs_all[i],
            "true_idx": true_idx,
        })

    return results


def main():
    args = parse_args()

    # ---- 设备 ----
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ---- 加载检查点 ----
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    class_names = checkpoint.get("class_names", None)
    model_cfg = config.get("model", {})
    num_classes = model_cfg.get("num_classes", 2)

    # 处理 class_names
    if num_classes <= 1:
        # 单 logit 二分类：有效类别数为 2
        if class_names is None or len(class_names) < 2:
            class_names = ["0", "1"]
    else:
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]

    is_binary = num_classes <= 2

    # ---- 构建模型 ----
    model_name = args.model_path
    local_files_only = model_cfg.get("local_files_only", True)

    model = MedSigLIPClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=model_cfg.get("classifier_dropout", 0.1),
        local_files_only=local_files_only,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ---- 预处理 ----
    data_cfg = config.get("data", {})
    image_size = data_cfg.get("image_size", 448)
    mean = data_cfg.get("mean", [0.5, 0.5, 0.5])
    std = data_cfg.get("std", [0.5, 0.5, 0.5])
    transform = get_val_transforms(image_size, mean, std)

    # ---- 加载标签（可选）----
    label_map = None
    if args.label_file:
        if not args.label_field:
            print("[ERROR] --label_field is required when --label_file is provided")
            sys.exit(1)
        label_map = load_labels(args.label_file, args.label_field)

    # ---- 收集图像 ----
    image_paths = collect_images(args.input)

    # ---- 批量推理 ----
    all_rows = []
    all_probs = []
    all_true_idx = []

    batch_tensors = []
    batch_paths = []

    # 打印配置
    print("=" * 60)
    print(f"权重:     {args.checkpoint}")
    print(f"数据:     {args.input}")
    print(f"类别数:   {num_classes}")
    if args.label_file:
        print(f"标签字段: {args.label_field}")
    print(f"设备:     {device}")
    print("=" * 60)

    for img_path in tqdm(image_paths, desc="推理"):
        # 读取图像（超声通常为灰度图）
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  [WARN] Failed to load: {img_path}")
            continue

        # 灰度图 -> 三通道 RGB
        image = np.stack([image] * 3, axis=-1)
        transformed = transform(image=image)
        batch_tensors.append(transformed["image"])
        batch_paths.append(img_path)

        if len(batch_tensors) >= args.batch_size:
            results = process_batch(
                model, device, batch_tensors, batch_paths,
                num_classes, class_names, label_map,
            )
            for r in results:
                all_rows.append(r["row"])
                all_probs.append(r["probs"])
                all_true_idx.append(r["true_idx"])
            batch_tensors = []
            batch_paths = []

    # 处理剩余
    if batch_tensors:
        results = process_batch(
            model, device, batch_tensors, batch_paths,
            num_classes, class_names, label_map,
        )
        for r in results:
            all_rows.append(r["row"])
            all_probs.append(r["probs"])
            all_true_idx.append(r["true_idx"])

    # ---- 保存 CSV ----
    df = pd.DataFrame(all_rows)
    cols = ["filename", "predicted_class", "confidence"]
    if label_map is not None:
        cols.append("true_label")
    df = df[[c for c in cols if c in df.columns]]

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output, index=False)

    # ---- 计算性能指标（如果有标签）----
    if label_map is not None:
        mask = [t is not None for t in all_true_idx]
        n_labeled = sum(mask)

        if n_labeled == 0:
            print("[WARN] No images matched labels in the label file. "
                  "Skipping metrics computation.")
            return

        # 提取有标签的样本
        probs = np.array([all_probs[i] for i in range(len(all_probs)) if mask[i]])
        labels = np.array([all_true_idx[i] for i in range(len(all_true_idx)) if mask[i]])
        preds = probs.argmax(axis=1)

        # 验证并校正标签范围
        n_effective_classes = probs.shape[1]
        max_label = int(labels.max())
        min_label = int(labels.min())

        # 确定偏移量
        if args.label_offset is not None:
            offset = args.label_offset
        elif min_label == 1 and max_label == n_effective_classes:
            # 自动检测：标签 1~N，模型 0~N-1，自动减 1
            offset = 1
        else:
            offset = 0

        if offset != 0:
            labels = labels - offset
            # 同步更新 CSV 中的 true_label
            for i in range(len(all_true_idx)):
                if all_true_idx[i] is not None:
                    all_true_idx[i] = all_true_idx[i] - offset
            # 重新构建 DataFrame 以反映偏移
            for r_idx, i in enumerate([j for j in range(len(all_true_idx)) if mask[j]]):
                all_rows[r_idx]["true_label"] = (
                    class_names[all_true_idx[i]]
                    if all_true_idx[i] is not None and 0 <= all_true_idx[i] < len(class_names)
                    else str(all_true_idx[i])
                )
            df = pd.DataFrame(all_rows)
            cols = ["filename", "predicted_class", "confidence"]
            if label_map is not None:
                cols.append("true_label")
            df = df[[c for c in cols if c in df.columns]]
            df.to_csv(args.output, index=False)

        # 再次检查范围
        max_label_new = int(labels.max())
        min_label_new = int(labels.min())
        if max_label_new >= n_effective_classes or min_label_new < 0:
            print(f"[ERROR] Label values out of range [0, {n_effective_classes - 1}]. "
                  f"After offset={offset}: min={min_label_new}, max={max_label_new}. "
                  f"Use --label_offset to adjust.")
            return

        metrics = compute_all_metrics(
            labels, preds, probs, is_binary,
            n_bootstrap=args.n_bootstrap,
        )

        report = format_metrics_report(
            metrics, is_binary, class_names, labels, preds,
            n_bootstrap=args.n_bootstrap,
            label_field=args.label_field or "",
        )
        print(report)

        # 保存到 .log 文件
        metrics_path = args.metrics_output
        if metrics_path is None:
            base = os.path.splitext(args.output)[0]
            metrics_path = base + "_metrics.log"

        metrics_dir = os.path.dirname(metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
