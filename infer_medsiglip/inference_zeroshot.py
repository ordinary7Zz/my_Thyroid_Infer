#!/usr/bin/env python3
"""
MedSigLIP 零样本分类推理脚本
==============================
不使用微调权重，直接用预训练 MedSigLIP 做零样本分类。
通过文本 prompt + 图像相似度进行分类，无需 finetune 分类头。

用法:
    # 二分类（良恶性）
    python inference_zeroshot.py \\
        --model_path /path/to/medsiglip-448 \\
        --input /path/to/images/ \\
        --output results.csv

    # TIRADS 五分类
    python inference_zeroshot.py \\
        --model_path /path/to/medsiglip-448 \\
        --input /path/to/images/ \\
        --num_classes 5 \\
        --class_names 1 2 3 4 5 \\
        --output results.csv

    # 自定义文本 prompt
    python inference_zeroshot.py \\
        --model_path /path/to/medsiglip-448 \\
        --input /path/to/images/ \\
        --prompts "prompt for class 0" "prompt for class 1" \\
        --output results.csv
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

# 确保可从任意目录运行
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# 使用项目级统一分类指标模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from cls_metrics import compute_all_metrics, format_metrics_report
from transforms import get_val_transforms


# ============================================================================
# 默认文本 prompt
# ============================================================================

DEFAULT_PROMPTS_BINARY = [
    "an ultrasound image of a benign thyroid nodule",
    "an ultrasound image of a malignant thyroid nodule",
]

DEFAULT_PROMPTS_TIRADS = [
    "an ultrasound image of a thyroid nodule, TI-RADS category 1",
    "an ultrasound image of a thyroid nodule, TI-RADS category 2",
    "an ultrasound image of a thyroid nodule, TI-RADS category 3",
    "an ultrasound image of a thyroid nodule, TI-RADS category 4",
    "an ultrasound image of a thyroid nodule, TI-RADS category 5",
]


def get_default_prompts(num_classes):
    if num_classes == 2:
        return list(DEFAULT_PROMPTS_BINARY)
    elif num_classes == 5:
        return list(DEFAULT_PROMPTS_TIRADS)
    else:
        raise ValueError(
            f"无默认 prompt for num_classes={num_classes}，请通过 --prompts 提供"
        )


# ============================================================================
# 模型加载
# ============================================================================

def load_medsiglip_model(model_path: str, device: torch.device):
    """加载完整 MedSigLIP 模型（视觉 + 文本编码器）和 tokenizer。

    Args:
        model_path: 预训练 MedSigLIP 权重目录
        device: 推理设备

    Returns:
        (model, tokenizer, image_size, max_seq_length)
    """
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    print(f"  预训练模型: {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 从 config 获取参数
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    image_size = config.vision_config.image_size  # 448
    max_seq_length = config.text_config.max_position_embeddings  # 64

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量:     {total_params:,}")
    print(f"  image_size:     {image_size}")
    print(f"  max_seq_length: {max_seq_length}")

    return model, tokenizer, image_size, max_seq_length


# ============================================================================
# 文本编码
# ============================================================================

@torch.no_grad()
def encode_text_prompts(model, tokenizer, prompts, max_seq_length, device):
    """编码文本 prompt，返回归一化的文本特征。

    Returns:
        text_features: (C, embed_dim) 归一化后的文本特征
    """
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    text_features = model.get_text_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    text_features = F.normalize(text_features, dim=-1)
    return text_features  # (C, embed_dim)


# ============================================================================
# 图像预处理与批量推理
# ============================================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(input_path):
    """收集输入路径下的所有图像。"""
    from pathlib import Path
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return sorted([
        p for p in input_path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])


@torch.no_grad()
def batch_infer(model, text_features, image_paths, transform, device, batch_size=32):
    """批量零样本推理。

    Returns:
        all_probs: (N, C) 概率矩阵
    """
    logit_scale = model.logit_scale.exp()
    all_probs = []

    batch_tensors = []
    batch_paths = []

    for img_path in tqdm(image_paths, desc="推理"):
        # 读取图像（超声通常为灰度图）
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"\n  [WARN] 读取失败: {img_path}，使用零张量替代")
            image = np.zeros((448, 448), dtype=np.uint8)

        # 灰度图 -> 三通道 RGB
        image = np.stack([image] * 3, axis=-1)
        transformed = transform(image=image)
        batch_tensors.append(transformed["image"])
        batch_paths.append(img_path)

        if len(batch_tensors) >= batch_size:
            probs = _process_batch(
                model, text_features, logit_scale,
                batch_tensors, device,
            )
            all_probs.append(probs)
            batch_tensors = []
            batch_paths = []

    # 处理剩余
    if batch_tensors:
        probs = _process_batch(
            model, text_features, logit_scale,
            batch_tensors, device,
        )
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)  # (N, C)


def _process_batch(model, text_features, logit_scale, tensors, device):
    """处理一个 batch，返回概率矩阵 (B, C)。"""
    pixel_values = torch.stack(tensors).to(device)
    image_features = model.get_image_features(pixel_values=pixel_values)
    image_features = F.normalize(image_features, dim=-1)

    logits = logit_scale * image_features @ text_features.T  # (B, C)
    probs = logits.softmax(dim=-1).cpu().numpy()
    return probs


# ============================================================================
# 标签加载（与 finetune 版逻辑一致）
# ============================================================================

def load_labels(label_file, label_field):
    """从 JSON 文件加载标签。"""
    with open(label_file, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Label file must be a JSON array, got {type(data)}")

    label_map = {}
    for item in data:
        filename = item["filename"]
        if label_field not in item:
            continue
        label_val = int(item[label_field])
        if label_val < 0:
            continue
        label_map[filename] = label_val

    return label_map


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MedSigLIP 零样本分类推理（不使用 finetune 权重）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="预训练 MedSigLIP 权重目录路径")
    parser.add_argument("--input", type=str, required=True,
                        help="输入图像文件或目录")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 CSV 路径")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="类别数 (默认 2)")
    parser.add_argument("--class_names", type=str, nargs="+", default=None,
                        help="类别名称列表（默认 0 1 或 1 2 3 4 5）")
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                        help="自定义文本 prompt（数量需与 num_classes 一致）")

    parser.add_argument("--device", type=str, default=None,
                        help="设备: cuda, cuda:0, cpu (默认自动)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批量大小 (默认 32)")

    parser.add_argument("--label_file", type=str, default=None,
                        help="标签 JSON 文件路径 (可选)")
    parser.add_argument("--label_field", type=str, default=None,
                        help="JSON 中的标签字段名")
    parser.add_argument("--metrics_output", type=str, default=None,
                        help="指标 .log 输出路径")
    parser.add_argument("--n_bootstrap", type=int, default=2000,
                        help="Bootstrap 迭代次数 (默认 2000)")
    parser.add_argument("--label_offset", type=int, default=None,
                        help="标签偏移量（1-indexed 自动减 1）")
    args = parser.parse_args()

    # 确定 class_names
    if args.class_names is None:
        if args.num_classes == 2:
            args.class_names = ["0", "1"]
        else:
            args.class_names = [str(i) for i in range(1, args.num_classes + 1)]

    if len(args.class_names) != args.num_classes:
        print(f"错误: class_names 长度 ({len(args.class_names)}) "
              f"与 num_classes ({args.num_classes}) 不一致")
        sys.exit(1)

    if args.label_file is not None and args.label_field is None:
        print("错误: 提供了 --label_file 时必须指定 --label_field")
        sys.exit(1)

    # 确定 prompt
    prompts = args.prompts if args.prompts else get_default_prompts(args.num_classes)
    if len(prompts) != args.num_classes:
        print(f"错误: prompts 数量 ({len(prompts)}) 与 num_classes "
              f"({args.num_classes}) 不一致")
        sys.exit(1)

    # 设备
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 加载模型
    print("=" * 60)
    print("MedSigLIP 零样本分类推理")
    print("=" * 60)
    model, tokenizer, image_size, max_seq_length = \
        load_medsiglip_model(args.model_path, device)

    # 编码文本 prompt
    print("\n  文本 prompt:")
    for i, p in enumerate(prompts):
        print(f"    [{i}] {p}")
    text_features = encode_text_prompts(
        model, tokenizer, prompts, max_seq_length, device
    )
    print(f"  文本特征形状: {text_features.shape}")

    # 预处理（SigLIP 标准预处理：mean=0.5, std=0.5）
    transform = get_val_transforms(image_size, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # 收集图像
    image_paths = collect_images(args.input)

    print(f"\n  数据:     {args.input}")
    print(f"  图片数:   {len(image_paths)}")
    print(f"  类别:     {args.class_names}")
    if args.label_file:
        print(f"  标签字段: {args.label_field}")
    print(f"  设备:     {device}")
    print("=" * 60)

    # 推理
    all_probs = batch_infer(
        model, text_features, image_paths, transform, device, args.batch_size,
    )

    # 加载标签（可选，用于 CSV true_label 列和评估）
    label_map = None
    if args.label_file:
        label_map = load_labels(args.label_file, args.label_field)

    # 保存 CSV
    import pandas as pd

    filenames = [os.path.basename(str(p)) for p in image_paths]
    rows = []
    for fname, probs in zip(filenames, all_probs):
        pred_idx = int(np.argmax(probs))
        pred_name = (
            args.class_names[pred_idx]
            if pred_idx < len(args.class_names)
            else str(pred_idx)
        )
        confidence = float(probs[pred_idx])
        row = {
            "filename": fname,
            "predicted_class": pred_name,
            "confidence": round(confidence, 6),
        }
        for j, cname in enumerate(args.class_names):
            if j < len(probs):
                row[f"prob_{cname}"] = round(float(probs[j]), 6)
        if label_map is not None:
            true_idx = label_map.get(fname)
            row["true_label"] = (
                args.class_names[true_idx]
                if true_idx is not None and 0 <= true_idx < len(args.class_names)
                else ""
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n  CSV 已保存: {args.output}")

    # 评估
    if label_map is not None:
        # 匹配标签
        y_true_list, y_pred_list, y_prob_list = [], [], []
        for fname, probs in zip(filenames, all_probs):
            if fname not in label_map:
                continue
            true_label = label_map[fname]
            pred_idx = int(np.argmax(probs))
            y_true_list.append(true_label)
            y_pred_list.append(pred_idx)
            y_prob_list.append(probs)

        if not y_true_list:
            print("  ⚠ 没有匹配到标签的样本，无法评估")
            return

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        y_prob = np.array(y_prob_list)

        # 标签偏移处理
        n_effective_classes = y_prob.shape[1]
        if args.label_offset is not None:
            offset = args.label_offset
        elif y_true.min() == 1 and y_true.max() == n_effective_classes:
            offset = 1
        else:
            offset = 0

        if offset != 0:
            y_true = y_true - offset

        if y_true.max() >= n_effective_classes or y_true.min() < 0:
            print(f"[ERROR] 标签范围错误 [0, {n_effective_classes - 1}]，"
                  f"offset={offset}, min={y_true.min()}, max={y_true.max()}")
            return

        is_binary = n_effective_classes <= 2
        metrics = compute_all_metrics(
            y_true, y_pred, y_prob, n_effective_classes,
            n_boot=args.n_bootstrap,
        )
        report = format_metrics_report(
            metrics, is_binary, args.class_names,
            labels=y_true, preds=y_pred,
            n_bootstrap=args.n_bootstrap,
            label_field=args.label_field or "",
        )
        print(report)

        metrics_path = args.metrics_output
        if metrics_path is None:
            base = os.path.splitext(args.output)[0]
            metrics_path = base + "_metrics.log"
        metrics_dir = os.path.dirname(metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            f.write(report + "\n")
        print(f"  评估结果已保存: {metrics_path}")

    print("\n  完成")


if __name__ == "__main__":
    main()
