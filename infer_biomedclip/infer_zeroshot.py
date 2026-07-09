#!/usr/bin/env python3
"""
BiomedCLIP 零样本分类推理脚本
================================
不使用微调权重，直接用预训练 BiomedCLIP 做零样本分类。
通过文本 prompt + 图像相似度进行分类，无需 finetune 分类头。

用法:
    # 二分类（良恶性）
    python infer_zeroshot.py \\
        --model_dir /path/to/biomedclip \\
        --folder /path/to/images/ \\
        --num_classes 2 \\
        --class_names 0 1 \\
        --output results.csv

    # TIRADS 五分类
    python infer_zeroshot.py \\
        --model_dir /path/to/biomedclip \\
        --folder /path/to/images/ \\
        --num_classes 5 \\
        --class_names 1 2 3 4 5 \\
        --output results.csv

    # 自定义文本 prompt
    python infer_zeroshot.py \\
        --model_dir /path/to/biomedclip \\
        --folder /path/to/images/ \\
        --num_classes 2 \\
        --class_names benign malignant \\
        --prompts "an ultrasound of a benign thyroid nodule" "an ultrasound of a malignant thyroid nodule" \\
        --output results.csv
"""

import os
import sys
import csv
import json
import argparse
import warnings
from datetime import datetime

# 强制 transformers / huggingface_hub 离线模式，避免构建文本塔时连 HF
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 使用项目级统一分类指标模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from cls_metrics import compute_all_metrics, format_metrics_report

warnings.filterwarnings("ignore")


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
    """根据类别数返回默认 prompt 列表。"""
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

def load_biomedclip_model(model_dir: str, device: torch.device):
    """从本地目录加载完整 BiomedCLIP 模型（视觉 + 文本编码器）。

    Args:
        model_dir: 包含 open_clip_config.json 和权重文件的目录
        device: 推理设备

    Returns:
        (visual, text, logit_scale, context_length, tokenizer)
    """
    import json as _json

    # 读取配置
    config_path = os.path.join(model_dir, "open_clip_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"未找到 {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = _json.load(f)
    model_cfg = cfg["model_cfg"]
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    context_length = text_cfg.get("context_length", 256)

    # 将 HF 模型名指向本地 model_dir（含 BiomedBERT config.json），
    # 并关闭 pretrained 标志：只用 from_config 建架构，权重从 .bin 加载
    text_cfg["hf_model_name"] = model_dir
    text_cfg["hf_model_pretrained"] = False

    # 查找本地权重文件
    weights_path = None
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".safetensors"):
            weights_path = os.path.join(model_dir, fname)
            break
        if fname.endswith((".bin", ".pt", ".pth")) and weights_path is None:
            weights_path = os.path.join(model_dir, fname)

    if weights_path is None:
        raise FileNotFoundError(
            f"在 {model_dir} 中未找到权重文件（.safetensors/.bin/.pt/.pth）"
        )

    print(f"  预训练权重:   {weights_path}")
    print(f"  embed_dim:    {embed_dim}")
    print(f"  context_length: {context_length}")

    # 加载权重
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        full_state = load_file(weights_path)
    else:
        full_state = torch.load(weights_path, map_location="cpu")

    # 构建视觉编码器和文本编码器
    from open_clip.model import _build_vision_tower, _build_text_tower

    print("  构建视觉编码器...")
    visual = _build_vision_tower(embed_dim, vision_cfg)
    visual_state = {
        k[len("visual."):]: v for k, v in full_state.items()
        if k.startswith("visual.")
    }
    missing_v, unexpected_v = visual.load_state_dict(visual_state, strict=False)
    if missing_v:
        print(f"  visual missing keys: {len(missing_v)}")
    visual.to(device).eval()

    print("  构建文本编码器...")
    text = _build_text_tower(embed_dim, text_cfg)
    text_state = {
        k[len("text."):]: v for k, v in full_state.items()
        if k.startswith("text.")
    }
    missing_t, unexpected_t = text.load_state_dict(text_state, strict=False)
    if missing_t:
        print(f"  text missing keys: {len(missing_t)}")
    text.to(device).eval()

    # logit_scale
    if "logit_scale" in full_state:
        logit_scale = full_state["logit_scale"].to(device)
    else:
        logit_scale = torch.ones([], device=device) * np.log(1 / 0.07)

    # 加载 tokenizer（BiomedBERT tokenizer，文件在 model_dir 中）
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    total_params = sum(p.numel() for p in visual.parameters()) + \
                   sum(p.numel() for p in text.parameters())
    print(f"  模型参数量:   {total_params:,}")

    return visual, text, logit_scale, context_length, tokenizer


# ============================================================================
# 文本编码
# ============================================================================

@torch.no_grad()
def encode_text_prompts(text_encoder, tokenizer, prompts, context_length, device):
    """编码文本 prompt，返回归一化的文本特征。

    Returns:
        text_features: (C, embed_dim) 归一化后的文本特征
    """
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=context_length,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # HFTextEncoder 可能接受 attention_mask，也可能不接受
    try:
        text_features = text_encoder(input_ids, attention_mask=attention_mask)
    except TypeError:
        text_features = text_encoder(input_ids)

    text_features = F.normalize(text_features, dim=-1)
    return text_features  # (C, embed_dim)


# ============================================================================
# 图像预处理与批量推理
# ============================================================================

def get_preprocess(image_size: int = 224):
    """CLIP 标准预处理（与 finetune 版一致）。"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


@torch.no_grad()
def batch_infer(visual, text_features, logit_scale, img_paths,
                preprocess, device, batch_size=32):
    """批量零样本推理。

    Returns:
        all_probs: (N, C) 概率矩阵
    """
    all_probs = []
    for i in tqdm(range(0, len(img_paths), batch_size), desc="推理"):
        batch_paths = img_paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
            except Exception as e:
                print(f"\n  ⚠ 读取图片失败: {p} ({e})，使用零张量替代")
                tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(tensors).to(device)
        image_features = visual(batch)
        image_features = F.normalize(image_features, dim=-1)

        logits = logit_scale.exp() * image_features @ text_features.T  # (B, C)
        probs = logits.softmax(dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)  # (N, C)


# ============================================================================
# CSV 输出（与 finetune 版格式一致）
# ============================================================================

def collect_images(folder: str):
    """收集文件夹中所有图片文件（按文件名排序）。"""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    ])


def save_csv(output_path, filenames, all_probs, class_names, label_map=None):
    """保存分类结果到 CSV（与 finetune 版格式一致）。"""
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["filename", "predicted_class", "confidence"]
    for cname in class_names:
        fieldnames.append(f"prob_{cname}")
    if label_map is not None:
        fieldnames.append("true_label")

    rows = []
    for fname, probs in zip(filenames, all_probs):
        pred_idx = int(np.argmax(probs))
        pred_name = class_names[pred_idx]
        pred_conf = float(probs[pred_idx])
        row = {
            "filename": fname,
            "predicted_class": pred_name,
            "confidence": round(pred_conf, 6),
        }
        for i, cname in enumerate(class_names):
            row[f"prob_{cname}"] = round(float(probs[i]), 6)
        if label_map is not None:
            true_idx = label_map.get(fname)
            row["true_label"] = (
                class_names[true_idx]
                if true_idx is not None and 0 <= true_idx < len(class_names)
                else ""
            )
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


# ============================================================================
# 标签加载（与 finetune 版逻辑一致）
# ============================================================================

def load_label_json(json_path, label_field, class_names):
    """加载标签 JSON，映射为 0-based 索引。"""
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    name_to_idx = {str(name): i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    label_map = {}
    missing = []
    for rec in records:
        fname = rec.get("filename")
        if fname is None:
            continue
        if label_field not in rec:
            missing.append(fname)
            continue

        raw_label = rec[label_field]
        label_str = str(raw_label)

        if label_str in name_to_idx:
            label_idx = name_to_idx[label_str]
        elif isinstance(raw_label, (int, float)) and 0 <= int(raw_label) < num_classes:
            label_idx = int(raw_label)
        elif isinstance(raw_label, (int, float)) and 1 <= int(raw_label) <= num_classes:
            label_idx = int(raw_label) - 1
        else:
            print(f"  ⚠ 无法映射标签: {fname} {label_field}={raw_label}, 跳过")
            continue

        label_map[fname] = label_idx

    if missing:
        print(f"  ⚠ {len(missing)} 条记录缺少字段 '{label_field}'，已跳过")

    return label_map


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BiomedCLIP 零样本分类推理（不使用 finetune 权重）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="本地 BiomedCLIP 预训练模型目录")
    parser.add_argument("--folder", type=str, required=True,
                        help="待推理的图片文件夹路径")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="类别数")
    parser.add_argument("--class_names", type=str, nargs="+", required=True,
                        help="类别名称列表")
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                        help="自定义文本 prompt（数量需与 num_classes 一致）")

    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备 (默认 cuda)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批推理大小 (默认 32)")
    parser.add_argument("--output", type=str, default="results.csv",
                        help="CSV 输出路径 (默认 results.csv)")

    parser.add_argument("--label_json", type=str, default=None,
                        help="标签 JSON 文件路径（可选）")
    parser.add_argument("--label_field", type=str, default=None,
                        help="JSON 中标签字段名")
    parser.add_argument("--eval_output", type=str, default=None,
                        help="评估结果保存路径 (.log)")
    parser.add_argument("--n_bootstrap", type=int, default=2000,
                        help="Bootstrap 迭代次数 (默认 2000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子 (默认 0)")
    args = parser.parse_args()

    # 校验
    if len(args.class_names) != args.num_classes:
        print(f"错误: --class_names 长度 ({len(args.class_names)}) "
              f"与 --num_classes ({args.num_classes}) 不一致")
        sys.exit(1)

    if args.label_json is not None and args.label_field is None:
        print("错误: 提供了 --label_json 时必须同时指定 --label_field")
        sys.exit(1)

    # 确定 prompt
    prompts = args.prompts if args.prompts else get_default_prompts(args.num_classes)
    if len(prompts) != args.num_classes:
        print(f"错误: prompts 数量 ({len(prompts)}) 与 num_classes "
              f"({args.num_classes}) 不一致")
        sys.exit(1)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 检查文件夹
    if not os.path.isdir(args.folder):
        print(f"错误: 图片文件夹不存在: {args.folder}")
        sys.exit(1)

    # 加载模型
    print("=" * 60)
    print("BiomedCLIP 零样本分类推理")
    print("=" * 60)
    visual, text_encoder, logit_scale, context_length, tokenizer = \
        load_biomedclip_model(args.model_dir, device)

    # 编码文本 prompt
    print("\n  文本 prompt:")
    for i, p in enumerate(prompts):
        print(f"    [{i}] {p}")
    text_features = encode_text_prompts(
        text_encoder, tokenizer, prompts, context_length, device
    )
    print(f"  文本特征形状: {text_features.shape}")

    # 收集图片
    filenames = collect_images(args.folder)
    if not filenames:
        print(f"错误: 文件夹中未找到图片: {args.folder}")
        sys.exit(1)
    img_paths = [os.path.join(args.folder, f) for f in filenames]

    print(f"\n  数据:     {args.folder}")
    print(f"  图片数:   {len(filenames)}")
    print(f"  类别:     {args.class_names}")
    if args.label_json:
        print(f"  标签字段: {args.label_field}")
    print(f"  设备:     {device}")
    print("=" * 60)

    # 推理
    preprocess = get_preprocess(image_size=224)
    all_probs = batch_infer(
        visual, text_features, logit_scale, img_paths,
        preprocess, device, args.batch_size,
    )

    # 加载标签（可选，用于 CSV true_label 列和评估）
    label_map = None
    if args.label_json:
        label_map = load_label_json(
            args.label_json, args.label_field, args.class_names
        )

    # 保存 CSV
    save_csv(args.output, filenames, all_probs, args.class_names, label_map)
    print(f"\n  CSV 已保存: {args.output}")

    # 评估
    if args.label_json:

        y_true_list, y_pred_list, y_prob_list = [], [], []
        for fname, probs in zip(filenames, all_probs):
            if fname not in label_map:
                continue
            true_label = label_map[fname]
            pred_idx = int(np.argmax(probs))
            if true_label < 0 or true_label >= args.num_classes:
                continue
            y_true_list.append(true_label)
            y_pred_list.append(pred_idx)
            y_prob_list.append(probs)

        if not y_true_list:
            print("  ⚠ 没有匹配到标签的样本，无法评估")
            return

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        y_prob = np.array(y_prob_list)

        is_binary = args.num_classes == 2
        metrics = compute_all_metrics(
            y_true, y_pred, y_prob, args.num_classes,
            n_boot=args.n_bootstrap,
        )
        report = format_metrics_report(
            metrics, is_binary, args.class_names,
            labels=y_true, preds=y_pred,
            n_bootstrap=args.n_bootstrap,
            label_field=args.label_field or "",
        )
        print(report)

        eval_path = args.eval_output
        if eval_path is None:
            out_dir = os.path.dirname(os.path.abspath(args.output))
            ts = datetime.now().strftime("%m%d_%H%M%S")
            eval_path = os.path.join(out_dir, f"eval_result_{ts}.log")
        out_dir = os.path.dirname(os.path.abspath(eval_path))
        os.makedirs(out_dir, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"  评估结果已保存: {eval_path}")

    print("\n  完成")


if __name__ == "__main__":
    main()
