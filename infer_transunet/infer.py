#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TransUNet 独立推理脚本
================================
功能：
  1. 对图像目录进行批量推理
  2. (可选) 输出推理掩码 PNG 到指定目录
  3. (可选) 输入 GT mask 目录，计算 Dice / HD95 及 95% CI，写入纯文本 log

用法示例:
  # 仅推理，不保存掩码，不计算指标
  python infer.py --ckpt model.pth --img_dir ./images

  # 推理 + 保存掩码
  python infer.py --ckpt model.pth --img_dir ./images --out_dir ./preds

  # 推理 + 保存掩码 + 计算指标并写 log
  python infer.py --ckpt model.pth --img_dir ./images --gt_dir ./masks \
      --out_dir ./preds --log ./eval_log.txt
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import torch
from tqdm import tqdm
from medpy import metric

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TransUNet 独立推理脚本（批量图像目录）"
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="训练好的模型权重路径 (.pth)")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="输入图像目录 (包含 png/jpg/jpeg 文件)")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="GT mask 目录 (可选)。提供则计算 Dice/HD95 指标。"
                             "掩码文件名需与图像文件名的 stem 一致。")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="输出掩码保存目录 (可选)。提供则保存推理掩码 PNG。")
    parser.add_argument("--log", type=str, default="./eval_log.txt",
                        help="指标 log 文件路径 (纯文本)。仅在提供 --gt_dir 时写入。"
                             "默认: ./eval_log.txt")

    parser.add_argument("--img_size", type=int, default=224,
                        help="网络输入尺寸 (需与训练一致)，默认 224")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="类别数 (二分类=2)，默认 2")
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16",
                        help="ViT 骨干名称 (需与训练一致)，默认 R50-ViT-B_16")
    parser.add_argument("--n_skip", type=int, default=3,
                        help="skip 连接数量 (需与训练一致)，默认 3")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备: 'cuda' 或 'cpu'。无 CUDA 时自动回退到 cpu。"
                             "默认 cuda")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")


def collect_images(img_dir: str) -> List[Tuple[str, str]]:
    """收集目录下所有图像文件，返回 (stem, filepath) 列表，按文件名排序。"""
    files = []
    for fname in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in IMAGE_EXTS:
            files.append((stem, os.path.join(img_dir, fname)))
    return files


def find_gt_file(gt_dir: str, stem: str) -> Optional[str]:
    """在 gt_dir 中查找与 stem 匹配的文件 (支持 png/jpg/jpeg，大小写不敏感)。"""
    for ext in IMAGE_EXTS:
        path = os.path.join(gt_dir, stem + ext)
        if os.path.exists(path):
            return path
        path_upper = os.path.join(gt_dir, stem + ext.upper())
        if os.path.exists(path_upper):
            return path_upper
    return None


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """加载 TransUNet 模型。"""
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(args.img_size / 16),
            int(args.img_size / 16),
        )
    net = ViT_seg(config_vit, img_size=args.img_size,
                  num_classes=config_vit.n_classes)
    state = torch.load(args.ckpt, map_location=device)
    # 兼容直接 state_dict 或包裹在字典中的情况
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


def infer_one(net: torch.nn.Module, img_path: str,
              img_size: int, device: torch.device) -> np.ndarray:
    """对单张图像推理，返回原图尺寸的预测掩码 (numpy uint8，值 0 或 1)。"""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.float32)
    h, w = img_np.shape

    # resize 到网络输入大小
    if h != img_size or w != img_size:
        img_resized = zoom(img_np, (img_size / h, img_size / w), order=3)
    else:
        img_resized = img_np

    input_tensor = (
        torch.from_numpy(img_resized)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.no_grad():
        logits = net(input_tensor)
        pred_resized = (
            torch.argmax(torch.softmax(logits, dim=1), dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    # resize 回原图大小
    if h != img_size or w != img_size:
        pred = zoom(pred_resized, (h / img_size, w / img_size), order=0)
        pred = np.round(pred).astype(np.uint8)
    else:
        pred = pred_resized

    return (pred > 0).astype(np.uint8)


def calculate_metric(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """计算单例 Dice 和 HD95 (二分类)。

    边界情况处理与项目原始 utils.calculate_metric_percase 保持一致：
      - 有预测、有GT: 正常计算
      - 有预测、无GT (假阳性): dice=1, hd95=0
      - 无预测、有GT (假阴性): dice=0, hd95=0
      - 都为空 (真阴性): dice=0, hd95=0

    Returns:
        (dice, hd95)
    """
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if pred_bin.sum() > 0 and gt_bin.sum() > 0:
        dice = metric.binary.dc(pred_bin, gt_bin)
        hd95 = metric.binary.hd95(pred_bin, gt_bin)
        return float(dice), float(hd95)
    elif pred_bin.sum() > 0 and gt_bin.sum() == 0:
        return 1.0, 0.0
    else:
        return 0.0, 0.0


def mean_ci95(values: List[float]) -> Tuple[float, float, float]:
    """正态近似法计算均值和 95% 置信区间。

    Returns:
        (mean, lower, upper)
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean
    std = float(arr.std(ddof=1))
    margin = 1.96 * std / np.sqrt(n)
    return mean, mean - margin, mean + margin


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # 设备
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    # 加载模型
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    net = load_model(args, device)
    print("[INFO] Model loaded successfully.")

    # 收集图像
    files = collect_images(args.img_dir)
    if not files:
        print(f"[ERROR] No images found in: {args.img_dir}")
        sys.exit(1)
    print(f"[INFO] Found {len(files)} images in: {args.img_dir}")

    # 准备输出目录
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[INFO] Predictions will be saved to: {args.out_dir}")
    else:
        print("[INFO] No --out_dir given, predictions will not be saved.")

    # 是否计算指标
    compute_metrics = args.gt_dir is not None
    if compute_metrics:
        print(f"[INFO] GT masks from: {args.gt_dir}, metrics will be computed.")
    else:
        print("[INFO] No --gt_dir given, metrics will not be computed.")

    # 推理循环
    dice_list: List[float] = []
    hd95_list: List[float] = []
    missing_gt: List[str] = []

    for stem, img_path in tqdm(files, desc="Inferring"):
        # 推理
        pred = infer_one(net, img_path, args.img_size, device)

        # 保存掩码 (0/1 → 0/255，单通道 PNG)
        if args.out_dir is not None:
            mask_img = Image.fromarray(pred * 255)
            out_path = os.path.join(args.out_dir, stem + ".png")
            mask_img.save(out_path)

        # 计算指标
        if compute_metrics:
            gt_path = find_gt_file(args.gt_dir, stem)
            if gt_path is None:
                missing_gt.append(stem)
                continue
            gt = Image.open(gt_path)
            gt_np = np.array(gt, dtype=np.int32)
            gt_bin = (gt_np > 0).astype(np.uint8)
            d, h95 = calculate_metric(pred, gt_bin)
            dice_list.append(d)
            hd95_list.append(h95)

    # 汇总输出
    print(f"\n[INFO] Inference complete. {len(files)} images processed.")

    if args.out_dir is not None:
        print(f"[INFO] Masks saved to: {args.out_dir}")

    if compute_metrics:
        if missing_gt:
            print(f"[WARN] {len(missing_gt)} images have no matching GT mask: "
                  f"{missing_gt[:5]}{'...' if len(missing_gt) > 5 else ''}")

        n_eval = len(dice_list)
        if n_eval == 0:
            print("[WARN] No valid GT pairs found, no metrics computed.")
            return

        dice_mean, dice_lo, dice_hi = mean_ci95(dice_list)
        hd95_mean, hd95_lo, hd95_hi = mean_ci95(hd95_list)

        # 打印到终端
        print(f"\n{'='*50}")
        print(f"Evaluation Results ({n_eval} cases)")
        print(f"{'='*50}")
        print(f"  Dice:  mean={dice_mean:.6f}  95% CI=({dice_lo:.6f}, {dice_hi:.6f})")
        print(f"  HD95:  mean={hd95_mean:.6f}  95% CI=({hd95_lo:.6f}, {hd95_hi:.6f})")
        print(f"{'='*50}")

        # 写纯文本 log
        log_path = args.log
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("TransUNet Inference Evaluation Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint:   {args.ckpt}\n")
            f.write(f"Image dir:    {args.img_dir}\n")
            f.write(f"GT dir:       {args.gt_dir}\n")
            f.write(f"Output dir:   {args.out_dir if args.out_dir else 'Not saved'}\n")
            f.write(f"Model config: vit_name={args.vit_name}, "
                    f"img_size={args.img_size}, "
                    f"num_classes={args.num_classes}, "
                    f"n_skip={args.n_skip}\n")
            f.write(f"Device:       {device}\n")
            f.write(f"Num cases:    {n_eval}\n")
            f.write("-" * 60 + "\n")
            f.write("Per-case results:\n")
            for i, (d, h95) in enumerate(zip(dice_list, hd95_list), 1):
                f.write(f"  [{i:04d}] Dice={d:.6f}  HD95={h95:.4f}\n")
            f.write("-" * 60 + "\n")
            f.write("Summary:\n")
            f.write(f"  Dice:  mean={dice_mean:.6f}  "
                    f"95% CI=({dice_lo:.6f}, {dice_hi:.6f})\n")
            f.write(f"  HD95:  mean={hd95_mean:.6f}  "
                    f"95% CI=({hd95_lo:.6f}, {hd95_hi:.6f})\n")
            if missing_gt:
                f.write("-" * 60 + "\n")
                f.write(f"Missing GT ({len(missing_gt)} cases):\n")
                for s in missing_gt:
                    f.write(f"  {s}\n")
            f.write("=" * 60 + "\n")

        print(f"[INFO] Log written to: {log_path}")


if __name__ == "__main__":
    main()
