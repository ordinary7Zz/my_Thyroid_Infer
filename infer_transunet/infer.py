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
# 使用项目级统一指标模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from seg_metrics import compute_dice, compute_hd95, bootstrap_ci

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
    """计算单例 Dice 和 HD95，使用统一指标模块。

    Returns:
        (dice, hd95)
    """
    return compute_dice(pred, gt), compute_hd95(pred, gt)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # 设备
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    # 加载模型
    net = load_model(args, device)

    # 收集图像
    files = collect_images(args.img_dir)
    if not files:
        print(f"[ERROR] No images found in: {args.img_dir}")
        sys.exit(1)

    # 准备输出目录
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    # 是否计算指标
    compute_metrics = args.gt_dir is not None

    # 打印配置
    print("=" * 60)
    print(f"权重:     {args.ckpt}")
    print(f"数据:     {args.img_dir}")
    print(f"GT:       {args.gt_dir if args.gt_dir else '(无)'}")
    print(f"设备:     {device}")
    print("=" * 60)

    # 推理循环
    dice_list: List[float] = []
    hd95_list: List[float] = []

    for stem, img_path in tqdm(files, desc="推理"):
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
                continue
            gt = Image.open(gt_path)
            gt_np = np.array(gt, dtype=np.int32)
            gt_bin = (gt_np > 0).astype(np.uint8)
            d, h95 = calculate_metric(pred, gt_bin)
            dice_list.append(d)
            hd95_list.append(h95)

    # 汇总输出
    if compute_metrics:
        n_eval = len(dice_list)
        if n_eval == 0:
            print("[WARN] No valid GT pairs found, no metrics computed.")
            return

        dice_mean, dice_lo, dice_hi = bootstrap_ci(dice_list)
        hd95_mean, hd95_lo, hd95_hi = bootstrap_ci(hd95_list)

        print("=" * 60)
        print(f"评估样本数: {n_eval}")
        print(f"Dice:  {dice_mean:.4f}  (95% CI: [{dice_lo:.4f}, {dice_hi:.4f}])")
        print(f"HD95:  {hd95_mean:.4f}  (95% CI: [{hd95_lo:.4f}, {hd95_hi:.4f}])")
        print("=" * 60)

        # 写 log 文件（仅指标）
        log_path = args.log
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"评估样本数: {n_eval}\n")
            f.write(f"Dice:  {dice_mean:.4f}  (95% CI: [{dice_lo:.4f}, {dice_hi:.4f}])\n")
            f.write(f"HD95:  {hd95_mean:.4f}  (95% CI: [{hd95_lo:.4f}, {hd95_hi:.4f}])\n")


if __name__ == "__main__":
    main()
