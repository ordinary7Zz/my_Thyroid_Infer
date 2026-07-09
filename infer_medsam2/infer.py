#!/usr/bin/env python3
"""
MedSAM2 2D 推理脚本（自包含版本）
==================================

对目录中的 2D 图像进行 MedSAM2 分割推理。

功能:
  - 对输入图像目录中所有图像进行分割推理（全图 box prompt）
  - 可选: 输出预测掩码图像 (PNG)
  - 可选: 输入 GT mask 目录，计算 Dice 和 HD95 指标及 95% 置信区间
  - 所有结果和运行信息保存到 log 文件
  - 若不指定输出目录也不提供 GT，仅执行推理，无任何输出（可接受）

用法示例:
  # 仅推理，不保存、不评估
  python infer.py --image_dir ./images/ --checkpoint ./medsam2.pt

  # 推理 + 保存 mask
  python infer.py --image_dir ./images/ --checkpoint ./medsam2.pt --output_dir ./predictions/

  # 推理 + 评估（需要 GT）
  python infer.py --image_dir ./images/ --checkpoint ./medsam2.pt --gt_dir ./masks/

  # 推理 + 保存 + 评估
  python infer.py --image_dir ./images/ --checkpoint ./medsam2.pt --gt_dir ./masks/ --output_dir ./predictions/

  # 指定设备和配置
  python infer.py --image_dir ./images/ --checkpoint ./medsam2.pt --config sam2.1_hiera_t512.yaml --device cuda
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 确保脚本所在目录在 Python path 中（使 sam2 包可导入）
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 使用项目级统一指标模块
_ROOT = os.path.dirname(SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sam2.build_sam import build_sam2_video_predictor_npz
from seg_metrics import compute_dice, compute_hd95, bootstrap_ci, logits_to_binary

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def collect_images(image_dir):
    """收集目录中所有支持的图像文件，按文件名排序。"""
    paths = []
    for name in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTS:
            paths.append(os.path.join(image_dir, name))
    return paths


def load_rgb_tensor(image_path):
    """加载图像为 [3, H, W] tensor，值域 [0, 1]。"""
    img = Image.open(image_path).convert("RGB")
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def preprocess(img_tensor, image_size):
    """Resize + ImageNet normalize。

    输入:  [3, H, W], 值域 [0, 1]
    输出:  [3, image_size, image_size], ImageNet normalized
    """
    _, H, W = img_tensor.shape
    if H != image_size or W != image_size:
        img = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    else:
        img = img_tensor.clone()
    return (img - IMAGENET_MEAN) / IMAGENET_STD


def load_gt_mask(gt_path, target_h, target_w):
    """加载 GT mask，返回二值 numpy 数组 (0/1)，尺寸为 (target_h, target_w)。"""
    mask = Image.open(gt_path).convert("L")
    if mask.height != target_h or mask.width != target_w:
        mask = mask.resize((target_w, target_h), Image.NEAREST)
    mask_np = np.array(mask)
    return (mask_np > 128).astype(np.uint8)


def find_gt_file(gt_dir, stem):
    """在 gt_dir 中查找与 stem 匹配的 mask 文件，返回路径或 None。"""
    for ext in SUPPORTED_EXTS:
        gt_path = os.path.join(gt_dir, f"{stem}{ext}")
        if os.path.isfile(gt_path):
            return gt_path
    return None


def resolve_config_path(config_arg):
    """将 config 参数解析为 hydra 可识别的 '//' 绝对路径。

    查找顺序:
      1. 已带 // 前缀 → 直接返回
      2. 绝对路径 → 加 // 前缀
      3. 相对于脚本目录查找 (sam2/configs/<config>, sam2/<config>, <config>)
    """
    if config_arg.startswith('//'):
        return config_arg
    if os.path.isabs(config_arg):
        return '//' + config_arg

    candidates = [
        os.path.join(SCRIPT_DIR, 'sam2', 'configs', config_arg),
        os.path.join(SCRIPT_DIR, 'sam2', config_arg),
        os.path.join(SCRIPT_DIR, config_arg),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return '//' + c

    # fallback: 默认位置
    return '//' + os.path.join(SCRIPT_DIR, 'sam2', 'configs', config_arg)


def setup_logger(log_dir):
    """设置日志：同时输出到控制台和文件。返回 (logger, log_file_path)。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"metrics_{timestamp}.log")

    logger = logging.getLogger("infer_medsam2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, log_file


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

def infer_one_image(predictor, img_tensor, device):
    """对单张图像推理（全图 box prompt）。

    参数:
        predictor: MedSAM2 video predictor
        img_tensor: [3, H, W], 值域 [0, 1]
        device: torch.device

    返回:
        mask_logits: [1, H, W] tensor (原图尺寸)
    """
    H_orig, W_orig = img_tensor.shape[1], img_tensor.shape[2]

    # 预处理 → [1, 3, image_size, image_size]
    img_processed = preprocess(img_tensor, predictor.image_size)
    images = img_processed.unsqueeze(0).to(device)

    # 初始化单帧 "视频" inference state
    inference_state = predictor.init_state(
        images=images,
        video_height=H_orig,
        video_width=W_orig,
    )

    # 全图作为 box prompt
    box = np.array([0, 0, W_orig - 1, H_orig - 1], dtype=np.float32)
    _, _, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        box=box,
    )

    # out_mask_logits: [num_obj, 1, H_orig, W_orig]
    mask_logits = out_mask_logits[0].float()  # [1, H, W]
    predictor.reset_state(inference_state)
    return mask_logits


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MedSAM2 2D 推理脚本（自包含版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅推理
  python infer.py --image_dir ./images/ --checkpoint ./medsam2.pt

  # 推理 + 保存 mask
  python infer.py --image_dir ./images/ --checkpoint ./medsam2.pt --output_dir ./pred/

  # 推理 + 评估
  python infer.py --image_dir ./images/ --checkpoint ./medsam2.pt --gt_dir ./masks/

  # 推理 + 保存 + 评估
  python infer.py --image_dir ./images/ --checkpoint ./medsam2.pt --gt_dir ./masks/ --output_dir ./pred/
        """,
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="输入图像目录")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="MedSAM2 权重文件路径 (.pt)")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="GT mask 目录（可选）。提供后计算 Dice/HD95 指标")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出预测 mask 目录（可选）。不提供则不保存 mask")
    parser.add_argument("--config", type=str, default="sam2.1_hiera_t512.yaml",
                        help="模型配置文件名或路径 (默认: sam2.1_hiera_t512.yaml)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备 (默认: cuda, 无 GPU 时自动回退到 cpu)")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="日志输出目录 (默认: ./logs)")
    args = parser.parse_args()

    # --- 设置日志 ---
    logger, log_file = setup_logger(args.log_dir)

    # --- 设备 ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- 记录配置 ---
    logger.info("=" * 60)
    logger.info(f"权重:     {args.checkpoint}")
    logger.info(f"数据:     {args.image_dir}")
    logger.info(f"GT:       {args.gt_dir if args.gt_dir else '(无)'}")
    logger.info(f"设备:     {device}")
    logger.info("=" * 60)

    # --- 检查输入 ---
    if not os.path.isdir(args.image_dir):
        logger.error(f"图像目录不存在: {args.image_dir}")
        return
    if not os.path.isfile(args.checkpoint):
        logger.error(f"权重文件不存在: {args.checkpoint}")
        return
    if args.gt_dir and not os.path.isdir(args.gt_dir):
        logger.error(f"GT 目录不存在: {args.gt_dir}")
        return
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- 加载模型 ---
    cfg_path = resolve_config_path(args.config)

    try:
        predictor = build_sam2_video_predictor_npz(
            config_file=cfg_path,
            ckpt_path=args.checkpoint,
            device=device,
            mode="eval",
        )
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 收集图像 ---
    image_paths = collect_images(args.image_dir)
    if not image_paths:
        logger.error(f"在 {args.image_dir} 中没有找到图像文件")
        return

    # --- 推理 + 评估 ---
    dice_values = []
    hd95_values = []
    per_sample = []  # (filename, dice, hd95)
    n_saved = 0
    n_no_gt = 0

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.nullcontext()
    )

    start_time = time.time()
    with torch.inference_mode(), autocast_ctx:
        for i, img_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="推理"):
            stem = os.path.splitext(os.path.basename(img_path))[0]

            # 加载图像
            img_tensor = load_rgb_tensor(img_path)  # [3, H, W], [0, 1]
            H, W = img_tensor.shape[1], img_tensor.shape[2]

            # 推理
            mask_logits = infer_one_image(predictor, img_tensor, device)

            # 确保输出尺寸与原图一致
            _, out_H, out_W = mask_logits.shape
            if out_H != H or out_W != W:
                mask_logits = F.interpolate(
                    mask_logits.unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)

            # 转为二值 mask (0/1)
            pred_bin = logits_to_binary(mask_logits)  # [H, W]

            # 保存预测 mask
            if args.output_dir:
                out_path = os.path.join(args.output_dir, f"{stem}.png")
                mask_save = (pred_bin * 255).astype(np.uint8)
                Image.fromarray(mask_save, mode="L").save(out_path)
                n_saved += 1

            # 计算 metrics
            if args.gt_dir:
                gt_path = find_gt_file(args.gt_dir, stem)
                if gt_path is None:
                    n_no_gt += 1
                    continue

                gt_bin = load_gt_mask(gt_path, H, W)
                # 统一 resize 到 224×224 计算指标
                pred_224 = np.array(Image.fromarray(pred_bin).resize(
                    (224, 224), Image.NEAREST))
                gt_224 = np.array(Image.fromarray(gt_bin).resize(
                    (224, 224), Image.NEAREST))
                dice = compute_dice(pred_224, gt_224)
                hd95 = compute_hd95(pred_224, gt_224)
                dice_values.append(dice)
                hd95_values.append(hd95)
                per_sample.append((os.path.basename(img_path), dice, hd95))

    elapsed = time.time() - start_time

    # --- 汇总 ---
    if args.gt_dir:
        if len(dice_values) > 0:
            dice_mean, dice_lo, dice_hi = bootstrap_ci(dice_values)
            hd95_mean, hd95_lo, hd95_hi = bootstrap_ci(hd95_values)
            logger.info("=" * 60)
            logger.info(f"评估样本数: {len(dice_values)}")
            logger.info(f"Dice:  {dice_mean:.4f}  (95% CI: [{dice_lo:.4f}, {dice_hi:.4f}])")
            logger.info(f"HD95:  {hd95_mean:.4f}  (95% CI: [{hd95_lo:.4f}, {hd95_hi:.4f}])")
            logger.info("=" * 60)

            # 追加每样本指标到 log 文件
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("\n--- Per-Sample Metrics ---\n")
                f.write("filename,dice,hd95\n")
                for fname, dsc, hd in per_sample:
                    f.write(f"{fname},{dsc:.4f},{hd:.4f}\n")
        else:
            logger.warning("未计算到任何有效指标（可能 GT mask 均未找到）")


if __name__ == "__main__":
    main()
