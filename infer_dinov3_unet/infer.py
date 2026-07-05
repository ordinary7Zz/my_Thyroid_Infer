#!/usr/bin/env python
"""
DINOv3-UNet 独立推理脚本

功能：
  - 输入 checkpoint + 图像目录，进行分割推理
  - 可选输出预测掩码图像（--output_dir）
  - 可选输入 GT mask 目录（--gt_dir），计算 Dice / HD95 / ECE 及其 CI95 置信区间
  - 指标结果保存到 log 文件（文本 + JSON）
  - 两个输出（掩码、指标）均可关闭；若无 GT 则自动跳过指标计算

用法示例：
  # 仅推理，不输出掩码，不计算指标
  python infer.py --checkpoint model.pth --input_dir ./images

  # 推理 + 输出掩码
  python infer.py --checkpoint model.pth --input_dir ./images --output_dir ./preds

  # 推理 + 输出掩码 + 计算指标
  python infer.py --checkpoint model.pth --input_dir ./images \\
      --gt_dir ./masks --output_dir ./preds --log_dir ./logs
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import DINOv3_S_UNet
from metrics import Dice, HD95, ECE, bootstrap_ci


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =========================
# 工具函数
# =========================
def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clean_path(path):
    if isinstance(path, str):
        if (path.startswith('"') and path.endswith('"')) or (
            path.startswith("'") and path.endswith("'")
        ):
            path = path[1:-1]
        path = path.strip()
    return path


def json_default(obj):
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# =========================
# 数据集
# =========================
class InferenceDataset(Dataset):
    """加载图像，可选加载按文件名 stem 匹配的 GT mask。"""

    def __init__(self, input_dir, gt_dir, img_size):
        self.input_dir = Path(input_dir)
        self.img_size = img_size

        # 收集输入图像（单层目录，不递归）
        self.image_paths = sorted(
            p
            for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        # 构建 GT stem -> path 映射
        self.gt_map = {}
        if gt_dir is not None:
            gt_dir = Path(gt_dir)
            for p in gt_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    self.gt_map[p.stem] = p

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            img_tensor = self.transform(img)

        # 查找匹配的 GT（按 stem 匹配，格式可不同）
        gt_array = None
        if self.gt_map:
            gt_path = self.gt_map.get(img_path.stem)
            if gt_path is not None:
                with Image.open(gt_path) as gt:
                    gt = gt.convert("L")
                    gt = gt.resize((self.img_size, self.img_size), resample=Image.NEAREST)
                    # 任何非零像素视为前景
                    gt_array = (np.array(gt) > 0).astype(np.uint8)

        return {
            "image": img_tensor,
            "filename": img_path.name,
            "stem": img_path.stem,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "gt": gt_array,
        }


def collate_fn(batch):
    """自定义 collate_fn，处理 GT 为 None 的情况。"""
    images = torch.stack([item["image"] for item in batch])
    filenames = [item["filename"] for item in batch]
    stems = [item["stem"] for item in batch]
    orig_ws = torch.tensor([item["orig_w"] for item in batch])
    orig_hs = torch.tensor([item["orig_h"] for item in batch])
    gts = [item["gt"] for item in batch]
    return {
        "image": images,
        "filename": filenames,
        "stem": stems,
        "orig_w": orig_ws,
        "orig_h": orig_hs,
        "gt": gts,
    }


# =========================
# Checkpoint 加载
# =========================
def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 兼容 DataParallel 保存的 checkpoint
    if len(state_dict) > 0 and next(iter(state_dict)).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return missing_keys, unexpected_keys


# =========================
# 掩码保存
# =========================
def save_binary_mask(mask_uint8, output_path, orig_size=None):
    """保存二值 mask（0/255），可选 resize 回原始尺寸。"""
    if orig_size is not None:
        ow, oh = orig_size
        if (mask_uint8.shape[1], mask_uint8.shape[0]) != (ow, oh):
            mask_uint8 = np.array(
                Image.fromarray(mask_uint8).resize((ow, oh), resample=Image.NEAREST)
            ).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_uint8).save(str(output_path))


# =========================
# 日志输出（同时写终端和文件）
# =========================
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        self._closed = False

    def write(self, message):
        self.terminal.write(message)
        if not self._closed:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self._closed:
            self.log.flush()

    def close(self):
        if not self._closed:
            self.log.close()
            self._closed = True


# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser(
        "DINOv3-UNet Standalone Inference",
        description="分割推理 + 可选掩码输出 + 可选指标计算 (Dice/HD95/ECE + CI95)",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重文件路径 (.pth)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="输入图像目录")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="GT mask 目录（可选，提供后计算指标；按文件名 stem 匹配）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="预测掩码输出目录（可选，不提供则不输出掩码）")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="指标 log 文件保存目录（默认: ./logs，仅当提供 --gt_dir 时生成）")
    parser.add_argument("--img_size", type=int, default=224,
                        help="模型输入图像尺寸 (默认: 224)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="推理 batch size (默认: 4)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader 工作进程数 (默认: 4)")
    parser.add_argument("--device", type=str, default=None,
                        help='设备，如 "cuda", "cuda:0", "cpu"')
    parser.add_argument("--dino_pretrained", type=str, default="false",
                        help="是否加载 DINO 预训练权重 (true/false, 默认: false)")
    parser.add_argument("--use_dilation", type=str, default="false",
                        help="是否使用 dilation 模块 (true/false, 默认: false)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="二值化阈值 (默认: 0.5)")
    parser.add_argument("--save_orig_size", type=str, default="true",
                        help="是否将预测掩码 resize 回原始尺寸 (true/false, 默认: true)")
    parser.add_argument("--n_boot", type=int, default=5000,
                        help="Bootstrap CI95 迭代次数 (默认: 5000)")
    parser.add_argument("--ci", type=float, default=0.95,
                        help="置信区间水平 (默认: 0.95)")
    args = parser.parse_args()

    # 清理路径
    args.input_dir = clean_path(args.input_dir)
    args.checkpoint = clean_path(args.checkpoint)
    if args.gt_dir:
        args.gt_dir = clean_path(args.gt_dir)
    if args.output_dir:
        args.output_dir = clean_path(args.output_dir)
    args.log_dir = clean_path(args.log_dir)
    args.dino_pretrained = str2bool(args.dino_pretrained)
    args.use_dilation = str2bool(args.use_dilation)
    args.save_orig_size = str2bool(args.save_orig_size)

    # 校验输入
    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"--input_dir 必须是目录: {args.input_dir}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint 不存在: {args.checkpoint}")
    if args.gt_dir and not os.path.isdir(args.gt_dir):
        raise NotADirectoryError(f"--gt_dir 必须是目录: {args.gt_dir}")

    compute_metrics = args.gt_dir is not None
    save_masks = args.output_dir is not None

    # 设备
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 设置 log（仅当计算指标时）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = None
    json_file = None
    original_stdout = sys.stdout

    if compute_metrics:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = os.path.join(args.log_dir, f"infer_{timestamp}.log")
        json_file = os.path.join(args.log_dir, f"infer_{timestamp}_metrics.json")
        sys.stdout = TeeLogger(log_file)

    # 打印配置
    print("=" * 60)
    print(f"权重:     {args.checkpoint}")
    print(f"数据:     {args.input_dir}")
    print(f"GT:       {args.gt_dir if args.gt_dir else '(无)'}")
    print(f"设备:     {device}")
    print("=" * 60)

    # 数据集
    dataset = InferenceDataset(args.input_dir, args.gt_dir, args.img_size)
    if len(dataset) == 0:
        print("错误: 输入目录中未找到图像文件。")
        _restore_stdout(original_stdout)
        return

    # 检查 GT 匹配情况
    if compute_metrics:
        matched = sum(1 for p in dataset.image_paths if p.stem in dataset.gt_map)
        if matched == 0:
            print("警告: 没有图像匹配到 GT，将跳过指标计算。")
            compute_metrics = False

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # 模型
    model = DINOv3_S_UNet(
        pretrained=args.dino_pretrained, use_dilation=args.use_dilation
    ).to(device)
    model.eval()

    load_checkpoint(model, args.checkpoint, device)

    # 指标计算器
    dice_calculator = Dice()
    hd_calculator = HD95()
    ece_calculator = ECE()

    all_dice_values = []
    all_hd_values = []
    all_ece_values = []
    case_records = []

    # 推理
    start_time = time.time()
    total = len(dataset)
    done = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="推理", unit="batch"):
            images = batch["image"].to(device)
            filenames = batch["filename"]
            stems = batch["stem"]
            orig_ws = batch["orig_w"]
            orig_hs = batch["orig_h"]

            pred = model(images)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

            probs = torch.sigmoid(pred)  # (B, 1, H, W)
            masks_binary = (probs > args.threshold).float()  # (B, 1, H, W)

            for i in range(masks_binary.shape[0]):
                prob_np = probs[i, 0].cpu().numpy()  # (H, W)
                mask_np = masks_binary[i, 0].cpu().numpy()  # (H, W), float 0/1
                mask_uint8 = (mask_np * 255).astype(np.uint8)

                ow = int(orig_ws[i].item())
                oh = int(orig_hs[i].item())
                filename = filenames[i]
                stem = stems[i]

                # 保存掩码
                if save_masks:
                    name = Path(filename).stem + ".png"
                    out_path = os.path.join(args.output_dir, name)
                    if args.save_orig_size:
                        save_binary_mask(mask_uint8, out_path, orig_size=(ow, oh))
                    else:
                        save_binary_mask(mask_uint8, out_path)

                # 计算指标
                if compute_metrics:
                    gt_array = batch["gt"][i]

                    if gt_array is None:
                        case_records.append({
                            "filename": str(filename),
                            "stem": str(stem),
                            "skipped": True,
                            "reason": "no matching GT",
                            "dice": None,
                            "hd95": None,
                            "ece": None,
                        })
                    else:
                        gt_bool = gt_array.astype(bool)
                        pred_bool = mask_np > 0.5

                        if not np.any(gt_bool):
                            # GT 为空，跳过该 case（保留现有行为）
                            case_records.append({
                                "filename": str(filename),
                                "stem": str(stem),
                                "skipped": True,
                                "reason": "empty GT (no foreground)",
                                "dice": None,
                                "hd95": None,
                                "ece": None,
                            })
                        else:
                            # 转为 tensor 供指标计算器使用
                            pred_tensor = torch.from_numpy(
                                pred_bool.astype(np.float32)
                            )
                            gt_tensor = torch.from_numpy(
                                gt_bool.astype(np.float32)
                            )
                            prob_tensor = torch.from_numpy(
                                prob_np.astype(np.float32)
                            )

                            # Dice
                            dice_i = dice_calculator(pred_tensor, gt_tensor)
                            dice_value = (
                                None if dice_i is None else float(dice_i.item())
                            )

                            # HD95
                            hd_value = None
                            try:
                                hd_i = hd_calculator(pred_tensor, gt_tensor)
                                if hd_i is not None:
                                    hd_value = float(hd_i.item())
                            except Exception as e:
                                print(
                                    f"[警告] HD95 计算失败 "
                                    f"({filename}): {e}"
                                )

                            # ECE
                            ece_value = None
                            try:
                                ece_i = ece_calculator(prob_tensor, gt_tensor)
                                ece_value = float(ece_i.item())
                            except Exception as e:
                                print(
                                    f"[警告] ECE 计算失败 "
                                    f"({filename}): {e}"
                                )

                            if dice_value is not None:
                                all_dice_values.append(dice_value)
                            if hd_value is not None:
                                all_hd_values.append(hd_value)
                            if ece_value is not None:
                                all_ece_values.append(ece_value)

                            case_records.append({
                                "filename": str(filename),
                                "stem": str(stem),
                                "skipped": False,
                                "reason": None,
                                "dice": round(dice_value, 4) if dice_value is not None else None,
                                "hd95": round(hd_value, 4) if hd_value is not None else None,
                                "ece": round(ece_value, 4) if ece_value is not None else None,
                            })

                done += 1

    elapsed = time.time() - start_time

    # 汇总指标 + CI95
    if compute_metrics:
        dice_mean, dice_ci95 = bootstrap_ci(
            all_dice_values, n_boot=args.n_boot, ci=args.ci, seed=0
        )
        hd95_mean, hd95_ci95 = bootstrap_ci(
            all_hd_values, n_boot=args.n_boot, ci=args.ci, seed=0
        )

        n_evaluated = len(all_dice_values)

        print("=" * 60)
        print(f"评估样本数: {n_evaluated}")
        print(f"Dice:  {dice_mean:.4f}  (95% CI: [{dice_ci95[0]:.4f}, {dice_ci95[1]:.4f}])")
        print(f"HD95:  {hd95_mean:.4f}  (95% CI: [{hd95_ci95[0]:.4f}, {hd95_ci95[1]:.4f}])")
        print("=" * 60)

        # 保存 JSON（逐样本明细，不打印）
        results = {
            "timestamp": timestamp,
            "checkpoint": args.checkpoint,
            "input_dir": args.input_dir,
            "gt_dir": args.gt_dir,
            "total_images": total,
            "evaluated_cases": n_evaluated,
            "Dice": {
                "mean": round(dice_mean, 4),
                "CI95": [round(dice_ci95[0], 4), round(dice_ci95[1], 4)],
                "values": [round(float(v), 4) for v in all_dice_values],
            },
            "HD95": {
                "mean": round(hd95_mean, 4),
                "CI95": [round(hd95_ci95[0], 4), round(hd95_ci95[1], 4)],
                "values": [round(float(v), 4) for v in all_hd_values],
            },
            "cases": case_records,
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=json_default)

    # 恢复 stdout
    _restore_stdout(original_stdout)


def _restore_stdout(original_stdout):
    """安全恢复原始 stdout。"""
    if isinstance(sys.stdout, TeeLogger):
        sys.stdout.close()
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
