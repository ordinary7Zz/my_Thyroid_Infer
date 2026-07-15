"""
UltraFedFM segmentation inference.

Usage examples:

  # Only output predicted masks
  python segment.py --data_path /path/to/images --resume /path/to/ckpt.pth \
      --output_dir /path/to/predicted_masks

  # Output masks + compute Dice/HD95 against GT
  python segment.py --data_path /path/to/images --resume /path/to/ckpt.pth \
      --output_dir /path/to/predicted_masks --gt_dir /path/to/gt_masks \
      --output_log metrics.log

  # Only compute metrics (no mask output)
  python segment.py --data_path /path/to/images --resume /path/to/ckpt.pth \
      --gt_dir /path/to/gt_masks --output_log metrics.log

  # Neither masks nor metrics (rare, but accepted — no output)
  python segment.py --data_path /path/to/images --resume /path/to/ckpt.pth
"""

import os
import sys
import cv2
import torch
import argparse
import datetime
import numpy as np
import torch.nn.functional as F
import albumentations as A

# 使用项目级统一指标模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from seg_metrics import compute_dice, compute_hd95, bootstrap_ci

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp


SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FlatImageDataset(Dataset):
    """Load all images from a flat directory (no subdirectories, no labels)."""

    def __init__(self, root, img_size=224):
        self.root = root
        self.img_size = img_size
        self.image_paths = []
        self.image_names = []
        for fname in sorted(os.listdir(root)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                self.image_paths.append(os.path.join(root, fname))
                self.image_names.append(fname)

        self.transform = A.Compose([
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        name = self.image_names[index]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        image = self.transform(image=image)['image']
        return image, name, orig_h, orig_w


# ---------------------------------------------------------------------------
# GT mask loading
# ---------------------------------------------------------------------------
def find_gt_mask(gt_dir, basename):
    """Find a GT mask file in gt_dir matching the given basename (any extension).

    支持大小写不敏感的扩展名匹配（如 .PNG / .png 均可）。
    """
    stem = os.path.splitext(basename)[0]
    for ext in SUPPORTED_EXTS:
        candidate = os.path.join(gt_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    for ext in SUPPORTED_EXTS:
        candidate = os.path.join(gt_dir, stem + ext.upper())
        if os.path.isfile(candidate):
            return candidate
    # also try exact name
    candidate = os.path.join(gt_dir, basename)
    if os.path.isfile(candidate):
        return candidate
    return None


def load_gt_mask(gt_dir, basename):
    """Load a GT mask and binarize it (>0 → foreground=1)."""
    path = find_gt_mask(gt_dir, basename)
    if path is None:
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = (mask > 0).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Metrics — 使用项目级统一指标模块 (seg_metrics)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(data_loader, model, device, threshold, output_dir, gt_dir):
    """Run inference. Returns lists of (dice, hd95) for matched samples.

    If output_dir is None, masks are not saved.
    If gt_dir is None, metrics are not computed (returns empty lists).
    """
    model.eval()
    dice_list = []
    hd95_list = []
    per_sample = []  # (filename, dice, hd95)

    for images, names, orig_hs, orig_ws in tqdm(data_loader, desc="推理"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)  # (B, 1, H, W), activation='sigmoid' already applied

        for i in range(images.size(0)):
            name = names[i]
            orig_h, orig_w = orig_hs[i].item(), orig_ws[i].item()

            pred = outputs[i]  # (1, H, W)
            pred = F.interpolate(
                pred.unsqueeze(0), size=(orig_h, orig_w),
                mode='bilinear', align_corners=False
            )
            pred = pred.squeeze().cpu().numpy()  # (orig_H, orig_W)
            mask = (pred > threshold).astype(np.uint8)

            # --- save mask ---
            if output_dir is not None:
                out_name = os.path.splitext(name)[0] + '.png'
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, mask * 255)

            # --- compute metrics ---
            if gt_dir is not None:
                gt_mask = load_gt_mask(gt_dir, name)
                if gt_mask is not None:
                    # resize gt to match pred if needed (should match if from same source)
                    if gt_mask.shape[0] != orig_h or gt_mask.shape[1] != orig_w:
                        gt_mask = cv2.resize(gt_mask, (orig_w, orig_h),
                                             interpolation=cv2.INTER_NEAREST)

                    # 统一 resize 到 224×224 计算指标
                    mask_224 = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
                    gt_224 = cv2.resize(gt_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
                    dice = compute_dice(mask_224, gt_224)
                    hd95 = compute_hd95(mask_224, gt_224)
                    dice_list.append(dice)
                    hd95_list.append(hd95)
                    per_sample.append((name, dice, hd95))

    return dice_list, hd95_list, per_sample


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('UltraFedFM standalone segmentation inference')

    parser.add_argument('--data_path', required=True, type=str,
                        help='Flat directory containing images')
    parser.add_argument('--resume', required=True, type=str,
                        help='Path to segmentation checkpoint .pth file')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Output directory for predicted masks (omit to skip mask output)')
    parser.add_argument('--gt_dir', default=None, type=str,
                        help='Directory of ground-truth masks (omit to skip metrics)')
    parser.add_argument('--output_log', default=None, type=str,
                        help='Output log path for Dice/HD95 metrics (default: seg_metrics_<timestamp>.log)')

    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='Binarization threshold (default: 0.5)')
    parser.add_argument('--n_bootstrap', default=2000, type=int,
                        help='Number of bootstrap iterations for CI95')

    return parser


def main():
    args = get_args_parser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- output dir ---
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- dataset ---
    dataset = FlatImageDataset(args.data_path, img_size=args.img_size)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    # --- model ---
    model = smp.Unet(
        encoder_name='mae',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid',
    )
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device)

    # --- 打印配置 ---
    print("=" * 60)
    print(f"权重:     {args.resume}")
    print(f"数据:     {args.data_path}")
    print(f"GT:       {args.gt_dir if args.gt_dir else '(无)'}")
    print(f"设备:     {device}")
    print("=" * 60)

    # --- inference ---
    dice_list, hd95_list, per_sample = run_inference(
        data_loader, model, device, args.threshold,
        args.output_dir, args.gt_dir
    )

    # --- metrics ---
    if args.gt_dir is not None and len(dice_list) > 0:
        if args.output_log is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output_log = 'seg_metrics_{}.log'.format(timestamp)

        dice_mean, dice_ci_lo, dice_ci_hi = bootstrap_ci(dice_list, n_boot=args.n_bootstrap)
        hd95_mean, hd95_ci_lo, hd95_ci_hi = bootstrap_ci(hd95_list, n_boot=args.n_bootstrap)

        print("=" * 60)
        print(f"评估样本数: {len(dice_list)}")
        print(f"Dice:  {dice_mean:.4f}  (95% CI: [{dice_ci_lo:.4f}, {dice_ci_hi:.4f}])")
        print(f"HD95:  {hd95_mean:.4f}  (95% CI: [{hd95_ci_lo:.4f}, {hd95_ci_hi:.4f}])")
        print("=" * 60)

        os.makedirs(os.path.dirname(os.path.abspath(args.output_log)), exist_ok=True)
        with open(args.output_log, 'w', encoding='utf-8') as f:
            f.write(f"评估样本数: {len(dice_list)}\n")
            f.write(f"Dice:  {dice_mean:.4f}  (95% CI: [{dice_ci_lo:.4f}, {dice_ci_hi:.4f}])\n")
            f.write(f"HD95:  {hd95_mean:.4f}  (95% CI: [{hd95_ci_lo:.4f}, {hd95_ci_hi:.4f}])\n")
            f.write("\n--- Per-Sample Metrics ---\n")
            f.write("filename,dice,hd95\n")
            for fname, dsc, hd in per_sample:
                f.write(f"{fname},{dsc:.4f},{hd:.4f}\n")
    elif args.gt_dir is not None and len(dice_list) == 0:
        print('No GT masks matched the image filenames — skipping metrics')



if __name__ == '__main__':
    main()
