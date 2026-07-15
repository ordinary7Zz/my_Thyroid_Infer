# -*- coding: utf-8 -*-
"""
MedSegX Inference — minimal self-contained toolkit.

Usage
-----
python inference.py \
    --input_dir     /path/to/images \
    --task_name     US_GlndThyroid \
    --checkpoint    /path/to/SAM \
    --model_weight  /path/to/medsegx_vit_b.pth \
    --log_file      ./result.log

Optional:
    --gt_dir        /path/to/gt_masks   # if provided, compute DSC + HD95 + CI95
    --output_dir    /path/to/pred_masks # if provided, save predicted masks as PNG
    --model_type    vit_b               # vit_b / vit_l / vit_h
    --device        cuda:0
    --n_boot        2000                # bootstrap iterations for CI95
    --ci            95                  # confidence level

Outputs
-------
1. Predicted masks  (only if --output_dir is given):  PNG, 0/255 binary.
2. Metrics + CI95    (only if --gt_dir is given):      written to --log_file.
3. If neither is given, the script runs inference only (no file output).
"""

import argparse
import os
import sys
import time

join = os.path.join

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm

from segment_anything import sam_model_registry, sam_model_checkpoint
from segment_anything.utils.transforms import ResizeLongestSide
from model import MedSAM, MedSegX
from data.datainfo import (
    modal_dict,
    modal_map,
    organ_level_1_dict,
    organ_level_1_map,
    organ_level_2_dict,
    organ_level_2_map,
    organ_level_3_dict,
    organ_level_3_map,
    task_idx,
)
from utils.metrics import dice_coeff, hd95
# 使用项目级统一指标模块（覆盖 bootstrap_ci 实现）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from seg_metrics import bootstrap_ci

IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
MASK_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


# ────────────────────────── helpers ──────────────────────────

def parse_task(task_name: str):
    """Parse 'US_GlndThyroid' → (modal_idx, (o1, o2, o3, o4))."""
    modal = task_name.split('_')[0]
    organ = ('').join(task_name.split('_')[1:]).rstrip('0123456789')

    modal_idx = modal_map[modal_dict[modal]]
    o1 = next(organ_level_1_map[k] for k, v in organ_level_1_dict.items() if organ in v)
    o2 = next(organ_level_2_map[k] for k, v in organ_level_2_dict.items() if organ in v)
    o3 = next(organ_level_3_map[k] for k, v in organ_level_3_dict.items() if organ in v)
    o4 = task_idx[organ]
    return modal_idx, (o1, o2, o3, o4)


def load_image(path: str) -> np.ndarray:
    """Load an image as HWC float32 RGB."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


def load_mask(path: str, h: int, w: int) -> np.ndarray:
    """Load a GT mask, binarise and resize to (h, w)."""
    mask = Image.open(path).convert("L")
    mask = mask.resize((w, h), Image.NEAREST)
    mask = np.array(mask, dtype=np.float32) / 255.0
    return (mask > 0.5).astype(np.uint8)


def find_gt_path(gt_dir: str, basename: str):
    """Try common extensions for a GT mask matching *basename*.

    支持大小写不敏感的扩展名匹配（如 .PNG / .png 均可）。
    """
    for ext in sorted(MASK_EXTENSIONS):
        candidate = join(gt_dir, basename + ext)
        if os.path.exists(candidate):
            return candidate
    for ext in sorted(MASK_EXTENSIONS):
        candidate = join(gt_dir, basename + ext.upper())
        if os.path.exists(candidate):
            return candidate
    return None


def mask_to_box(mask: np.ndarray, perturb: int = 0, rng=None):
    """Extract bounding box from a binary mask [x1, y1, x2, y2].

    If perturb > 0, randomly expand each side by [0, perturb] pixels
    (consistent with data/dataset_copy.py).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return np.array([0, 0, w, h], dtype=np.float32)
    x_min, x_max = int(np.min(xs)), int(np.max(xs))
    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    if perturb > 0:
        r = rng or np.random
        H, W = mask.shape[:2]
        if hasattr(r, "integers"):
            p = lambda: r.integers(0, perturb + 1)
        else:
            p = lambda: r.randint(0, perturb)
        x_min = max(0, x_min - p())
        x_max = min(W, x_max + p())
        y_min = max(0, y_min - p())
        y_max = min(H, y_max + p())
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


@torch.no_grad()
def infer_single(model, image_np: np.ndarray, modal: int, organ: tuple,
                 img_size: int, device: torch.device, box=None):
    """Run inference on a single image, return (H, W) uint8 binary mask.

    If *box* is None, use the full image as the box prompt.
    Otherwise *box* should be [x1, y1, x2, y2] in original image coords.
    """
    model.eval()
    h_orig, w_orig = image_np.shape[:2]

    # Image → tensor
    img_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # C,H,W
    if box is None:
        box = np.array([0, 0, w_orig, h_orig], dtype=np.float32)
    box = torch.tensor([box], dtype=torch.float32)

    # Transform box & image to model input size
    box_transform = ResizeLongestSide(img_size)
    box = box_transform.apply_boxes_torch(
        box.reshape(-1, 2, 2), (h_orig, w_orig)).reshape(-1, 4)

    img_resize = Resize((img_size, img_size), antialias=True)
    img_tensor = img_resize(img_tensor.unsqueeze(0))  # (1,3,img_size,img_size)

    img_tensor = img_tensor.to(device)
    box = box.to(device)
    img_tensor = model.sam.preprocess(img_tensor)

    # Prompt encoder
    sparse_emb, dense_emb = model.sam.prompt_encoder(
        points=None, boxes=box[:, None, :], masks=None)

    # Modal & organ embedding
    batch_size = img_tensor.shape[0]
    modal_t = torch.tensor([modal], dtype=torch.long, device=device)
    modal_index = model.sam.image_encoder.modal_index[modal_t]
    modal_embed = model.sam.image_encoder.modal_embed(modal_index)

    o1, o2, o3, o4 = organ
    o1, o2, o3, o4 = [torch.tensor([x], dtype=torch.long, device=device)
                      for x in (o1, o2, o3, o4)]
    organ_index_0 = torch.zeros(batch_size, dtype=torch.long, device=device)
    organ_embed = (
        model.sam.image_encoder.organ_embed[0](organ_index_0),
        model.sam.image_encoder.organ_embed[1](
            model.sam.image_encoder.organ_index_1[o1]),
        model.sam.image_encoder.organ_embed[2](
            model.sam.image_encoder.organ_index_2[o2]),
        model.sam.image_encoder.organ_embed[3](
            model.sam.image_encoder.organ_index_3[o3]),
        model.sam.image_encoder.organ_embed[4](o4),
    )

    # Image encoder
    image_embedding, _ = model.sam.image_encoder(
        img_tensor, modal_embed, organ_embed)

    # Mask decoder
    mask_pred, iou_pred = model.sam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=True,
    )

    # Pick best mask by IoU prediction
    best_idx = iou_pred.argmax(dim=1)
    mask_prob = torch.sigmoid(mask_pred)
    chosen = mask_prob[0, best_idx[0]]  # (img_size, img_size)
    chosen = (chosen > 0.5).to(torch.uint8)

    # Resize back to original resolution
    chosen = chosen.unsqueeze(0).unsqueeze(0).float()
    chosen = F.interpolate(chosen, size=(h_orig, w_orig),
                           mode="bilinear", antialias=True)
    mask_np = (chosen.squeeze() > 0.5).to(torch.uint8).cpu().numpy()
    return mask_np


# ────────────────────────── main ──────────────────────────

def main():
    parser = argparse.ArgumentParser(
        "MedSegX Inference",
        description="Run MedSegX segmentation inference. "
                    "Optionally compute DSC/HD95 with CI95 when GT is provided.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- data ---
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input image directory (PNG/JPG/BMP/TIFF)")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="GT mask directory. If provided, compute DSC + HD95 + CI95. "
                             "Masks matched by filename (without extension).")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Task name, e.g. US_GlndThyroid, US_ThyroidNodule")

    # --- model ---
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="SAM checkpoint directory (containing sam_vit_b_01ec64.pth etc.)")
    parser.add_argument("--model_type", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM backbone scale")
    parser.add_argument("--model_weight", type=str, required=True,
                        help="MedSegX weight file (.pth)")
    parser.add_argument("--method", type=str, default="medsegx",
                        choices=["medsegx", "medsam"],
                        help="Model method")
    parser.add_argument("--bottleneck_dim", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--expert_num", type=int, default=4)

    # --- output ---
    parser.add_argument("--output_dir", type=str, default=None,
                        help="If provided, save predicted masks as PNG to this directory.")
    parser.add_argument("--log_file", type=str, required=True,
                        help="Log file path for metrics and CI95 results.")

    # --- runtime ---
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_boot", type=int, default=2000,
                        help="Bootstrap iterations for CI95")
    parser.add_argument("--ci", type=float, default=95,
                        help="Confidence interval level (e.g. 95)")

    # --- box prompt ---
    parser.add_argument("--box_mode", type=str, default="full",
                        choices=["full", "gt"],
                        help="Box prompt mode: 'full' = full image box [0,0,W,H]; "
                             "'gt' = bounding box extracted from GT mask (requires --gt_dir)")
    parser.add_argument("--box_perturb", type=int, default=20,
                        help="Max perturbation (pixels) added to GT box on each side. "
                             "Only used when --box_mode gt. Set 0 for exact GT box.")
    parser.add_argument("--box_seed", type=int, default=42,
                        help="Random seed for GT box perturbation (reproducibility)")

    args = parser.parse_args()

    # ── Collect images ──
    img_files = sorted([
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ])
    if not img_files:
        raise RuntimeError(f"No image files found in {args.input_dir} "
                           f"(supported: {IMG_EXTENSIONS})")

    # ── Parse task ──
    modal, organ = parse_task(args.task_name)

    # ── Load model ──
    device = torch.device(args.device)
    sam_ckpt = join(args.checkpoint, sam_model_checkpoint[args.model_type])
    sam_model = sam_model_registry[args.model_type](
        image_size=256, keep_resolution=True, checkpoint=sam_ckpt)

    if args.method == "medsegx":
        model = MedSegX(sam_model, bottleneck_dim=args.bottleneck_dim,
                        embedding_dim=args.embedding_dim,
                        expert_num=args.expert_num).to(device)
    elif args.method == "medsam":
        model = MedSAM(sam_model).to(device)
    else:
        raise NotImplementedError(f"Unsupported method: {args.method}")

    ckpt = torch.load(args.model_weight, map_location=device)
    model.load_parameters(ckpt["model"])

    img_size = model.sam.image_encoder.img_size

    # ── Prepare output dir if requested ──
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    compute_metrics = args.gt_dir is not None

    use_gt_box = (args.box_mode == "gt")
    box_rng = None
    if use_gt_box:
        if args.gt_dir is None:
            raise ValueError("--box_mode gt requires --gt_dir to be provided")
        box_rng = np.random.default_rng(args.box_seed)

    # ── 打印配置 ──
    print("=" * 60)
    print(f"权重:     {args.model_weight}")
    print(f"数据:     {args.input_dir}")
    print(f"GT:       {args.gt_dir if args.gt_dir else '(无)'}")
    print(f"设备:     {device}")
    print("=" * 60)

    # ── Inference loop ──
    dsc_list, hd95_list = [], []
    per_sample = []  # (filename, dice, hd95)
    skipped = 0

    for fname in tqdm(img_files, desc="推理"):
        img_path = join(args.input_dir, fname)
        image_np = load_image(img_path)
        h, w = image_np.shape[:2]

        # Load GT mask (used for both GT box prompt and metric evaluation)
        gt_mask = None
        if use_gt_box or compute_metrics:
            basename = os.path.splitext(fname)[0]
            gt_path = find_gt_path(args.gt_dir, basename)
            if gt_path is None:
                skipped += 1
                continue
            gt_mask = load_mask(gt_path, h, w)

        # Determine box prompt
        box = None
        if use_gt_box:
            box = mask_to_box(gt_mask, perturb=args.box_perturb, rng=box_rng)

        # Run model
        pred_mask = infer_single(model, image_np, modal, organ, img_size, device, box=box)

        # Save predicted mask (optional)
        if args.output_dir:
            base = os.path.splitext(fname)[0]
            out_path = join(args.output_dir, f"{base}.png")
            Image.fromarray(pred_mask * 255).save(out_path)

        # Evaluate (optional)
        if compute_metrics:
            # 统一 resize 到 224×224 计算指标
            pred_224 = np.array(Image.fromarray(pred_mask.astype(np.uint8)).resize(
                (224, 224), Image.NEAREST))
            gt_224 = np.array(Image.fromarray(gt_mask.astype(np.uint8)).resize(
                (224, 224), Image.NEAREST))
            dsc = dice_coeff(pred_224, gt_224)
            hd = hd95(pred_224, gt_224)
            dsc_list.append(dsc)
            hd95_list.append(hd)
            per_sample.append((fname, dsc, hd))

    # ── Summary ──
    evaluated = len(dsc_list)
    mean_dsc = mean_hd = float('nan')
    dsc_lo = dsc_hi = float('nan')
    hd_lo = hd_hi = float('nan')

    if compute_metrics and evaluated > 0:
        dsc_arr = np.array(dsc_list, dtype=float)
        hd95_arr = np.array(hd95_list, dtype=float)

        # Compute bootstrap CI (with fallback to simple mean)
        try:
            mean_dsc, dsc_lo, dsc_hi = bootstrap_ci(
                dsc_arr, n_boot=args.n_boot, ci=args.ci / 100.0)
            mean_hd, hd_lo, hd_hi = bootstrap_ci(
                hd95_arr, n_boot=args.n_boot, ci=args.ci / 100.0)
        except Exception as e:
            print(f"[Warning] bootstrap_ci failed: {e}")
            mean_dsc = float(np.nanmean(dsc_arr))
            mean_hd = float(np.nanmean(hd95_arr))
            dsc_lo = dsc_hi = mean_dsc
            hd_lo = hd_hi = mean_hd

    # ── 统一输出 ──
    if compute_metrics and evaluated > 0:
        print("=" * 60)
        print(f"评估样本数: {evaluated}")
        print(f"Dice:  {mean_dsc:.4f}  (95% CI: [{dsc_lo:.4f}, {dsc_hi:.4f}])")
        print(f"HD95:  {mean_hd:.4f}  (95% CI: [{hd_lo:.4f}, {hd_hi:.4f}])")
        print("=" * 60)

        # 写 log 文件（仅指标）
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        with open(args.log_file, 'w', encoding='utf-8') as f:
            f.write(f"评估样本数: {evaluated}\n")
            f.write(f"Dice:  {mean_dsc:.4f}  (95% CI: [{dsc_lo:.4f}, {dsc_hi:.4f}])\n")
            f.write(f"HD95:  {mean_hd:.4f}  (95% CI: [{hd_lo:.4f}, {hd_hi:.4f}])\n")
            f.write("\n--- Per-Sample Metrics ---\n")
            f.write("filename,dice,hd95\n")
            for fname, dsc, hd in per_sample:
                f.write(f"{fname},{dsc:.4f},{hd:.4f}\n")


if __name__ == "__main__":
    main()
