"""
UltraFedFM classification inference (binary or multi-class, e.g. TIRADS 5-class).

Usage examples:

  # Without labels — just produce predictions CSV
  python classify.py --data_path /path/to/images --resume /path/to/ckpt.pth \
      --nb_classes 2 --output_csv predictions.csv

  # With labels — also produce a metrics log (AUROC, AUPRC, etc. + CI95)
  python classify.py --data_path /path/to/images --resume /path/to/ckpt.pth \
      --nb_classes 5 --label_file labels.json --label_field tirads \
      --output_csv predictions.csv --output_log metrics.log
"""

import os
import sys
import csv
import json
import math
import torch
import argparse
import datetime
import numpy as np
import torch.backends.cudnn as cudnn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm

import models_vit

# 使用项目级统一分类指标模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from cls_metrics import compute_all_metrics


SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FlatImageDataset(Dataset):
    """Load all images from a flat directory (no subdirectories)."""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        for fname in sorted(os.listdir(root)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                self.image_paths.append(os.path.join(root, fname))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, path


# ---------------------------------------------------------------------------
# Transform (inlined from util/datasets.py to avoid external dependency)
# ---------------------------------------------------------------------------
def build_eval_transform(input_size):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(t)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(model, resume_path, nb_classes):
    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

    # 过滤掉形状不匹配的参数（如 head 层类别数不一致）
    model_state_dict = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in checkpoint_model.items():
        if k in model_state_dict and v.shape != model_state_dict[k].shape:
            skipped.append((k, list(v.shape), list(model_state_dict[k].shape)))
        else:
            filtered[k] = v

    if skipped:
        print("跳过形状不匹配的参数:")
        for k, old_shape, new_shape in skipped:
            print(f"  {k}: checkpoint {old_shape} -> model {new_shape}")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"缺失的参数: {missing}")
    if unexpected:
        print(f"多余的参数: {unexpected}")
    print('Loaded checkpoint from {}'.format(resume_path))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(data_loader, model, device, nb_classes):
    model.eval()
    all_paths = []
    all_preds = []
    all_scores = []  # list of [prob_0, prob_1, ...]

    for images, paths in tqdm(data_loader, desc="推理"):
        images = images.to(device, non_blocking=True)
        output = model(images)
        probabilities = torch.softmax(output, dim=1)
        preds = torch.argmax(probabilities, dim=1)

        all_paths.extend(list(paths))
        all_preds.extend(preds.cpu().flatten().tolist())
        all_scores.extend(probabilities.cpu().tolist())

    return all_paths, all_preds, all_scores


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------
def load_labels(label_file, label_field, image_names, nb_classes):
    """Load labels from JSON, keyed by image filename.

    自动检测标签偏移：若标签范围为 [1, nb_classes] 则减 1 转为 0-indexed。
    例如 TIRADS 标签 1-5 → 0-4，与模型输出对齐。

    Returns a list of true labels aligned with image_names (None if not found).
    """
    with open(label_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    label_map = {}
    raw_labels = []
    for rec in records:
        fname = rec['filename']
        if label_field in rec:
            label_val = int(rec[label_field])
            if label_val < 0:
                continue
            label_map[fname] = label_val
            raw_labels.append(label_val)

    # 自动检测偏移
    offset = 0
    if raw_labels:
        min_val = min(raw_labels)
        max_val = max(raw_labels)
        if min_val == 1 and max_val == nb_classes:
            offset = 1
            print(f"[自动检测] 标签范围 [{min_val}, {max_val}]，自动偏移 {offset}（{min_val}->{min_val - offset}, {max_val}->{max_val - offset}）")

    true_labels = []
    for name in image_names:
        val = label_map.get(name)
        if val is not None:
            val = val - offset
        true_labels.append(val)

    return true_labels


# ---------------------------------------------------------------------------
# Metrics — 使用项目级统一指标模块 (cls_metrics)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('UltraFedFM standalone classification inference')
    parser.add_argument('--data_path', required=True, type=str,
                        help='Flat directory containing images')
    parser.add_argument('--resume', required=True, type=str,
                        help='Path to checkpoint .pth file')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='Number of classes (2 for binary, 5 for TIRADS)')
    parser.add_argument('--label_file', default=None, type=str,
                        help='Optional JSON file with ground-truth labels')
    parser.add_argument('--label_field', default=None, type=str,
                        help='Field name in JSON for the label (e.g. malignancy, tirads)')
    parser.add_argument('--output_csv', default=None, type=str,
                        help='Output CSV path (default: predictions_<timestamp>.csv)')
    parser.add_argument('--output_log', default=None, type=str,
                        help='Output log path for metrics (default: metrics_<timestamp>.log)')

    parser.add_argument('--model', default='vit_base_patch16', type=str,
                        help='Model architecture (vit_base_patch16 / vit_large_patch16)')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.0)
    parser.add_argument('--global_pool', action='store_true', default=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--n_bootstrap', default=2000, type=int,
                        help='Number of bootstrap iterations for CI95')
    return parser


def main():
    args = get_args_parser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # --- default output paths ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_csv is None:
        args.output_csv = 'predictions_{}.csv'.format(timestamp)
    if args.output_log is None:
        args.output_log = 'metrics_{}.log'.format(timestamp)

    # --- dataset ---
    transform = build_eval_transform(args.input_size)
    dataset = FlatImageDataset(args.data_path, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    # --- model ---
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    load_model_from_checkpoint(model, args.resume, args.nb_classes)
    model.to(device)

    # --- 打印配置 ---
    print("=" * 60)
    print(f"权重:     {args.resume}")
    print(f"数据:     {args.data_path}")
    print(f"类别数:   {args.nb_classes}")
    if args.label_file:
        print(f"标签字段: {args.label_field}")
    print(f"设备:     {device}")
    print("=" * 60)

    # --- inference ---
    paths, preds, scores = run_inference(data_loader, model, device, args.nb_classes)

    image_names = [os.path.basename(p) for p in paths]

    # --- load labels if provided ---
    true_labels = None
    if args.label_file is not None:
        if args.label_field is None:
            raise ValueError('--label_field is required when --label_file is provided')
        true_labels = load_labels(args.label_file, args.label_field, image_names, args.nb_classes)

    # --- save CSV ---
    header = ['filename', 'predicted_class', 'confidence']
    for c in range(args.nb_classes):
        header.append('prob_{}'.format(c))
    if true_labels is not None:
        header.append('true_label')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (name, pred, score) in enumerate(zip(image_names, preds, scores)):
            conf = score[pred]
            row = [name, pred, '{:.6f}'.format(conf)]
            row.extend(['{:.6f}'.format(s) for s in score])
            if true_labels is not None:
                row.append(true_labels[i] if true_labels[i] is not None else '')
            writer.writerow(row)

    # --- compute & save metrics ---
    if true_labels is not None:
        valid_idx = [i for i, t in enumerate(true_labels) if t is not None]
        if len(valid_idx) == 0:
            print('No matched labels found — skipping metrics computation')
            return

        y_true = np.array([true_labels[i] for i in valid_idx])
        y_pred = np.array([preds[i] for i in valid_idx])
        y_score = np.array([scores[i] for i in valid_idx])

        metrics = compute_all_metrics(
            y_true, y_pred, y_score, args.nb_classes, n_boot=args.n_bootstrap
        )

        # 统一输出
        print("=" * 60)
        print(f"评估样本数: {len(valid_idx)}")
        for name in ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'F1', 'Recall']:
            m = metrics[name]
            print(f"{name:<12s}: {m['value']:.4f}  (95% CI: [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}])")
        print("=" * 60)

        os.makedirs(os.path.dirname(os.path.abspath(args.output_log)), exist_ok=True)
        with open(args.output_log, 'w', encoding='utf-8') as f:
            f.write(f"评估样本数: {len(valid_idx)}\n")
            for name in ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'F1', 'Recall']:
                m = metrics[name]
                f.write(f"{name:<12s}: {m['value']:.4f}  (95% CI: [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}])\n")



if __name__ == '__main__':
    main()
