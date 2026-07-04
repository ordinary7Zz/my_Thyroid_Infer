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

import models_vit


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
    checkpoint = torch.load(resume_path, map_location='cpu')
    checkpoint_model = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(checkpoint_model, strict=True)
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

    for images, paths in data_loader:
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
def load_labels(label_file, label_field, image_names):
    """Load labels from JSON, keyed by image filename.

    Returns a dict {image_name: int_label} and a list of true labels
    aligned with image_names (None if not found).
    """
    with open(label_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    label_map = {}
    for rec in records:
        fname = rec['filename']
        if label_field in rec:
            label_map[fname] = int(rec[label_field])

    true_labels = []
    for name in image_names:
        true_labels.append(label_map.get(name))

    return true_labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def bootstrap_ci(metric_fn, y_true, y_pred, y_score, n_bootstrap=2000, seed=42, alpha=0.05):
    """Bootstrap 95% confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    indices = rng.randint(0, n, size=(n_bootstrap, n))
    values = []
    for idx in indices:
        yt = y_true[idx]
        yp = y_pred[idx]
        ys = y_score[idx] if y_score is not None else None
        try:
            v = metric_fn(yt, yp, ys)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                values.append(v)
        except Exception:
            pass

    if len(values) == 0:
        return float('nan'), float('nan')
    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def compute_all_metrics(y_true, y_pred, y_score, nb_classes, n_bootstrap=2000):
    """Compute AUROC, AUPRC, Accuracy, Precision, F1, Recall (macro) + CI95."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        accuracy_score, precision_score, f1_score, recall_score,
    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # Auto-remap 1-indexed labels (e.g. TIRADS 1-5) to 0-indexed (0-4)
    # Only y_true needs remapping; y_pred is already 0-indexed from argmax
    label_min = y_true.min()
    label_max = y_true.max()
    if label_min >= 1 and label_max >= nb_classes:
        offset = label_min
        y_true = y_true - offset
        print('Auto-remapped true labels from {}-{} to 0-{}'.format(
            offset, label_max, label_max - offset))

    metrics = {}

    # --- point estimates ---
    if nb_classes == 2:
        # binary: use positive-class probability
        score_pos = y_score[:, 1]
        auroc = roc_auc_score(y_true, score_pos)
        auprc = average_precision_score(y_true, score_pos)
    else:
        auroc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        auprc = average_precision_score(
            np.eye(nb_classes)[y_true], y_score, average='macro'
        )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    metrics['AUROC'] = float(auroc)
    metrics['AUPRC'] = float(auprc)
    metrics['Accuracy'] = float(accuracy)
    metrics['Precision'] = float(precision)
    metrics['F1'] = float(f1)
    metrics['Recall'] = float(recall)

    # --- CI95 via bootstrap ---
    def _auroc_fn(yt, yp, ys):
        if nb_classes == 2:
            return roc_auc_score(yt, ys[:, 1])
        else:
            return roc_auc_score(yt, ys, multi_class='ovr', average='macro')

    def _auprc_fn(yt, yp, ys):
        if nb_classes == 2:
            return average_precision_score(yt, ys[:, 1])
        else:
            return average_precision_score(np.eye(nb_classes)[yt], ys, average='macro')

    def _acc_fn(yt, yp, ys):
        return accuracy_score(yt, yp)

    def _prec_fn(yt, yp, ys):
        return precision_score(yt, yp, average='macro', zero_division=0)

    def _f1_fn(yt, yp, ys):
        return f1_score(yt, yp, average='macro', zero_division=0)

    def _rec_fn(yt, yp, ys):
        return recall_score(yt, yp, average='macro', zero_division=0)

    fns = {
        'AUROC': _auroc_fn,
        'AUPRC': _auprc_fn,
        'Accuracy': _acc_fn,
        'Precision': _prec_fn,
        'F1': _f1_fn,
        'Recall': _rec_fn,
    }

    for name, fn in fns.items():
        lo, hi = bootstrap_ci(fn, y_true, y_pred, y_score, n_bootstrap=n_bootstrap)
        metrics['{}_CI95'.format(name)] = (lo, hi)

    return metrics


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
    print('Found {} images in {}'.format(len(dataset), args.data_path))

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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model: {}, Params: {:.2f}M'.format(args.model, n_params / 1e6))

    # --- inference ---
    paths, preds, scores = run_inference(data_loader, model, device, args.nb_classes)

    image_names = [os.path.basename(p) for p in paths]

    # --- load labels if provided ---
    true_labels = None
    if args.label_file is not None:
        if args.label_field is None:
            raise ValueError('--label_field is required when --label_file is provided')
        true_labels = load_labels(args.label_file, args.label_field, image_names)
        n_matched = sum(1 for t in true_labels if t is not None)
        print('Labels loaded: {}/{} matched'.format(n_matched, len(image_names)))

    # --- save CSV ---
    header = ['image_name', 'predicted_class', 'confidence']
    for c in range(args.nb_classes):
        header.append('prob_class_{}'.format(c))
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
    print('Saved {} predictions to {}'.format(len(image_names), args.output_csv))

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
            y_true, y_pred, y_score, args.nb_classes, n_bootstrap=args.n_bootstrap
        )

        os.makedirs(os.path.dirname(os.path.abspath(args.output_log)), exist_ok=True)
        with open(args.output_log, 'w', encoding='utf-8') as f:
            f.write('=' * 60 + '\n')
            f.write('UltraFedFM Classification Metrics\n')
            f.write('=' * 60 + '\n')
            f.write('Model: {}\n'.format(args.model))
            f.write('Checkpoint: {}\n'.format(args.resume))
            f.write('Data path: {}\n'.format(args.data_path))
            f.write('Nb classes: {}\n'.format(args.nb_classes))
            f.write('Label file: {}\n'.format(args.label_file))
            f.write('Label field: {}\n'.format(args.label_field))
            f.write('Num samples (matched): {}\n'.format(len(valid_idx)))
            f.write('Bootstrap iterations: {}\n'.format(args.n_bootstrap))
            f.write('Averaging: macro\n')
            f.write('-' * 60 + '\n')
            f.write('{:<15s} {:>10s}   {:>12s}\n'.format('Metric', 'Value', 'CI95'))
            f.write('-' * 60 + '\n')
            for name in ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'F1', 'Recall']:
                val = metrics[name]
                ci = metrics['{}_CI95'.format(name)]
                f.write('{:<15s} {:>10.4f}   [{:.4f}, {:.4f}]\n'.format(
                    name, val, ci[0], ci[1]
                ))
            f.write('-' * 60 + '\n')
            f.write('Timestamp: {}\n'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))

        print('Saved metrics to {}'.format(args.output_log))
        print('\n--- Metrics Summary ---')
        for name in ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'F1', 'Recall']:
            val = metrics[name]
            ci = metrics['{}_CI95'.format(name)]
            print('{:<15s}: {:.4f}  CI95=[{:.4f}, {:.4f}]'.format(name, val, ci[0], ci[1]))


if __name__ == '__main__':
    main()
