import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from timm.models import create_model

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import utils  # 来自项目根目录的 utils.py
import model  # 导入以注册 RepViT 模型到 timm


METRIC_ORDER = [
    "AUROC",
    "AUPRC",
    "ACC",
    "PREC",
    "RECALL",
    "F1",
    "SPEC",
    "ECE",
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def resolve_output_path(output_file, checkpoint_path):
    if output_file:
        return Path(output_file)
    project_root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    checkpoint_stem = Path(checkpoint_path).resolve().parent.name
    return project_root / "test_log" / f"eval_multi_test_{checkpoint_stem}_{timestamp}.txt"


def build_test_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_model(model_name, ckpt_path, num_classes=2, device="cuda"):
    print(f"Loading model {model_name} from {ckpt_path}")
    model = create_model(
        model_name,
        num_classes=num_classes,
        distillation=False,
        pretrained=False,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    model_state = model.state_dict()
    removed_keys = []
    for k in list(state_dict.keys()):
        if k in model_state and state_dict[k].shape != model_state[k].shape:
            removed_keys.append(k)
            del state_dict[k]
    if removed_keys:
        print(f"Removed incompatible keys from checkpoint (shape mismatch): {removed_keys}")

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded state_dict with msg: {msg}")

    model.to(device)
    model.eval()
    utils.replace_batchnorm(model)

    return model


def collect_predictions(model, dataloader, device="cuda"):
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_score = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_true.astype(int), y_score.astype(np.float64)


def compute_ece(y_true, y_score, n_bins=10):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bin_edges[1:-1], right=True)

    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_score[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.mean()
    return float(ece)


def compute_metrics(y_true, y_score, threshold=0.5, ece_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)

    unique_labels = np.unique(y_true)
    if len(unique_labels) == 1:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ece = compute_ece(y_true, y_score, n_bins=ece_bins)

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "ACC": float(acc),
        "PREC": float(prec),
        "RECALL": float(recall),
        "F1": float(f1),
        "SPEC": float(specificity),
        "ECE": float(ece),
    }


def bootstrap_ci(y_true, y_score, threshold=0.5, ece_bins=10, n_boot=2000, alpha=0.95, seed=42):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)

    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)

    boot_metrics = {name: [] for name in METRIC_ORDER}

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        for _ in range(n_boot):
            sample_idx = rng.integers(0, len(y_true), size=len(y_true))
            metrics = compute_metrics(
                y_true[sample_idx],
                y_score[sample_idx],
                threshold=threshold,
                ece_bins=ece_bins,
            )
            for name, value in metrics.items():
                if not np.isnan(value):
                    boot_metrics[name].append(value)
    else:
        for _ in range(n_boot):
            sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            sample_idx = np.concatenate([sample_pos, sample_neg])
            rng.shuffle(sample_idx)
            metrics = compute_metrics(
                y_true[sample_idx],
                y_score[sample_idx],
                threshold=threshold,
                ece_bins=ece_bins,
            )
            for name, value in metrics.items():
                if not np.isnan(value):
                    boot_metrics[name].append(value)

    ci = {}
    for name in METRIC_ORDER:
        values = np.asarray(boot_metrics[name], dtype=np.float64)
        if values.size == 0:
            ci[name] = (float("nan"), float("nan"))
        else:
            ci[name] = (
                float(np.quantile(values, lower_q)),
                float(np.quantile(values, upper_q)),
            )
    return ci


def format_metric(name, value, ci):
    lower, upper = ci
    if np.isnan(value):
        return f"{name}: nan [nan, nan]"
    if np.isnan(lower) or np.isnan(upper):
        return f"{name}: {value:.4f} [nan, nan]"
    return f"{name}: {value:.4f} [{lower:.4f}, {upper:.4f}]"


def main():
    parser = argparse.ArgumentParser("Multi-dataset evaluation for RepViT binary classification")

    parser.add_argument("--model", type=str, default="repvit_m1_0",
                        help="RepViT model name, e.g. repvit_m0_9, repvit_m1_0")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint, e.g. checkpoints/.../checkpoint_best.pth")
    parser.add_argument("--test-dirs", type=str, nargs="+", required=True,
                        help="One or more test dataset roots (ImageFolder style)")
    parser.add_argument("--test-names", type=str, nargs="*", default=None,
                        help="Optional names for each test dataset (same order as test-dirs)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--ci-alpha", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to save evaluation output; defaults to test_log directory")

    args = parser.parse_args()

    output_path = resolve_output_path(args.output_file, args.checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with output_path.open("w", encoding="utf-8") as output_handle:
        sys.stdout = Tee(original_stdout, output_handle)
        try:
            print(f"Saving evaluation output to: {output_path}")

            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            transform = build_test_transform(args.input_size)

            model = load_model(args.model, args.checkpoint, num_classes=2, device=device)

            print("\n========== Evaluation ==========")
            for idx, test_root in enumerate(args.test_dirs):
                if not os.path.isdir(test_root):
                    print(f"[WARN] Test dir not found, skip: {test_root}")
                    continue

                dataset = datasets.ImageFolder(root=test_root, transform=transform)
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )

                if args.test_names is not None and len(args.test_names) == len(args.test_dirs):
                    name = args.test_names[idx]
                else:
                    name = Path(test_root).name

                print(f"\n--- Dataset: {name} ({test_root}), size={len(dataset)} ---")
                y_true, y_score = collect_predictions(model, dataloader, device=device)
                metrics = compute_metrics(
                    y_true,
                    y_score,
                    threshold=args.threshold,
                    ece_bins=args.ece_bins,
                )
                ci = bootstrap_ci(
                    y_true,
                    y_score,
                    threshold=args.threshold,
                    ece_bins=args.ece_bins,
                    n_boot=args.bootstrap_iters,
                    alpha=args.ci_alpha,
                    seed=args.seed,
                )

                print("  ".join(format_metric(metric_name, metrics[metric_name], ci[metric_name]) for metric_name in METRIC_ORDER))
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
