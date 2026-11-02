import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np


def resolve_output_dir(output_dir, checkpoint_path):
    if output_dir:
        return Path(output_dir)
    project_root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    checkpoint_stem = Path(checkpoint_path).resolve().parent.name
    return project_root / "test_log" / f"infer_multi_to_json_{checkpoint_stem}_{timestamp}"


def sanitize_filename(name):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return sanitized or "dataset"


def collect_prediction_records(model, dataloader, dataset, model_name, device="cuda"):
    import torch

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

    if all_probs:
        y_score = np.concatenate(all_probs, axis=0).astype(np.float64)
        y_true = np.concatenate(all_labels, axis=0).astype(int)
    else:
        y_score = np.empty((0,), dtype=np.float64)
        y_true = np.empty((0,), dtype=int)

    sample_entries = getattr(dataset, "samples", None)
    if sample_entries is None:
        sample_entries = getattr(dataset, "imgs", [])

    if len(sample_entries) != len(y_true):
        raise ValueError(
            f"Dataset samples ({len(sample_entries)}) do not match predictions ({len(y_true)})"
        )

    records = []
    for (image_path, dataset_label), true_label, prob_class_1 in zip(sample_entries, y_true, y_score):
        if int(dataset_label) != int(true_label):
            raise ValueError(
                f"Label mismatch for {image_path}: dataset={dataset_label}, predicted_batch_order={true_label}"
            )

        prob_1 = float(prob_class_1)
        prob_0 = float(1.0 - prob_1)
        predicted_class = int(prob_1 >= 0.5)
        confidence = float(max(prob_0, prob_1))

        records.append({
            "record_type": "sample",
            "image_file": str(Path(image_path).resolve()),
            "image_name": Path(image_path).name,
            "selected_model": model_name,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "prob_class_0": prob_0,
            "prob_class_1": prob_1,
            "true_label": int(true_label),
        })

    return records, y_true, y_score


def save_records(records, output_path):
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def save_metrics(metrics_summary, output_path):
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_summary, handle, ensure_ascii=False, indent=2)


def format_metric(name, value, ci):
    lower, upper = ci
    if np.isnan(value):
        return f"{name}: nan [nan, nan]"
    if np.isnan(lower) or np.isnan(upper):
        return f"{name}: {value:.4f} [nan, nan]"
    return f"{name}: {value:.4f} [{lower:.4f}, {upper:.4f}]"


def main():
    parser = argparse.ArgumentParser("Multi-dataset inference to AUROC JSON for RepViT binary classification")

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
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save per-dataset JSON files; defaults to test_log directory")

    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    from eval_multi_test import (
        METRIC_ORDER,
        bootstrap_ci,
        build_test_transform,
        compute_metrics,
        load_model,
    )
    import model  # noqa: F401  # 导入以注册 RepViT 模型到 timm

    output_dir = resolve_output_dir(args.output_dir, args.checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = build_test_transform(args.input_size)
    inference_model = load_model(args.model, args.checkpoint, num_classes=2, device=device)
    checkpoint_stem = Path(args.checkpoint).resolve().parent.name

    print(f"Saving per-dataset JSON outputs to: {output_dir}")
    print("\n========== JSON Export Inference ==========")

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
            dataset_name = args.test_names[idx]
        else:
            dataset_name = Path(test_root).name

        print(f"\n--- Dataset: {dataset_name} ({test_root}), size={len(dataset)} ---")
        records, y_true, y_score = collect_prediction_records(
            inference_model,
            dataloader,
            dataset,
            model_name=args.model,
            device=device,
        )

        filename = f"{sanitize_filename(dataset_name)}__{sanitize_filename(checkpoint_stem)}__predictions.json"
        output_path = output_dir / filename
        save_records(records, output_path)

        metrics_filename = f"{sanitize_filename(dataset_name)}__{sanitize_filename(checkpoint_stem)}__metrics.json"
        metrics_output_path = output_dir / metrics_filename

        if len(y_true) > 0:
            metrics = compute_metrics(
                y_true,
                y_score,
                threshold=args.threshold,
                ece_bins=args.ece_bins,
            )
            if args.bootstrap_iters > 0:
                ci = bootstrap_ci(
                    y_true,
                    y_score,
                    threshold=args.threshold,
                    ece_bins=args.ece_bins,
                    n_boot=args.bootstrap_iters,
                    alpha=args.ci_alpha,
                    seed=args.seed,
                )
            else:
                ci = {name: (float("nan"), float("nan")) for name in METRIC_ORDER}
            label_set = np.unique(y_true).tolist()
        else:
            metrics = {name: float("nan") for name in METRIC_ORDER}
            ci = {name: (float("nan"), float("nan")) for name in METRIC_ORDER}
            label_set = []

        metrics_summary = {
            "record_type": "classification_metrics_summary",
            "dataset_name": dataset_name,
            "dataset_root": str(Path(test_root).resolve()),
            "model": args.model,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "threshold": args.threshold,
            "ece_bins": args.ece_bins,
            "bootstrap_iters": args.bootstrap_iters,
            "ci_alpha": args.ci_alpha,
            "seed": args.seed,
            "n_samples": int(len(y_true)),
            "label_set": label_set,
            "metrics": metrics,
            "confidence_intervals": {
                name: {
                    "lower": ci[name][0],
                    "upper": ci[name][1],
                }
                for name in METRIC_ORDER
            },
        }
        save_metrics(metrics_summary, metrics_output_path)

        print(
            "  ".join(format_metric(metric_name, metrics[metric_name], ci[metric_name]) for metric_name in METRIC_ORDER)
        )
        print(
            f"Saved {len(records)} records to {output_path} | "
            f"Saved metrics to {metrics_output_path} | "
            f"label_set={label_set} | "
            f"prob_range=[{float(np.min(y_score)):.6f}, {float(np.max(y_score)):.6f}]"
            if len(records) > 0
            else f"Saved 0 records to {output_path} | Saved metrics to {metrics_output_path}"
        )


if __name__ == "__main__":
    main()
