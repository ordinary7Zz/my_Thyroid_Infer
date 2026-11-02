import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from timm.models import create_model
from model import build as _build  # noqa: F401

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
except ImportError as e:
    raise ImportError(
        "scikit-learn is required for eval_thyroid_auroc_json.py. "
        "Please install it with `pip install scikit-learn`."
    ) from e


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LSNet thyroid classifier and export sample-level AUROC JSON."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint_best.pth (or similar).",
    )
    parser.add_argument(
        "--test-dirs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more test dataset roots. Each root should be an ImageFolder-style "
            "directory with subfolders for each class."
        ),
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help="Optional human-readable names for each test set (same length as --test-dirs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--pos-class-index",
        type=int,
        default=1,
        help=(
            "Class index to treat as the positive class. Exported prob_class_1 "
            "and true_label=1 will always refer to this class."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=(
            "Optional path to a log file. If provided, all printed results will also "
            "be appended to this file."
        ),
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of bins for ECE.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save one AUROC-compatible JSON file per test dataset.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional prefix for generated JSON filenames.",
    )
    return parser.parse_args()


def build_transform(input_size: int = 224) -> transforms.Compose:
    size = int((256 / 224) * input_size)
    t = [
        transforms.Resize(size, interpolation=3),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return transforms.Compose(t)


def load_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, int, str]:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "args" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain 'args'. Make sure it was saved by main.py in this repo."
        )

    args_ckpt = checkpoint["args"]
    model_name = getattr(args_ckpt, "model", None)
    num_classes = getattr(args_ckpt, "nb_classes", None)

    if model_name is None or num_classes is None:
        raise RuntimeError(
            "Checkpoint 'args' does not contain 'model' or 'nb_classes'. "
            "Please ensure the model was trained with this repository's main.py."
        )

    distillation_type = getattr(args_ckpt, "distillation_type", "none")
    model = create_model(
        model_name,
        num_classes=num_classes,
        distillation=(distillation_type != "none"),
    )

    state_dict = checkpoint.get("model", checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model state dict with message: {msg}")

    model.to(device)
    model.eval()
    return model, num_classes, model_name


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for bin_idx in range(n_bins):
        lower = bin_edges[bin_idx]
        upper = bin_edges[bin_idx + 1]
        if bin_idx == n_bins - 1:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        else:
            in_bin = (y_prob >= lower) & (y_prob < upper)

        if not np.any(in_bin):
            continue

        bin_prob = y_prob[in_bin]
        bin_true = y_true[in_bin]
        bin_acc = np.mean(bin_true)
        bin_conf = np.mean(bin_prob)
        ece += (len(bin_prob) / len(y_prob)) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_point_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ece_bins: int,
) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Acc": float(acc),
        "Prec": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "Specificity": float(specificity),
        "ECE": compute_ece(y_true, y_prob, n_bins=ece_bins),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "N": int(len(y_true)),
    }


def bootstrap_metrics_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ece_bins: int,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Bootstrap CI requires both positive and negative samples.")

    metric_names = ["AUROC", "AUPRC", "Acc", "Prec", "Recall", "F1", "Specificity", "ECE"]
    bootstrap_values = {name: [] for name in metric_names}

    for _ in range(n_bootstrap):
        sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sampled_idx = np.concatenate([sampled_pos, sampled_neg])
        rng.shuffle(sampled_idx)

        sampled_true = y_true[sampled_idx]
        sampled_prob = y_prob[sampled_idx]
        sampled_metrics = compute_point_metrics(sampled_true, sampled_prob, ece_bins)

        for name in metric_names:
            bootstrap_values[name].append(sampled_metrics[name])

    alpha = 1.0 - ci_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    return {
        name: (
            float(np.percentile(values, lower_q)),
            float(np.percentile(values, upper_q)),
        )
        for name, values in bootstrap_values.items()
    }


@torch.no_grad()
def evaluate_one_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    sample_entries: List[Tuple[str, int]],
    device: torch.device,
    pos_class_index: int,
    ece_bins: int,
    n_bootstrap: int,
    ci_level: float,
    bootstrap_seed: int,
    model_name: str,
) -> Tuple[List[dict], dict]:
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    sample_records: List[dict] = []
    sample_offset = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(1)

        labels_np = targets.detach().cpu().numpy().astype(int)
        true_labels_np = (labels_np == pos_class_index).astype(int)

        if outputs.size(1) == 1:
            probs_class_1 = torch.sigmoid(outputs.squeeze(1))
            if pos_class_index == 1:
                probs_pos = probs_class_1
            elif pos_class_index == 0:
                probs_pos = 1.0 - probs_class_1
            else:
                raise ValueError(
                    "Single-logit binary models only support --pos-class-index 0 or 1."
                )
        else:
            probs = torch.softmax(outputs, dim=1)
            probs_pos = probs[:, pos_class_index]

        probs_pos_np = probs_pos.detach().cpu().numpy().astype(float)
        probs_neg_np = 1.0 - probs_pos_np

        batch_size = len(labels_np)
        batch_entries = sample_entries[sample_offset: sample_offset + batch_size]
        if len(batch_entries) != batch_size:
            raise RuntimeError("Prediction count does not match dataset sample entries.")

        for (image_file, _), true_label, prob_class_1, prob_class_0 in zip(
            batch_entries,
            true_labels_np.tolist(),
            probs_pos_np.tolist(),
            probs_neg_np.tolist(),
        ):
            predicted_class = int(prob_class_1 >= 0.5)
            sample_records.append(
                {
                    "record_type": "sample",
                    "image_file": image_file,
                    "image_name": os.path.basename(image_file),
                    "selected_model": model_name,
                    "predicted_class": predicted_class,
                    "confidence": float(max(prob_class_0, prob_class_1)),
                    "prob_class_0": float(prob_class_0),
                    "prob_class_1": float(prob_class_1),
                    "true_label": int(true_label),
                }
            )

        sample_offset += batch_size
        all_probs.append(probs_pos_np)
        all_labels.append(true_labels_np)

    if sample_offset != len(sample_entries):
        raise RuntimeError("Processed sample count does not match dataset size.")

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            f"Expected binary labels for AUROC/AUPRC, got classes: {np.unique(y_true)}"
        )

    point_metrics = compute_point_metrics(y_true, y_prob, ece_bins)
    ci_metrics = bootstrap_metrics_ci(
        y_true=y_true,
        y_prob=y_prob,
        ece_bins=ece_bins,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=bootstrap_seed,
    )

    return sample_records, {
        **point_metrics,
        "CI": ci_metrics,
    }


def write_sample_level_results(records: List[dict], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def build_output_path(output_dir: str, dataset_name: str, output_prefix: str = None) -> str:
    stem = f"{output_prefix}_{dataset_name}" if output_prefix else dataset_name
    return os.path.join(output_dir, f"{stem}.json")


def main():
    args = parse_args()

    if args.names is not None and len(args.names) != len(args.test_dirs):
        raise ValueError("--names must be the same length as --test-dirs if provided.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    log_f = None
    if args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True) if os.path.dirname(args.log_file) else None
        log_f = open(args.log_file, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        if log_f is not None:
            log_f.write(msg + "\n")
            log_f.flush()

    log(f"Using device: {device}")

    model, num_classes, model_name = load_model_from_checkpoint(
        args.checkpoint, device
    )
    log(f"Loaded model '{model_name}' with {num_classes} classes.")

    transform = build_transform(input_size=224)

    for idx, test_root in enumerate(args.test_dirs):
        if not os.path.isdir(test_root):
            log(f"[WARN] Test dir does not exist, skipped: {test_root}")
            continue

        name = args.names[idx] if args.names is not None else os.path.basename(
            os.path.normpath(test_root)
        )
        log(f"\n=== Evaluating on test set: {name} ===")
        log(f"Path: {test_root}")

        dataset = datasets.ImageFolder(test_root, transform=transform, loader=default_loader)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        log(f"Classes: {dataset.classes}")

        sample_records, metrics = evaluate_one_dataset(
            model=model,
            dataloader=dataloader,
            sample_entries=dataset.samples,
            device=device,
            pos_class_index=args.pos_class_index,
            ece_bins=args.ece_bins,
            n_bootstrap=args.bootstrap_samples,
            ci_level=args.ci_level,
            bootstrap_seed=args.bootstrap_seed,
            model_name=model_name,
        )

        output_path = build_output_path(
            output_dir=args.output_dir,
            dataset_name=name,
            output_prefix=args.output_prefix,
        )
        write_sample_level_results(sample_records, output_path)

        log(f"Samples: {metrics['N']}")
        log(f"TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']} TP={metrics['TP']}")
        ci = metrics["CI"]
        log(
            f"AUROC={metrics['AUROC']:.4f} [{ci['AUROC'][0]:.4f}, {ci['AUROC'][1]:.4f}]  "
            f"AUPRC={metrics['AUPRC']:.4f} [{ci['AUPRC'][0]:.4f}, {ci['AUPRC'][1]:.4f}]"
        )
        log(
            f"Acc={metrics['Acc']:.4f} [{ci['Acc'][0]:.4f}, {ci['Acc'][1]:.4f}]  "
            f"Prec={metrics['Prec']:.4f} [{ci['Prec'][0]:.4f}, {ci['Prec'][1]:.4f}]  "
            f"Recall={metrics['Recall']:.4f} [{ci['Recall'][0]:.4f}, {ci['Recall'][1]:.4f}]"
        )
        log(
            f"F1={metrics['F1']:.4f} [{ci['F1'][0]:.4f}, {ci['F1'][1]:.4f}]  "
            f"Specificity={metrics['Specificity']:.4f} [{ci['Specificity'][0]:.4f}, {ci['Specificity'][1]:.4f}]  "
            f"ECE={metrics['ECE']:.4f} [{ci['ECE'][0]:.4f}, {ci['ECE'][1]:.4f}]"
        )
        log(f"Saved AUROC JSON: {output_path}")

    if log_f is not None:
        log_f.close()


if __name__ == "__main__":
    main()
