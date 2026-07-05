#!/usr/bin/env python3
"""
分类推理脚本（独立可运行版本）。

功能：
  1. 使用 DINOv3_S_UNet_MULTITASK 的分类头对图像目录进行批量推理。
  2. 输出 CSV 文件（filename, predicted_class, confidence [, true_label]）。
  3. 若提供标签文件，额外输出分类指标 .log 文件（含 CI95 置信区间）。

支持的标签文件格式（JSON 列表）：
  [
    {"filename": "path/to/image.jpg", "label_field_A": 0, "label_field_B": 1},
    ...
  ]
  通过 --label_field 指定当前任务对应的字段名。

用法示例：
  # 二分类推理（无标签文件）
  python infer_classification.py \
      --image_dir /path/to/images \
      --checkpoint /path/to/model.pth \
      --num_classes 2 \
      --output results/preds.csv

  # 五分类推理（有标签文件，输出指标）
  python infer_classification.py \
      --image_dir /path/to/images \
      --checkpoint /path/to/model.pth \
      --num_classes 5 \
      --output results/preds.csv \
      --label_file /path/to/labels.json \
      --label_field tirads \
      --log_file results/metrics.log
"""

import os
import csv
import json
import argparse
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import DINOv3_S_UNet_MULTITASK

# 使用项目级统一分类指标模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from cls_metrics import (
    METRIC_ORDER,
    binary_bootstrap_metrics,
    multiclass_bootstrap_metrics,
)


# ---------------------------------------------------------------------------
# 常量与工具函数
# ---------------------------------------------------------------------------

VALID_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')


def str2bool(value: str) -> bool:
    return str(value).lower() in ('true', '1', 'yes', 'y')


def normalize_rel_path(path: str) -> str:
    """规范化相对路径，统一使用正斜杠。"""
    return os.path.normpath(path.replace('\\', '/'))


def scan_image_dir(image_dir: str) -> dict[str, str]:
    """递归扫描图像目录。

    返回: {normalized_rel_path: full_path}
    同时用 basename 作为额外 key（冲突时保留先扫描到的）。
    """
    mapping: dict[str, str] = {}
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if not fname.lower().endswith(VALID_SUFFIXES):
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, image_dir)
            norm_rel = normalize_rel_path(rel_path)
            mapping[norm_rel] = full_path
            basename = os.path.basename(norm_rel)
            if basename not in mapping:
                mapping[basename] = full_path
    return mapping


def load_labels(
    label_file: str,
    label_field: str,
    image_mapping: dict[str, str],
    num_classes: int,
    label_offset: int,
) -> dict[str, int]:
    """加载标签文件，返回 {full_image_path: label}。

    label_offset:
      -1: 自动检测（五分类且 min(label)==1 时偏移 1，否则 0）
       0: 不偏移
       1: 标签减 1（如 TIRADS 1-5 → 0-4）
    """
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    if isinstance(label_data, dict):
        label_data = list(label_data.values())

    # 收集所有标签值，用于自动检测偏移
    raw_labels = []
    for item in label_data:
        if not isinstance(item, dict):
            continue
        if label_field not in item:
            continue
        val = item[label_field]
        if val is not None and val != "":
            raw_labels.append(int(val))

    if not raw_labels:
        raise ValueError(
            f"标签文件中没有找到字段 '{label_field}' 的有效值。"
            f"请检查 --label_field 参数是否正确。"
        )

    # 自动检测偏移
    if label_offset == -1:
        min_val = min(raw_labels)
        max_val = max(raw_labels)
        if num_classes == 5 and min_val == 1 and max_val == 5:
            actual_offset = 1
        elif num_classes == 2 and min_val == 0 and max_val == 1:
            actual_offset = 0
        elif min_val == 1 and max_val == num_classes:
            actual_offset = 1
        else:
            actual_offset = 0
        if actual_offset != 0:
            print(f"[自动检测] 标签范围 [{min_val}, {max_val}]，"
                  f"自动偏移 {actual_offset}（{min_val}->{min_val - actual_offset}, "
                  f"{max_val}->{max_val - actual_offset}）")
    else:
        actual_offset = label_offset

    # 构建映射
    label_mapping: dict[str, int] = {}
    matched = 0
    unmatched = []

    for item in label_data:
        if not isinstance(item, dict):
            continue
        filename = item.get('filename')
        if not filename or label_field not in item:
            continue

        raw_label = item[label_field]
        if raw_label is None or raw_label == "":
            continue

        label = int(raw_label) - actual_offset

        if label < 0 or label >= num_classes:
            print(f"[警告] 标签 {raw_label}（偏移后 {label}）超出范围 "
                  f"[0, {num_classes - 1}]，跳过: {filename}")
            continue

        norm_filename = normalize_rel_path(filename)
        full_path = image_mapping.get(norm_filename)
        if full_path is None:
            basename = os.path.basename(norm_filename)
            full_path = image_mapping.get(basename)
        if full_path is None:
            unmatched.append(filename)
            continue

        label_mapping[full_path] = label
        matched += 1

    if matched == 0:
        raise ValueError(
            f"标签文件中没有任何条目能匹配到图像目录中的文件。\n"
            f"请检查标签文件中的 filename 路径是否与图像目录结构一致。"
        )

    if unmatched:
        print(f"[警告] {len(unmatched)} 个标签条目未匹配到图像文件，"
              f"前 5 个: {unmatched[:5]}")

    print(f"标签加载完成: {matched} 个样本已匹配，偏移量={actual_offset}")
    return label_mapping


# ---------------------------------------------------------------------------
# 推理 Dataset
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """纯分类推理用 Dataset：递归扫描目录中所有图像文件。"""

    def __init__(self, image_dir: str, img_size: int = 224):
        self.image_dir = image_dir
        self.img_size = img_size

        mapping = scan_image_dir(image_dir)
        # 去重（同一个文件可能通过 rel_path 和 basename 两个 key 指向）
        seen = set()
        unique_paths = []
        for p in mapping.values():
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
        self.image_paths = sorted(unique_paths)

        if not self.image_paths:
            raise FileNotFoundError(f"No image files found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        filename = os.path.relpath(path, self.image_dir)
        with Image.open(path) as img:
            img = img.convert('RGB')
        tensor = self.transform(img)
        return tensor, filename, path


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, pretrained: bool, use_dilation: bool,
               device: torch.device) -> DINOv3_S_UNet_MULTITASK:
    """加载模型并恢复权重。"""
    model = DINOv3_S_UNet_MULTITASK(pretrained=pretrained, use_dilation=use_dilation)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 推理主流程
# ---------------------------------------------------------------------------

def run_inference(model, loader, device, num_classes):
    """执行推理，返回每个样本的结果列表（含预测类别、置信度、概率）。

    每个结果 dict 包含：
      filename, predicted_class, confidence
      _prob_1  (二分类: P(class=1))
      _probs   (五分类: [p0, p1, ..., p4])
      _path    (图像完整路径，用于标签匹配)
    """
    results = []

    with torch.no_grad():
        for images, filenames, paths in tqdm(loader, desc="推理"):
            images = images.to(device)
            _, benign_malignant, tirads = model(images)

            if num_classes == 2:
                logits = benign_malignant.squeeze(1)  # (B,)
                probs_1 = torch.sigmoid(logits)       # (B,)
                probs_0 = 1.0 - probs_1
                preds = (probs_1 > 0.5).long()
                confidence = torch.max(
                    torch.stack([probs_0, probs_1], dim=1), dim=1
                )[0]

                for i, (fname, fpath) in enumerate(zip(filenames, paths)):
                    results.append({
                        'filename': fname,
                        'predicted_class': int(preds[i].item()),
                        'confidence': float(confidence[i].item()),
                        '_prob_1': float(probs_1[i].item()),
                        '_path': fpath,
                    })
            else:
                probs = F.softmax(tirads, dim=1)  # (B, 5)
                preds = probs.argmax(dim=1)
                confidence = probs.max(dim=1)[0]

                for i, (fname, fpath) in enumerate(zip(filenames, paths)):
                    results.append({
                        'filename': fname,
                        'predicted_class': int(preds[i].item()),
                        'confidence': float(confidence[i].item()),
                        '_probs': probs[i].cpu().numpy().tolist(),
                        '_path': fpath,
                    })

    return results


def save_csv(results, output_path, has_labels):
    """保存结果到 CSV。"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if has_labels:
        fieldnames = ['filename', 'predicted_class', 'confidence', 'true_label']
    else:
        fieldnames = ['filename', 'predicted_class', 'confidence']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames}
            writer.writerow(row)




def compute_and_save_metrics(results, num_classes, label_field, log_path,
                             n_boot, ci, seed):
    """计算指标并保存到 .log 文件（统一格式）。"""
    y_true = np.array([r['true_label'] for r in results], dtype=np.int32)
    valid = y_true != -1

    if num_classes == 2:
        y_prob = np.array([r['_prob_1'] for r in results], dtype=np.float64)
        y_true_valid = y_true[valid]
        y_prob_valid = y_prob[valid]

        metrics = binary_bootstrap_metrics(
            y_prob_valid, y_true_valid,
            threshold=0.5, n_boot=n_boot, ci=ci, seed=seed,
        )
    else:
        y_probs = np.array([r['_probs'] for r in results], dtype=np.float64)
        y_true_valid = y_true[valid]
        y_probs_valid = y_probs[valid]

        metrics = multiclass_bootstrap_metrics(
            y_probs_valid, y_true_valid,
            num_classes=num_classes,
            n_boot=n_boot, ci=ci, seed=seed,
        )

    # 构建统一格式输出
    n_valid = int(valid.sum())
    out_lines = []
    out_lines.append("=" * 60)
    out_lines.append(f"评估样本数: {n_valid}")

    metric_names = ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'F1', 'Recall']
    for name in metric_names:
        for key in metrics:
            if key.upper() == name.upper():
                mean_v, (low_v, high_v) = metrics[key]
                out_lines.append(
                    f"{name:<12s}: {mean_v:.4f}  (95% CI: [{low_v:.4f}, {high_v:.4f}])"
                )
                break

    out_lines.append("=" * 60)
    report = "\n".join(out_lines)

    # 打印到终端
    print(report)

    # 写 log 文件
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(report + "\n")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main(args):
    if args.num_classes not in (2, 5):
        raise ValueError(f"num_classes must be 2 or 5, got {args.num_classes}")

    if args.label_file and not args.label_field:
        raise ValueError("提供 --label_file 时必须同时提供 --label_field")

    # ---------- 设备 ----------
    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    )

    # ---------- 加载模型 ----------
    model = load_model(
        args.checkpoint, args.dino_pretrained, args.use_dilation, device
    )

    # ---------- 构建数据加载器 ----------
    dataset = InferenceDataset(args.image_dir, img_size=args.img_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # ---------- 加载标签（如果提供） ----------
    label_mapping = None
    if args.label_file:
        image_mapping = scan_image_dir(args.image_dir)
        label_mapping = load_labels(
            args.label_file, args.label_field, image_mapping,
            args.num_classes, args.label_offset,
        )

    # ---------- 打印配置 ----------
    print("=" * 60)
    print(f"权重:     {args.checkpoint}")
    print(f"数据:     {args.image_dir}")
    print(f"类别数:   {args.num_classes}")
    if args.label_file:
        print(f"标签字段: {args.label_field}")
    print(f"设备:     {device}")
    print("=" * 60)

    # ---------- 推理 ----------
    results = run_inference(model, loader, device, args.num_classes)

    # ---------- 附加 true_label ----------
    if label_mapping is not None:
        for r in results:
            fpath = r['_path']
            r['true_label'] = label_mapping.get(fpath, -1)

    # ---------- 保存 CSV ----------
    has_labels = label_mapping is not None
    save_csv(results, args.output, has_labels)

    # ---------- 计算并保存指标 ----------
    if label_mapping is not None:
        log_path = args.log_file
        if not log_path:
            csv_abs = os.path.abspath(args.output)
            log_path = os.path.splitext(csv_abs)[0] + '.log'

        compute_and_save_metrics(
            results, args.num_classes, args.label_field, log_path,
            args.n_boot, args.ci, args.seed,
        )

    # ---------- 清理临时字段 ----------
    for r in results:
        for key in list(r.keys()):
            if key.startswith('_'):
                del r[key]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DINOv3-UNet Multitask Classification Inference (Standalone)"
    )

    # 必填参数
    parser.add_argument("--image_dir", type=str, required=True,
                        help="待推理图像所在目录（支持递归扫描子目录）")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重文件路径 (.pth)")
    parser.add_argument("--num_classes", type=int, required=True,
                        choices=[2, 5],
                        help="分类类别数 (2: 二分类, 5: TIRADS五分类)")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 CSV 文件路径")

    # 标签相关（可选）
    parser.add_argument("--label_file", type=str, default=None,
                        help="标签 JSON 文件路径（可选）。提供后将计算指标并输出 .log")
    parser.add_argument("--label_field", type=str, default=None,
                        help="标签文件中对应的任务字段名（如 malignancy, tirads, LNM_CN01 等）")
    parser.add_argument("--label_offset", type=int, default=-1,
                        help="标签偏移量。-1=自动检测, 0=不偏移, 1=标签减1（如 TIRADS 1-5 → 0-4）。默认 -1")
    parser.add_argument("--log_file", type=str, default=None,
                        help="指标日志输出路径。不指定时默认与 CSV 同名 .log 文件")

    # 图像与模型配置
    parser.add_argument("--img_size", type=int, default=224,
                        help="输入图像尺寸 (默认: 224)")
    parser.add_argument("--dino_pretrained", type=str, default='False',
                        help="DINO backbone 是否使用预训练权重 (True/False)。推理时建议 False")
    parser.add_argument("--use_dilation", type=str, default='False',
                        help="模型是否使用 dilation 层 (True/False)，需与训练时一致")

    # 硬件
    parser.add_argument("--cuda_device", type=int, default=0,
                        help="CUDA 设备索引 (默认: 0)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="推理批大小 (默认: 16)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader 子进程数 (默认: 4)")

    # Bootstrap 参数
    parser.add_argument("--n_boot", type=int, default=2000,
                        help="Bootstrap 采样次数 (默认: 2000)")
    parser.add_argument("--ci", type=float, default=0.95,
                        help="置信区间水平 (默认: 0.95)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Bootstrap 随机种子 (默认: 0)")

    args = parser.parse_args()

    # 布尔类型转换
    args.dino_pretrained = str2bool(args.dino_pretrained)
    args.use_dilation = str2bool(args.use_dilation)

    main(args)
