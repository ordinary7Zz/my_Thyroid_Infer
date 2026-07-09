"""
BiomedCLIP 分类推理脚本（独立版，带 Bootstrap CI95 评估）
=========================================================
对图片文件夹进行批量推理，输出分类结果 CSV。
若提供标签 JSON 文件，额外计算分类性能指标（含 95% 置信区间）并保存到 .log 文件。

评估指标: AUROC, AUPRC, Accuracy, Precision, F1, Recall（均含 CI95）

用法:
    # 仅推理，输出 CSV
    python infer.py \\
        --ckpt /path/to/best_model.pth \\
        --folder /path/to/images/ \\
        --num_classes 2 \\
        --class_names benign malignant \\
        --output results.csv

    # 推理 + 评估（二分类）
    python infer.py \\
        --ckpt /path/to/best_model.pth \\
        --folder /path/to/images/ \\
        --num_classes 2 \\
        --class_names benign malignant \\
        --label_json /path/to/labels.json \\
        --label_field malignancy \\
        --output results.csv \\
        --eval_output eval_result.log

    # 推理 + 评估（TIRADS 五分类）
    python infer.py \\
        --ckpt /path/to/best_model.pth \\
        --folder /path/to/images/ \\
        --num_classes 5 \\
        --class_names 1 2 3 4 5 \\
        --label_json /path/to/labels.json \\
        --label_field tirads \\
        --output results.csv \\
        --eval_output eval_result.log

标签 JSON 格式示例:
    [
        {"filename": "a.jpg", "malignancy": 0, "tirads": 2},
        {"filename": "b.jpg", "malignancy": 1, "tirads": 4}
    ]

注意:
    - 标签值为整数索引，与 --class_names 的顺序对应
    - 模型路径通过 --model_dir 指定本地预训练模型目录（须含 open_clip_config.json 和权重文件）
"""

import os
import sys
import csv
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 使用项目级统一分类指标模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from cls_metrics import bootstrap_ci as _bootstrap_ci, format_metrics_report

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================================
# 模型定义（内联自 model.py，无需外部依赖）
# ============================================================================

def _load_biomedclip_backbone(model_dir: str):
    """
    从本地目录加载 BiomedCLIP 预训练骨干模型（仅视觉编码器，不加载文本编码器）。

    BiomedCLIP 的视觉编码器使用 timm 的 ViT，不需要 transformers/torchaudio。
    直接用 _build_vision_tower 构建，从 .bin/.safetensors 中只加载 visual 参数。

    Args:
        model_dir: 包含 open_clip_config.json 和 .bin/.safetensors 权重的本地目录
    """
    import json
    from open_clip.model import _build_vision_tower

    config_path = os.path.join(model_dir, "open_clip_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"未找到 {config_path}，请确认 --model_dir 指向正确的 BiomedCLIP 模型目录"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_cfg = cfg["model_cfg"]

    # 查找本地权重文件（优先 .safetensors，其次 .bin）
    local_weights = None
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".safetensors"):
            local_weights = os.path.join(model_dir, fname)
            break
        if fname.endswith((".bin", ".pt", ".pth")) and local_weights is None:
            local_weights = os.path.join(model_dir, fname)

    if local_weights is None:
        raise FileNotFoundError(
            f"在 {model_dir} 中未找到权重文件（.safetensors/.bin/.pt/.pth）"
        )

    print(f"  预训练骨干配置: {config_path}")
    print(f"  预训练骨干权重: {local_weights}")

    # 只构建视觉编码器（跳过文本编码器，避免 transformers/torchaudio 依赖）
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    visual = _build_vision_tower(embed_dim, vision_cfg)

    # 从权重文件中加载 visual 相关参数
    if local_weights.endswith(".safetensors"):
        from safetensors.torch import load_file
        full_state = load_file(local_weights)
    else:
        full_state = torch.load(local_weights, map_location="cpu")

    # 只保留 visual. 开头的参数
    visual_state = {}
    for k, v in full_state.items():
        if k.startswith("visual."):
            visual_state[k[len("visual."):]] = v

    missing, unexpected = visual.load_state_dict(visual_state, strict=False)
    if missing:
        print(f"  visual missing keys: {len(missing)} (前5: {missing[:5]})")
    if unexpected:
        print(f"  visual unexpected keys: {len(unexpected)} (前5: {unexpected[:5]})")

    return visual

class BiomedCLIPClassifier(nn.Module):
    """
    在 BiomedCLIP 图像编码器之上添加分类头。
    推理时策略不影响结果，此处固定 full_finetune 结构。
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
        model_dir: str = None,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        if model_dir is None:
            raise ValueError("model_dir 不能为空，请通过 --model_dir 指定本地预训练模型目录")

        self.visual = _load_biomedclip_backbone(model_dir)
        self.embed_dim = self._get_embed_dim(self.visual)

        # 分类头（与训练时结构完全一致）
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

    @staticmethod
    def _get_embed_dim(visual) -> int:
        """获取视觉编码器输出维度，兼容不同 open_clip 版本"""
        try:
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                out = visual(dummy)
            return out.shape[-1]
        except Exception:
            pass
        if hasattr(visual, "output_dim"):
            return visual.output_dim
        if hasattr(visual, "trunk"):
            trunk = visual.trunk
            for attr in ("num_features", "embed_dim"):
                if hasattr(trunk, attr):
                    return getattr(trunk, attr)
        return 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.visual(x)
        logits = self.classifier(features)
        return logits


# ============================================================================
# 工具函数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="BiomedCLIP 分类推理（支持可选标签文件进行 Bootstrap CI95 性能评估）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必选参数
    parser.add_argument("--ckpt", type=str, required=True,
                        help="训练好的模型权重路径 (.pth)")
    parser.add_argument("--folder", type=str, required=True,
                        help="待推理的图片文件夹路径")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="类别数，二分类填 2，TIRADS 五分类填 5")
    parser.add_argument("--class_names", type=str, nargs="+", required=True,
                        help="类别名称列表，顺序与训练时一致，例如: benign malignant 或 1 2 3 4 5")

    # 模型相关
    parser.add_argument("--model_dir", type=str, required=True,
                        help="本地 BiomedCLIP 预训练模型目录（须含 open_clip_config.json 和 .bin/.safetensors 权重文件）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备: cuda 或 cpu（默认 cuda，无 GPU 自动回退 cpu）")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批推理大小（默认 32）")

    # 输出
    parser.add_argument("--output", type=str, default="results.csv",
                        help="分类结果 CSV 输出路径（默认 results.csv）")

    # 可选：标签文件与评估
    parser.add_argument("--label_json", type=str, default=None,
                        help="标签 JSON 文件路径（可选）；提供后将额外输出分类性能指标（含 CI95）")
    parser.add_argument("--label_field", type=str, default=None,
                        help="JSON 中用于评估的标签字段名，例如 malignancy 或 tirads（提供 --label_json 时必填）")
    parser.add_argument("--eval_output", type=str, default=None,
                        help="评估结果保存路径 (.log)；未指定时自动在 --output 同级目录生成")
    parser.add_argument("--n_bootstrap", type=int, default=2000,
                        help="Bootstrap 迭代次数（默认 2000，用于计算 95%% 置信区间）")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子（默认 0，确保 Bootstrap 结果可复现）")

    return parser.parse_args()


def get_preprocess(image_size: int = 224):
    """与训练一致的推理预处理（CLIP 标准归一化）"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def load_model(ckpt_path: str, model_dir: str, num_classes: int,
               device: torch.device) -> BiomedCLIPClassifier:
    """加载训练好的分类模型"""
    model = BiomedCLIPClassifier(
        num_classes=num_classes,
        model_dir=model_dir,
    )
    print(f"  加载分类权重:   {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量:     {total_params:,}")
    return model


def collect_images(folder: str):
    """收集文件夹中所有图片文件（按文件名排序）"""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    ])
    return files


def load_label_json(json_path: str, label_field: str, class_names: list):
    """
    加载标签 JSON 文件，并自动将标签值映射为 0-based 索引。

    映射策略（按优先级）:
      1. 标签值（转字符串）能在 class_names 中找到 → 用其索引
         例如 class_names=["1","2","3","4","5"], tirads=3 → 索引 2
      2. 标签值已是 0-based（0 ~ num_classes-1）→ 直接用
         例如 malignancy=0 → 索引 0, malignancy=1 → 索引 1
      3. 标签值是 1-based（1 ~ num_classes）→ 减 1
         例如 tirads=3 → 索引 2

    返回: dict {filename: label_index(0-based)}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 构建 class_names 映射表: {"1": 0, "2": 1, ...} 或 {"benign": 0, "malignant": 1}
    name_to_idx = {str(name): i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    label_map = {}
    missing = []
    remapped = 0
    for rec in records:
        fname = rec.get("filename")
        if fname is None:
            continue
        if label_field not in rec:
            missing.append(fname)
            continue

        raw_label = rec[label_field]
        label_str = str(raw_label)

        # 策略 1: 标签值在 class_names 中找到
        if label_str in name_to_idx:
            label_idx = name_to_idx[label_str]
            remapped += 1
        # 策略 2: 已是 0-based
        elif isinstance(raw_label, (int, float)) and 0 <= int(raw_label) < num_classes:
            label_idx = int(raw_label)
        # 策略 3: 1-based, 减 1
        elif isinstance(raw_label, (int, float)) and 1 <= int(raw_label) <= num_classes:
            label_idx = int(raw_label) - 1
            remapped += 1
        else:
            print(f"  ⚠ 无法映射标签: {fname} {label_field}={raw_label}, 跳过")
            continue

        label_map[fname] = label_idx

    if missing:
        print(f"  ⚠ 以下 {len(missing)} 条记录缺少字段 '{label_field}'，将跳过: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    # 打印映射信息
    if remapped > 0:
        print(f"  标签映射: {remapped} 条标签通过 class_names 映射为 0-based 索引")

    return label_map


@torch.no_grad()
def batch_infer(model: BiomedCLIPClassifier, img_paths: list,
                preprocess, device: torch.device, batch_size: int):
    """
    批量推理，返回每张图片的概率数组。
    返回: np.ndarray, shape (N, num_classes)
    """
    all_probs = []
    for i in tqdm(range(0, len(img_paths), batch_size), desc="推理"):
        batch_paths = img_paths[i: i + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
            except Exception as e:
                print(f"\n  ⚠ 读取图片失败: {p} ({e})，使用零张量替代")
                tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(tensors).to(device)
        logits = model(batch)
        probs = logits.softmax(dim=1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)  # (N, num_classes)


def save_csv(output_path: str, filenames: list, all_probs: np.ndarray,
             class_names: list):
    """保存分类结果到 CSV"""
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["filename", "predict_label", "predict_confidence"]
    for cname in class_names:
        fieldnames.append(f"prob_{cname}")

    rows = []
    for fname, probs in zip(filenames, all_probs):
        pred_idx = int(np.argmax(probs))
        pred_name = class_names[pred_idx]
        pred_conf = float(probs[pred_idx])

        row = {
            "filename": fname,
            "predict_label": pred_name,
            "predict_confidence": round(pred_conf, 6),
        }
        for i, cname in enumerate(class_names):
            row[f"prob_{cname}"] = round(float(probs[i]), 6)
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


# ============================================================================
# 评估：使用项目级统一指标模块 (cls_metrics)
# ============================================================================

def run_evaluation(filenames, all_probs, label_map, class_names,
                   num_classes, eval_output, n_bootstrap, seed):
    """对有标签的样本进行评估，保存结果到 .log 文件。"""
    if not SKLEARN_AVAILABLE:
        print("  ⚠ scikit-learn 未安装，无法进行性能评估。请执行: pip install scikit-learn")
        return

    # 匹配有标签的样本
    y_true_list, y_pred_list, y_prob_list = [], [], []
    per_sample = []  # (filename, true_label, pred_label)
    skipped = []
    out_of_range = []

    for fname, probs in zip(filenames, all_probs):
        if fname not in label_map:
            skipped.append(fname)
            continue
        true_label = label_map[fname]
        pred_idx = int(np.argmax(probs))

        # 校验标签是否在有效范围 [0, num_classes-1] 内
        if true_label < 0 or true_label >= num_classes:
            out_of_range.append((fname, true_label))
            continue

        y_true_list.append(true_label)
        y_pred_list.append(pred_idx)
        y_prob_list.append(probs)
        per_sample.append((fname, true_label, pred_idx))

    if out_of_range:
        print(f"  ⚠ 以下 {len(out_of_range)} 条样本标签超出范围 [0, {num_classes-1}]，已跳过:")
        for fname, label in out_of_range[:10]:
            print(f"    {fname}: {label}")
        if len(out_of_range) > 10:
            print(f"    ... 共 {len(out_of_range)} 条")

    if not y_true_list:
        print("  ⚠ 没有找到任何匹配的标签记录，无法评估。"
              "请检查 JSON 中的 filename 是否与图片文件名一致。")
        return

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(y_prob_list)

    # Bootstrap CI95 (统一模块)
    results, valid_iters = _bootstrap_ci(
        y_true, y_pred, y_prob, num_classes,
        n_boot=n_bootstrap, seed=seed,
    )

    # 生成报告
    report_str = format_metrics_report(results, labels=y_true)

    # 打印到终端
    print(report_str)

    # 保存到 .log 文件
    out_dir = os.path.dirname(os.path.abspath(eval_output))
    os.makedirs(out_dir, exist_ok=True)
    with open(eval_output, "w", encoding="utf-8") as f:
        f.write(report_str)
        f.write("\n")


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()

    # 参数校验
    if len(args.class_names) != args.num_classes:
        print(f"错误: --class_names 长度 ({len(args.class_names)}) "
              f"与 --num_classes ({args.num_classes}) 不一致")
        sys.exit(1)

    if args.label_json is not None and args.label_field is None:
        print("错误: 提供了 --label_json 时必须同时指定 --label_field")
        sys.exit(1)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 检查文件夹
    if not os.path.isdir(args.folder):
        print(f"错误: 图片文件夹不存在: {args.folder}")
        sys.exit(1)

    # 检查本地模型目录
    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        print(f"错误: 预训练模型目录不存在: {model_dir}")
        sys.exit(1)

    # 加载模型
    model = load_model(args.ckpt, model_dir, args.num_classes, device)
    preprocess = get_preprocess(image_size=224)

    # 收集图片
    filenames = collect_images(args.folder)
    if not filenames:
        print(f"错误: 文件夹中未找到图片文件: {args.folder}")
        sys.exit(1)

    img_paths = [os.path.join(args.folder, f) for f in filenames]

    # 打印配置
    print("=" * 60)
    print(f"权重:     {args.ckpt}")
    print(f"数据:     {args.folder}")
    print(f"类别数:   {args.num_classes}")
    if args.label_json:
        print(f"标签字段: {args.label_field}")
    print(f"设备:     {device}")
    print("=" * 60)

    # 批量推理
    all_probs = batch_infer(model, img_paths, preprocess, device,
                            args.batch_size)

    # 保存 CSV
    save_csv(args.output, filenames, all_probs, args.class_names)

    # 可选：性能评估（含 CI95）
    if args.label_json:
        label_map = load_label_json(args.label_json, args.label_field, args.class_names)

        # 确定评估结果保存路径
        if args.eval_output:
            eval_output = args.eval_output
        else:
            out_dir = os.path.dirname(os.path.abspath(args.output))
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            eval_output = os.path.join(out_dir, f"eval_result_{timestamp}.log")

        run_evaluation(filenames, all_probs, label_map, args.class_names,
                       args.num_classes, eval_output,
                       args.n_bootstrap, args.seed)

    print(f"\n  完成")


if __name__ == "__main__":
    main()
