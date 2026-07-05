#!/usr/bin/env python3
"""
精简模型权重 checkpoint 文件
============================
读取 config.yaml 中的权重路径，去除 checkpoint 中的冗余权重，
减小磁盘占用。修改后的 checkpoint 与现有推理代码完全兼容。

优化内容:
  1. MedSigLIP:     去除文本编码器权重 (~2.4GB/文件)
  2. UltraFedFM 分类: 去除 optimizer/scaler 状态 (~700MB/文件)
  3. MedSegX:        检查 SAM 预训练目录，报告不必要的大文件

用法:
  # 预览（不修改任何文件）
  python slim_checkpoints.py --dry_run

  # 执行精简（自动备份原文件为 .bak）
  python slim_checkpoints.py

  # 恢复备份
  python slim_checkpoints.py --restore

  # 指定其他配置文件
  python slim_checkpoints.py --config /path/to/config.yaml
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


# ============================================================================
# 工具函数
# ============================================================================

def get_file_size(path):
    """返回文件大小（MB）。"""
    return Path(path).stat().st_size / (1024 * 1024)


def backup_file(path):
    """将原文件备份为 .bak（如果 .bak 已存在则跳过）。"""
    bak = Path(path).with_suffix(Path(path).suffix + ".bak")
    if bak.exists():
        return False
    shutil.copy2(path, bak)
    return True


def restore_file(path):
    """从 .bak 恢复原文件。"""
    bak = Path(path).with_suffix(Path(path).suffix + ".bak")
    if not bak.exists():
        return False
    shutil.copy2(bak, path)
    return True


def count_keys_by_prefix(state_dict, prefixes):
    """统计 state_dict 中以指定前缀开头的 key 数量。"""
    count = 0
    for k in state_dict:
        for prefix in prefixes:
            if k.startswith(prefix):
                count += 1
                break
    return count


# ============================================================================
# 1. MedSigLIP: 去除文本编码器权重
# ============================================================================

def slim_medsiglip(ckpt_path, dry_run=False):
    """
    MedSigLIP checkpoint 包含完整的 full_model（视觉+文本编码器）。
    推理只用视觉编码器，文本编码器是冗余的。

    state_dict 中的文本编码器 keys:
      - full_model.text_model.*
      - full_model.text_projection.*
      - full_model.text_*

    去除后从 ~4GB 降到 ~1.6GB。
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)

    text_prefixes = [
        "full_model.text_model",
        "full_model.text_projection",
        "full_model.text",
    ]
    text_keys = [k for k in sd if any(k.startswith(p) for p in text_prefixes)]

    if not text_keys:
        print(f"  [跳过] 未找到文本编码器 keys，可能已精简")
        return 0

    # 统计文本编码器参数量
    text_params = sum(sd[k].numel() for k in text_keys)
    total_params = sum(v.numel() for v in sd.values())

    print(f"  文本编码器 keys: {len(text_keys)}")
    print(f"  文本编码器参数: {text_params / 1e6:.1f}M / 总 {total_params / 1e6:.1f}M "
          f"({text_params / total_params * 100:.1f}%)")

    if dry_run:
        print(f"  [DRY RUN] 可节省约 {text_params * 4 / 1024 / 1024:.0f} MB")
        return text_params * 4 / (1024 * 1024)

    # 过滤掉文本编码器 keys
    new_sd = {k: v for k, v in sd.items() if k not in text_keys}
    ckpt["model_state_dict"] = new_sd

    backup_file(ckpt_path)
    torch.save(ckpt, ckpt_path)

    saved = get_file_size(f"{ckpt_path}.bak") - get_file_size(ckpt_path)
    print(f"  已精简: {get_file_size(f'{ckpt_path}.bak'):.1f} MB -> {get_file_size(ckpt_path):.1f} MB "
          f"(节省 {saved:.1f} MB)")
    return saved


# ============================================================================
# 2. UltraFedFM 分类: 去除 optimizer/scaler
# ============================================================================

def slim_ultrafedfm_classify(ckpt_path, dry_run=False):
    """
    UltraFedFM 分类 checkpoint 使用 MAE 训练框架保存，格式:
      {'model': state_dict, 'optimizer': ..., 'scaler': ..., 'epoch': ...}

    推理只用 'model'，optimizer/scaler 是冗余的。
    去除后从 ~1GB 降到 ~350MB。
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        print(f"  [跳过] 不是 MAE 框架格式（无 'model' key），可能已精简")
        return 0

    redundant_keys = [k for k in ckpt if k != "model"]
    print(f"  冗余 keys: {redundant_keys}")

    if dry_run:
        # 估算冗余大小
        model_size = sum(v.numel() * v.element_size() for v in ckpt["model"].values())
        total_size = Path(ckpt_path).stat().st_size
        redundant_size = total_size - model_size
        print(f"  [DRY RUN] 模型 ~{model_size / 1024 / 1024:.1f} MB, "
              f"冗余 ~{redundant_size / 1024 / 1024:.1f} MB")
        return redundant_size / (1024 * 1024)

    new_ckpt = {"model": ckpt["model"]}
    # 保留可能有用的元数据
    for key in ["class_names", "epoch"]:
        if key in ckpt:
            new_ckpt[key] = ckpt[key]

    backup_file(ckpt_path)
    torch.save(new_ckpt, ckpt_path)

    saved = get_file_size(f"{ckpt_path}.bak") - get_file_size(ckpt_path)
    print(f"  已精简: {get_file_size(f'{ckpt_path}.bak'):.1f} MB -> {get_file_size(ckpt_path):.1f} MB "
          f"(节省 {saved:.1f} MB)")
    return saved


# ============================================================================
# 3. MedSegX: 检查 SAM 预训练目录
# ============================================================================

def check_medsegx_sam(sam_dir, dry_run=False):
    """
    MedSegX 的 checkpoint 本身已精简（只含 adapter + decoder）。
    但 SAM 预训练目录下可能存了多个 ViT 变体（vit_b / vit_l / vit_h），
    其中只有 vit_b 被使用。

    列出所有文件及大小，提示用户手动删除不用的。
    """
    sam_dir = Path(sam_dir)
    if not sam_dir.exists():
        print(f"  [跳过] SAM 预训练目录不存在: {sam_dir}")
        return 0

    print(f"  SAM 预训练目录: {sam_dir}")
    total = 0
    files_info = []
    for f in sorted(sam_dir.iterdir()):
        if not f.is_file():
            continue
        # 跳过 .gitkeep 等隐藏文件
        if f.name.startswith("."):
            continue
        size = get_file_size(f)
        total += size
        used = "✓ 使用" if "vit_b" in f.name else "✗ 未使用"
        files_info.append((f.name, size, used))

    for name, size, used in files_info:
        print(f"    {used}  {name}: {size:.1f} MB")

    print(f"  总计: {total:.1f} MB")

    # 提示未使用的文件
    unused = [(name, size) for name, size, used in files_info if used.startswith("✗")]
    if unused:
        total_unused = sum(s for _, s in unused)
        print(f"\n  ⚠ 以下文件未被使用，可手动删除（节省 {total_unused:.1f} MB）:")
        for name, size in unused:
            print(f"    rm '{sam_dir / name}'  # {size:.1f} MB")
        return total_unused

    return 0


# ============================================================================
# 4. BiomedCLIP: 检查是否有 optimizer
# ============================================================================

def slim_biomedclip(ckpt_path, dry_run=False):
    """
    BiomedCLIP checkpoint 是纯 state_dict（visual.* + classifier.*）。
    如果 visual 编码器是微调过的，不能删除（与预训练权重不同）。
    这里只检查是否有意外冗余。
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 如果是嵌套字典格式，提取 model
    if isinstance(ckpt, dict) and "model" in ckpt and not any(
        k.startswith("visual.") or k.startswith("classifier.")
        for k in ckpt.keys()
    ):
        redundant = [k for k in ckpt if k != "model"]
        print(f"  嵌套格式，冗余 keys: {redundant}")

        if dry_run:
            model_size = sum(v.numel() * v.element_size() for v in ckpt["model"].values())
            total_size = Path(ckpt_path).stat().st_size
            saved = (total_size - model_size) / (1024 * 1024)
            print(f"  [DRY RUN] 可节省约 {saved:.1f} MB")
            return saved

        new_ckpt = {"model": ckpt["model"]}
        backup_file(ckpt_path)
        torch.save(new_ckpt, ckpt_path)
        saved = get_file_size(f"{ckpt_path}.bak") - get_file_size(ckpt_path)
        print(f"  已精简: {get_file_size(f'{ckpt_path}.bak'):.1f} MB -> {get_file_size(ckpt_path):.1f} MB "
              f"(节省 {saved:.1f} MB)")
        return saved

    # 统计 visual 和 classifier 参数
    if isinstance(ckpt, dict):
        visual_keys = [k for k in ckpt if k.startswith("visual.")]
        classifier_keys = [k for k in ckpt if k.startswith("classifier.")]
        other_keys = [k for k in ckpt if not k.startswith("visual.") and not k.startswith("classifier.")]

        print(f"  visual.* keys: {len(visual_keys)}")
        print(f"  classifier.* keys: {len(classifier_keys)}")
        if other_keys:
            print(f"  其他 keys: {len(other_keys)} (前5: {other_keys[:5]})")

    print(f"  [跳过] 纯 state_dict 格式，无 optimizer 可去除")
    print(f"  注意: visual.* 是微调后的权重，不能删除（与预训练目录不同）")
    return 0


# ============================================================================
# 主流程
# ============================================================================

def load_config(config_path):
    """加载 config.yaml，返回 weights 和 pretrained 段。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def collect_targets(cfg):
    """从 config 中收集所有需要精简的 checkpoint 路径和对应处理函数。"""
    weights = cfg.get("weights", {})
    pretrained = cfg.get("pretrained", {})

    targets = []

    # MedSigLIP (分类): 去除文本编码器
    for task in ["binary", "tirads"]:
        w = weights.get(task, {}).get("medsiglip")
        if w:
            targets.append({
                "name": f"MedSigLIP ({task})",
                "path": w,
                "fn": slim_medsiglip,
            })

    # UltraFedFM 分类: 去除 optimizer
    for task in ["binary", "tirads"]:
        w = weights.get(task, {}).get("ultrafedfm")
        if w:
            targets.append({
                "name": f"UltraFedFM 分类 ({task})",
                "path": w,
                "fn": slim_ultrafedfm_classify,
            })

    # BiomedCLIP 分类: 检查冗余
    for task in ["binary", "tirads"]:
        w = weights.get(task, {}).get("biomedclip")
        if w:
            targets.append({
                "name": f"BiomedCLIP ({task})",
                "path": w,
                "fn": slim_biomedclip,
            })

    # MedSegX: 检查 SAM 预训练目录
    sam_dir = pretrained.get("medsegx_sam_dir")
    if sam_dir:
        targets.append({
            "name": "MedSegX SAM 预训练目录",
            "path": sam_dir,
            "fn": check_medsegx_sam,
            "is_dir": True,
        })

    return targets


def main():
    parser = argparse.ArgumentParser(
        description="精简模型权重 checkpoint 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览（不修改任何文件）
  python slim_checkpoints.py --dry_run

  # 执行精简
  python slim_checkpoints.py

  # 恢复备份
  python slim_checkpoints.py --restore
        """,
    )
    parser.add_argument(
        "--config", type=str, default=str(ROOT / "config.yaml"),
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="只预览不修改",
    )
    parser.add_argument(
        "--restore", action="store_true",
        help="从 .bak 备份恢复原文件",
    )
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)
    targets = collect_targets(cfg)

    if not targets:
        print("未找到需要精简的 checkpoint，请检查 config.yaml")
        return

    # 恢复模式
    if args.restore:
        print("\n" + "=" * 70)
        print("  恢复备份模式")
        print("=" * 70)
        for t in targets:
            path = ROOT / t["path"] if not os.path.isabs(t["path"]) else Path(t["path"])
            if t.get("is_dir"):
                continue
            if path.exists():
                if restore_file(path):
                    print(f"  ✓ 已恢复: {path}")
                else:
                    print(f"  - 无备份: {path}")
            else:
                print(f"  - 文件不存在: {path}")
        return

    # 精简模式
    print("\n" + "=" * 70)
    print(f"  {'[DRY RUN] ' if args.dry_run else ''}精简 Checkpoint 文件")
    print("=" * 70)
    print(f"  配置文件: {args.config}")
    print(f"  目标数量: {len(targets)}")
    print()

    total_saved = 0
    total_saved_mb = 0

    for t in targets:
        name = t["name"]
        rel_path = t["path"]
        path = ROOT / rel_path if not os.path.isabs(rel_path) else Path(rel_path)

        print("-" * 70)
        print(f"  {name}")
        print(f"  路径: {path}")

        if not path.exists():
            print(f"  [跳过] 文件不存在")
            print()
            continue

        if t.get("is_dir"):
            saved = t["fn"](path, dry_run=args.dry_run)
            if saved:
                total_saved_mb += saved
        else:
            size_before = get_file_size(path)
            print(f"  当前大小: {size_before:.1f} MB")
            saved = t["fn"](path, dry_run=args.dry_run)
            if saved > 0:
                total_saved_mb += saved

        print()

    # 汇总
    print("=" * 70)
    if args.dry_run:
        print(f"  [DRY RUN] 预计可节省约 {total_saved_mb:.0f} MB ({total_saved_mb / 1024:.1f} GB)")
    else:
        print(f"  精简完成! 共节省约 {total_saved_mb:.0f} MB ({total_saved_mb / 1024:.1f} GB)")
        print(f"  备份文件已保存为 .bak，可用 --restore 恢复")
    print("=" * 70)


if __name__ == "__main__":
    main()
