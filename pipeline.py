#!/usr/bin/env python3
"""
端到端流水线脚本
================
一键完成：数据预处理 → 生成配置 → 运行推理

流程:
  Step 1. 调用 prepare_data.py 处理原始数据（ROI 裁剪 + 掩码对齐 + 标签提取）
  Step 2. 基于现有 config.yaml 生成新配置，将数据路径指向 processed/ 目录
  Step 3. 调用 run_all.py 用新配置运行全部推理

配置优先级:
  命令行参数 > config.yaml 中 prepare 段 > 代码默认值

config.yaml 中 prepare 段格式:
  prepare:
    input_dir:      ./datasets/甲状腺私有数据/新建文件夹
    output_dir:     ./datasets/processed
    roi_checkpoint: /path/to/best_dice_model.pth

用法:
  # 完整流水线（从 config.yaml 读取输入目录和 ROI 权重）
  python pipeline.py

  # 命令行覆盖输入目录
  python pipeline.py --input_dir /path/to/raw_data

  # 跳过 ROI 提取
  python pipeline.py --skip_roi

  # 只运行特定任务
  python pipeline.py --tasks gland nodule

  # 只打印命令不执行
  python pipeline.py --dry_run

  # 跳过预处理，直接用已有的 processed/ 数据生成配置并运行
  python pipeline.py --skip_prepare
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ============================================================================
# 路径常量
# ============================================================================

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"
PREPARE_DATA_SCRIPT = ROOT / "my_ThyroidROI" / "newcode" / "prepare_data.py"

# 代码级默认值（当 config.yaml 中没有 prepare 段时使用）
_FALLBACK_INPUT_DIR = ROOT / "datasets" / "甲状腺私有数据" / "新建文件夹"
_FALLBACK_PROCESSED_DIR = ROOT / "datasets" / "processed"
_FALLBACK_CHECKPOINT = "/mnt/wangbd8/workspace/ThyroidAgent/ThyroidROI/outputs/best_dice_model.pth"


def _load_prepare_config(config_path):
    """从 config.yaml 中读取 prepare 段，返回 (input_dir, processed_dir, checkpoint)。

    如果 config.yaml 不存在或没有 prepare 段，返回代码级默认值。
    """
    try:
        import yaml
    except ImportError:
        return _FALLBACK_INPUT_DIR, _FALLBACK_PROCESSED_DIR, _FALLBACK_CHECKPOINT

    p = Path(config_path)
    if not p.is_file():
        return _FALLBACK_INPUT_DIR, _FALLBACK_PROCESSED_DIR, _FALLBACK_CHECKPOINT

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "prepare" not in cfg:
        return _FALLBACK_INPUT_DIR, _FALLBACK_PROCESSED_DIR, _FALLBACK_CHECKPOINT

    prep = cfg["prepare"]
    input_dir = prep.get("input_dir", _FALLBACK_INPUT_DIR)
    processed_dir = prep.get("output_dir", _FALLBACK_PROCESSED_DIR)
    checkpoint = prep.get("roi_checkpoint", _FALLBACK_CHECKPOINT)

    # 相对路径以 ROOT 为基准
    def _resolve(p):
        p = Path(p)
        if not p.is_absolute():
            p = ROOT / p
        return p

    return _resolve(input_dir), _resolve(processed_dir), checkpoint


# ============================================================================
# Step 1: 数据预处理
# ============================================================================

def run_prepare_data(input_dir, processed_dir, checkpoint, skip_roi, dry_run):
    """调用 prepare_data.py 进行数据预处理。"""
    print("\n" + "=" * 70)
    print("  Step 1: 数据预处理 (prepare_data.py)")
    print("=" * 70)

    cmd = [
        sys.executable, str(PREPARE_DATA_SCRIPT),
        "--input_dir", str(input_dir),
        "--output_dir", str(processed_dir),
    ]

    if skip_roi:
        cmd.append("--skip_roi")
    elif checkpoint and Path(checkpoint).exists():
        cmd.extend(["--checkpoint", checkpoint])
    else:
        print("  ⚠ 未提供有效 checkpoint，自动跳过 ROI 提取")
        cmd.append("--skip_roi")

    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {processed_dir}")
    if not skip_roi and "--checkpoint" in cmd:
        print(f"  ROI 权重: {checkpoint}")
    else:
        print(f"  ROI 提取: 跳过")
    print(f"  命令: {' '.join(cmd)}")
    print()

    if dry_run:
        print("  [DRY RUN] 跳过执行")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ✗ 预处理失败 (退出码 {result.returncode})，耗时 {elapsed:.1f}s")
        return False

    print(f"\n  ✓ 预处理完成，耗时 {elapsed:.1f}s")
    return True


# ============================================================================
# Step 2: 生成新配置文件
# ============================================================================

def generate_config(base_config_path, processed_dir, output_path, use_dino_mask=False):
    """基于基础 config.yaml 生成新配置，将数据路径指向 processed/ 目录。

    修改的字段:
      - datasets.gland_images  → processed/images
      - datasets.gland_masks   → processed/gland_masks
      - datasets.nodule_images → processed/images
      - datasets.nodule_masks  → processed/nodule_masks
      - datasets.binary_images → processed/images
      - datasets.tirads_images → processed/images
      - labels.binary_json     → processed/labels.json
      - labels.tirads_json     → processed/labels.json
      - labels.binary_field    → malignancy (保持)
      - labels.tirads_field    → tirads (保持)

    其余字段（weights, pretrained, device 等）保持不变。
    """
    try:
        import yaml
    except ImportError:
        print("  [错误] 未安装 PyYAML，请运行: pip install pyyaml")
        sys.exit(1)

    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 相对路径，以项目根目录为基准
    rel_processed = os.path.relpath(str(processed_dir), str(ROOT))

    cfg["datasets"]["gland_images"]  = f"./{rel_processed}/images"
    cfg["datasets"]["gland_masks"]   = f"./{rel_processed}/gland_masks"
    cfg["datasets"]["nodule_images"] = f"./{rel_processed}/images"
    cfg["datasets"]["nodule_masks"]  = f"./{rel_processed}/nodule_masks"
    cfg["datasets"]["binary_images"] = f"./{rel_processed}/images"
    cfg["datasets"]["tirads_images"]  = f"./{rel_processed}/images"

    cfg["labels"]["binary_json"] = f"./{rel_processed}/labels.json"
    cfg["labels"]["tirads_json"] = f"./{rel_processed}/labels.json"
    cfg["labels"]["binary_field"] = "malignancy"
    cfg["labels"]["tirads_field"]  = "tirads"

    if use_dino_mask:
        cfg["use_dino_mask_for_cls"] = True

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  ✓ 新配置已保存: {output_path}")
    print(f"    数据集路径指向: {rel_processed}/")
    return output_path


# ============================================================================
# Step 3: 运行推理
# ============================================================================

def run_inference(new_config, tasks, models, dry_run):
    """调用 run_all.py 运行推理。"""
    print("\n" + "=" * 70)
    print("  Step 3: 运行推理 (run_all.py)")
    print("=" * 70)

    cmd = [
        sys.executable, str(ROOT / "run_all.py"),
        "--config", str(new_config),
    ]

    if tasks:
        cmd.extend(["--tasks"] + tasks)
    if models:
        cmd.extend(["--models"] + models)
    if dry_run:
        cmd.append("--dry_run")

    print(f"  配置文件: {new_config}")
    print(f"  任务: {', '.join(tasks) if tasks else '全部'}")
    if models:
        print(f"  模型筛选: {', '.join(models)}")
    print(f"  命令: {' '.join(cmd)}")
    print()

    if dry_run:
        print("  [DRY RUN] 跳过执行")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ✗ 推理失败 (退出码 {result.returncode})，耗时 {elapsed:.1f}s")
        return False

    print(f"\n  ✓ 推理完成，耗时 {elapsed:.1f}s")
    return True


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="端到端流水线：数据预处理 → 生成配置 → 运行推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流水线（从 config.yaml 读取输入目录和 ROI 权重）
  python pipeline.py

  # 命令行覆盖输入目录
  python pipeline.py --input_dir /path/to/raw_data

  # 跳过 ROI 提取
  python pipeline.py --skip_roi

  # 只运行分割任务
  python pipeline.py --tasks gland nodule

  # 只打印命令不执行
  python pipeline.py --dry_run

  # 跳过预处理（已有 processed/ 数据）
  python pipeline.py --skip_prepare

配置:
  修改 config.yaml 中 prepare 段即可配置输入目录、输出目录和 ROI 权重:
    prepare:
      input_dir:      ./datasets/甲状腺私有数据/新建文件夹
      output_dir:     ./datasets/processed
      roi_checkpoint: /path/to/best_dice_model.pth

  命令行参数 --input_dir / --processed_dir / --checkpoint 优先于配置文件。
        """,
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="原始数据目录（默认: 从 config.yaml 的 prepare.input_dir 读取）",
    )
    parser.add_argument(
        "--processed_dir", type=str, default=None,
        help="预处理输出目录（默认: 从 config.yaml 的 prepare.output_dir 读取）",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="ROI 提取模型权重（默认: 从 config.yaml 的 prepare.roi_checkpoint 读取）",
    )
    parser.add_argument(
        "--skip_roi", action="store_true",
        help="跳过 ROI 提取，仅复制图像和掩码",
    )
    parser.add_argument(
        "--skip_prepare", action="store_true",
        help="跳过预处理步骤（使用已有的 processed/ 数据）",
    )
    parser.add_argument(
        "--base_config", type=str, default=str(DEFAULT_CONFIG),
        help=f"基础配置文件（默认: {DEFAULT_CONFIG}）",
    )
    parser.add_argument(
        "--new_config", type=str, default="",
        help="生成的新配置文件路径（默认: <processed_dir>/config.yaml）",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        choices=["gland", "nodule", "binary", "tirads"],
        help="要运行的任务（默认全部）",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="只运行指定的模型",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="只打印命令不执行",
    )
    parser.add_argument(
        "--use_dino_mask", action="store_true",
        help="使用 dinov3_unet 腺体掩码作为分类任务的 mask（自动开启 save_masks）",
    )
    args = parser.parse_args()

    # 从 config.yaml 读取 prepare 段作为默认值
    cfg_input_dir, cfg_processed_dir, cfg_checkpoint = _load_prepare_config(args.base_config)

    # 命令行参数优先，否则用 config 中的值
    input_dir = Path(args.input_dir) if args.input_dir else cfg_input_dir
    processed_dir = Path(args.processed_dir) if args.processed_dir else cfg_processed_dir
    checkpoint = args.checkpoint if args.checkpoint else cfg_checkpoint
    new_config_path = Path(args.new_config) if args.new_config else processed_dir / "config.yaml"

    print("=" * 70)
    print("  甲状腺推理流水线")
    print("=" * 70)
    print(f"  输入目录:   {input_dir}")
    print(f"  预处理目录: {processed_dir}")
    print(f"  基础配置:   {args.base_config}")
    print(f"  新配置:     {new_config_path}")
    if checkpoint and not args.skip_roi:
        print(f"  ROI 权重:   {checkpoint}")
    print(f"  任务:       {', '.join(args.tasks) if args.tasks else '全部'}")
    if args.models:
        print(f"  模型筛选:   {', '.join(args.models)}")
    print(f"  Dry run:    {args.dry_run}")
    if args.use_dino_mask:
        print(f"  DINO掩码:   开启（autogluon 使用 dinov3_unet 腺体掩码）")
    print("=" * 70)

    # ---- Step 1: 数据预处理 ----
    if not args.skip_prepare:
        ok = run_prepare_data(
            input_dir, processed_dir, checkpoint, args.skip_roi, args.dry_run
        )
        if not ok:
            sys.exit(1)
    else:
        print("\n  跳过预处理（--skip_prepare）")
        if not processed_dir.exists():
            print(f"  [错误] 预处理目录不存在: {processed_dir}")
            sys.exit(1)

    # ---- Step 2: 生成新配置 ----
    print("\n" + "-" * 70)
    print("  Step 2: 生成新配置")
    print("-" * 70)

    # 配置生成只是写文件，dry_run 也执行，以便后续 run_all 能加载
    new_config_path.parent.mkdir(parents=True, exist_ok=True)
    generate_config(args.base_config, processed_dir, new_config_path, args.use_dino_mask)

    # ---- Step 3: 运行推理 ----
    ok = run_inference(new_config_path, args.tasks, args.models, args.dry_run)
    if not ok:
        sys.exit(1)

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("  流水线完成!")
    print("=" * 70)
    print(f"  预处理目录: {processed_dir}")
    print(f"  配置文件:   {new_config_path}")
    print(f"  推理结果:   {processed_dir.parent / 'results'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
