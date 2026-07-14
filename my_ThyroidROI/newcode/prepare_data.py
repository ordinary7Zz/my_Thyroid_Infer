#!/usr/bin/env python3
"""
数据预处理脚本：将甲状腺私有数据转换为模型可用的格式。

流程:
  1. 用 ROIExtractor (Swin-UNet) 对原图提取甲状腺 ROI 并裁剪
  2. 用相同的裁剪参数处理腺体掩码 (ORG1) 和结节掩码 (ROI1)
  3. 从 INI 文件提取良恶性和 TI-RADS 标签

本脚本应放在 my_ThyroidROI/newcode/ 目录下，与 roi_extractor.py 同级。
直接 import 同目录下的 roi_extractor 模块，无需 sys.path hack。

扫描方式: 以 INI 标签文件为驱动，对每个 INI 查找同名原图和掩码（_ORG1/_ROI1）。
          INI 数量 >= 图像数量（部分 INI 可能无对应图像，将被跳过并记录）。
输入: <repo>/datasets/甲状腺私有数据/新建文件夹/ 下的 INI 标签文件及对应的原图(_ORG1/_ROI1)
输出: <repo>/datasets/processed/ 下的:
  - images/        裁剪后的原始图像
  - gland_masks/   裁剪后的腺体掩码 (文件名与原图一致)
  - nodule_masks/  裁剪后的结节掩码 (文件名与原图一致)
  - labels.json    分类标签 (malignancy + tirads, 缺失用 -1)

标签映射:
  良恶性 (pathologic 字段):
    良/良性/0/benign → 0
    恶/恶性/1/malignant/癌 → 1
    空/未知 → -1
  TI-RADS (birads 字段):
    1类→1, 2类→2, 3类→3, 4a/4b/4c类→4, 5类→5
    空/未知 → -1

用法:
  # 带 ROI 提取（需要模型权重）
  python prepare_data.py --checkpoint /path/to/best_dice_model.pth

  # 跳过 ROI 提取，仅复制和标签处理
  python prepare_data.py --skip_roi

  # 关闭文件名匿名化（默认开启，将中心+患者信息替换为哈希，保留时间戳）
  python prepare_data.py --no_anonymize

  # 不保存 原名→匿名名 映射（默认保存，便于回溯）
  python prepare_data.py --no_save_name_mapping
"""

import os
import re
import json
import shutil
import hashlib
import argparse
from pathlib import Path
from configparser import ConfigParser

import cv2
import numpy as np

# ========================= 配置 =========================
# SCRIPT_DIR = my_ThyroidROI/newcode/, 项目根目录 = 上两级
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "datasets" / "甲状腺私有数据" / "新建文件夹"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# 后缀标识
ORG_SUFFIX = "_ORG1"  # 腺体掩码
ROI_SUFFIX = "_ROI1"  # 结节掩码


# ========================= 文件名匿名化 =========================
def anonymize_filename(stem: str, ext: str) -> str:
    """
    匿名化文件名：分别抹除中心信息（前 3 段）和患者信息（第 4 段），
    保留时间戳（第 5 段及以后）。

    文件名结构: THYB_S_AN01_ND000091_202052842015
                └── 中心 ──┘ └ 患者者 ┘ └ 时间戳 ┘

    匿名化后:   3d0b487c4929_5f8a2b1c9d3e_202052842015
                └─ 中心哈希 ─┘└─ 患者哈希 ┘└ 时间戳 ┘

    规则:
      - 中心（前 3 段用 _ 连接）和患者（第 4 段）分别计算 SHA-256 前 12 位
      - 同中心 → 同哈希，同患者 → 同哈希（确定性，非随机）
      - 时间戳（第 5 段及以后）原样保留
      - 段数 < 4 时回退为对完整 stem 哈希

    例: THYB_S_AN01_ND000091_202052842015 → 3d0b487c4929_5f8a2b1c9d3e_202052842015
        THYB_S_AN01_ND000092_20206154032  → 3d0b487c4929_a1b2c3d4e5f6_20206154032
    """
    parts = stem.split("_")

    if len(parts) >= 4:
        # 前 3 段 = 中心，第 4 段 = 患者，第 5 段及以后 = 时间戳（保留）
        center_key = "_".join(parts[:3])
        patient_key = parts[3]
        suffix = "_".join(parts[4:])  # 可能为空

        center_hash = hashlib.sha256(center_key.encode("utf-8")).hexdigest()[:12]
        patient_hash = hashlib.sha256(patient_key.encode("utf-8")).hexdigest()[:12]

        new_stem = f"{center_hash}_{patient_hash}"
        if suffix:
            new_stem = f"{new_stem}_{suffix}"
    else:
        # 段数 < 4：回退为对完整 stem 哈希
        new_stem = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:12]

    return f"{new_stem}{ext}"


# ========================= INI 解析 =========================
def parse_ini(filepath: Path) -> ConfigParser:
    """解析 INI 文件，自动尝试多种编码。"""
    cp = ConfigParser(strict=False, interpolation=None)
    for encoding in ["gb18030", "gbk", "utf-8", "latin-1"]:
        try:
            cp.read(str(filepath), encoding=encoding)
            if cp.sections():
                return cp
        except Exception:
            continue
    return cp


def get_first_label_from_roi(cp: ConfigParser, field: str) -> str:
    """从第一个有非空标签的 ROI 节获取指定字段的值。"""
    for section in cp.sections():
        if not section.startswith("ROI"):
            continue
        if cp.has_option(section, field):
            val = cp.get(section, field, fallback="").strip()
            if val:
                return val
    return ""


# ========================= 标签映射 =========================
def extract_malignancy(pathologic_value: str) -> int:
    """
    从 pathologic 字段提取良恶性标签。

    映射规则 (健壮匹配):
      良/良性/0/benign → 0
      恶/恶性/1/malignant/癌 → 1
      空/未知 → -1
    """
    if not pathologic_value:
        return -1
    val = pathologic_value.strip()
    val_lower = val.lower()

    # 良性: 包含"良"、值为0、包含benign
    if "良" in val or val == "0" or "benign" in val_lower:
        return 0
    # 恶性: 包含"恶"、值为1、包含malign、包含"癌"
    if "恶" in val or val == "1" or "malign" in val_lower or "癌" in val:
        return 1
    return -1


def extract_tirads(birads_value: str) -> int:
    """
    从 birads 字段提取 TI-RADS 分类。

    映射规则:
      1类 → 1, 2类 → 2, 3类 → 3
      4a/4b/4c类 → 4
      5类 → 5
      空/未知 → -1

    匹配策略 (健壮): 搜索字符串中第一个数字字符
      "2类" → 2, "4a类" → 4, "TR3" → 3, "第5类" → 5
    """
    if not birads_value:
        return -1
    val = birads_value.strip()

    # 搜索第一个数字
    match = re.search(r"(\d)", val)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 5:
            return num
        return -1

    # 中文数字
    chinese_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    for ch, num in chinese_map.items():
        if ch in val:
            return num

    return -1


# ========================= ROI 提取 =========================
def init_roi_extractor(checkpoint_path: str):
    """初始化 ROIExtractor（同目录直接 import）。"""
    from roi_extractor import ROIExtractor

    extractor = ROIExtractor(checkpoint_path)
    return extractor


def extract_roi_for_image(extractor, image_path: Path):
    """
    对单张图像提取 ROI，返回裁剪后的 BGR 图像和裁剪参数。

    返回:
      roi_bgr: np.ndarray (H, W, 3) uint8 BGR
      crop_params: dict {'x', 'y', 'w', 'h', 'mask'}
    """
    roi_rgb, crop_params = extractor.extract_roi_with_crop_params(str(image_path))
    # RGB float32 [0,1] → BGR uint8 [0,255]
    roi_bgr = (roi_rgb[:, :, ::-1] * 255).astype("uint8")
    return roi_bgr, crop_params


def crop_mask_with_params(mask_path: Path, crop_params: dict):
    """
    用与原图相同的裁剪参数处理掩码。

    1. 按原图的 x, y, w, h 裁剪掩码
    2. 用处理后的 ROI 掩码屏蔽非甲状腺区域（置 0）

    返回: np.ndarray (H, W) uint8, 裁剪后的二值掩码
    """
    # 读取掩码为灰度图
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩码: {mask_path}")

    # 按原图坐标裁剪
    x, y, w, h = crop_params["x"], crop_params["y"], crop_params["w"], crop_params["h"]
    cropped_mask = mask[y : y + h, x : x + w].copy()

    # 用处理后的 ROI 掩码屏蔽区域外像素
    processed_mask = crop_params["mask"]
    cropped_processed = processed_mask[y : y + h, x : x + w]
    cropped_mask = np.where(cropped_processed > 0, cropped_mask, 0).astype("uint8")

    return cropped_mask


# ========================= 文件查找 =========================
def _find_file_for_stem(input_dir, stem, suffix=""):
    """根据 stem 查找对应的图像/掩码文件，尝试所有支持的图像扩展名。

    参数:
        input_dir: 输入目录 (Path)
        stem:      文件名 stem（不含扩展名）
        suffix:    文件名后缀标识（如 "_ORG1", "_ROI1"，默认空表示原图）

    返回: 找到的 Path，或 None
    """
    for ext in sorted(IMAGE_EXTS):
        candidate = input_dir / f"{stem}{suffix}{ext}"
        if candidate.is_file():
            return candidate
    return None


# ========================= 主流程 =========================
def main():
    parser = argparse.ArgumentParser(
        description="甲状腺数据预处理：ROI 裁剪 + 掩码对齐 + 标签提取"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="ROI 提取模型权重路径 (Swin-UNet)。不提供则跳过 ROI 提取",
    )
    parser.add_argument(
        "--skip_roi",
        action="store_true",
        help="跳过 ROI 提取，仅复制原图和掩码 + 提取标签",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(INPUT_DIR),
        help="输入数据目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="输出目录",
    )
    parser.add_argument(
        "--no_anonymize",
        action="store_true",
        help="关闭文件名匿名化（默认开启：将中心+患者信息替换为哈希，保留时间戳）",
    )
    parser.add_argument(
        "--no_save_name_mapping",
        action="store_true",
        help="不保存 原名→匿名名 的映射文件 name_mapping.json（默认保存，"
        "便于回溯；注意其中包含原始病人信息）",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 决定是否启用 ROI 提取
    use_roi = not args.skip_roi
    extractor = None
    if use_roi:
        if not args.checkpoint:
            print("⚠ 未提供 --checkpoint，将跳过 ROI 提取（仅复制 + 标签）")
            use_roi = False
        elif not Path(args.checkpoint).exists():
            print(f"⚠ 模型权重不存在: {args.checkpoint}，将跳过 ROI 提取")
            use_roi = False

    if use_roi:
        print("=" * 60)
        print("启用 ROI 提取 (Swin-UNet)")
        print(f"  权重: {args.checkpoint}")
        print("=" * 60)
        extractor = init_roi_extractor(args.checkpoint)
    else:
        print("=" * 60)
        print("跳过 ROI 提取，仅复制图像和掩码")
        print("=" * 60)

    anonymize = not args.no_anonymize
    if anonymize:
        print("文件名匿名化: 开启（中心+患者信息替换为哈希，保留时间戳）")
    else:
        print("文件名匿名化: 关闭")

    # 创建输出目录
    images_dir = output_dir / "images"
    gland_masks_dir = output_dir / "gland_masks"
    nodule_masks_dir = output_dir / "nodule_masks"
    for d in [images_dir, gland_masks_dir, nodule_masks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 收集所有 INI 标签文件（以 INI 为驱动扫描图像和掩码）
    ini_files = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".ini":
            continue
        ini_files.append(p)

    print(f"找到 {len(ini_files)} 个 INI 标签文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print()

    labels = []
    name_map = {}  # 原始文件名 → 匿名化文件名（仅 anonymize 时填充）
    stats = {
        "total": 0,
        "roi_success": 0,
        "roi_failed": 0,
        "has_malignancy": 0,
        "has_tirads": 0,
        "has_gland_mask": 0,
        "has_nodule_mask": 0,
        "missing_gland_mask": [],
        "missing_nodule_mask": [],
        "missing_image": [],
        "roi_errors": [],
    }

    for ini_path in ini_files:
        stem = ini_path.stem  # 去掉 .ini 后缀

        # 根据 INI 的 stem 查找对应的原图（尝试所有图像扩展名）
        img_path = _find_file_for_stem(input_dir, stem, suffix="")
        if img_path is None:
            stats["missing_image"].append(ini_path.name)
            continue

        ext = img_path.suffix
        stats["total"] += 1

        # 对应的掩码文件（尝试所有图像扩展名，不再限定与原图同扩展名）
        org_path = _find_file_for_stem(input_dir, stem, suffix=ORG_SUFFIX)
        roi_path = _find_file_for_stem(input_dir, stem, suffix=ROI_SUFFIX)

        # 输出文件名：匿名化时将中心+患者信息替换为哈希
        if anonymize:
            out_name = anonymize_filename(stem, ext)
            name_map[img_path.name] = out_name
        else:
            out_name = img_path.name

        out_image = images_dir / out_name
        out_gland = gland_masks_dir / out_name
        out_nodule = nodule_masks_dir / out_name

        # --- 处理原图 ---
        if use_roi:
            try:
                roi_bgr, crop_params = extract_roi_for_image(extractor, img_path)
                cv2.imwrite(str(out_image), roi_bgr)
                stats["roi_success"] += 1
            except Exception as e:
                print(f"  [ROI 错误] {img_path.name}: {e}，回退为直接复制")
                shutil.copy2(img_path, out_image)
                crop_params = None
                stats["roi_failed"] += 1
                stats["roi_errors"].append((img_path.name, str(e)))
        else:
            shutil.copy2(img_path, out_image)
            crop_params = None

        # --- 处理腺体掩码 (ORG1) ---
        if org_path is not None:
            if use_roi and crop_params is not None:
                try:
                    cropped_gland = crop_mask_with_params(org_path, crop_params)
                    cv2.imwrite(str(out_gland), cropped_gland)
                    stats["has_gland_mask"] += 1
                except Exception as e:
                    print(f"  [腺体掩码错误] {img_path.name}: {e}，回退为直接复制")
                    shutil.copy2(org_path, out_gland)
                    stats["has_gland_mask"] += 1
            else:
                shutil.copy2(org_path, out_gland)
                stats["has_gland_mask"] += 1
        else:
            stats["missing_gland_mask"].append(img_path.name)

        # --- 处理结节掩码 (ROI1) ---
        if roi_path is not None:
            if use_roi and crop_params is not None:
                try:
                    cropped_nodule = crop_mask_with_params(roi_path, crop_params)
                    cv2.imwrite(str(out_nodule), cropped_nodule)
                    stats["has_nodule_mask"] += 1
                except Exception as e:
                    print(f"  [结节掩码错误] {img_path.name}: {e}，回退为直接复制")
                    shutil.copy2(roi_path, out_nodule)
                    stats["has_nodule_mask"] += 1
            else:
                shutil.copy2(roi_path, out_nodule)
                stats["has_nodule_mask"] += 1
        else:
            stats["missing_nodule_mask"].append(img_path.name)

        # --- 提取标签（INI 一定存在，因为以 INI 为驱动扫描）---
        malignancy = -1
        tirads = -1

        cp = parse_ini(ini_path)
        pathologic_val = get_first_label_from_roi(cp, "pathologic")
        malignancy = extract_malignancy(pathologic_val)
        if malignancy != -1:
            stats["has_malignancy"] += 1

        birads_val = get_first_label_from_roi(cp, "birads")
        tirads = extract_tirads(birads_val)
        if tirads != -1:
            stats["has_tirads"] += 1

        labels.append({
            "filename": out_name,
            "malignancy": malignancy,
            "tirads": tirads,
        })

    # 保存标签 JSON
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    # 保存 原名→匿名名 映射（默认保存，便于回溯；含原始病人信息）
    if anonymize and not args.no_save_name_mapping:
        mapping_path = output_dir / "name_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(name_map, f, ensure_ascii=False, indent=4)
        print(f"  名称映射已保存: {mapping_path}")

    # ========================= 统计报告 =========================
    print(f"\n{'=' * 60}")
    print("处理完成!")
    print(f"  INI 标签文件: {len(ini_files)}")
    print(f"  有图像的:     {stats['total']}")
    print(f"  无图像的:     {len(stats['missing_image'])}")
    if use_roi:
        print(f"  ROI 提取成功: {stats['roi_success']} / {stats['total']}")
        print(f"  ROI 提取失败: {stats['roi_failed']} / {stats['total']}")
    print(f"  腺体掩码:     {stats['has_gland_mask']} / {stats['total']}")
    print(f"  结节掩码:     {stats['has_nodule_mask']} / {stats['total']}")
    print(f"  良恶性标签:   {stats['has_malignancy']} / {stats['total']}")
    print(f"  TI-RADS标签:  {stats['has_tirads']} / {stats['total']}")
    if anonymize:
        print(f"  匿名化文件名: {len(name_map)} 个")

    if stats["missing_image"]:
        print(f"\n  INI 无对应图像 ({len(stats['missing_image'])}):")
        for name in stats["missing_image"]:
            print(f"    - {name}")

    if stats["missing_gland_mask"]:
        print(f"\n  缺失腺体掩码 ({len(stats['missing_gland_mask'])}):")
        for name in stats["missing_gland_mask"]:
            print(f"    - {name}")

    if stats["missing_nodule_mask"]:
        print(f"\n  缺失结节掩码 ({len(stats['missing_nodule_mask'])}):")
        for name in stats["missing_nodule_mask"]:
            print(f"    - {name}")

    if stats["roi_errors"]:
        print(f"\n  ROI 提取错误 ({len(stats['roi_errors'])}):")
        for name, err in stats["roi_errors"]:
            print(f"    - {name}: {err}")

    # 标签分布
    malignancy_dist = {}
    tirads_dist = {}
    for rec in labels:
        malignancy_dist[rec["malignancy"]] = malignancy_dist.get(rec["malignancy"], 0) + 1
        tirads_dist[rec["tirads"]] = tirads_dist.get(rec["tirads"], 0) + 1

    print(f"\n  良恶性标签分布:")
    for k in sorted(malignancy_dist.keys()):
        label = {-1: "N/A(缺失)", 0: "良性(0)", 1: "恶性(1)"}.get(k, str(k))
        print(f"    {label}: {malignancy_dist[k]}")

    print(f"\n  TI-RADS标签分布:")
    for k in sorted(tirads_dist.keys()):
        label = {-1: "N/A(缺失)"}.get(k, f"TR{k}")
        print(f"    {label}: {tirads_dist[k]}")

    print(f"\n输出目录:")
    print(f"  原图:     {images_dir}")
    print(f"  腺体掩码: {gland_masks_dir}")
    print(f"  结节掩码: {nodule_masks_dir}")
    print(f"  标签文件: {labels_path}")


if __name__ == "__main__":
    main()
