#!/usr/bin/env python3
"""
B7：NHC 私有数据预处理（自 my_Thyroid_Infer prepare_data.py 复制并扩展）。

在原有 ROI 裁剪 / 掩码对齐 / 标签提取基础上增加：
  - labels.json 每条写入 center_id、patient_id（来自原始 stem）
  - center_stats.json：中心数、每中心图像数 / 患者数（ND 段去重）
  - center_manifest.json：输出文件名 stem -> center_id（供 B7/B8 join）

ROI 模型仍从 my_Thyroid_Infer 的 my_ThyroidROI/newcode 加载；
可设置环境变量 THYROID_INFER_ROOT 指向 my_Thyroid_Infer 根目录。

用法:
  python scripts/revision/b7/run_nhc_prepare.py \\
    --input_dir /path/to/甲状腺私有数据/新建文件夹 \\
    --output_dir output/b7/nhc_prepared \\
    --skip_roi
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from configparser import ConfigParser
from pathlib import Path

import cv2
import numpy as np

from pipeline.revision.b7.nhc_stem import (
    CenterStatsCollector,
    anonymize_filename,
    parse_nhc_stem,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
ORG_SUFFIX = "_ORG1"
ROI_SUFFIX = "_ROI1"


def _thyroid_infer_newcode_dir() -> Path:
    env = os.environ.get("THYROID_INFER_ROOT", "").strip()
    if env:
        return Path(env) / "my_ThyroidROI" / "newcode"
    cascade_root = Path(__file__).resolve().parents[3]
    return cascade_root.parent / "my_Thyroid_Infer" / "my_ThyroidROI" / "newcode"


def _ensure_roi_extractor_import() -> None:
    newcode = _thyroid_infer_newcode_dir()
    if not newcode.is_dir():
        raise FileNotFoundError(
            f"未找到 ROI 代码目录: {newcode}；请设置 THYROID_INFER_ROOT 或 --skip_roi"
        )
    path = str(newcode)
    if path not in sys.path:
        sys.path.insert(0, path)


def parse_ini(filepath: Path) -> ConfigParser:
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
    for section in cp.sections():
        if not section.startswith("ROI"):
            continue
        if cp.has_option(section, field):
            val = cp.get(section, field, fallback="").strip()
            if val:
                return val
    return ""


def extract_malignancy(pathologic_value: str) -> int:
    if not pathologic_value:
        return -1
    val = pathologic_value.strip()
    val_lower = val.lower()
    if "良" in val or val == "0" or "benign" in val_lower:
        return 0
    if "恶" in val or val == "1" or "malign" in val_lower or "癌" in val:
        return 1
    return -1


def extract_tirads(birads_value: str) -> int:
    if not birads_value:
        return -1
    val = birads_value.strip()
    match = re.search(r"(\d)", val)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 5:
            return num
        return -1
    chinese_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    for ch, num in chinese_map.items():
        if ch in val:
            return num
    return -1


def init_roi_extractor(checkpoint_path: str):
    _ensure_roi_extractor_import()
    from roi_extractor import ROIExtractor

    return ROIExtractor(checkpoint_path)


def extract_roi_for_image(extractor, image_path: Path):
    roi_rgb, crop_params = extractor.extract_roi_with_crop_params(str(image_path))
    roi_bgr = (roi_rgb[:, :, ::-1] * 255).astype("uint8")
    return roi_bgr, crop_params


def crop_mask_with_params(mask_path: Path, crop_params: dict):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩码: {mask_path}")
    x, y, w, h = crop_params["x"], crop_params["y"], crop_params["w"], crop_params["h"]
    cropped_mask = mask[y : y + h, x : x + w].copy()
    processed_mask = crop_params["mask"]
    cropped_processed = processed_mask[y : y + h, x : x + w]
    return np.where(cropped_processed > 0, cropped_mask, 0).astype("uint8")


def _find_file_for_stem(input_dir: Path, stem: str, suffix: str = ""):
    for ext in sorted(IMAGE_EXTS):
        candidate = input_dir / f"{stem}{suffix}{ext}"
        if candidate.is_file():
            return candidate
    return None


def run_prepare(
    input_dir: Path,
    output_dir: Path,
    *,
    use_roi: bool,
    checkpoint: str | None,
    anonymize: bool,
    save_name_mapping: bool,
) -> dict:
    extractor = None
    if use_roi:
        if not checkpoint or not Path(checkpoint).is_file():
            raise ValueError("启用 ROI 需要有效 --checkpoint")
        extractor = init_roi_extractor(checkpoint)

    images_dir = output_dir / "images"
    gland_masks_dir = output_dir / "gland_masks"
    nodule_masks_dir = output_dir / "nodule_masks"
    for d in (images_dir, gland_masks_dir, nodule_masks_dir):
        d.mkdir(parents=True, exist_ok=True)

    ini_files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".ini"
    )

    labels: list[dict] = []
    name_map: dict[str, str] = {}
    center_manifest: dict[str, str] = {}
    center_stats = CenterStatsCollector()
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
        stem = ini_path.stem
        img_path = _find_file_for_stem(input_dir, stem, suffix="")
        if img_path is None:
            stats["missing_image"].append(ini_path.name)
            continue

        ext = img_path.suffix
        stats["total"] += 1
        org_path = _find_file_for_stem(input_dir, stem, suffix=ORG_SUFFIX)
        roi_path = _find_file_for_stem(input_dir, stem, suffix=ROI_SUFFIX)

        parsed = parse_nhc_stem(stem)
        center_stats.record(parsed["center_id"], parsed["patient_id"])

        if anonymize:
            out_name = anonymize_filename(stem, ext)
            name_map[img_path.name] = out_name
        else:
            out_name = img_path.name

        out_stem = Path(out_name).stem
        if parsed["center_id"]:
            center_manifest[out_stem] = parsed["center_id"]

        out_image = images_dir / out_name
        out_gland = gland_masks_dir / out_name
        out_nodule = nodule_masks_dir / out_name

        crop_params = None
        if use_roi and extractor is not None:
            try:
                roi_bgr, crop_params = extract_roi_for_image(extractor, img_path)
                cv2.imwrite(str(out_image), roi_bgr)
                stats["roi_success"] += 1
            except Exception as e:
                print(f"  [ROI 错误] {img_path.name}: {e}，回退为直接复制")
                shutil.copy2(img_path, out_image)
                stats["roi_failed"] += 1
                stats["roi_errors"].append((img_path.name, str(e)))
        else:
            shutil.copy2(img_path, out_image)

        if org_path is not None:
            if use_roi and crop_params is not None:
                try:
                    cv2.imwrite(str(out_gland), crop_mask_with_params(org_path, crop_params))
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

        if roi_path is not None:
            if use_roi and crop_params is not None:
                try:
                    cv2.imwrite(str(out_nodule), crop_mask_with_params(roi_path, crop_params))
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

        cp = parse_ini(ini_path)
        malignancy = extract_malignancy(get_first_label_from_roi(cp, "pathologic"))
        tirads = extract_tirads(get_first_label_from_roi(cp, "birads"))
        if malignancy != -1:
            stats["has_malignancy"] += 1
        if tirads != -1:
            stats["has_tirads"] += 1

        rec: dict = {
            "filename": out_name,
            "malignancy": malignancy,
            "tirads": tirads,
        }
        if parsed["center_id"]:
            rec["center_id"] = parsed["center_id"]
        if parsed["patient_id"]:
            rec["patient_id"] = parsed["patient_id"]
        labels.append(rec)

    labels_path = output_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    if anonymize and save_name_mapping:
        with open(output_dir / "name_mapping.json", "w", encoding="utf-8") as f:
            json.dump(name_map, f, ensure_ascii=False, indent=2)

    summary = center_stats.to_summary()
    with open(output_dir / "center_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_dir / "center_manifest.json", "w", encoding="utf-8") as f:
        json.dump(center_manifest, f, ensure_ascii=False, indent=2)

    stats["center_summary"] = summary
    stats["n_ini"] = len(ini_files)
    stats["labels_path"] = str(labels_path)
    return stats


def _print_report(stats: dict, output_dir: Path, use_roi: bool, anonymize: bool) -> None:
    print(f"\n{'=' * 60}")
    print("B7 NHC 预处理完成")
    print(f"  INI: {stats['n_ini']}  有图: {stats['total']}  无图: {len(stats['missing_image'])}")
    if use_roi:
        print(f"  ROI 成功/失败: {stats['roi_success']} / {stats['roi_failed']}")
    cs = stats["center_summary"]
    print(f"  中心数: {cs['n_centers']}  图像: {cs['n_images']}  患者键(全局去重): {cs['n_patients_total']}")
    for cid, block in sorted(cs["centers"].items()):
        print(f"    {cid}: n_images={block['n_images']}, n_patients={block['n_patients']}")
    print(f"  输出: {output_dir}")
    print(f"  center_stats.json / center_manifest.json / labels.json (含 center_id)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B7 NHC 数据预处理（含中心统计）")
    parser.add_argument("--input_dir", type=Path, required=True, help="原始 NHC 平铺目录（含 .ini）")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/b7/nhc_prepared"),
        help="输出目录（默认 output/b7/nhc_prepared）",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Swin-UNet ROI 权重")
    parser.add_argument("--skip_roi", action="store_true", help="跳过 ROI，仅复制与标签")
    parser.add_argument("--no_anonymize", action="store_true", help="关闭文件名匿名化")
    parser.add_argument("--no_save_name_mapping", action="store_true", help="不写 name_mapping.json")
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"错误: 输入目录不存在: {args.input_dir}", file=sys.stderr)
        return 1

    use_roi = not args.skip_roi
    if use_roi and (not args.checkpoint or not Path(args.checkpoint).is_file()):
        print("⚠ 未提供有效 checkpoint，自动 --skip_roi")
        use_roi = False

    anonymize = not args.no_anonymize
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = run_prepare(
        args.input_dir,
        args.output_dir,
        use_roi=use_roi,
        checkpoint=args.checkpoint,
        anonymize=anonymize,
        save_name_mapping=not args.no_save_name_mapping,
    )
    _print_report(stats, args.output_dir, use_roi, anonymize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
