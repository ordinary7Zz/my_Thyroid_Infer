#!/usr/bin/env python3
"""
根据匿名化文件名反查所属中心。

背景:
  prepare_data.py 的 anonymize_filename() 会把文件名前 3 段（中心）做
  SHA-256 取前 12 位，替换为 center_hash；第 4 段（患者）同样哈希；
  时间戳原样保留。

  匿名后格式: <center_hash>_<patient_hash>_<timestamp>.<ext>
  原始格式:   THYB_S_<中心编号>_<患者ID>_<时间戳>.<ext>

本脚本:
  1. 维护已知中心名 → 哈希 的映射表（与 prepare_data.py 完全一致的算法）
  2. 支持从原始数据目录自动扫描补充未知中心
  3. 对给定的匿名文件名，提取 center_hash 段反查出中心名

用法:
  # 反查单个文件
  python identify_center.py d49e75ad16e4_5f8a2b1c9d3e_202052842015.png

  # 批量反查一个目录下所有文件
  python identify_center.py --dir datasets/processed/images

  # 指定额外原始数据目录用于自动补充中心表
  python identify_center.py --scan datasets/甲状腺私有数据 --dir datasets/processed/images
"""

import os
import re
import sys
import hashlib
import argparse
from pathlib import Path
from collections import Counter

# ============================================================
# 已知中心列表（来自私有数据集统计图，共 35 个）
# 如需新增，直接在此追加即可
# ============================================================
KNOWN_CENTERS = [
    "THYB_S_SX04", "THYB_S_SH01", "THYB_S_YN05", "THYB_S_ZJ05",
    "THYB_S_ZJ24", "THYB_S_SH05", "THYB_S_AN01", "THYB_S_ZJ06",
    "THYB_S_AH04", "THYB_S_QX07", "THYB_S_BJ01", "THYB_S_JS02",
    "THYB_S_NX01", "THYB_S_EN02", "THYB_S_CQ03", "THYB_S_EN04",
    "THYB_S_JX06", "THYB_S_XJ01", "THYB_S_YN01", "THYB_S_JS01",
    "THYB_S_GS03", "THYB_S_GZ02", "THYB_S_SH06", "THYB_S_BJ09",
    "THYB_S_FJ03", "THYB_S_JL04", "THYB_S_SD12", "THYB_S_ZJ29",
    "THYB_S_SD13", "THYB_S_NM02", "THYB_S_SC06", "THYB_S_SD14",
    "THYB_S_GX01", "THYB_S_FJ01", "THYB_S_HB07",
]


def center_hash(center_key: str) -> str:
    """与 prepare_data.py 完全一致的哈希算法：SHA-256 前 12 位。"""
    return hashlib.sha256(center_key.encode("utf-8")).hexdigest()[:12]


def build_hash_to_center(extra_centers=None):
    """构建 {hash: center_name} 反查表。"""
    centers = list(KNOWN_CENTERS)
    if extra_centers:
        for c in extra_centers:
            if c not in centers:
                centers.append(c)
    return {center_hash(c): c for c in centers}


def scan_centers_from_dir(*dirs):
    """从原始数据目录扫描所有文件名，提取中心前缀（前 3 段）。

    原始文件名格式: THYB_S_XXX_ND..._时间戳.ext
    """
    centers = set()
    pat = re.compile(r"^(THYB_S_[A-Z]{2}\d{2})_")
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if m:
                centers.add(m.group(1))
    return sorted(centers)


def parse_anonymized_filename(filename):
    """解析匿名化文件名，返回 (center_hash, patient_hash, timestamp)。

    匿名格式: <12位hex>_<12位hex>_<timestamp>.<ext>
    回退: 若不是匿名格式，尝试按原始格式解析。
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    # 匿名化格式：前两段均为 12 位十六进制
    hex12 = re.compile(r"^[0-9a-f]{12}$")
    if len(parts) >= 2 and hex12.match(parts[0]) and hex12.match(parts[1]):
        center_hash_val = parts[0]
        patient_hash_val = parts[1]
        timestamp = "_".join(parts[2:]) if len(parts) > 2 else ""
        return center_hash_val, patient_hash_val, timestamp, "anonymized"

    # 原始格式：THYB_S_XXX_ND..._时间戳
    if len(parts) >= 4 and parts[0] == "THYB" and parts[1] == "S":
        center_key = "_".join(parts[:3])
        patient_key = parts[3]
        timestamp = "_".join(parts[4:]) if len(parts) > 4 else ""
        return center_hash(center_key), patient_hash(patient_key), timestamp, "original"

    return None, None, None, "unknown"


def patient_hash(patient_key: str) -> str:
    return hashlib.sha256(patient_key.encode("utf-8")).hexdigest()[:12]


def identify(filename, hash2center):
    """识别单个文件名对应的中心。"""
    ch, ph, ts, fmt = parse_anonymized_filename(filename)
    if ch is None:
        return None
    center = hash2center.get(ch)
    return {
        "filename": filename,
        "center": center if center else f"(未知 hash={ch})",
        "center_hash": ch,
        "patient_hash": ph,
        "timestamp": ts,
        "format": fmt,
    }


def main():
    parser = argparse.ArgumentParser(
        description="根据匿名化文件名反查所属中心"
    )
    parser.add_argument("filename", nargs="?", default=None,
                        help="单个文件名（匿名或原始格式均可）")
    parser.add_argument("--dir", default=None,
                        help="批量处理：扫描该目录下所有文件")
    parser.add_argument("--scan", nargs="*", default=[],
                        help="额外的原始数据目录，用于自动扫描补充中心表")
    parser.add_argument("--list", action="store_true",
                        help="仅打印已知中心 → 哈希 映射表")
    parser.add_argument("--top", type=int, default=0,
                        help="批量模式下只显示前 N 个中心（按样本数降序），0=全部")
    args = parser.parse_args()

    # 1. 补充中心表
    extra = scan_centers_from_dir(*args.scan) if args.scan else []
    if extra:
        # 保存到内存，不修改源码
        pass
    hash2center = build_hash_to_center(extra)

    if args.list:
        print(f"已知中心 → 哈希 映射表（共 {len(hash2center)} 个）")
        print("=" * 50)
        for h, c in sorted(hash2center.items(), key=lambda x: x[1]):
            print(f"  {c:15s} -> {h}")
        return

    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"错误: 目录不存在: {d}", file=sys.stderr)
            sys.exit(1)

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
        results = []
        unknown = Counter()
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            r = identify(p.name, hash2center)
            if r is None:
                continue
            results.append(r)
            if r["center"].startswith("(未知"):
                unknown[r["center_hash"]] += 1

        # 统计每个中心的样本数
        center_counts = Counter(r["center"] for r in results)
        print(f"共扫描 {len(results)} 个文件")
        print(f"识别出 {len(center_counts)} 个不同中心")
        print("=" * 60)
        print(f"{'中心':<18s} {'样本数':>8s}   中心哈希")
        print("-" * 60)

        items = center_counts.most_common(args.top if args.top > 0 else None)
        for center, cnt in items:
            # 找一个哈希
            h = next((r["center_hash"] for r in results if r["center"] == center), "?")
            print(f"{center:<18s} {cnt:>8d}   {h}")

        if unknown:
            print("-" * 60)
            print(f"未知中心哈希 ({len(unknown)} 种):")
            for h, cnt in unknown.most_common():
                print(f"  {h} : {cnt} 个文件")
        return

    if args.filename:
        r = identify(args.filename, hash2center)
        if r is None:
            print(f"无法解析文件名: {args.filename}", file=sys.stderr)
            sys.exit(1)
        print(f"文件名:     {r['filename']}")
        print(f"格式:       {r['format']}")
        print(f"中心:       {r['center']}")
        print(f"中心哈希:   {r['center_hash']}")
        print(f"患者哈希:   {r['patient_hash']}")
        print(f"时间戳:     {r['timestamp']}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
