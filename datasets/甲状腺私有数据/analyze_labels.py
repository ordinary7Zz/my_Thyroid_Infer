#!/usr/bin/env python3
"""分析两个目录下 ini 文件的良恶性分类和 TI-RADS 分类的取值范围"""

import os
import glob
from collections import defaultdict

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import SafeConfigParser as ConfigParser


def analyze_ini(filepath):
    """解析单个 ini 文件，提取每个 ROI 的 pathologic 和 birads 字段"""
    cp = ConfigParser(strict=False, interpolation=None)
    try:
        cp.read(filepath, encoding="utf-8")
    except Exception:
        try:
            cp.read(filepath, encoding="gbk")
        except Exception as e:
            print(f"  [跳过] 无法解析: {filepath} ({e})")
            return []

    results = []
    for section in cp.sections():
        if not section.startswith("ROI"):
            continue
        pathologic = cp.get(section, "pathologic", fallback="").strip()
        birads = cp.get(section, "birads", fallback="").strip()
        type_ = cp.get(section, "type", fallback="").strip()
        results.append({
            "file": os.path.basename(filepath),
            "roi": section,
            "type": type_,
            "pathologic": pathologic,
            "birads": birads,
        })
    return results


def main():
    dirs = [
        "/Users/wangbd/sysu/甲状腺私有数据/toWangshijie260703",
        "/Users/wangbd/sysu/甲状腺私有数据/新建文件夹",
    ]

    all_records = []
    for d in dirs:
        ini_files = sorted(glob.glob(os.path.join(d, "*.ini")))
        print(f"\n{'='*60}")
        print(f"目录: {d}")
        print(f"  找到 {len(ini_files)} 个 ini 文件")
        for f in ini_files:
            records = analyze_ini(f)
            all_records.extend(records)

    # 统计
    print(f"\n{'='*60}")
    print(f"共解析 {len(all_records)} 条 ROI 记录\n")

    # --- 良恶性分类 (pathologic) ---
    pathologic_values = [r["pathologic"] for r in all_records]
    pathologic_filled = [v for v in pathologic_values if v]
    pathologic_counts = defaultdict(int)
    for v in pathologic_filled:
        pathologic_counts[v] += 1

    print("【良恶性分类 (pathologic)】")
    print(f"  非空记录数: {len(pathologic_filled)} / {len(all_records)}")
    print(f"  唯一取值 ({len(pathologic_counts)} 种):")
    for val, cnt in sorted(pathologic_counts.items(), key=lambda x: -x[1]):
        print(f"    '{val}' → {cnt} 次")

    # --- TI-RADS 分类 (birads) ---
    birads_values = [r["birads"] for r in all_records]
    birads_filled = [v for v in birads_values if v]
    birads_counts = defaultdict(int)
    for v in birads_filled:
        birads_counts[v] += 1

    print(f"\n【TI-RADS 分类 (birads)】")
    print(f"  非空记录数: {len(birads_filled)} / {len(all_records)}")
    print(f"  唯一取值 ({len(birads_counts)} 种):")
    for val, cnt in sorted(birads_counts.items(), key=lambda x: -x[1]):
        print(f"    '{val}' → {cnt} 次")

    # --- 交叉表: pathologic × birads ---
    print(f"\n{'='*60}")
    print("【交叉表: pathologic × birads】")
    crosstab = defaultdict(lambda: defaultdict(int))
    for r in all_records:
        p = r["pathologic"] or "(空)"
        b = r["birads"] or "(空)"
        crosstab[p][b] += 1

    all_birads = sorted({r["birads"] or "(空)" for r in all_records})
    print(f"  {'pathologic':<12}", end="")
    for b in all_birads:
        print(f"{b:>8}", end="")
    print()
    for p in sorted(crosstab.keys()):
        print(f"  {p:<12}", end="")
        for b in all_birads:
            print(f"{crosstab[p][b]:>8}", end="")
        print()

    # --- type 字段 ---
    print(f"\n{'='*60}")
    print("【type 字段取值分布】")
    type_counts = defaultdict(int)
    for r in all_records:
        t = r["type"] or "(空)"
        type_counts[t] += 1
    for val, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  '{val}' → {cnt} 次")


if __name__ == "__main__":
    main()
