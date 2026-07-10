#!/usr/bin/env python3
"""
对比本地项目与 ModelScope 远程仓库的文件一致性。
排除与 upload_modelscope.sh 相同的目录和文件类型。

用法: python compare_modelscope.py
"""

import os
import sys
from pathlib import Path

REPO_ID = "instincts/my_Thyroid_infer"

# 与 upload_modelscope.sh --exclude 一致
EXCLUDE_DIRS = {
    "results", "results_pretrained", "results_dino_mask_pretrained",
    "datasets", "wheels", "gptoss", "gptoss_pth", "zips",
    "__pycache__", ".git", ".codebuddy", ".ipynb_checkpoints",
}
EXCLUDE_EXTS = {".pyc", ".pyo", ".so", ".zip"}
EXCLUDE_FILES = {".DS_Store"}


def should_skip(name):
    """检查文件/目录名是否应排除。"""
    if name in EXCLUDE_DIRS or name in EXCLUDE_FILES:
        return True
    ext = os.path.splitext(name)[1]
    if ext in EXCLUDE_EXTS:
        return True
    return False


def collect_local(root_dir):
    """收集本地文件列表: {relative_path: size_bytes}"""
    local = {}
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        # 原地过滤排除目录
        dirnames[:] = [d for d in dirnames if not should_skip(d)]
        for f in filenames:
            if should_skip(f):
                continue
            filepath = os.path.join(dirpath, f)
            relpath = os.path.relpath(filepath, root_dir)
            try:
                size = os.path.getsize(filepath)
            except OSError:
                continue
            local[relpath] = size
    return local


def collect_remote():
    """收集远程文件列表: {relative_path: size_bytes}"""
    from modelscope.hub.api import HubApi
    api = HubApi()
    files = api.get_model_files(REPO_ID, recursive=True)

    remote = {}
    for item in files:
        path = item.get("Path", "")
        size = item.get("Size", 0)
        if size == 0:
            # 跳过目录（Size=0 的可能是目录或空文件）
            # 但也跳过空文件，因为上传时通常不关心
            continue
        # 应用相同的排除规则
        parts = Path(path).parts
        if any(should_skip(p) for p in parts):
            continue
        remote[path] = size
    return remote


def main():
    root_dir = os.getcwd()

    print(f"本地目录: {root_dir}")
    print(f"远程仓库: {REPO_ID}")
    print()

    # 收集文件列表
    print("正在收集本地文件...")
    local = collect_local(root_dir)
    print(f"  本地文件数: {len(local)}")

    print("正在获取远程文件列表...")
    try:
        remote = collect_remote()
    except Exception as e:
        print(f"  获取远程文件失败: {e}")
        print("  请确保已登录: ms login --token YOUR_TOKEN")
        sys.exit(1)
    print(f"  远程文件数: {len(remote)}")
    print()

    # 对比
    local_set = set(local.keys())
    remote_set = set(remote.keys())

    only_local = sorted(local_set - remote_set)
    only_remote = sorted(remote_set - local_set)
    common = local_set & remote_set

    size_mismatch = []
    for f in sorted(common):
        if local[f] != remote[f]:
            size_mismatch.append((f, local[f], remote[f]))

    # 输出报告
    total_issues = len(only_local) + len(only_remote) + len(size_mismatch)

    if total_issues == 0:
        print("=" * 70)
        print("  ✅ 完全一致！本地与远程文件列表和大小均匹配。")
        print("=" * 70)
        return

    print("=" * 70)
    print(f"  发现 {total_issues} 处差异")
    print("=" * 70)

    if only_local:
        print(f"\n━━━ 仅本地有（未上传到远程）: {len(only_local)} 个文件 ━━━")
        for f in only_local:
            print(f"  + {local[f]:>12,}  {f}")

    if only_remote:
        print(f"\n━━━ 仅远程有（本地已删除）: {len(only_remote)} 个文件 ━━━")
        for f in only_remote:
            print(f"  - {remote[f]:>12,}  {f}")

    if size_mismatch:
        print(f"\n━━━ 大小不一致: {len(size_mismatch)} 个文件 ━━━")
        print(f"  {'文件':<60} {'本地':>12}  {'远程':>12}  {'差异':>12}")
        for f, lsz, rsz in size_mismatch:
            diff = lsz - rsz
            sign = "+" if diff > 0 else ""
            print(f"  {f:<60} {lsz:>12,}  {rsz:>12,}  {sign}{diff:>11,}")

    print()
    print(f"汇总: 仅本地={len(only_local)}  仅远程={len(only_remote)}  大小不一致={len(size_mismatch)}")


if __name__ == "__main__":
    main()
