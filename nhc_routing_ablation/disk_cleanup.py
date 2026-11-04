"""Remove heavy NHC ablation caches after metrics/json outputs are written."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from paths import STRATEGY_ROWS


def disk_cleanup_enabled(*, keep_cache: bool = False) -> bool:
    if keep_cache:
        return False
    env_keep = os.environ.get("NHC_ABLATION_KEEP_CACHE", "").strip().lower()
    if env_keep in ("1", "true", "yes"):
        return False
    env_skip = os.environ.get("NHC_ABLATION_SKIP_DISK_CLEANUP", "").strip().lower()
    if env_skip in ("1", "true", "yes"):
        return False
    return True


def _strategy_output_dirs(output_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for row in STRATEGY_ROWS:
        p = output_root / row["mode"]
        if p.is_dir():
            dirs.append(p)
    learned = output_root / "learned"
    if learned.is_dir():
        dirs.append(learned)
    return dirs


def _dir_byte_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() and not path.is_symlink():
        return path.stat().st_size
    total = 0
    if path.is_dir() and not path.is_symlink():
        for p in path.rglob("*"):
            if p.is_file() and not p.is_symlink():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    return total


def _remove_path_tree(path: Path) -> int:
    """Delete file or real directory tree; return bytes freed (approx)."""
    if not path.exists() and not path.is_symlink():
        return 0
    freed = _dir_byte_size(path)
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return freed
    shutil.rmtree(path)
    return freed


def _strip_strategy_intermediate(inter: Path) -> tuple[int, bool]:
    """Remove per-strategy intermediate/ without following symlinks into shared cache."""
    if not inter.exists() and not inter.is_symlink():
        return 0, False
    freed = 0
    removed_npz = False
    if inter.is_symlink():
        inter.unlink(missing_ok=True)
        return 0, False
    if not inter.is_dir():
        return 0, False
    for child in list(inter.iterdir()):
        if child.name == "selected_masks.npz" and child.is_file():
            freed += child.stat().st_size
            child.unlink(missing_ok=True)
            removed_npz = True
            continue
        if child.is_symlink():
            child.unlink(missing_ok=True)
        elif child.is_file():
            freed += child.stat().st_size
            child.unlink(missing_ok=True)
        elif child.is_dir() and not child.is_symlink():
            freed += _remove_path_tree(child)
    try:
        inter.rmdir()
    except OSError:
        pass
    return freed, removed_npz


@dataclass
class DiskCleanupReport:
    bytes_freed: int = 0
    selected_masks_removed: int = 0
    strategy_intermediate_dirs_removed: int = 0
    shared_seg_removed: bool = False
    shared_cls_removed: bool = False
    bundle_staging_removed: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def freed_mib(self) -> float:
        return self.bytes_freed / (1024 * 1024)


def cleanup_nhc_ablation_disk_cache(
    output_root: Path,
    shared_intermediate: Path,
    *,
    remove_cls_cache: bool = True,
) -> DiskCleanupReport:
    """
    Drop Phase1–2 seg/cls caches and per-strategy intermediate symlinks/npz.

    Keeps: results.json, b3_metrics.json, per_image_*_metrics.jsonl, _summary/, _center_analysis/.
    """
    report = DiskCleanupReport()
    output_root = output_root.resolve()
    shared = shared_intermediate.resolve()

    staging = output_root / ".bundle_staging"
    if staging.is_dir() or staging.is_symlink():
        report.bytes_freed += _remove_path_tree(staging)
        report.bundle_staging_removed = True

    for strategy_dir in _strategy_output_dirs(output_root):
        inter = strategy_dir / "intermediate"
        freed, had_npz = _strip_strategy_intermediate(inter)
        report.bytes_freed += freed
        if had_npz:
            report.selected_masks_removed += 1
        if not inter.exists():
            report.strategy_intermediate_dirs_removed += 1

    seg_dir = shared / "seg"
    if seg_dir.is_dir() and not seg_dir.is_symlink():
        report.bytes_freed += _remove_path_tree(seg_dir)
        report.shared_seg_removed = True
    elif seg_dir.is_symlink():
        seg_dir.unlink(missing_ok=True)
        report.shared_seg_removed = True

    if remove_cls_cache:
        cls_dir = shared / "cls"
        if cls_dir.is_dir() and not cls_dir.is_symlink():
            report.bytes_freed += _remove_path_tree(cls_dir)
            report.shared_cls_removed = True
        elif cls_dir.is_symlink():
            cls_dir.unlink(missing_ok=True)
            report.shared_cls_removed = True

    for name in ("autogluon.json",):
        p = shared / name
        if p.is_file() or p.is_symlink():
            try:
                report.bytes_freed += p.stat().st_size if p.is_file() else 0
            except OSError:
                pass
            p.unlink(missing_ok=True)

    return report


def format_cleanup_summary(report: DiskCleanupReport) -> str:
    parts: list[str] = []
    if report.shared_seg_removed:
        parts.append("Phase1–2 seg/")
    if report.shared_cls_removed:
        parts.append("cls/")
    if report.strategy_intermediate_dirs_removed:
        parts.append(f"{report.strategy_intermediate_dirs_removed}×策略 intermediate/")
    if report.selected_masks_removed:
        parts.append(f"{report.selected_masks_removed}× selected_masks.npz")
    if report.bundle_staging_removed:
        parts.append(".bundle_staging/")
    if not parts:
        return "（无大体积缓存需清理）"
    return "已删除: " + "、".join(parts) + f"；约释放 {report.freed_mib:.1f} MiB"
