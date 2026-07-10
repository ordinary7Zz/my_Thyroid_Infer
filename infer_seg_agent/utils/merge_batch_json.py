"""
Merge multiple batch result JSON files (e.g. partial runs) into one file with unified performance_stats.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.performance_stats import build_performance_stats


def merge_batch_result_json_files(
    json_paths: List[Union[str, Path]],
    *,
    output_path: Optional[Union[str, Path]] = None,
    dedupe_key: str = "image_name",
    prefer_last: bool = True,
) -> Dict[str, Any]:
    """
    Load several batch JSON files, merge their ``results`` lists, recompute aggregate performance.

    Args:
        json_paths: Ordered list of JSON file paths (later files override earlier on duplicate keys if prefer_last).
        output_path: If set, write merged JSON here.
        dedupe_key: Field used to deduplicate rows (default ``image_name``).
        prefer_last: If True, when the same dedupe_key appears in multiple files, keep the entry from the later file.

    Returns:
        Merged dictionary with ``results``, ``num_images``, ``performance_stats``, ``source_files``, etc.
    """
    paths = [Path(p).resolve() for p in json_paths]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"JSON not found: {p}")

    merged_by_key: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    source_meta: List[Dict[str, Any]] = []
    first_type: Optional[str] = None
    first_timestamp: Optional[str] = None
    no_key_seq = 0

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid JSON root (expected object): {p}")

        results = data.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Missing or invalid 'results' list in {p}")

        if first_type is None:
            first_type = data.get("type", "merged_results")
        if first_timestamp is None:
            first_timestamp = data.get("timestamp")

        source_meta.append(
            {
                "path": str(p),
                "num_images_in_file": data.get("num_images", len(results)),
                "timestamp": data.get("timestamp"),
            }
        )

        for item in results:
            if not isinstance(item, dict):
                continue
            key = item.get(dedupe_key)
            if key is None:
                key = f"__no_key_{no_key_seq}"
                no_key_seq += 1
            if prefer_last:
                merged_by_key[key] = item
            else:
                if key not in merged_by_key:
                    merged_by_key[key] = item

    merged_results = list(merged_by_key.values())
    out_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    merged: Dict[str, Any] = {
        "timestamp": out_ts,
        "type": first_type or "merged_results",
        "num_images": len(merged_results),
        "results": merged_results,
        "source_files": source_meta,
        "merged_from_original_timestamps": [m.get("timestamp") for m in source_meta],
        "performance_stats": None,
    }
    if first_timestamp:
        merged["first_source_timestamp"] = first_timestamp

    perf = build_performance_stats(merged_results)
    if perf is not None:
        merged["performance_stats"] = perf

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

    return merged


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple batch_results_*.json files and write aggregate performance_stats."
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="Input JSON paths (order matters for tie-breaking when deduping).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output merged JSON path.",
    )
    parser.add_argument(
        "--prefer-first-on-dedupe",
        action="store_true",
        help="On duplicate image_name, keep the first file's row instead of the last.",
    )
    args = parser.parse_args()

    merged = merge_batch_result_json_files(
        args.json_files,
        output_path=args.output,
        prefer_last=not args.prefer_first_on_dedupe,
    )
    print(f"Merged {len(merged['results'])} samples -> {args.output}")
    ps = merged.get("performance_stats") or {}
    if ps.get("mean_dice") is not None:
        print(f"  mean_dice: {ps['mean_dice']:.4f} (std {ps.get('std_dice', 0):.4f})")
    if ps.get("mean_hd95") is not None:
        print(f"  mean_hd95: {ps['mean_hd95']:.4f} (std {ps.get('std_hd95', 0):.4f})")
    if ps.get("mean_ece") is not None:
        print(f"  mean_ece: {ps['mean_ece']:.6f} (std {ps.get('std_ece', 0):.6f})")


if __name__ == "__main__":
    _cli()
