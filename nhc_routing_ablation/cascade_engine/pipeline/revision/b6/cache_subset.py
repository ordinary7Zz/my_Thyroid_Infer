"""B6：从大 intermediate 中只加载 stability 子集，避免 ZJH-8K 等全量 npz OOM。"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_seg_npz_subset(
    path: Path,
    image_names: list[str],
    *,
    load_confidence: bool = True,
) -> dict[str, Any]:
    """mmap 读 npz，仅 materialize ``image_names`` 对应行的 mask/conf。"""
    from segmentation.base_model import SegModelOutput

    wanted = set(image_names)
    if not wanted:
        return {}

    # object-dtype mask 数组在 mmap 下必须 allow_pickle=True
    data = np.load(str(path), mmap_mode="r", allow_pickle=True)
    try:
        all_names = data["image_names"]
        name_to_i: dict[str, int] = {}
        for i in range(len(all_names)):
            n = str(all_names[i])
            if n in wanted:
                name_to_i[n] = i

        masks = data["masks"]
        metas = pickle.loads(data["metadata"].item())
        model_names_arr = data["model_names"]
        confs = data["confidence_maps"] if load_confidence and "confidence_maps" in data else None

        out: dict[str, Any] = {}
        for name, i in name_to_i.items():
            mask = np.asarray(masks[i]).astype(np.uint8)
            conf_val = None
            if confs is not None:
                conf = np.asarray(confs[i])
                has_conf = conf.shape != (1, 1)
                conf_val = conf.astype(np.float32) if has_conf else None
            out[name] = SegModelOutput(
                model_name=str(model_names_arr[i]),
                mask=mask,
                confidence_map=conf_val,
                metadata=metas[i],
            )
        return out
    finally:
        if hasattr(data, "close"):
            data.close()


def load_cls_json_map_subset(path: Path, image_names: list[str]) -> dict[str, Any]:
    from classification.base_model import ClsModelOutput

    wanted = set(image_names)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, Any] = {}
    for img_name, pred_dict in data.get("results", {}).items():
        if str(img_name) not in wanted:
            continue
        out[str(img_name)] = ClsModelOutput(
            model_name=pred_dict["model_name"],
            predictions=pred_dict["predictions"],
            top_class=pred_dict["top_class"],
            top_confidence=pred_dict["top_confidence"],
            requires_mask=pred_dict["requires_mask"],
            metadata=pred_dict.get("metadata", {}),
        )
    return out
