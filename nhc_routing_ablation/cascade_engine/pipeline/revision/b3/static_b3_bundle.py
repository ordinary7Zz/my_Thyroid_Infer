"""
B3 静态策略 bundle：单次扫图，对多种 routing_mode 同时产出结果。

共享：预加载 intermediate、逐图 judge_batch（受 pipeline.phase3_expert_judge 控制）、共识 Path A/B。
每 entry：独立 output 子目录（selected_masks、results、b3_metrics）。
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from classification.agent import LLMClassificationAgent
from pipeline.revision.b3.ablation_metrics import build_b3_metrics, save_b3_metrics
from pipeline.cascade import (
    CascadePipeline,
    build_intermediate_row_index,
    discover_intermediate_image_names,
)
from pipeline.evaluate import evaluate_pipeline
from pipeline.revision.b3.intermediate_manifest import write_intermediate_manifest
from routing.best_single_expert import get_best_single_names
from pipeline.revision.b3.routing_ablation import (
    AblationRoutingEntry,
    clear_phase3_cache,
    link_shared_intermediate,
)
from pipeline.revision.b3.routing_methods import write_routing_methods_report
from radiomics_judge import RadiomicsJudge
from routing.oracle import resolve_cls_oracle
from routing.path_b_cls import resolve_path_b_indie_only
from routing.seg_strategies import select_mask_for_mode
from segmentation.agent import SegAgentDecision, SegmentationAgent
from segmentation.quality_evaluator import SegmentationQualityEvaluator
from shared.image_io import ImageIO
from shared.phase3_options import phase3_expert_judge_enabled
from shared.routing_modes import normalize_routing_mode
from shared.sklearn_warnings import configure_sklearn_pickle_warnings


def _write_bundle_progress(
    output_root: Path,
    *,
    phase: str,
    index: int,
    total: int,
    image_name: str = "",
) -> None:
    """写入 output_root/.bundle_progress.json，避免终端被 warning 淹没时无法看 N/total。"""
    payload = {
        "phase": phase,
        "index": index,
        "total": total,
        "image_name": image_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = output_root / ".bundle_progress.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_gt_stem_map(gt_mask_dir: Optional[str]) -> dict[str, Path]:
    if not gt_mask_dir:
        return {}
    gt_dir = Path(gt_mask_dir)
    if not gt_dir.is_dir():
        return {}
    return {p.stem: p for p in gt_dir.iterdir() if p.is_file()}


def _load_gt_mask_for_image(
    gt_stem_map: dict[str, Path],
    img_name: str,
    image_io: ImageIO,
) -> Optional[np.ndarray]:
    gt_path = gt_stem_map.get(Path(img_name).stem)
    if gt_path is None or not gt_path.is_file():
        return None
    return image_io.binarize_mask(image_io.load_mask(gt_path))


class _LazySegModelCache:
    """大队列 B3 bundle：mmap seg npz，按 index 取预测，避免 5×全队列 mask 常驻 RAM。"""

    def __init__(self, path: Path, *, load_confidence: bool) -> None:
        self._path = path
        self._load_confidence = load_confidence
        self._data = np.load(str(path), allow_pickle=True, mmap_mode="r")
        self._metas = pickle.loads(self._data["metadata"].item())
        self._n = len(self._data["image_names"])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        from segmentation.base_model import SegModelOutput

        mask = np.asarray(self._data["masks"][i], dtype=np.uint8)
        conf_val = None
        if self._load_confidence and "confidence_maps" in self._data:
            conf = self._data["confidence_maps"][i]
            has_conf = conf.shape != (1, 1)
            conf_val = np.asarray(conf, dtype=np.float32) if has_conf else None
        meta = self._metas[i] if i < len(self._metas) else {}
        if "model_names" in self._data:
            model_name = str(self._data["model_names"][i])
        else:
            model_name = str(meta.get("model_name", ""))
        return SegModelOutput(
            model_name=model_name,
            mask=mask,
            confidence_map=conf_val,
            metadata=meta,
        )

    def close(self) -> None:
        self._data.close()


SegCacheValue = Union[list, _LazySegModelCache]


def _seg_pred_at(cache: SegCacheValue, idx: int):
    if isinstance(cache, _LazySegModelCache):
        return cache[idx] if idx < len(cache) else None
    if idx < len(cache):
        return cache[idx]
    return None


def _bundle_lazy_seg_threshold(config: dict, n_images: int) -> bool:
    ab = config.get("pipeline", {}).get("ablation", {}) or {}
    raw = ab.get("bundle_lazy_seg_min_images", 1500)
    try:
        threshold = int(raw)
    except (TypeError, ValueError):
        threshold = 1500
    return n_images >= threshold


STATIC_B3_MODES = frozenset({
    "best_single",
    "soft_voting",
    "average",
    "majority",
    "static_rule",
    "max_confidence",
    "oracle",
})  # 不含 path_b_autogluon（B4 专用）

BUNDLE_SEG_MODES = STATIC_B3_MODES | {"learned"}
BUNDLE_CLS_MODES = STATIC_B3_MODES | {"learned"}


def _assert_static_mode(mode: str, *, side: str) -> str:
    m = normalize_routing_mode(mode) or mode
    if m == "llm":
        raise ValueError(f"bundle 不支持 {side} routing_mode=llm，请用 run_routing_ablation")
    allowed = BUNDLE_SEG_MODES if side == "seg" else BUNDLE_CLS_MODES
    if m not in allowed:
        raise ValueError(f"未知 bundle {side} mode: {mode}")
    return m


def _seg_method_name(mode: str) -> str:
    if mode == "static_rule":
        return "routing_static_rule_seg"
    if mode == "learned":
        return "routing_learned_seg"
    return f"routing_{mode}_seg"


def _serialize_seg_decision(
    img_name: str,
    consensus: bool,
    anchor_class: Optional[str],
    decision: SegAgentDecision,
    method: str,
) -> dict:
    return {
        "img_name": img_name,
        "consensus": consensus,
        "anchor_class": anchor_class,
        "selected_model": decision.selected_model,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "path": decision.path,
        "method": method,
        "all_predictions": decision.all_predictions,
    }


def _pick_seg_decision(
    pipeline: CascadePipeline,
    seg_mode: str,
    *,
    image: np.ndarray,
    seg_preds: list,
    anchor_class: Optional[str],
    gt_mask: Optional[np.ndarray],
    judge_results: Optional[list],
    best_single_seg: Optional[str],
    input_device_info: Optional[list],
    input_data_info: Optional[dict],
    skip_radiomics: bool = False,
) -> SegAgentDecision:
    if seg_mode == "learned":
        quality_results = pipeline.seg_agent.quality_evaluator.evaluate_batch(
            [p.mask for p in seg_preds],
            [p.model_name for p in seg_preds],
        )
        return pipeline._learned_router("seg").select_mask(
            seg_preds, quality_results, anchor_class
        )
    if seg_mode == "static_rule":
        decision, _ = pipeline._select_mask_static(
            image,
            seg_preds,
            anchor_class,
            gt_mask=gt_mask,
            input_device_info=input_device_info,
            input_data_info=input_data_info,
            skip_radiomics=skip_radiomics,
            judge_results=judge_results,
        )
        return decision
    static_fn = None
    return select_mask_for_mode(
        seg_mode,
        seg_preds,
        anchor_class,
        gt_mask=gt_mask,
        best_single_model=best_single_seg,
        static_selector=static_fn,
    )


def _resolve_cls_for_image(
    pipeline: CascadePipeline,
    cls_mode: str,
    *,
    consensus: bool,
    anchor_class: Optional[str],
    indie_preds: list,
    img_name: str,
    labels: dict,
    best_single_cls: Optional[str],
    seg_decision_dict: Optional[dict],
    selected_mask: Optional[np.ndarray],
    has_seg: bool,
) -> Any:
    from classification.base_model import ClsModelOutput

    if consensus:
        return pipeline._make_anchor_classification_decision(anchor_class, indie_preds)

    seg_stub = None
    if has_seg and seg_decision_dict and selected_mask is not None:
        seg_stub = SegAgentDecision(
            selected_model=seg_decision_dict["selected_model"],
            selected_mask=selected_mask,
            confidence=seg_decision_dict["confidence"],
            reasoning=seg_decision_dict["reasoning"],
            all_predictions=seg_decision_dict["all_predictions"],
            classification_anchor=anchor_class,
            path=seg_decision_dict.get("path", "B"),
        )

    if cls_mode == "oracle":
        img_stem = Path(img_name).stem
        tl = labels.get(img_stem)
        if tl is None:
            return resolve_path_b_indie_only(
                "max_confidence",
                indie_preds,
                best_single_model=best_single_cls,
            )
        bin_tl = pipeline._label_to_binary(tl)
        if bin_tl is None:
            bin_tl = int(tl) if isinstance(tl, (int, float)) else 0
        return resolve_cls_oracle(indie_preds, bin_tl)

    if cls_mode == "learned":
        return resolve_path_b_indie_only(
            "learned",
            indie_preds,
            best_single_model=best_single_cls,
            learned_router=pipeline._learned_router("cls"),
        )

    return resolve_path_b_indie_only(
        cls_mode,
        indie_preds,
        best_single_model=best_single_cls,
    )


def _write_ablation_run_meta(
    output_dir: Path,
    entry: AblationRoutingEntry,
    *,
    task: str,
    shared_intermediate: Path,
    force_phase3: bool,
) -> None:
    meta = {
        "routing_mode": entry.run_id,
        "routing_mode_seg": entry.seg,
        "routing_mode_cls": entry.cls,
        "task": task,
        "shared_intermediate": str(shared_intermediate),
        "force_phase3": force_phase3,
        "seg_mode": entry.seg,
        "cls_mode": entry.cls,
        "uses_llm": False,
        "bundle": True,
    }
    with open(output_dir / "ablation_run.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def run_static_b3_bundle(
    pipeline: CascadePipeline,
    config: dict,
    *,
    entries: list[AblationRoutingEntry],
    image_dir: str,
    output_root: Path,
    shared_intermediate: Path,
    gt_mask_dir: Optional[str],
    labels: dict[str, Any],
    img_names: list[str],
    image_paths: list[Path],
    task: str = "both",
    force_phase3: bool = True,
    skip_eval: bool = False,
    input_device_info: Optional[list] = None,
    input_data_info: Optional[dict] = None,
) -> list[str]:
    """
    单次扫图，为每个 AblationRoutingEntry 写入 output_root/<run_id>/。
    返回成功写盘的 run_id 列表。
    """
    configure_sklearn_pickle_warnings()
    if not entries:
        return []

    seg_modes: set[str] = set()
    cls_modes: set[str] = set()
    for e in entries:
        if task in ("seg", "both"):
            seg_modes.add(_assert_static_mode(e.seg, side="seg"))
        if task in ("cls", "both"):
            cls_modes.add(_assert_static_mode(e.cls, side="cls"))

    seg_model_configs = config.get("segmentation", {}).get("models", [])
    cls_model_configs = config.get("classification", {}).get("models", [])
    cls_agent_cfg = config.get("classification", {}).get("agent", {})
    consensus_threshold = cls_agent_cfg.get("consensus_min_confidence", 0.6)
    best_single_seg, best_single_cls = get_best_single_names(config)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    staging = output_root / ".bundle_staging"
    staging_inter = staging / "intermediate"
    staging_inter.mkdir(parents=True, exist_ok=True)
    link_shared_intermediate(staging, shared_intermediate)
    seg_cache_dir = staging_inter / "seg"
    cls_cache_dir = staging_inter / "cls"

    need_seg_conf = "max_confidence" in seg_modes
    run_segmentation = task in ("seg", "both")
    run_classification = task in ("cls", "both")

    seg_cache_by_model: dict[str, SegCacheValue] = {}
    cls_cache_by_model: dict[str, dict[str, Any]] = {}
    lazy_seg = run_segmentation and _bundle_lazy_seg_threshold(config, len(img_names))
    intermediate_names: list[str] = []

    if run_segmentation and run_classification and not lazy_seg:
        seg_cache_by_model, cls_cache_by_model, intermediate_names = pipeline._preload_seg_cls_caches(
            seg_model_configs,
            cls_model_configs,
            seg_cache_dir,
            cls_cache_dir,
            need_seg_conf=need_seg_conf,
        )
    else:
        if run_classification:
            _, cls_cache_by_model, intermediate_names = pipeline._preload_seg_cls_caches(
                [],
                cls_model_configs,
                seg_cache_dir,
                cls_cache_dir,
                need_seg_conf=False,
            )
        if run_segmentation:
            if lazy_seg:
                ab = config.get("pipeline", {}).get("ablation", {}) or {}
                print(
                    f"  seg 缓存：mmap 按图读取（阈值 {ab.get('bundle_lazy_seg_min_images', 1500)} 张），"
                    f"不整包预加载 npz",
                    flush=True,
                )
                for model_cfg in seg_model_configs:
                    model_name = model_cfg.get("name", "")
                    if not model_name:
                        continue
                    path = seg_cache_dir / f"{model_name}.npz"
                    if path.is_file():
                        seg_cache_by_model[model_name] = _LazySegModelCache(
                            path, load_confidence=need_seg_conf
                        )
            else:
                seg_only, _, inter_from_seg = pipeline._preload_seg_cls_caches(
                    seg_model_configs,
                    [],
                    seg_cache_dir,
                    cls_cache_dir,
                    need_seg_conf=need_seg_conf,
                )
                seg_cache_by_model = seg_only
                if inter_from_seg:
                    intermediate_names = inter_from_seg

    if not intermediate_names:
        intermediate_names = discover_intermediate_image_names(seg_cache_dir)
    intermediate_row_index = build_intermediate_row_index(intermediate_names)

    judge_obj = getattr(pipeline.seg_agent, "radiomics_judge", None)
    phase3_expert_judge = phase3_expert_judge_enabled(config)
    skip_static_radiomics = not phase3_expert_judge
    need_radiomics_judge = (
        phase3_expert_judge
        and judge_obj is not None
        and run_segmentation
        and "static_rule" in seg_modes
    )
    if not phase3_expert_judge:
        print(
            "  ⚡ phase3_expert_judge=false：bundle 跳过专家 mask 的 Radiomics judge_batch",
            flush=True,
        )
    elif run_segmentation and "static_rule" not in seg_modes:
        print(
            "  ⚡ 本批 seg_modes 无 static_rule：跳过 Radiomics judge_batch（大队列加速）",
            flush=True,
        )
    seg_store: dict[str, dict[str, dict]] = {m: {} for m in seg_modes}
    path_store: dict[str, dict] = {}

    print(f"\n{'='*60}")
    print(f"B3 static bundle：{len(img_names)} 张图，seg_modes={sorted(seg_modes)} cls_modes={sorted(cls_modes)}")
    print(f"{'='*60}")
    _write_bundle_progress(
        output_root, phase="scan", index=0, total=len(img_names), image_name=""
    )

    gt_stem_map = _build_gt_stem_map(gt_mask_dir)

    for idx, img_name in enumerate(img_names):
        gt_mask = _load_gt_mask_for_image(gt_stem_map, img_name, pipeline.image_io)
        image: Optional[np.ndarray] = None
        inter_row = intermediate_row_index.get(img_name)

        seg_preds = []
        if run_segmentation:
            if inter_row is None:
                print(f"  ✗ {img_name} 不在 intermediate image_names 中，跳过", flush=True)
                continue
            for model_cfg in seg_model_configs:
                model_name = model_cfg.get("name", "")
                cache = seg_cache_by_model.get(model_name)
                if cache is not None:
                    pred = _seg_pred_at(cache, inter_row)
                    if pred is not None:
                        seg_preds.append(pred)

        indie_preds = []
        if run_classification:
            for model_cfg in cls_model_configs:
                model_name = model_cfg.get("name", "")
                cls_map = cls_cache_by_model.get(model_name)
                if cls_map is not None:
                    pred = cls_map.get(img_name)
                    if pred is not None:
                        indie_preds.append(pred)

        consensus, anchor_class = False, None
        if indie_preds:
            consensus, anchor_class = CascadePipeline._check_classification_consensus(
                indie_preds, min_confidence=consensus_threshold
            )

        path_store[img_name] = {
            "consensus": consensus,
            "anchor_class": anchor_class,
            "path": "A" if consensus else "B",
            "indie_preds": indie_preds,
        }

        judge_results = None
        if need_radiomics_judge and seg_preds:
            try:
                image = pipeline.image_io.load_image(image_paths[idx])
                judge_results = judge_obj.judge_batch(image, [p.mask for p in seg_preds])
            except Exception as e:
                print(f"  ⚠️ {img_name} radiomics judge 失败: {e}")

        if run_segmentation and seg_preds:
            for seg_mode in seg_modes:
                if seg_mode == "static_rule" and image is None:
                    image = pipeline.image_io.load_image(image_paths[idx])
                pick_image = image if image is not None else np.zeros((1, 1, 3), dtype=np.uint8)
                decision = _pick_seg_decision(
                    pipeline,
                    seg_mode,
                    image=pick_image,
                    seg_preds=seg_preds,
                    anchor_class=anchor_class,
                    gt_mask=gt_mask,
                    judge_results=judge_results,
                    best_single_seg=best_single_seg,
                    input_device_info=input_device_info,
                    input_data_info=input_data_info,
                    skip_radiomics=skip_static_radiomics,
                )
                method = _seg_method_name(seg_mode)
                seg_store[seg_mode][img_name] = {
                    "mask": decision.selected_mask,
                    "decision": _serialize_seg_decision(
                        img_name, consensus, anchor_class, decision, method
                    ),
                }

        if (idx + 1) % 10 == 0 or idx == len(img_names) - 1:
            print(f"  bundle 扫图 [{idx + 1}/{len(img_names)}] {img_name}", flush=True)
            _write_bundle_progress(
                output_root,
                phase="scan",
                index=idx + 1,
                total=len(img_names),
                image_name=img_name,
            )

    completed: list[str] = []
    for entry in entries:
        out_dir = output_root / entry.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        link_shared_intermediate(out_dir, shared_intermediate)
        if force_phase3:
            clear_phase3_cache(out_dir)

        seg_mode = _assert_static_mode(entry.seg, side="seg") if run_segmentation else None
        cls_mode = _assert_static_mode(entry.cls, side="cls") if run_classification else None

        sel_names: list[str] = []
        sel_masks: list[np.ndarray] = []
        seg_decisions_serialized: list[dict] = []

        if run_segmentation and seg_mode:
            for img_name in img_names:
                row = seg_store[seg_mode].get(img_name)
                if row is None:
                    continue
                sel_names.append(img_name)
                sel_masks.append(row["mask"])
                seg_decisions_serialized.append(row["decision"])

            sel_path = out_dir / "intermediate" / "selected_masks.npz"
            CascadePipeline._save_selected_masks_npz(
                sel_path, sel_names, sel_masks, seg_decisions_serialized
            )

        results: list[dict] = []
        if run_classification and cls_mode:
            for img_name in img_names:
                pinfo = path_store.get(img_name, {})
                consensus = pinfo.get("consensus", False)
                anchor_class = pinfo.get("anchor_class")
                indie_preds_img = pinfo.get("indie_preds") or []

                if not indie_preds_img:
                    results.append({"image_name": img_name, "error": "无分类预测"})
                    continue

                has_seg = seg_mode is not None and img_name in seg_store.get(seg_mode, {})
                seg_d = None
                sel_mask = None
                if has_seg:
                    row = seg_store[seg_mode][img_name]
                    seg_d = row["decision"]
                    sel_mask = row["mask"]

                cls_decision = _resolve_cls_for_image(
                    pipeline,
                    cls_mode,
                    consensus=consensus,
                    anchor_class=anchor_class,
                    indie_preds=indie_preds_img,
                    img_name=img_name,
                    labels=labels,
                    best_single_cls=best_single_cls,
                    seg_decision_dict=seg_d,
                    selected_mask=sel_mask,
                    has_seg=has_seg,
                )

                raw_label = cls_decision.selected_class
                final_label = pipeline._label_to_binary(raw_label)
                if final_label is None:
                    final_label = raw_label

                result: dict[str, Any] = {
                    "path": "A" if consensus else "B",
                    "classification_consensus": consensus,
                    "cls_decision": cls_decision.to_dict(),
                    "final_label": final_label,
                    "final_confidence": cls_decision.confidence,
                    "image_name": img_name,
                }
                if has_seg and seg_d is not None and sel_mask is not None:
                    result["seg_decision"] = seg_d
                    result["selected_mask_shape"] = list(sel_mask.shape)
                    result["selected_mask_area"] = int(np.sum(sel_mask))

                img_stem = Path(img_name).stem
                if img_stem in labels:
                    result["true_label"] = labels[img_stem]
                results.append(result)

        elif run_segmentation and seg_mode:
            for img_name in img_names:
                row = seg_store[seg_mode].get(img_name)
                if not row:
                    continue
                seg_d = row["decision"]
                result = {
                    "path": seg_d.get("path"),
                    "seg_decision": seg_d,
                    "image_name": img_name,
                    "classification_consensus": seg_d.get("consensus"),
                }
                img_stem = Path(img_name).stem
                if img_stem in labels:
                    result["true_label"] = labels[img_stem]
                results.append(result)

        CascadePipeline._save_results_json(results, out_dir / "results.json")
        write_intermediate_manifest(out_dir / "intermediate", out_dir / "intermediate_manifest.json")
        write_routing_methods_report(out_dir, results)
        _write_ablation_run_meta(
            out_dir,
            entry,
            task=task,
            shared_intermediate=shared_intermediate,
            force_phase3=force_phase3,
        )

        if not skip_eval and results:
            metrics = evaluate_pipeline(
                results,
                str(out_dir),
                image_io=pipeline.image_io,
                gt_mask_dir=gt_mask_dir,
            )
            run_label = entry.run_id
            b3 = build_b3_metrics(
                results,
                metrics,
                routing_mode=run_label,
                task=task,
                seg_mode=seg_mode or entry.seg,
                cls_mode=cls_mode or entry.cls,
            )
            save_b3_metrics(out_dir, b3)
            print(f"  ✓ {entry.run_id}: b3_metrics.json")

        completed.append(entry.run_id)

    print(f"\nBundle 完成: {completed}")
    for cache in seg_cache_by_model.values():
        if isinstance(cache, _LazySegModelCache):
            cache.close()
    return completed


def load_labels(label_file: Optional[str], label_key: str) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    if not label_file:
        return labels
    with open(label_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        labels = {Path(k).stem: v for k, v in raw.items()}
    elif isinstance(raw, list):
        for item in raw:
            fname = item.get("filename", "")
            if fname:
                labels[Path(fname).stem] = item.get(label_key)
    return labels


def collect_image_paths(
    image_dir: str,
    start_index: int = 0,
    max_images: Optional[int] = None,
) -> tuple[list[str], list[Path]]:
    img_dir = Path(image_dir)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_paths = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in exts],
        key=lambda x: x.name,
    )
    if start_index > 0:
        image_paths = image_paths[start_index:]
    if max_images is not None:
        image_paths = image_paths[:max_images]
    img_names = [p.name for p in image_paths]
    return img_names, image_paths


def build_pipeline_for_bundle(config: dict, device: str) -> CascadePipeline:
    """构造仅用于静态 bundle 的 CascadePipeline（无 LLM）。"""
    judge_cfg = config.get("radiomics_judge", {})
    pipe_cfg = config.get("pipeline", {})
    judge = None
    if judge_cfg.get("enabled", False) and pipe_cfg.get("run_segmentation", True):
        try:
            judge = RadiomicsJudge(
                model_dir=judge_cfg["model_dir"],
                top_k_features=judge_cfg.get("top_k_features", 5),
                shap_reference_path=judge_cfg.get("shap_reference_path"),
                radiomics_params=judge_cfg.get("radiomics_params"),
            )
        except Exception as e:
            print(f"⚠️ Radiomics 裁判加载失败: {e}")

    seg_cfg = config.get("segmentation", {})
    cls_cfg = config.get("classification", {})
    seg_agent = SegmentationAgent(
        llm_client=None,
        quality_evaluator=SegmentationQualityEvaluator(),
        radiomics_judge=judge,
        config=seg_cfg.get("agent", {}),
        routing_logger=None,
    )
    cls_agent = LLMClassificationAgent(llm_client=None, config=cls_cfg.get("agent", {}))
    return CascadePipeline(
        seg_agent=seg_agent,
        cls_agent=cls_agent,
        image_io=ImageIO(),
        config=config,
        device=device,
    )
