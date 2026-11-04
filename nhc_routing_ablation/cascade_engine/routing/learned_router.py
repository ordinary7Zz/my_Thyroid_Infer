"""
B3 策略 6：验证集多项逻辑回归「选专家」路由器（分割全图 / 分类仅 Path B 训练标签）。

推理只读 intermediate 衍生的表格特征，与 routing_log 摘要同源；不含 AutoGluon。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import numpy as np

from shared.routing_log import (
    cls_disagreement_tier,
    seg_disagreement_tier,
    summarize_cls_candidates,
    summarize_seg_candidates,
)


def _tier_code(tier: str) -> float:
    if tier == "close":
        return 1.0
    if tier == "easy":
        return 0.0
    return 0.5


def _dice_coeff(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    denom = float(a.sum() + b.sum())
    if denom <= 0:
        return 0.0
    return 2.0 * inter / denom


def _normalize_binary_label(label: Any) -> Optional[int]:
    if label is None:
        return None
    if isinstance(label, (int, float)):
        return int(label)
    s = str(label).strip().lower()
    if s in ("1", "恶性", "malignant", "m"):
        return 1
    if s in ("0", "良性", "benign", "b"):
        return 0
    return None


def _cls_top_binary(pred: Any) -> Optional[int]:
    return _normalize_binary_label(pred.top_class)


@dataclass
class LearnedRouterParams:
    """与 config pipeline.ablation.learned_router.params 对齐。"""

    C: float = 0.1
    max_iter: int = 200
    random_state: int = 42
    random_init: bool = False
    random_coef_scale: float = 0.05

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> LearnedRouterParams:
        raw = raw or {}
        return cls(
            C=float(raw.get("C", 0.1)),
            max_iter=int(raw.get("max_iter", 200)),
            random_state=int(raw.get("random_state", 42)),
            random_init=bool(raw.get("random_init", False)),
            random_coef_scale=float(raw.get("random_coef_scale", 0.05)),
        )


def seg_feature_names(model_names: Sequence[str]) -> list[str]:
    names: list[str] = []
    for m in model_names:
        for suffix in (
            "mask_area",
            "mean_confidence",
            "average_agreement",
            "circularity",
            "num_components",
        ):
            names.append(f"{m}.{suffix}")
    names.extend(["global.path_b", "global.seg_disagreement_tier"])
    return names


def cls_feature_names(model_names: Sequence[str]) -> list[str]:
    names: list[str] = []
    for m in model_names:
        for suffix in ("top_confidence", "entropy", "margin_top2", "top_class_bin"):
            names.append(f"{m}.{suffix}")
    names.extend(
        [
            "global.malignant_vote_frac",
            "global.conf_margin",
            "global.cls_disagreement_tier",
        ]
    )
    return names


def build_seg_features(
    predictions: list[Any],
    quality_results: dict,
    *,
    anchor_class: Optional[str],
    model_names: Sequence[str],
) -> np.ndarray:
    summary = {row["model_name"]: row for row in summarize_seg_candidates(predictions, quality_results)}
    parts: list[float] = []
    for name in model_names:
        row = summary.get(name, {})
        parts.append(float(row.get("mask_area", 0)))
        parts.append(float(row.get("mean_confidence", 0.0) or 0.0))
        parts.append(float(row.get("average_agreement", 0.0) or 0.0))
        parts.append(float(row.get("circularity", 0.0) or 0.0))
        parts.append(float(row.get("num_components", 0) or 0))
    parts.append(1.0 if anchor_class is None else 0.0)
    parts.append(_tier_code(seg_disagreement_tier(quality_results)))
    return np.asarray(parts, dtype=np.float64)


def build_cls_features(
    predictions: list[Any],
    *,
    model_names: Sequence[str],
) -> np.ndarray:
    summary = {row["model_name"]: row for row in summarize_cls_candidates(predictions)}
    parts: list[float] = []
    votes_m = 0
    n_votes = 0
    for name in model_names:
        row = summary.get(name, {})
        parts.append(float(row.get("top_confidence", 0.0) or 0.0))
        parts.append(float(row.get("entropy", 0.0) or 0.0))
        parts.append(float(row.get("margin_top2", 0.0) or 0.0))
        bin_cls = _normalize_binary_label(row.get("top_class"))
        parts.append(float(bin_cls) if bin_cls is not None else 0.5)
        if bin_cls == 1:
            votes_m += 1
        if bin_cls is not None:
            n_votes += 1
    confs = sorted(float(p.top_confidence) for p in predictions)
    margin = confs[-1] - confs[-2] if len(confs) >= 2 else 1.0
    parts.append(votes_m / max(n_votes, 1))
    parts.append(margin)
    parts.append(_tier_code(cls_disagreement_tier(predictions)))
    return np.asarray(parts, dtype=np.float64)


@dataclass
class LearnedExpertRouter:
    task: str  # "seg" | "cls"
    model_names: list[str]
    feature_names: list[str]
    params: LearnedRouterParams = field(default_factory=LearnedRouterParams)
    init_mode: str = "fitted"  # fitted | random
    _clf: Any = None

    def predict_expert_index(self, features: np.ndarray) -> int:
        if self._clf is None:
            raise RuntimeError("LearnedExpertRouter 未加载分类器")
        x = features.reshape(1, -1)
        pred = int(self._clf.predict(x)[0])
        return max(0, min(pred, len(self.model_names) - 1))

    def predict_model_name(self, features: np.ndarray) -> str:
        idx = self.predict_expert_index(features)
        idx = max(0, min(idx, len(self.model_names) - 1))
        return self.model_names[idx]

    def select_mask(
        self,
        predictions: list[Any],
        quality_results: dict,
        anchor_class: Optional[str],
    ):
        from segmentation.agent import SegAgentDecision

        if not predictions:
            raise ValueError("无分割候选")
        feats = build_seg_features(
            predictions, quality_results, anchor_class=anchor_class, model_names=self.model_names
        )
        name = self.predict_model_name(feats)
        pred = next((p for p in predictions if p.model_name == name), predictions[0])
        path = "A" if anchor_class else "B"
        conf = float((pred.metadata or {}).get("mean_confidence", 0.5))
        return SegAgentDecision(
            selected_model=pred.model_name,
            selected_mask=pred.mask,
            confidence=conf,
            reasoning=f"learned 路由 → {pred.model_name} (init={self.init_mode})",
            all_predictions=[p.to_dict() for p in predictions],
            classification_anchor=anchor_class,
            path=path,
        )

    def resolve_cls(
        self,
        indie_predictions: list[Any],
    ):
        from classification.agent import ClsAgentDecision

        if not indie_predictions:
            raise ValueError("没有独立分类模型预测")
        feats = build_cls_features(indie_predictions, model_names=self.model_names)
        name = self.predict_model_name(feats)
        pred = next((p for p in indie_predictions if p.model_name == name), indie_predictions[0])
        return ClsAgentDecision(
            selected_model=pred.model_name,
            selected_class=pred.top_class,
            confidence=float(pred.top_confidence),
            reasoning=f"learned 路由 → {pred.model_name} (init={self.init_mode})",
            all_predictions=[p.to_dict() for p in indie_predictions],
            method="path_b_learned",
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "task": self.task,
            "model_names": self.model_names,
            "feature_names": self.feature_names,
            "init_mode": self.init_mode,
            "params": {
                "C": self.params.C,
                "max_iter": self.params.max_iter,
                "random_state": self.params.random_state,
                "random_init": self.params.random_init,
                "random_coef_scale": self.params.random_coef_scale,
            },
            "sklearn_model": self._clf,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LearnedExpertRouter:
        params = LearnedRouterParams.from_dict(payload.get("params"))
        router = cls(
            task=str(payload["task"]),
            model_names=list(payload["model_names"]),
            feature_names=list(payload["feature_names"]),
            params=params,
            init_mode=str(payload.get("init_mode", "fitted")),
            _clf=payload.get("sklearn_model"),
        )
        return router


def _make_logistic(params: LearnedRouterParams):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=params.C,
        max_iter=params.max_iter,
        random_state=params.random_state,
        multi_class="multinomial",
        solver="lbfgs",
    )


def _init_random_classifier(
    n_classes: int,
    n_features: int,
    params: LearnedRouterParams,
) -> Any:
    """可部署前占位：随机系数的多项逻辑回归（仅推理，未在 val 上拟合）。"""
    rng = np.random.RandomState(params.random_state)
    clf = _make_logistic(params)
    clf.classes_ = np.arange(n_classes, dtype=np.int64)
    clf.coef_ = rng.randn(n_classes, n_features) * params.random_coef_scale
    clf.intercept_ = rng.randn(n_classes) * params.random_coef_scale
    clf.n_features_in_ = n_features
    return clf


def build_router_random(
    task: str,
    model_names: list[str],
    params: LearnedRouterParams,
) -> LearnedExpertRouter:
    if task == "seg":
        feat_names = seg_feature_names(model_names)
    elif task == "cls":
        feat_names = cls_feature_names(model_names)
    else:
        raise ValueError(f"未知 task: {task}")
    clf = _init_random_classifier(len(model_names), len(feat_names), params)
    return LearnedExpertRouter(
        task=task,
        model_names=model_names,
        feature_names=feat_names,
        params=params,
        init_mode="random",
        _clf=clf,
    )


def save_learned_router(router: LearnedExpertRouter, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(router.to_payload(), path)
    meta = {
        "task": router.task,
        "model_names": router.model_names,
        "init_mode": router.init_mode,
        "n_features": len(router.feature_names),
    }
    with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_learned_router(path: Path | str) -> LearnedExpertRouter:
    payload = joblib.load(path)
    if isinstance(payload, LearnedExpertRouter):
        return payload
    return LearnedExpertRouter.from_payload(payload)


def model_names_from_config(config: dict, task: str) -> list[str]:
    if task == "seg":
        models = config.get("segmentation", {}).get("models", [])
    else:
        models = config.get("classification", {}).get("models", [])
    return [str(m.get("name", "")) for m in models if m.get("name")]


def learned_router_config(config: dict) -> dict:
    ab = config.get("pipeline", {}).get("ablation", {}) or {}
    return ab.get("learned_router") or config.get("learned_router") or {}


def artifact_path_for_task(config: dict, task: str) -> Optional[Path]:
    lr = learned_router_config(config)
    key = "seg_artifact" if task == "seg" else "cls_artifact"
    raw = lr.get(key)
    if not raw:
        default = Path("artifacts") / f"learned_router_{task}.joblib"
        return default
    return Path(raw)


def load_learned_router_from_config(
    config: dict,
    task: str,
    *,
    create_if_missing: bool = True,
) -> Optional[LearnedExpertRouter]:
    lr_cfg = learned_router_config(config)
    params = LearnedRouterParams.from_dict(lr_cfg.get("params"))
    path = artifact_path_for_task(config, task)
    if path is not None and path.is_file():
        return load_learned_router(path)
    if not create_if_missing:
        return None
    if params.random_init or lr_cfg.get("allow_missing_artifact", True):
        names = model_names_from_config(config, task)
        if not names:
            return None
        router = build_router_random(task, names, params)
        if path is not None and lr_cfg.get("write_random_artifact", False):
            save_learned_router(router, path)
        return router
    return None


def _seg_label_for_image(
    predictions: list[Any],
    gt_mask: np.ndarray,
    model_names: list[str],
) -> int:
    best_i = 0
    best_dice = -1.0
    name_to_idx = {n: i for i, n in enumerate(model_names)}
    for p in predictions:
        d = _dice_coeff(p.mask, gt_mask)
        i = name_to_idx.get(p.model_name, 0)
        if d > best_dice:
            best_dice = d
            best_i = i
    return best_i


def _cls_label_for_image(
    predictions: list[Any],
    gt_bin: int,
    model_names: list[str],
    fallback_model: str,
) -> int:
    name_to_idx = {n: i for i, n in enumerate(model_names)}
    correct = [p for p in predictions if _cls_top_binary(p) == gt_bin]
    if correct:
        best = max(correct, key=lambda p: float(p.top_confidence))
        return name_to_idx.get(best.model_name, 0)
    fb = next((p for p in predictions if p.model_name == fallback_model), predictions[0])
    return name_to_idx.get(fb.model_name, 0)


def fit_learned_seg_router(
    config: dict,
    intermediate_dir: Path,
    *,
    val_stems: list[str],
    gt_mask_dir: Path,
    image_io: Any = None,
) -> LearnedExpertRouter:
    from pipeline.compare_models import (
        build_gt_mask_stem_map,
        load_seg_model_masks,
        resolve_gt_mask_path,
    )
    from routing.best_single_expert import _image_names_for_stems
    from shared.image_io import ImageIO

    image_io = image_io or ImageIO()
    model_names = model_names_from_config(config, "seg")
    lr_cfg = learned_router_config(config)
    params = LearnedRouterParams.from_dict(lr_cfg.get("params"))

    seg_dir = intermediate_dir / "seg"
    by_model: dict[str, tuple[list[str], list]] = {}
    for npz_path in sorted(seg_dir.glob("*.npz")):
        img_names, masks, model_name = load_seg_model_masks(npz_path)
        if model_name in model_names:
            by_model[model_name] = (img_names, masks)

    if not by_model:
        raise RuntimeError("intermediate/seg 无可用专家")

    ref_names, _ = next(iter(by_model.values()))
    stem_set = set(val_stems)
    val_img_names = _image_names_for_stems(ref_names, stem_set)

    from segmentation.quality_evaluator import SegmentationQualityEvaluator

    qeval = SegmentationQualityEvaluator()
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    from segmentation.base_model import SegModelOutput

    gt_stem_map = build_gt_mask_stem_map(gt_mask_dir)

    for img_name in val_img_names:
        preds: list[Any] = []
        for mname in model_names:
            if mname not in by_model:
                continue
            names, masks = by_model[mname]
            if img_name not in names:
                continue
            idx = names.index(img_name)
            preds.append(
                SegModelOutput(model_name=mname, mask=masks[idx], confidence_map=None, metadata={})
            )
        if len(preds) < 2:
            continue
        gt_path = resolve_gt_mask_path(gt_mask_dir, img_name, stem_map=gt_stem_map)
        if gt_path is None:
            continue
        gt_mask = image_io.binarize_mask(image_io.load_mask(gt_path))
        quality = qeval.evaluate_batch([p.mask for p in preds], [p.model_name for p in preds])
        feats = build_seg_features(preds, quality, anchor_class=None, model_names=model_names)
        X_rows.append(feats)
        y_rows.append(_seg_label_for_image(preds, gt_mask, model_names))

    if not X_rows:
        raise RuntimeError("val 上无可用分割样本用于拟合 learned 路由")

    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=np.int64)
    clf = _make_logistic(params)
    clf.fit(X, y)

    return LearnedExpertRouter(
        task="seg",
        model_names=model_names,
        feature_names=seg_feature_names(model_names),
        params=params,
        init_mode="fitted",
        _clf=clf,
    )


def _load_cls_predictions_by_image(json_path: Path) -> tuple[str, dict[str, Any]]:
    from classification.base_model import ClsModelOutput

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "results" not in data:
        raise ValueError(f"分类缓存格式不符: {json_path}")
    model_name = str(data.get("model_name") or json_path.stem)
    out: dict[str, Any] = {}
    for img_name, pred_dict in data["results"].items():
        out[img_name] = ClsModelOutput(
            model_name=pred_dict.get("model_name", model_name),
            predictions=pred_dict["predictions"],
            top_class=pred_dict["top_class"],
            top_confidence=float(pred_dict["top_confidence"]),
            requires_mask=pred_dict["requires_mask"],
            metadata=pred_dict.get("metadata", {}),
        )
    return model_name, out


def _image_names_from_cls_by_model(
    by_model: dict[str, dict[str, Any]],
    model_names: Sequence[str],
) -> list[str]:
    """cls-only intermediate（无 seg）时，用各专家 predictions 的交集对齐图像名。"""
    maps = [by_model[m] for m in model_names if m in by_model]
    if not maps:
        raise FileNotFoundError("intermediate/cls 无可用专家预测")
    common = set(maps[0].keys())
    for img_map in maps[1:]:
        common &= set(img_map.keys())
    if not common:
        raise RuntimeError("各分类专家无共同图像名，无法拟合 learned 路由")
    return sorted(common)


def fit_learned_cls_router(
    config: dict,
    intermediate_dir: Path,
    *,
    val_stems: list[str],
    label_file: str,
    label_key: str = "malignancy",
    consensus_min_confidence: float = 0.6,
    fallback_cls_model: Optional[str] = None,
    image_names: Optional[list[str]] = None,
) -> LearnedExpertRouter:
    from pipeline.cascade import CascadePipeline
    from pipeline.compare_models import load_labels, load_seg_model_masks
    from routing.best_single_expert import _image_names_for_stems

    model_names = model_names_from_config(config, "cls")
    lr_cfg = learned_router_config(config)
    params = LearnedRouterParams.from_dict(lr_cfg.get("params"))
    labels = load_labels(label_file, label_key)
    ab = config.get("pipeline", {}).get("ablation", {}) or {}
    fallback = fallback_cls_model or ab.get("best_single_cls") or model_names[0]

    cls_dir = intermediate_dir / "cls"
    by_model: dict[str, dict[str, Any]] = {}
    for json_path in sorted(cls_dir.glob("*.json")):
        mname, img_map = _load_cls_predictions_by_image(json_path)
        if mname in model_names:
            by_model[mname] = img_map

    if image_names is None:
        seg_dir = intermediate_dir / "seg"
        sample_npz = next(seg_dir.glob("*.npz"), None) if seg_dir.is_dir() else None
        if sample_npz is not None:
            image_names, _, _ = load_seg_model_masks(sample_npz)
        else:
            image_names = _image_names_from_cls_by_model(by_model, model_names)

    val_names = _image_names_for_stems(image_names, val_stems)
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    for img_name in val_names:
        preds: list[Any] = []
        for mname in model_names:
            pred = by_model.get(mname, {}).get(img_name)
            if pred is not None:
                preds.append(pred)
        if len(preds) < 2:
            continue
        consensus, _ = CascadePipeline._check_classification_consensus(
            preds, min_confidence=consensus_min_confidence
        )
        if consensus:
            continue
        stem = Path(img_name).stem
        gt_bin = labels.get(stem)
        if gt_bin is None:
            gt_bin = _normalize_binary_label(labels.get(img_name))
        if gt_bin is None:
            continue
        feats = build_cls_features(preds, model_names=model_names)
        X_rows.append(feats)
        y_rows.append(_cls_label_for_image(preds, int(gt_bin), model_names, fallback))

    if not X_rows:
        raise RuntimeError("val 上无 Path B 分类样本用于拟合 learned 路由")

    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=np.int64)
    clf = _make_logistic(params)
    clf.fit(X, y)

    return LearnedExpertRouter(
        task="cls",
        model_names=model_names,
        feature_names=cls_feature_names(model_names),
        params=params,
        init_mode="fitted",
        _clf=clf,
    )


def ablation_task(config: dict) -> str:
    ab = config.get("pipeline", {}).get("ablation", {}) or {}
    return str(ab.get("task") or "both")


def fit_learned_on_val(
    config: dict,
    intermediate_dir: Path,
    *,
    gt_mask_dir: Path,
    label_file: str,
    label_key: str = "malignancy",
    seg_artifact: Optional[Path] = None,
    cls_artifact: Optional[Path] = None,
    task: Optional[str] = None,
    fit_split: str = "val",
) -> dict[str, Any]:
    from routing.best_single_expert import resolve_learned_fit_stems
    from pipeline.compare_models import load_seg_model_masks

    task = task or ablation_task(config)
    if task not in ("seg", "cls", "both"):
        raise ValueError(f"未知 ablation.task: {task}")

    seg_dir = intermediate_dir / "seg"
    cls_dir = intermediate_dir / "cls"
    sample_npz = next(seg_dir.glob("*.npz"), None) if seg_dir.is_dir() else None
    if sample_npz is None and task in ("seg", "both"):
        raise FileNotFoundError(f"无分割 npz: {seg_dir}")

    if sample_npz is not None:
        img_names, _, _ = load_seg_model_masks(sample_npz)
    else:
        sample_cls = next(cls_dir.glob("*.json"), None) if cls_dir.is_dir() else None
        if sample_cls is None:
            raise FileNotFoundError(f"task={task} 需要 intermediate/seg 或 intermediate/cls")
        _, img_map = _load_cls_predictions_by_image(sample_cls)
        img_names = sorted(img_map.keys())

    val_stems = resolve_learned_fit_stems(config, img_names, fit_split=fit_split)

    cls_cfg = config.get("classification", {}).get("agent", {})
    consensus_th = float(cls_cfg.get("consensus_min_confidence", 0.6))

    seg_router: Optional[LearnedExpertRouter] = None
    cls_router: Optional[LearnedExpertRouter] = None

    if task in ("seg", "both"):
        seg_router = fit_learned_seg_router(
            config,
            intermediate_dir,
            val_stems=val_stems,
            gt_mask_dir=gt_mask_dir,
        )
    if task in ("cls", "both"):
        if not label_file:
            raise ValueError("拟合 cls learned 路由须配置 val_data.label_file 或 pipeline.data.label_file")
        cls_router = fit_learned_cls_router(
            config,
            intermediate_dir,
            val_stems=val_stems,
            label_file=label_file,
            label_key=label_key,
            consensus_min_confidence=consensus_th,
        )

    lr_cfg = learned_router_config(config)
    seg_path = seg_artifact or artifact_path_for_task(config, "seg")
    cls_path = cls_artifact or artifact_path_for_task(config, "cls")
    if seg_router is not None and seg_path:
        save_learned_router(seg_router, seg_path)
    if cls_router is not None and cls_path:
        save_learned_router(cls_router, cls_path)

    out: dict[str, Any] = {
        "task": task,
        "fit_split": fit_split,
        "val_stems": val_stems,
        "n_val_stems": len(val_stems),
        "seg_artifact": str(seg_path) if seg_router is not None and seg_path else None,
        "cls_artifact": str(cls_path) if cls_router is not None and cls_path else None,
    }
    if seg_router is not None:
        out["seg_init_mode"] = seg_router.init_mode
    if cls_router is not None:
        out["cls_init_mode"] = cls_router.init_mode
    return out
