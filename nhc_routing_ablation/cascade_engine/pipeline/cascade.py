"""
串联 Pipeline：5 阶段级联推理（按需加载/卸载模型）。

Phase 1: 依次加载分割模型 → 推理全部图像 → 保存 npz → 卸载
Phase 2: 依次加载独立分类模型 → 推理全部图像 → 保存 json → 卸载
Phase 3: 从缓存加载结果 → 分割 Agent 选 mask → 保存 selected_masks.npz
Phase 4: AutoGluon 加载 → 推理全集图像 → 保存 json → 卸载
Phase 5: 分类裁决 + 保存 results.json

agent_enabled=true  → Phase 3 调 LLM, Phase 5 可能调 LLM
agent_enabled=false → Phase 3/5 纯规则（不调 LLM）
"""

from __future__ import annotations

import json
import os
import pickle
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np

from shared.image_io import ImageIO
from segmentation.agent import SegmentationAgent, SegAgentDecision
from segmentation.model_registry import SegModelRegistry
from classification.agent import LLMClassificationAgent, ClsAgentDecision
from classification.model_registry import ClsModelRegistry
from shared.routing_modes import (
    cls_uses_llm_api,
    get_cls_routing_mode,
    get_seg_routing_mode,
    is_b4_only_cls_mode,
    normalize_routing_mode,
    path_b_includes_autogluon,
    seg_uses_llm_api,
)
from routing.seg_strategies import select_mask_for_mode
from routing.path_b_cls import resolve_path_b_indie_only
from routing.oracle import resolve_cls_oracle
from routing.best_single_expert import get_best_single_names
from shared.center_manifest import load_center_lookup, resolve_center_id
from shared.phase3_options import (
    phase3_checkpoint_every,
    phase3_expert_judge_enabled,
    validate_phase3_checkpoint,
)
from shared.uncertainty_signals import build_uncertainty_block, seg_disagreement_from_masks


def read_seg_npz_image_names(path: Path) -> list[str]:
    """intermediate/seg/*.npz 内 image_names 列（Phase 1 写入顺序）。"""
    with np.load(str(path), allow_pickle=True) as data:
        return [str(x) for x in data["image_names"]]


def discover_intermediate_image_names(seg_cache_dir: Path) -> list[str]:
    """任取一个 seg npz 的 image_names；B5 for_llm 子集须用其名查行，不能用 batch idx。"""
    for npz in sorted(seg_cache_dir.glob("*.npz")):
        return read_seg_npz_image_names(npz)
    return []


def build_intermediate_row_index(intermediate_names: list[str]) -> dict[str, int]:
    return {str(n): i for i, n in enumerate(intermediate_names)}


def preload_workers_for_config(config: dict) -> int:
    """Phase 3 等阶段并行读 intermediate 的线程数（与「多策略 batch 并行」无关）。"""
    pipe = config.get("pipeline", {}) or {}
    raw = pipe.get("preload_workers")
    if raw is not None:
        return max(1, int(raw))
    return min(8, os.cpu_count() or 4)


def phase12_intermediate_complete(
    seg_cache_dir: Path,
    cls_cache_dir: Path,
    seg_model_configs: list,
    cls_model_configs: list,
    *,
    run_segmentation: bool,
    run_classification: bool,
) -> tuple[bool, list[str]]:
    """检查 config 中每个专家是否已有 Phase 1–2 落盘缓存。"""
    missing: list[str] = []
    if run_segmentation:
        for i, model_cfg in enumerate(seg_model_configs):
            name = model_cfg.get("name", f"seg_{i}")
            path = seg_cache_dir / f"{name}.npz"
            if not path.is_file():
                missing.append(str(path))
    if run_classification:
        for i, model_cfg in enumerate(cls_model_configs):
            name = model_cfg.get("name", f"cls_{i}")
            path = cls_cache_dir / f"{name}.json"
            if not path.is_file():
                missing.append(str(path))
    return len(missing) == 0, missing


class CascadePipeline:
    """串联分割筛选与分类筛选（按需加载/卸载模型）。"""

    def __init__(
        self,
        seg_agent: SegmentationAgent,
        cls_agent: LLMClassificationAgent,
        seg_registry: Optional[SegModelRegistry] = None,
        cls_registry: Optional[ClsModelRegistry] = None,
        image_io: ImageIO = None,
        config: Optional[dict] = None,
        device: str = "cuda",
    ):
        self.seg_agent = seg_agent
        self.cls_agent = cls_agent
        self.seg_registry = seg_registry
        self.cls_registry = cls_registry
        self.image_io = image_io or ImageIO()
        self.config = config or {}
        self.output_cfg = config.get("pipeline", {}).get("output", {}) if config else {}
        self.device = device
        self._learned_routers: dict[str, Any] = {}

    def _learned_router(self, task: str):
        """懒加载 B3 learned 路由器（CPU 推理，不占额外显存）。"""
        if task not in self._learned_routers:
            from routing.learned_router import load_learned_router_from_config

            router = load_learned_router_from_config(self.config, task)
            if router is None:
                raise ValueError(
                    f"routing_mode=learned 但无法加载 {task} 路由器；"
                    "请运行 fit_learned_router.py 或开启 learned_router.params.random_init"
                )
            self._learned_routers[task] = router
        return self._learned_routers[task]

    def _enrich_result_record(
        self,
        result: dict[str, Any],
        *,
        img_name: str,
        indie_preds: list,
        seg_decision_dict: Optional[dict],
        consensus: bool,
        final_confidence: float,
        llm_used_seg: bool,
        llm_used_cls: bool,
        center_lookup: dict[str, str],
    ) -> None:
        cid = resolve_center_id(img_name, center_lookup)
        if cid is not None:
            result["center_id"] = cid
        seg_conf = seg_decision_dict.get("confidence") if seg_decision_dict else None
        seg_ag = (seg_decision_dict or {}).get("seg_agreement")
        result["uncertainty"] = build_uncertainty_block(
            indie_preds,
            seg_ag,
            final_confidence=final_confidence,
            seg_confidence=seg_conf,
            llm_used_seg=llm_used_seg,
            llm_used_cls=llm_used_cls and not consensus,
            classification_consensus=consensus,
        )

    # ===== 缓存读写工具 =====

    @staticmethod
    def _save_seg_npz(
        path: Path,
        image_names: list[str],
        seg_outputs: list,
    ) -> None:
        """将一个分割模型对所有图像的结果保存为 npz。

        npz 内部结构:
            image_names: np.ndarray[str]
            masks: np.ndarray (N, H, W) uint8
            confidence_maps: np.ndarray (N, H, W) float32  (无 conf 时为空数组)
            metadata: bytes (pickle 序列化的 list[dict])
            model_names: np.ndarray[str]
        """
        masks = []
        confs = []
        metas = []
        model_names = []
        for pred in seg_outputs:
            masks.append(pred.mask if pred.mask is not None else np.zeros((1, 1), dtype=np.uint8))
            confs.append(
                pred.confidence_map
                if pred.confidence_map is not None
                else np.zeros((1, 1), dtype=np.float32)
            )
            metas.append(pred.metadata or {})
            model_names.append(pred.model_name)

        # 统一 shape（可能不同图像尺寸不同，用 object 数组）
        masks_arr = np.empty(len(masks), dtype=object)
        confs_arr = np.empty(len(confs), dtype=object)
        for i in range(len(masks)):
            masks_arr[i] = masks[i]
            confs_arr[i] = confs[i]

        np.savez_compressed(
            str(path),
            image_names=np.array(image_names),
            masks=masks_arr,
            confidence_maps=confs_arr,
            metadata=np.array(pickle.dumps(metas)),
            model_names=np.array(model_names),
        )

    @staticmethod
    def _load_seg_npz(path: Path, load_confidence: bool = True) -> list:
        """加载分割 npz，返回每张图的 SegModelOutput。

        Args:
            load_confidence: False 时跳过 confidence_map（静态选 mask 不需要，
                可大幅省内存；medsam2 等全分辨率 float32 conf map 体积是 mask 的 4 倍）。
        """
        from segmentation.base_model import SegModelOutput

        data = np.load(str(path), allow_pickle=True)
        image_names = data["image_names"]
        masks = data["masks"]
        metas = pickle.loads(data["metadata"].item())
        model_names = data["model_names"]
        confs = data["confidence_maps"] if load_confidence else None

        results = []
        for i in range(len(image_names)):
            mask = masks[i]
            conf_val = None
            if confs is not None:
                conf = confs[i]
                # 恢复为 SegModelOutput 对象
                has_conf = conf.shape != (1, 1)
                conf_val = conf.astype(np.float32) if has_conf else None
            results.append(SegModelOutput(
                model_name=str(model_names[i]),
                mask=mask.astype(np.uint8),
                confidence_map=conf_val,
                metadata=metas[i],
            ))
        return results

    @staticmethod
    def _save_cls_json(path: Path, image_names: list[str], cls_outputs: list) -> None:
        """将一个分类模型对所有图像的结果保存为 json。"""
        results = {}
        for img_name, pred in zip(image_names, cls_outputs):
            results[img_name] = pred.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model_name": cls_outputs[0].model_name if cls_outputs else "", "results": results}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _load_cls_json(path: Path) -> list:
        """加载分类 json，返回 list[ClsModelOutput]（顺序不保证；Phase 3 请用 _load_cls_json_map）。"""
        return list(CascadePipeline._load_cls_json_map(path).values())

    @staticmethod
    def _load_cls_json_map(path: Path) -> dict[str, Any]:
        """按 image_name 索引的 ClsModelOutput（与 Phase 2 json results 键一致）。"""
        from classification.base_model import ClsModelOutput

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        out: dict[str, Any] = {}
        for img_name, pred_dict in data.get("results", {}).items():
            out[str(img_name)] = ClsModelOutput(
                model_name=pred_dict["model_name"],
                predictions=pred_dict["predictions"],
                top_class=pred_dict["top_class"],
                top_confidence=pred_dict["top_confidence"],
                requires_mask=pred_dict["requires_mask"],
                metadata=pred_dict.get("metadata", {}),
            )
        return out

    def _preload_seg_cls_caches(
        self,
        seg_model_configs: list,
        cls_model_configs: list,
        seg_cache_dir: Path,
        cls_cache_dir: Path,
        *,
        need_seg_conf: bool,
    ) -> tuple[dict[str, list], dict[str, dict[str, Any]], list[str]]:
        """并行加载 Phase 3 所需的 seg npz / cls json（策略推理仍按图串行）。"""
        workers = preload_workers_for_config(self.config)
        seg_cache_by_model: dict[str, list] = {}
        cls_cache_by_model: dict[str, dict[str, Any]] = {}
        intermediate_names: list[str] = []

        seg_jobs: list[tuple[str, Path, bool]] = []
        for model_cfg in seg_model_configs:
            model_name = model_cfg.get("name", "")
            if not model_name:
                continue
            path = seg_cache_dir / f"{model_name}.npz"
            if path.is_file():
                seg_jobs.append((model_name, path, need_seg_conf))

        cls_jobs: list[tuple[str, Path]] = []
        for model_cfg in cls_model_configs:
            model_name = model_cfg.get("name", "")
            if not model_name:
                continue
            path = cls_cache_dir / f"{model_name}.json"
            if path.is_file():
                cls_jobs.append((model_name, path))

        if seg_jobs:
            intermediate_names = read_seg_npz_image_names(seg_jobs[0][1])

        n_tasks = len(seg_jobs) + len(cls_jobs)
        if n_tasks == 0:
            return seg_cache_by_model, cls_cache_by_model, intermediate_names
        if n_tasks == 1:
            workers = 1

        print(
            f"  预加载 intermediate：{len(seg_jobs)} seg + {len(cls_jobs)} cls"
            f"（workers={min(workers, n_tasks)}）...",
            flush=True,
        )

        def _load_seg(job: tuple[str, Path, bool]) -> tuple[str, list]:
            name, path, need_conf = job
            return name, self._load_seg_npz(path, load_confidence=need_conf)

        def _load_cls(job: tuple[str, Path]) -> tuple[str, dict[str, Any]]:
            name, path = job
            return name, self._load_cls_json_map(path)

        with ThreadPoolExecutor(max_workers=min(workers, n_tasks)) as pool:
            futs: list[tuple[str, Any]] = []
            for job in seg_jobs:
                futs.append(("seg", pool.submit(_load_seg, job)))
            for job in cls_jobs:
                futs.append(("cls", pool.submit(_load_cls, job)))
            pending = {fut: kind for kind, fut in futs}
            for fut in as_completed(pending):
                kind = pending[fut]
                name, rows = fut.result()
                if kind == "seg":
                    seg_cache_by_model[name] = rows
                else:
                    cls_cache_by_model[name] = rows

        return seg_cache_by_model, cls_cache_by_model, intermediate_names

    @staticmethod
    def _save_selected_masks_npz(
        path: Path,
        image_names: list[str],
        masks: list[np.ndarray],
        seg_decisions_serialized: list[dict],
    ) -> None:
        """保存 Phase 3 选定的 mask + 分割决策摘要。"""
        masks_arr = np.empty(len(masks), dtype=object)
        for i, m in enumerate(masks):
            masks_arr[i] = m

        np.savez_compressed(
            str(path),
            image_names=np.array(image_names),
            selected_masks=masks_arr,
            seg_decisions=np.array(pickle.dumps(seg_decisions_serialized)),
        )

    @staticmethod
    def _load_selected_masks_npz(path: Path) -> tuple[list[str], list[np.ndarray], list[dict]]:
        """加载选定 mask + 分割决策摘要。"""
        data = np.load(str(path), allow_pickle=True)
        image_names = list(data["image_names"])
        masks = list(data["selected_masks"])
        decisions = pickle.loads(data["seg_decisions"].item())
        return image_names, masks, decisions

    @staticmethod
    def _save_results_json(results: list[dict], path: Path) -> None:
        """保存最终结果到 JSON（去除 numpy 类型）。"""
        serializable = []
        for r in results:
            s = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    s[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    s[k] = float(v)
                else:
                    s[k] = v
            serializable.append(s)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    # ===== Phase 辅助方法 =====

    @staticmethod
    def _label_to_binary(label):
        """将标签统一转为 0/1（良性=0, 恶性=1）。无法转换返回 None。"""
        if isinstance(label, (int, float)):
            return int(label)
        if isinstance(label, str):
            if label in ("恶性", "malignant", "M", "1", "1.0"):
                return 1
            if label in ("良性", "benign", "B", "0", "0.0"):
                return 0
        return None

    @staticmethod
    def _weighted_soft_vote(predictions, voting_weights=None):
        """加权软投票：按 voting_weights 对各类概率加权平均。

        标签键统一归一化（良性/benign/0 → 良性侧，恶性/malignant/1 → 恶性侧）。
        voting_weights: {model_name: weight}，缺失或 None 时等权(1.0)。

        Returns:
            (selected_class_str, confidence)
        """
        voting_weights = voting_weights or {}
        w_sum = 0.0
        p_benign = 0.0
        p_malign = 0.0
        for p in predictions:
            w = float(voting_weights.get(p.model_name, 1.0))
            preds = p.predictions or {}
            pb = preds.get("良性", preds.get("benign", preds.get("0", preds.get("B", 0.0))))
            pm = preds.get("恶性", preds.get("malignant", preds.get("1", preds.get("M", 0.0))))
            try:
                pb = float(pb)
                pm = float(pm)
            except (TypeError, ValueError):
                pb = 0.0
                pm = 0.0
            p_benign += w * pb
            p_malign += w * pm
            w_sum += w
        if w_sum <= 0:
            w_sum = 1.0
        p_benign /= w_sum
        p_malign /= w_sum
        if p_malign >= p_benign:
            return "恶性", p_malign
        return "良性", p_benign

    def _get_voting_weights(self):
        """从 config 提取分类模型的投票权重 {model_name: weight}。

        独立分类模型的权重从 classification.models 的 voting_weight 读取。
        AutoGluon (radiomics_judge) 的权重从 radiomics_judge.voting_weight 读取（默认 0.5），
        仅在 Path B 裁决时作为第4个投票成员参与。
        """
        weights = {}
        for model_cfg in self.config.get("classification", {}).get("models", []):
            if model_cfg.get("type") == "autogluon_radiomics":
                continue  # mask-dependent，不参与独立投票
            name = model_cfg.get("name", "")
            weights[name] = float(model_cfg.get("voting_weight", 1.0))
        # AutoGluon 的投票权重（Path B 时参与加权软投票）
        judge_cfg = self.config.get("radiomics_judge", {})
        if judge_cfg.get("enabled", False):
            weights["radiomics_judge"] = float(judge_cfg.get("voting_weight", 0.5))
        return weights

    @staticmethod
    def _check_classification_consensus(predictions, min_confidence=0.6):
        """判断独立分类模型是否有共识。

        标签先归一化（良性=0/恶性=1），避免 medsiglip 输出 "0"/"1"
        而其他模型输出 "良性"/"恶性" 导致的字符串不一致误判。

        弱化共识条件：只要归一化后 top_class 一致即视为共识，
        不再要求 min(confidence) > min_confidence。一致性本身即强信号，
        低置信一致也直接采纳锚点，避免引入 AutoGluon 仲裁的不确定性。
        （min_confidence 参数保留以兼容调用，但不再用作门槛。）

        Returns:
            (consensus: bool, anchor_class: str | None)
        """
        if len(predictions) < 2:
            return False, None
        classes = [
            CascadePipeline._label_to_binary(p.top_class) for p in predictions
        ]
        if all(c is not None for c in classes) and len(set(classes)) == 1:
            return True, ("恶性" if classes[0] == 1 else "良性")
        return False, None

    @staticmethod
    def _make_anchor_classification_decision(
        anchor_class, indie_preds
    ) -> "ClsAgentDecision":
        """Path A: 分类模型一致 → 直接用锚点。"""
        best = max(indie_preds, key=lambda p: p.top_confidence)
        min_conf = min(p.top_confidence for p in indie_preds)
        reasoning = (
            f"独立分类模型一致预测为「{anchor_class}」"
            f"(conf={min_conf:.2f}~{best.top_confidence:.2f})，直接采纳。"
        )
        return ClsAgentDecision(
            selected_model="independent_consensus",
            selected_class=anchor_class,
            confidence=float(best.top_confidence),
            reasoning=reasoning,
            all_predictions=[p.to_dict() for p in indie_preds],
            method="classification_consensus",
        )

    def _select_mask_static(
        self, image, seg_predictions, anchor_class,
        gt_mask=None, input_device_info=None, input_data_info=None,
        skip_radiomics=False, judge_results=None,
    ) -> tuple["SegAgentDecision", dict[str, Any]]:
        """静态规则选择最佳分割 mask。

        Args:
            skip_radiomics: 跳过 RadiomicsJudge（省内存，仅用 agreement 选 mask）。
            judge_results: 外部预先跑好的 radiomics 裁判结果（与 seg_predictions 等长）。
                传入则跳过内部 _run_judge，避免与 Phase 4 重复计算。
        """
        from segmentation.agent import SegAgentDecision

        pred_map = {p.model_name: p for p in seg_predictions}
        names = [p.model_name for p in seg_predictions]
        masks = [p.mask for p in seg_predictions]
        preds_list = list(seg_predictions)

        quality_results = self.seg_agent.quality_evaluator.evaluate_batch(masks, names)
        # average_agreement 是 list（每个模型与其他模型的平均 IoU）
        avg_agreement_list = (
            quality_results.get("agreement_metrics", {})
            .get("average_agreement", [])
        )
        # 转成 dict: {model_name: avg_agreement_score}
        if isinstance(avg_agreement_list, list):
            agreement_dict = {
                n: float(avg_agreement_list[i]) if i < len(avg_agreement_list) else 0.0
                for i, n in enumerate(names)
            }
        else:
            agreement_dict = avg_agreement_list or {}

        remaining_names = set(names)
        if anchor_class and not skip_radiomics:
            if judge_results is None:
                judge_results = self.seg_agent._run_judge(image, preds_list)
            if judge_results:
                anchor_numeric = 1 if anchor_class == "恶性" else 0
                for p in preds_list:
                    j = judge_results[preds_list.index(p)] if judge_results else {}
                    if (j.get("valid") and j.get("confidence", 0) > 0.6
                            and abs(j.get("malignant_prob", 0.5) - anchor_numeric) > 0.4):
                        remaining_names.discard(p.model_name)
                if not remaining_names:
                    remaining_names = set(names)

        best_name = max(remaining_names, key=lambda n: agreement_dict.get(n, 0))
        best_pred = pred_map[best_name]
        best_idx = names.index(best_name)

        quality = quality_results["individual_quality"][best_idx]
        agreement = agreement_dict.get(best_name, 0) if agreement_dict else None
        reason = (
            f"静态规则：{best_name}（average_agreement={agreement:.2f}）"
            if agreement else f"静态规则：{best_name}"
        )
        if anchor_class:
            reason += f"，与分类锚点「{anchor_class}」吻合"

        decision = SegAgentDecision(
            selected_model=best_name,
            selected_mask=best_pred.mask,
            confidence=agreement or 0.5,
            reasoning=reason,
            all_predictions=[p.to_dict() for p in preds_list],
            quality_metrics=quality,
            agreement_score=agreement,
            is_ensemble=False,
            classification_anchor=anchor_class,
            path="A" if anchor_class else "B",
        )
        return decision, quality_results

    @staticmethod
    def _resolve_classification_path_b_static(
        indie_preds, autogluon_pred, voting_weights=None
    ) -> "ClsAgentDecision":
        """Path B 静态分类裁决：AutoGluon 加入加权软投票。

        AutoGluon 基于 ROI radiomics 的分类结果作为第4个投票成员，
        与3个独立分类模型一起做加权软投票，使 Path B 引入 ROI 级信号，
        与 Path A（纯独立模型共识）形成互补。

        AutoGluon 权重由 config radiomics_judge.voting_weight 控制（默认 0.5），
        低于独立模型（0.79~0.91），反映其在难样本上较低的 AUROC。
        若 AutoGluon 无有效预测（predictions 为空），则仅用独立模型软投票。

        voting_weights: {model_name: weight}，None 或缺失时等权(1.0)。
        """
        # 把 AutoGluon 加入投票列表（跳过无有效预测的）
        all_preds = list(indie_preds)
        if autogluon_pred is not None and autogluon_pred.predictions:
            all_preds.append(autogluon_pred)

        sv_class, sv_conf = CascadePipeline._weighted_soft_vote(
            all_preds, voting_weights
        )

        if autogluon_pred is not None and autogluon_pred.predictions:
            al_binary = CascadePipeline._label_to_binary(autogluon_pred.top_class)
            al_class_str = "恶性" if al_binary == 1 else "良性"
            al_conf = autogluon_pred.top_confidence
            reasoning = (
                f"独立模型无共识，AutoGluon(ROI,{al_class_str},{al_conf:.2f})"
                f"加入加权软投票，结果={sv_class}({sv_conf:.2f})。"
            )
            selected_model = "weighted_soft_vote_with_autogluon"
        else:
            reasoning = f"独立模型无共识，AutoGluon无有效结果，加权软投票={sv_class}({sv_conf:.2f})"
            selected_model = "weighted_soft_vote"

        return ClsAgentDecision(
            selected_model=selected_model,
            selected_class=sv_class,
            confidence=float(sv_conf),
            reasoning=reasoning,
            all_predictions=[p.to_dict() for p in all_preds],
            method="path_b_static_softvote",
        )

    # ===== 主入口 =====

    def run_single(
        self,
        image: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        input_device_info: Optional[list[str]] = None,
        input_data_info: Optional[dict] = None,
    ) -> dict[str, Any]:
        """单图推理（需预加载模型，仅用于调试）。"""
        cls_cfg = self.config.get("classification", {}).get("agent", {})
        seg_cfg = self.config.get("segmentation", {}).get("agent", {})
        enable_seg_agent = seg_cfg.get("enable_agent", True)
        enable_cls_agent = cls_cfg.get("enable_agent", True)
        consensus_threshold = cls_cfg.get("consensus_min_confidence", 0.6)

        indie_preds = self._get_independent_classification_predictions(image)
        consensus, anchor_class = self._check_classification_consensus(
            indie_preds, min_confidence=consensus_threshold
        )

        seg_predictions = self.seg_registry.predict_all(image)
        if not seg_predictions:
            raise RuntimeError("分割模型全部失败，无法继续")

        if enable_seg_agent:
            seg_decision: SegAgentDecision = self.seg_agent.select_best_mask(
                image, seg_predictions,
                classification_anchor=anchor_class,
                gt_mask=gt_mask,
                input_device_info=input_device_info,
                input_data_info=input_data_info,
                image_name=None,
            )
        else:
            seg_decision, _ = self._select_mask_static(
                image, seg_predictions, anchor_class,
                gt_mask=gt_mask,
                input_device_info=input_device_info,
                input_data_info=input_data_info,
            )

        selected_mask = seg_decision.selected_mask

        if consensus:
            cls_decision = self._make_anchor_classification_decision(anchor_class, indie_preds)
        else:
            autogluon_pred = self._run_autogluon_classification(image, selected_mask)
            if enable_cls_agent:
                cls_decision = self.cls_agent.resolve_path_b(
                    indie_preds, autogluon_pred, seg_decision,
                    voting_weights=self._get_voting_weights(),
                    image_name=None,
                )
            else:
                cls_decision = self._resolve_classification_path_b_static(
                    indie_preds, autogluon_pred, self._get_voting_weights()
                )

        return {
            "path": "A" if consensus else "B",
            "classification_consensus": consensus,
            "seg_decision": seg_decision.to_simplified_dict(),
            "selected_mask": selected_mask,
            "selected_mask_shape": selected_mask.shape,
            "selected_mask_area": int(np.sum(selected_mask)),
            "cls_decision": cls_decision.to_dict(),
            "final_label": cls_decision.selected_class,
            "final_confidence": cls_decision.confidence,
        }

    def _get_independent_classification_predictions(self, image):
        """单图模式：跑独立分类模型（需 cls_registry 已加载）。"""
        preds = []
        for name, model in self.cls_registry.get_independent_models():
            try:
                preds.append(model.predict(image, mask=None))
            except Exception as e:
                print(f"  ✗ 独立分类模型 {name} 失败: {e}")
        return preds

    def _run_autogluon_classification(self, image, mask):
        """单图模式：用 seg_agent.radiomics_judge 跑 AutoGluon 分类。"""
        from classification.base_model import ClsModelOutput
        judge = getattr(self.seg_agent, "radiomics_judge", None)
        if judge is None:
            return None
        try:
            result = judge.judge(image, mask)
            if not result.get("valid"):
                return None
            mp = float(result["malignant_prob"])
            predictions = {"良性": 1.0 - mp, "恶性": mp}
            return ClsModelOutput(
                model_name="radiomics_judge",
                predictions=predictions,
                top_class="恶性" if mp > 0.5 else "良性",
                top_confidence=float(result["confidence"]),
                requires_mask=True,
                metadata={"source": "radiomics_judge"},
            )
        except Exception as e:
            print(f"  ✗ AutoGluon 分类失败: {e}")
            return None

    # ===== 批量推理（5 阶段按需加载）=====

    def run_batch(
        self,
        image_dir: str,
        output_dir: str,
        gt_mask_dir: Optional[str] = None,
        label_file: Optional[str] = None,
        input_device_info: Optional[list[str]] = None,
        input_data_info: Optional[dict] = None,
        start_index: int = 0,
        max_images: Optional[int] = None,
        image_paths: Optional[list] = None,
        label_key: str = "malignancy",
        skip_phase1_2: Optional[bool] = None,
        force_phase3: bool = False,
    ) -> list[dict[str, Any]]:
        """批量级联推理（按需加载/卸载模型，中间结果落盘）。

        5 个 Phase:
          Phase 1: 依次加载分割模型 → 推理全部图像 → 保存 npz → 卸载
          Phase 2: 依次加载独立分类模型 → 推理全部图像 → 保存 json → 卸载
          Phase 3: 从缓存加载 → 分割 Agent 选择 → 保存 selected_masks.npz
          Phase 4: AutoGluon 加载 → 推理全集图像 → 保存 json → 卸载
          Phase 5: 分类裁决 + 保存 results.json

        skip_phase1_2:
          None（默认）— 若 intermediate 已含全部专家 seg/cls 缓存则跳过整段 Phase 1–2，
          否则进入 Phase 1–2，并按文件断点续跑（缺哪个专家补哪个）。
          True — 强制跳过 Phase 1–2；False — 始终进入 Phase 1–2 循环（仍可按文件跳过）。
        """
        from segmentation.model_factory import build_seg_model
        from classification.model_factory import build_cls_model

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        inter_dir = out_path / "intermediate"
        seg_cache_dir = inter_dir / "seg"
        cls_cache_dir = inter_dir / "cls"
        seg_cache_dir.mkdir(parents=True, exist_ok=True)
        cls_cache_dir.mkdir(parents=True, exist_ok=True)

        # 加载标签
        # 支持两种格式:
        #   1. dict: {"0001": 0, "0002": 1, ...}
        #   2. list[dict]: [{"filename": "0001.jpg", "malignancy": 0, ...}, ...]
        # label_key 指定 list 格式中标签字段名（如 "malignancy"）
        # key 统一用不带后缀的文件名（stem）
        labels: dict[str, Any] = {}
        label_rows: dict[str, dict[str, Any]] = {}
        if label_file:
            with open(label_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                labels = {Path(k).stem: v for k, v in raw.items()}
            elif isinstance(raw, list):
                for item in raw:
                    fname = item.get("filename", "")
                    if fname:
                        stem = Path(fname).stem
                        labels[stem] = item.get(label_key)
                        label_rows[stem] = item

        pipe_data_cfg = self.config.get("pipeline", {}).get("data", {})
        static_data_info = pipe_data_cfg.get("input_data_info")
        data_info_label_keys = pipe_data_cfg.get("input_data_info_label_keys")

        # 收集图像
        if image_paths is not None:
            image_paths = [Path(p) for p in image_paths]
            img_names = [p.name for p in image_paths]
            print(f"级联推理: {len(img_names)} 张图像 (显式子集)")
        else:
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
            print(f"级联推理: {len(img_names)} 张图像 (start={start_index})")

        self.last_batch_image_count = len(img_names)

        # 获取配置
        seg_cfg_full = self.config.get("segmentation", {})
        cls_cfg_full = self.config.get("classification", {})
        seg_model_configs = seg_cfg_full.get("models", [])
        cls_model_configs = cls_cfg_full.get("models", [])
        seg_agent_cfg = seg_cfg_full.get("agent", {})
        cls_agent_cfg = cls_cfg_full.get("agent", {})
        seg_routing_mode = get_seg_routing_mode(self.config)
        cls_routing_mode = get_cls_routing_mode(self.config)
        enable_seg_agent = seg_agent_cfg.get("enable_agent", True) and seg_uses_llm_api(seg_routing_mode)
        enable_cls_agent = cls_agent_cfg.get("enable_agent", True) and cls_uses_llm_api(cls_routing_mode)
        consensus_threshold = cls_agent_cfg.get("consensus_min_confidence", 0.6)
        best_single_seg, best_single_cls = get_best_single_names(self.config)

        # 阶段开关：可单独关闭分割或分类
        run_segmentation = self.config.get("pipeline", {}).get("run_segmentation", True)
        run_classification = self.config.get("pipeline", {}).get("run_classification", True)
        if not run_segmentation and not run_classification:
            print("⚠️ 分割与分类均已关闭，无可执行阶段，返回空结果")
            return []
        print(f"阶段开关: run_segmentation={run_segmentation}, run_classification={run_classification}")

        skip_phase12_run = False
        if skip_phase1_2 is True:
            skip_phase12_run = True
            print("  ⏭ skip_phase1_2=True，强制跳过 Phase 1–2")
        elif skip_phase1_2 is False:
            skip_phase12_run = False
        else:
            ready, missing = phase12_intermediate_complete(
                seg_cache_dir,
                cls_cache_dir,
                seg_model_configs,
                cls_model_configs,
                run_segmentation=run_segmentation,
                run_classification=run_classification,
            )
            if ready:
                skip_phase12_run = True
                print("  ⏭ Phase 1–2 缓存齐全，跳过专家推理（auto）")
            elif missing:
                print(f"  Phase 1–2 待补缓存: {len(missing)} 个文件（将按专家断点续跑）")

        # ===== Phase 1: 分割模型（逐个加载 → 推理全部图像 → 保存 npz → 卸载）=====
        print(f"\n{'='*60}")
        print("Phase 1: 依次加载分割模型并推理全部图像")
        print(f"{'='*60}")

        if skip_phase12_run:
            print("  ⏭ 已跳过 Phase 1")
        if not run_segmentation:
            print("  ⏭ 分割阶段已关闭，跳过 Phase 1")
        for i, model_cfg in enumerate(seg_model_configs if run_segmentation and not skip_phase12_run else []):
            model_cfg = dict(model_cfg)
            model_name = model_cfg.get("name", f"seg_{i}")
            model_cfg["device"] = self.device
            cache_path = seg_cache_dir / f"{model_name}.npz"

            if cache_path.exists():
                print(f"  [{i+1}/{len(seg_model_configs)}] ⏭ {model_name} 已有缓存，跳过")
                continue

            model = build_seg_model(model_cfg)
            model.load_model()

            seg_outputs = []
            for img_path in image_paths:
                try:
                    image = self.image_io.load_image(img_path)
                    pred = model.predict(image)
                    seg_outputs.append(pred)
                except Exception as e:
                    print(f"  ✗ {model.model_name} 推理 {img_path.name} 失败: {e}")
                    from segmentation.base_model import SegModelOutput
                    seg_outputs.append(SegModelOutput(
                        model_name=model.model_name,
                        mask=np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),
                        confidence_map=None,
                        metadata={"error": str(e)},
                    ))

            model.unload_model()
            self._save_seg_npz(cache_path, img_names, seg_outputs)
            print(f"  [{i+1}/{len(seg_model_configs)}] ✓ {model.model_name} 推理完成，已保存 {cache_path.name}，已卸载")

        # ===== Phase 2: 独立分类模型（逐个加载 → 推理全部图像 → 保存 json → 卸载）=====
        print(f"\n{'='*60}")
        print("Phase 2: 依次加载独立分类模型并推理全部图像")
        print(f"{'='*60}")

        if not run_classification:
            print("  ⏭ 分类阶段已关闭，跳过 Phase 2")
        if skip_phase12_run and run_classification:
            print("  ⏭ 已跳过 Phase 2")
        for i, model_cfg in enumerate(cls_model_configs if run_classification and not skip_phase12_run else []):
            model_cfg = dict(model_cfg)
            model_name = model_cfg.get("name", f"cls_{i}")

            cache_path = cls_cache_dir / f"{model_name}.json"
            if cache_path.exists():
                print(f"  [{i+1}/{len(cls_model_configs)}] ⏭ {model_name} 已有缓存，跳过")
                continue

            model_cfg["device"] = self.device
            model = build_cls_model(model_cfg)
            model.load_model()

            cls_outputs = []
            for img_path in image_paths:
                try:
                    image = self.image_io.load_image(img_path)
                    pred = model.predict(image, mask=None)
                    cls_outputs.append(pred)
                except Exception as e:
                    print(f"  ✗ {model.model_name} 推理 {img_path.name} 失败: {e}")
                    from classification.base_model import ClsModelOutput
                    cls_outputs.append(ClsModelOutput(
                        model_name=model.model_name,
                        predictions={},
                        top_class="unknown",
                        top_confidence=0.0,
                        requires_mask=False,
                        metadata={"error": str(e)},
                    ))

            model.unload_model()
            self._save_cls_json(cache_path, img_names, cls_outputs)
            print(f"  [{i+1}/{len(cls_model_configs)}] ✓ {model.model_name} 推理完成，已保存 {cache_path.name}，已卸载")

        if self.config.get("pipeline", {}).get("stop_after_phase2", False):
            print("  ⏭ stop_after_phase2=True，跳过 Phase 3–5（仅保留 intermediate 供 B3 等）")
            print(f"  ✓ Phase 1–2 intermediate 对应 {len(img_names)} 张图像")
            return []

        # ===== Phase 3: 分类共识 + 分割 Agent（从缓存加载，CPU 运行）=====
        print(f"\n{'='*60}")
        print("Phase 3: 分类共识判断 + 分割 Agent 选择")
        print(f"{'='*60}")

        selected_masks_path = inter_dir / "selected_masks.npz"
        phase3_checkpoint_path = inter_dir / "phase3_checkpoint.npz"
        phase3_expert_judge = phase3_expert_judge_enabled(self.config)
        checkpoint_every = phase3_checkpoint_every(self.config)
        if force_phase3:
            for p in (selected_masks_path, phase3_checkpoint_path):
                if p.exists():
                    p.unlink()
            print("  ✓ force_phase3: 已清除 Phase 3 落盘缓存，将重跑")

        if not run_segmentation:
            print("  ⏭ 分割阶段已关闭，跳过 Phase 3（mask 选择）")
            sel_img_names, sel_masks, seg_decisions_serialized = [], [], []
        elif selected_masks_path.exists():
            print("  ⏭ selected_masks.npz 已存在，跳过 Phase 3")
            sel_img_names, sel_masks, seg_decisions_serialized = \
                self._load_selected_masks_npz(selected_masks_path)
        else:
            import gc

            sel_img_names = []
            sel_masks = []
            seg_decisions_serialized = []
            start_idx = 0
            if phase3_checkpoint_path.exists():
                ck_names, ck_masks, ck_decisions = self._load_selected_masks_npz(
                    phase3_checkpoint_path
                )
                if validate_phase3_checkpoint(img_names, ck_names):
                    sel_img_names = ck_names
                    sel_masks = ck_masks
                    seg_decisions_serialized = ck_decisions
                    start_idx = len(sel_img_names)
                    print(
                        f"  ↻ Phase 3 断点续跑: 已完成 {start_idx}/{len(img_names)}",
                        flush=True,
                    )
                else:
                    print(
                        "  ⚠️ phase3_checkpoint 与当前图像列表不一致，从头重跑",
                        flush=True,
                    )

            if not phase3_expert_judge:
                print(
                    "  ⚡ phase3_expert_judge=false：跳过专家 mask 的 Radiomics judge_batch",
                    flush=True,
                )

            # 预加载每个模型的缓存「仅一次」，避免逐图重复全量解压 npz。
            # 旧实现每张图都把每个模型的完整 npz（约 4GB）解压一遍：N 图 × M 模型
            # 次解压，是 Phase 3 的性能/内存瓶颈（ZJH-8K 上卡死 1.5 天零产出）。
            # 静态模式（enable_seg_agent=False）下不使用 confidence_map，跳过以省内存。
            need_seg_conf = enable_seg_agent or seg_routing_mode == "max_confidence"
            seg_cache_by_model, cls_cache_by_model, intermediate_names = self._preload_seg_cls_caches(
                seg_model_configs,
                cls_model_configs,
                seg_cache_dir,
                cls_cache_dir,
                need_seg_conf=need_seg_conf,
            )
            if not intermediate_names:
                intermediate_names = discover_intermediate_image_names(seg_cache_dir)
            intermediate_row_index = build_intermediate_row_index(intermediate_names)

            # radiomics judge 对象（Phase 3 统一跑一次，结果同时用于 mask 选择
            # 和填充 autogluon 缓存，避免 Phase 4 对 all_datasets mask 重复计算）
            judge_obj = getattr(self.seg_agent, "radiomics_judge", None)
            autogluon_cache_path = inter_dir / "autogluon.json"
            autogluon_results: dict[str, Any] = {}
            if judge_obj is not None and autogluon_cache_path.exists():
                with open(autogluon_cache_path, "r", encoding="utf-8") as f:
                    autogluon_results = json.load(f)
                print(f"  已加载 autogluon 缓存: {len(autogluon_results)} 张")

            from shared.metadata_prompt import merge_per_image_input_data_info

            skip_static_radiomics = not phase3_expert_judge

            gt_stem_map: dict[str, Path] = {}
            if gt_mask_dir:
                gt_dir = Path(gt_mask_dir)
                if gt_dir.is_dir():
                    gt_stem_map = {
                        p.stem: p for p in gt_dir.iterdir() if p.is_file()
                    }

            for idx in range(start_idx, len(img_names)):
                img_name = img_names[idx]
                img_path = image_paths[idx]
                image = self.image_io.load_image(img_path)
                img_stem = Path(img_name).stem
                per_data_info = merge_per_image_input_data_info(
                    input_data_info if input_data_info is not None else static_data_info,
                    label_rows.get(img_stem),
                    label_keys=data_info_label_keys,
                    label_key=label_key,
                )

                gt_mask = None
                if gt_stem_map:
                    gt_path = gt_stem_map.get(img_stem)
                    if gt_path is not None and gt_path.is_file():
                        gt_mask = self.image_io.binarize_mask(
                            self.image_io.load_mask(gt_path)
                        )

                # 按 image_name 在 intermediate npz 中查行（支持 for_llm 子集 ≠ Phase1 列表顺序）
                inter_row = intermediate_row_index.get(img_name)
                seg_preds = []
                for model_cfg in seg_model_configs:
                    model_name = model_cfg.get("name", "")
                    cache = seg_cache_by_model.get(model_name)
                    if cache is not None and inter_row is not None and inter_row < len(cache):
                        seg_preds.append(cache[inter_row])

                indie_preds = []
                for model_cfg in cls_model_configs:
                    model_name = model_cfg.get("name", "")
                    cls_map = cls_cache_by_model.get(model_name)
                    if cls_map is not None:
                        pred = cls_map.get(img_name)
                        if pred is not None:
                            indie_preds.append(pred)

                # 分类共识
                consensus, anchor_class = self._check_classification_consensus(
                    indie_preds, min_confidence=consensus_threshold
                )

                # 分割 Agent
                if inter_row is None and run_segmentation:
                    print(f"  ✗ {img_name} 不在 intermediate image_names 中，跳过")
                    continue
                if not seg_preds:
                    print(f"  ✗ {img_name} 无分割结果，跳过")
                    continue

                # 统一跑 radiomics judge（对所有候选 mask，一次完成）
                # 结果同时用于：1) mask 选择（anchor 过滤 / LLM prompt）2) 填充 autogluon 缓存
                judge_results = None
                if judge_obj is not None and phase3_expert_judge:
                    try:
                        judge_results = judge_obj.judge_batch(
                            image, [p.mask for p in seg_preds], image_name=img_name
                        )
                    except Exception as e:
                        print(f"  ⚠️ {img_name} radiomics judge 失败: {e}")

                seg_method_used = f"routing_{seg_routing_mode}_seg"

                if seg_routing_mode == "llm" and enable_seg_agent:
                    seg_decision = self.seg_agent.select_best_mask(
                        image, seg_preds,
                        classification_anchor=anchor_class,
                        gt_mask=gt_mask,
                        input_device_info=input_device_info,
                        input_data_info=per_data_info,
                        judge_results=judge_results,
                        image_name=img_name,
                    )
                elif seg_routing_mode == "learned":
                    seg_method_used = "routing_learned_seg"
                    quality_results = self.seg_agent.quality_evaluator.evaluate_batch(
                        [p.mask for p in seg_preds],
                        [p.model_name for p in seg_preds],
                    )
                    seg_decision = self._learned_router("seg").select_mask(
                        seg_preds, quality_results, anchor_class
                    )
                    self.seg_agent._seg_log_ctx = {
                        "image_name": img_name,
                        "path": seg_decision.path,
                        "classification_anchor": anchor_class,
                        "predictions": seg_preds,
                        "quality_results": quality_results,
                        "judge_results": judge_results,
                        "metadata_mode": self.seg_agent.metadata_mode,
                    }
                    self.seg_agent._finish_seg_decision(
                        seg_decision,
                        llm_invoked=False,
                        method=seg_method_used,
                    )
                elif seg_routing_mode == "static_rule":
                    seg_method_used = "routing_static_rule_seg"
                    seg_decision, quality_results = self._select_mask_static(
                        image, seg_preds, anchor_class,
                        gt_mask=gt_mask,
                        input_device_info=input_device_info,
                        input_data_info=per_data_info,
                        skip_radiomics=skip_static_radiomics,
                        judge_results=judge_results,
                    )
                    self.seg_agent._seg_log_ctx = {
                        "image_name": img_name,
                        "path": seg_decision.path,
                        "classification_anchor": anchor_class,
                        "predictions": seg_preds,
                        "quality_results": quality_results,
                        "judge_results": judge_results,
                        "metadata_mode": self.seg_agent.metadata_mode,
                    }
                    self.seg_agent._finish_seg_decision(
                        seg_decision,
                        llm_invoked=False,
                        method="routing_static_rule_seg",
                    )
                else:
                    static_fn = None
                    if seg_routing_mode == "static_rule":
                        static_fn = lambda: self._select_mask_static(
                            image, seg_preds, anchor_class,
                            gt_mask=gt_mask,
                            input_device_info=input_device_info,
                            input_data_info=per_data_info,
                            skip_radiomics=skip_static_radiomics,
                            judge_results=judge_results,
                        )[0]
                    seg_decision = select_mask_for_mode(
                        seg_routing_mode,
                        seg_preds,
                        anchor_class,
                        gt_mask=gt_mask,
                        best_single_model=best_single_seg,
                        static_selector=static_fn,
                    )
                    self.seg_agent._seg_log_ctx = {
                        "image_name": img_name,
                        "path": seg_decision.path,
                        "classification_anchor": anchor_class,
                        "predictions": seg_preds,
                        "quality_results": None,
                        "judge_results": judge_results,
                        "metadata_mode": self.seg_agent.metadata_mode,
                    }
                    quality_results = self.seg_agent.quality_evaluator.evaluate_batch(
                        [p.mask for p in seg_preds],
                        [p.model_name for p in seg_preds],
                    )
                    self.seg_agent._seg_log_ctx["quality_results"] = quality_results
                    self.seg_agent._finish_seg_decision(
                        seg_decision,
                        llm_invoked=False,
                        method=f"routing_{seg_routing_mode}_seg",
                    )

                # 从 judge_results 提取 all_datasets 的结果，填充 autogluon 缓存
                # （Phase 4 检测到缓存已存在则跳过，避免对 all_datasets mask 重复计算）
                if judge_results:
                    all_ds_idx = next(
                        (i for i, p in enumerate(seg_preds)
                         if p.model_name == "all_datasets"),
                        None,
                    )
                    if all_ds_idx is not None and all_ds_idx < len(judge_results):
                        jr = judge_results[all_ds_idx]
                        if jr.get("valid"):
                            mp = float(jr["malignant_prob"])
                            autogluon_results[img_name] = {
                                "model_name": "radiomics_judge",
                                "predictions": {"良性": 1.0 - mp, "恶性": mp},
                                "top_class": "恶性" if mp > 0.5 else "良性",
                                "top_confidence": float(jr["confidence"]),
                                "requires_mask": True,
                                "metadata": {"source": "radiomics_judge"},
                            }
                        elif img_name not in autogluon_results:
                            autogluon_results[img_name] = {
                                "model_name": "radiomics_judge",
                                "predictions": {},
                                "top_class": "unknown",
                                "top_confidence": 0.0,
                                "requires_mask": True,
                                "metadata": {"error": "invalid mask"},
                            }

                sel_img_names.append(img_name)
                sel_masks.append(seg_decision.selected_mask)
                seg_decisions_serialized.append({
                    "img_name": img_name,
                    "consensus": consensus,
                    "anchor_class": anchor_class,
                    "selected_model": seg_decision.selected_model,
                    "confidence": seg_decision.confidence,
                    "reasoning": seg_decision.reasoning,
                    "path": seg_decision.path,
                    "method": seg_method_used,
                    "all_predictions": seg_decision.all_predictions,
                    "seg_agreement": seg_disagreement_from_masks([p.mask for p in seg_preds]),
                })
                print(f"  [{idx+1}/{len(img_names)}] {img_name}: {seg_decision.selected_model} (consensus={consensus})")

                if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0:
                    self._save_selected_masks_npz(
                        phase3_checkpoint_path,
                        sel_img_names,
                        sel_masks,
                        seg_decisions_serialized,
                    )
                    if judge_obj is not None and autogluon_results:
                        with open(autogluon_cache_path, "w", encoding="utf-8") as f:
                            json.dump(autogluon_results, f, ensure_ascii=False, indent=2)
                    print(
                        f"  💾 Phase 3 checkpoint: {len(sel_img_names)}/{len(img_names)}",
                        flush=True,
                    )

                # 释放本图临时对象（预加载缓存不在此释放）
                del image, seg_preds, indie_preds, gt_mask, seg_decision, judge_results

            # 释放预加载缓存（sel_masks 已持有被选中 mask 的引用，不会被回收）
            del seg_cache_by_model, cls_cache_by_model
            gc.collect()

            self._save_selected_masks_npz(
                selected_masks_path, sel_img_names, sel_masks, seg_decisions_serialized
            )
            print(f"  ✓ 已保存 {selected_masks_path.name}")
            if phase3_checkpoint_path.exists():
                phase3_checkpoint_path.unlink()

            # 保存 autogluon 缓存（Phase 3 已填充 all_datasets 的 judge 结果，
            # Phase 4 检测到缓存存在则跳过重跑）
            if judge_obj is not None and autogluon_results:
                with open(autogluon_cache_path, "w", encoding="utf-8") as f:
                    json.dump(autogluon_results, f, ensure_ascii=False, indent=2)
                print(f"  ✓ autogluon 缓存已更新: {len(autogluon_results)} 张（Phase 4 将复用）")

        # 构建 img_name → (mask, decision, consensus, anchor) 映射
        sel_map: dict[str, dict] = {}
        for i, name in enumerate(sel_img_names):
            d = seg_decisions_serialized[i]
            sel_map[name] = {
                "mask": sel_masks[i],
                "decision": d,
                "consensus": d["consensus"],
                "anchor_class": d["anchor_class"],
            }

        # ===== Phase 4: AutoGluon 分类（全集推理，便于评估其分类性能）=====
        # 注：PathA（有共识）图像在 Phase 5 裁决时不会使用 AutoGluon 结果，
        # 但这里仍对其推理，以便评估 AutoGluon 在全集上的分类性能。
        print(f"\n{'='*60}")
        print("Phase 4: AutoGluon 分类（全集推理，基于 all_datasets 分割模型输出）")
        print(f"{'='*60}")

        autogluon_cache_path = inter_dir / "autogluon.json"

        # PathB（无共识）图像：Phase 5 裁决时会使用 AutoGluon 结果
        needs_autogluon_names = [
            name for name in img_names
            if name in sel_map and not sel_map[name]["consensus"]
        ]
        # 推理范围：所有有分割结果的图像（用于评估 AutoGluon 全集性能）
        autogluon_target_names = [name for name in img_names if name in sel_map]

        if not autogluon_target_names or not run_classification:
            print("  无可用图像或分类阶段已关闭，跳过 AutoGluon")
            autogluon_results: dict[str, Any] = {}
        elif getattr(self.seg_agent, "radiomics_judge", None) is None:
            print("  ⚠️ radiomics_judge 未启用，跳过 AutoGluon")
            autogluon_results = {}
        else:
            # 加载已有缓存（若存在），支持断点续跑：仅补全缺失的图像
            autogluon_results: dict[str, Any] = {}
            if autogluon_cache_path.exists():
                with open(autogluon_cache_path, "r", encoding="utf-8") as f:
                    autogluon_results = json.load(f)
                print(f"  已加载缓存 {autogluon_cache_path.name}: {len(autogluon_results)} 张")

            # 待推理：缓存中缺失的图像（首次运行时为全集）
            pending_names = [n for n in autogluon_target_names if n not in autogluon_results]
            if pending_names:
                # 从 all_datasets.npz 加载分割模型输出（而非 agent 选出的 mask）
                all_datasets_npz = seg_cache_dir / "all_datasets.npz"
                if not all_datasets_npz.exists():
                    print(f"  ⚠️ {all_datasets_npz.name} 不存在，回退到 selected_mask")
                    all_datasets_masks: dict[str, np.ndarray] = {}
                else:
                    seg_data = np.load(str(all_datasets_npz), allow_pickle=True)
                    all_datasets_masks = {
                        str(n): np.asarray(m)
                        for n, m in zip(seg_data["image_names"], seg_data["masks"])
                    }
                    print(f"  已加载 all_datasets 分割输出: {len(all_datasets_masks)} 张")

                # 复用 seg_agent.radiomics_judge（与分割阶段同一 AutoGluon 模型）
                judge = self.seg_agent.radiomics_judge
                print(
                    f"  待推理 {len(pending_names)}/{len(autogluon_target_names)} 张"
                    f"（其中 PathB 无共识 {len(needs_autogluon_names)} 张）"
                )
                for idx, img_name in enumerate(pending_names):
                    img_path = image_paths[img_names.index(img_name)]
                    image = self.image_io.load_image(img_path)
                    # 优先用 all_datasets 模型的 mask，回退到 selected_mask
                    mask = all_datasets_masks.get(img_name)
                    if mask is None:
                        mask = sel_map[img_name]["mask"]

                    try:
                        result = judge.judge(image, mask, image_name=img_name)
                        if result.get("valid"):
                            mp = float(result["malignant_prob"])
                            autogluon_results[img_name] = {
                                "model_name": "radiomics_judge",
                                "predictions": {"良性": 1.0 - mp, "恶性": mp},
                                "top_class": "恶性" if mp > 0.5 else "良性",
                                "top_confidence": float(result["confidence"]),
                                "requires_mask": True,
                                "metadata": {"source": "radiomics_judge"},
                            }
                        else:
                            autogluon_results[img_name] = {
                                "model_name": "radiomics_judge",
                                "predictions": {},
                                "top_class": "unknown",
                                "top_confidence": 0.0,
                                "requires_mask": True,
                                "metadata": {"error": "invalid mask"},
                            }
                    except Exception as e:
                        print(f"  ✗ radiomics_judge 推理 {img_name} 失败: {e}")
                        autogluon_results[img_name] = {
                            "model_name": "radiomics_judge",
                            "predictions": {},
                            "top_class": "unknown",
                            "top_confidence": 0.0,
                            "requires_mask": True,
                            "metadata": {"error": str(e)},
                        }

                print(f"  ✓ radiomics_judge 推理完成: 全集 {len(autogluon_results)} 张")
            else:
                print(f"  ⏭ 全部 {len(autogluon_target_names)} 张已有缓存，跳过推理")

            with open(autogluon_cache_path, "w", encoding="utf-8") as f:
                json.dump(autogluon_results, f, ensure_ascii=False, indent=2)

        # ===== Phase 5: 分类裁决 + 保存最终结果 =====
        print(f"\n{'='*60}")
        print("Phase 5: 分类裁决 + 保存最终结果")
        print(f"{'='*60}")

        results: list[dict[str, Any]] = []
        pipe_cfg = self.config.get("pipeline", {}) or {}
        rev_cfg = pipe_cfg.get("revision") or {}
        center_manifest = rev_cfg.get("center_manifest") or pipe_cfg.get("data", {}).get(
            "center_manifest"
        )
        center_lookup = load_center_lookup(center_manifest)

        if not run_classification:
            # 分类关闭：仅保存分割结果（selected_masks.npz 已在 Phase 3 落盘）
            print("  ⏭ 分类阶段已关闭，仅保存分割结果")
            for idx, img_name in enumerate(img_names):
                if img_name not in sel_map:
                    results.append({"image_name": img_name, "error": "无分割结果"})
                    continue
                sel_info = sel_map[img_name]
                sel_d = sel_info["decision"]
                selected_mask = sel_info["mask"]
                result = {
                    "path": sel_d.get("path"),
                    "seg_decision": sel_d,
                    "selected_mask_shape": list(selected_mask.shape),
                    "selected_mask_area": int(np.sum(selected_mask)),
                    "image_name": img_name,
                }
                img_stem = Path(img_name).stem
                if img_stem in labels:
                    result["true_label"] = labels[img_stem]
                self._enrich_result_record(
                    result,
                    img_name=img_name,
                    indie_preds=[],
                    seg_decision_dict=sel_d,
                    consensus=bool(sel_d.get("consensus")),
                    final_confidence=float(sel_d.get("confidence") or 0.0),
                    llm_used_seg=enable_seg_agent,
                    llm_used_cls=False,
                    center_lookup=center_lookup,
                )
                results.append(result)
                print(f"  [{idx+1}/{len(img_names)}] {img_name}: seg={sel_d.get('selected_model')}")
                # 增量保存
                if (idx + 1) % 10 == 0 or idx == len(img_names) - 1:
                    self._save_results_json(results, out_path / "results.json")
        else:
            # 从缓存加载独立分类结果（用于裁决）
            all_indie_cls: dict[str, list] = {name: [] for name in img_names}
            for model_cfg in cls_model_configs:
                model_name = model_cfg.get("name", "")
                cache_path = cls_cache_dir / f"{model_name}.json"
                if cache_path.exists():
                    cls_map = self._load_cls_json_map(cache_path)
                    for img_name in img_names:
                        pred = cls_map.get(img_name)
                        if pred is not None:
                            all_indie_cls[img_name].append(pred)

            for idx, img_name in enumerate(img_names):
                indie_preds = all_indie_cls[img_name]
                has_seg = img_name in sel_map

                if has_seg:
                    sel_info = sel_map[img_name]
                    consensus = sel_info["consensus"]
                    anchor_class = sel_info["anchor_class"]
                    seg_decision_dict = sel_info["decision"]
                    selected_mask = sel_info["mask"]
                else:
                    # 无分割结果（仅分类模式）：从独立分类预测直接算共识
                    consensus, anchor_class = self._check_classification_consensus(
                        indie_preds, min_confidence=consensus_threshold
                    )
                    seg_decision_dict = None
                    selected_mask = None

                if not indie_preds:
                    results.append({
                        "image_name": img_name,
                        "error": "无分割与分类结果" if not has_seg else "无分类预测",
                    })
                    continue

                # 分类裁决
                if consensus:
                    cls_decision = self._make_anchor_classification_decision(anchor_class, indie_preds)
                    self.cls_agent.log_path_a_decision(
                        img_name, anchor_class, indie_preds, cls_decision
                    )
                else:
                    from classification.base_model import ClsModelOutput
                    from segmentation.agent import SegAgentDecision

                    use_ag = path_b_includes_autogluon(cls_routing_mode)
                    ag_dict = autogluon_results.get(img_name) if use_ag else None
                    autogluon_pred = None
                    if ag_dict:
                        autogluon_pred = ClsModelOutput(
                            model_name=ag_dict["model_name"],
                            predictions=ag_dict["predictions"],
                            top_class=ag_dict["top_class"],
                            top_confidence=ag_dict["top_confidence"],
                            requires_mask=ag_dict["requires_mask"],
                            metadata=ag_dict.get("metadata", {}),
                        )

                    seg_stub = None
                    if has_seg and seg_decision_dict:
                        seg_stub = SegAgentDecision(
                            selected_model=seg_decision_dict["selected_model"],
                            selected_mask=selected_mask,
                            confidence=seg_decision_dict["confidence"],
                            reasoning=seg_decision_dict["reasoning"],
                            all_predictions=seg_decision_dict["all_predictions"],
                            classification_anchor=anchor_class,
                            path=seg_decision_dict.get("path", "B"),
                        )

                    if cls_routing_mode == "llm" and enable_cls_agent and has_seg:
                        cls_decision = self.cls_agent.resolve_path_b(
                            indie_preds, autogluon_pred, seg_stub,
                            voting_weights=self._get_voting_weights(),
                            image_name=img_name,
                        )
                    elif cls_routing_mode == "oracle":
                        img_stem = Path(img_name).stem
                        tl = labels.get(img_stem)
                        if tl is None:
                            cls_decision = resolve_path_b_indie_only(
                                "max_confidence",
                                indie_preds,
                                best_single_model=best_single_cls,
                            )
                        else:
                            bin_tl = self._label_to_binary(tl)
                            if bin_tl is None:
                                bin_tl = int(tl) if isinstance(tl, (int, float)) else 0
                            cls_decision = resolve_cls_oracle(indie_preds, bin_tl)
                        self.cls_agent._emit_cls_routing_log(
                            cls_decision,
                            task="cls_path_b",
                            llm_invoked=False,
                            method=cls_decision.method,
                            image_name=img_name,
                            path="B",
                            classification_consensus=False,
                            predictions=indie_preds,
                            extra=self.cls_agent._path_b_extra(None, seg_stub) if seg_stub else None,
                        )
                    elif cls_routing_mode == "llm" and not has_seg:
                        cls_decision = self._resolve_classification_path_b_static(
                            indie_preds, autogluon_pred, self._get_voting_weights()
                        )
                    elif is_b4_only_cls_mode(cls_routing_mode):
                        cls_decision = self._resolve_classification_path_b_static(
                            indie_preds, autogluon_pred, self._get_voting_weights()
                        )
                        cls_decision.method = "path_b_autogluon_static"
                        self.cls_agent._emit_cls_routing_log(
                            cls_decision,
                            task="cls_path_b",
                            llm_invoked=False,
                            method=cls_decision.method,
                            image_name=img_name,
                            path="B",
                            classification_consensus=False,
                            predictions=indie_preds,
                            extra=self.cls_agent._path_b_extra(None, seg_stub) if seg_stub else None,
                        )
                    else:
                        learned_cls = (
                            self._learned_router("cls")
                            if normalize_routing_mode(cls_routing_mode) == "learned"
                            else None
                        )
                        cls_decision = resolve_path_b_indie_only(
                            cls_routing_mode,
                            indie_preds,
                            best_single_model=best_single_cls,
                            learned_router=learned_cls,
                        )
                        self.cls_agent._emit_cls_routing_log(
                            cls_decision,
                            task="cls_path_b",
                            llm_invoked=False,
                            method=cls_decision.method,
                            image_name=img_name,
                            path="B",
                            classification_consensus=False,
                            predictions=indie_preds,
                            extra=self.cls_agent._path_b_extra(None, seg_stub) if seg_stub else None,
                        )

                # 统一标签为 0/1（良性=0, 恶性=1）
                raw_label = cls_decision.selected_class
                final_label = self._label_to_binary(raw_label)
                if final_label is None:
                    final_label = raw_label  # 无法转换时保留原值

                result = {
                    "path": "A" if consensus else "B",
                    "classification_consensus": consensus,
                    "cls_decision": cls_decision.to_dict(),
                    "final_label": final_label,
                    "final_confidence": cls_decision.confidence,
                    "image_name": img_name,
                }
                if has_seg:
                    result["seg_decision"] = seg_decision_dict
                    result["selected_mask_shape"] = list(selected_mask.shape)
                    result["selected_mask_area"] = int(np.sum(selected_mask))

                # 标签匹配：统一用不带后缀的文件名（stem）
                img_stem = Path(img_name).stem
                if img_stem in labels:
                    result["true_label"] = labels[img_stem]

                self._enrich_result_record(
                    result,
                    img_name=img_name,
                    indie_preds=indie_preds,
                    seg_decision_dict=seg_decision_dict,
                    consensus=consensus,
                    final_confidence=float(result["final_confidence"]),
                    llm_used_seg=enable_seg_agent,
                    llm_used_cls=enable_cls_agent,
                    center_lookup=center_lookup,
                )
                results.append(result)
                print(f"  [{idx+1}/{len(img_names)}] {img_name}: {result['final_label']} (conf={result['final_confidence']:.3f})")

                # 增量保存
                if (idx + 1) % 10 == 0 or idx == len(img_names) - 1:
                    self._save_results_json(results, out_path / "results.json")

        self._save_results_json(results, out_path / "results.json")
        print(f"\n级联推理完成: {len(results)} 张图像")
        return results
