"""
LLM 驱动的分割 Agent：从多个模型输出中选择最佳 mask。

重写自 Segmentation_Agent/agent/segmentation_agent.py。
关键改动：
  - 用 shared/llm_client.LLMClient 代替内联 OpenAI 调用
  - 预留 radiomics_judge 接口（Phase 7 注入），select_best_mask 增加 image 参数
  - AgentDecision 增加 judge_scores 字段
  - 用 shared/base_datasets_info.BASE_DATASETS_INFO 作为单一来源
  - 保留全部分歧度量、ensemble、降级选择、JSON 解析逻辑
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from shared.base_datasets_info import BASE_DATASETS_INFO
from shared.llm_client import BRIEF_ROUTING_REASONING, LLMClient
from shared.routing_log import (
    RoutingLogger,
    llm_config_dict,
    seg_disagreement_tier,
    summarize_judge_results,
    summarize_seg_candidates,
)

from .base_model import SegModelOutput
from .quality_evaluator import SegmentationQualityEvaluator
from .metrics import compute_dice, compute_hd95, compute_ece

# B6 / Supp.: 运行时追加到分割 system prompt（Path A 有锚点，Path B 无共识）
SEG_LLM_SYSTEM_APPEND_PATH_A_TEMPLATE = """
【分类锚点（关键信号）】
独立分类模型（3个，无需mask）一致判断该图像为「{classification_anchor}」。
此锚点用于评估各分割结果的合理性：
- 与锚点矛盾的Radiomics裁判分类 → 该mask可能覆盖了错误的ROI → 可信度显著降低。
- 与锚点吻合的mask → 额外加分。
- 若所有mask的Radiomics分类都与锚点矛盾 → 选confidence最高的mask并解释原因。"""

SEG_LLM_SYSTEM_APPEND_PATH_B = """
【无分类锚点】
独立分类模型对该图像的判断不一致，未能形成共识。
整图级别特征模糊，分割评估应以以下为主要依据：
1. Radiomics裁判的内部一致性（不同的分割得到相似的分类方向→更可信）
2. 形态学合理性（circularity 0.6-0.9、单连通）
3. 设备匹配与历史性能"""


def seg_llm_system_append(classification_anchor: Optional[str]) -> str:
    if classification_anchor:
        return SEG_LLM_SYSTEM_APPEND_PATH_A_TEMPLATE.format(
            classification_anchor=classification_anchor,
        )
    return SEG_LLM_SYSTEM_APPEND_PATH_B


# 导出规格时占位，使 base system 含 radiomics judge 说明（与 pipeline 一致）
RADIOMICS_JUDGE_SPEC_PLACEHOLDER = object()


@dataclass
class SegAgentDecision:
    """Agent 对最佳分割的选择结果。"""

    selected_model: str
    selected_mask: np.ndarray
    confidence: float
    reasoning: str
    all_predictions: list[dict[str, Any]]
    quality_metrics: Optional[dict[str, Any]] = None
    agreement_score: Optional[float] = None
    dice_score: Optional[float] = None
    hd95_score: Optional[float] = None
    ece_metrics: Optional[dict[str, float]] = None
    selected_models: Optional[list[str]] = None
    model_weights: Optional[list[float]] = None
    is_ensemble: bool = False
    judge_scores: Optional[dict[str, Any]] = None  # radiomics 裁判结果
    classification_anchor: Optional[str] = None  # 分类锚点 (Path A) 或 None (Path B)
    path: str = "unknown"  # "A" | "B"

    def to_dict(self, include_mask: bool = False) -> dict:
        result = {
            "selected_model": self.selected_model,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "all_predictions": self.all_predictions,
            "quality_metrics": self.quality_metrics,
            "agreement_score": self.agreement_score,
            "is_ensemble": self.is_ensemble,
        }
        if self.is_ensemble and self.selected_models:
            result["selected_models"] = self.selected_models
            if self.model_weights:
                result["model_weights"] = self.model_weights
        if include_mask:
            result["selected_mask"] = self.selected_mask.tolist()
        else:
            result["mask_shape"] = self.selected_mask.shape
            result["mask_area"] = int(np.sum(self.selected_mask))
        if self.dice_score is not None:
            result["dice_score"] = self.dice_score
        if self.hd95_score is not None:
            result["hd95_score"] = self.hd95_score
        if self.ece_metrics:
            result["ece_metrics"] = self.ece_metrics
            result["ece_mean"] = float(np.mean(list(self.ece_metrics.values())))
        if self.judge_scores:
            result["judge_scores"] = self.judge_scores
        if self.classification_anchor:
            result["classification_anchor"] = self.classification_anchor
        result["path"] = self.path
        return result

    def to_simplified_dict(self) -> dict:
        simplified_preds = []
        for pred in self.all_predictions:
            p = {"model_name": pred["model_name"], "mask_area": pred.get("mask_area", 0)}
            if pred.get("has_confidence_map"):
                p["mean_confidence"] = pred.get("mean_confidence", 0.0)
            simplified_preds.append(p)
        result = {
            "selected_model": self.selected_model,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "all_predictions": simplified_preds,
            "is_ensemble": self.is_ensemble,
        }
        if self.is_ensemble and self.selected_models:
            result["selected_models"] = self.selected_models
            if self.model_weights:
                result["model_weights"] = [float(w) for w in self.model_weights]
        if self.ece_metrics:
            result["ece_metrics"] = self.ece_metrics
            result["ece_mean"] = float(np.mean(list(self.ece_metrics.values())))
        if self.classification_anchor:
            result["classification_anchor"] = self.classification_anchor
        result["path"] = self.path
        return result


class SegmentationAgent:
    """LLM 驱动的分割选择 Agent。"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        quality_evaluator: Optional[SegmentationQualityEvaluator] = None,
        radiomics_judge=None,
        config: Optional[dict] = None,
        routing_logger: Optional[RoutingLogger] = None,
    ):
        """
        Args:
            llm_client: 共享的 LLM 客户端（None 时仅在 enable_agent=False 模式可用）。
            quality_evaluator: 质量评估器（None 时内部创建）。
            radiomics_judge: GT-trained radiomics 裁判（None 时不启用，Phase 7 注入）。
            config: agent 配置（ensemble, disagreement 等）。
        """
        self.llm_client = llm_client
        self.quality_evaluator = quality_evaluator or SegmentationQualityEvaluator()
        self.radiomics_judge = radiomics_judge
        self.base_datasets_info = BASE_DATASETS_INFO
        cfg = config or {}
        self.enable_agent = cfg.get("enable_agent", True)
        self.routing_mode = cfg.get("routing_mode")
        self.ensemble_enabled = cfg.get("ensemble", {}).get("enabled", True)
        self.ensemble_top_k = cfg.get("ensemble", {}).get("top_k", 1)
        self.ensemble_method = cfg.get("ensemble", {}).get("method", "weighted_average")
        self.ensemble_threshold = cfg.get("ensemble", {}).get("threshold", 0.5)
        self.include_disagreement = cfg.get("include_disagreement_metrics_in_prompt", True)

        # 阶段 1 预筛选参数（从 config 读取，可在 config.yaml 中调整）
        pf_cfg = cfg.get("pre_filter", {})
        self.pf_cosine = pf_cfg.get("cosine_threshold", 0.8)
        self.pf_prob_outlier = pf_cfg.get("prob_outlier_threshold", 0.4)
        self.pf_min_keep = pf_cfg.get("min_keep", 3)
        self.metadata_mode = cfg.get("metadata_mode", "full")
        self.brief_routing_prompt = bool(cfg.get("brief_routing_prompt", False))
        self.routing_logger = routing_logger
        self._seg_log_ctx: dict[str, Any] = {}

        self.system_prompt = self._generate_system_prompt()

    def _generate_system_prompt(self) -> str:
        if self.brief_routing_prompt:
            return self._generate_system_prompt_brief()
        if self.ensemble_enabled and self.ensemble_top_k > 1:
            m3 = ', "模型3名称"' if self.ensemble_top_k >= 3 else ''
            w3 = ', 0.0' if self.ensemble_top_k >= 3 else ''
            fmt = f'{{"selected_models": ["模型1", "模型2"{m3}], "weights": [0.6, 0.4{w3}], "confidence": 0.95, "reasoning": "选择理由"}}'
            desc = (
                f'返回JSON: "selected_models"(长度{self.ensemble_top_k}数组), '
                f'"weights"(和为1.0), "confidence"(0-1), '
                f'"reasoning"(3-4句中文含数值, 约120-180字)。'
            )
        else:
            fmt = '{"selected_model": "模型名称", "confidence": 0.95, "reasoning": "选择理由"}'
            desc = (
                '返回JSON: "selected_model", "confidence"(0-1), '
                '"reasoning"(3-4句中文含数值, 约120-180字)。'
            )

        disagreement_block = ""
        if self.include_disagreement:
            disagreement_block = """
当输入含 group_uncertainty 与 disagreement 时：
- group_uncertainty.area_cv：面积离散度，越大共识越差；
- pairwise_hd95_mean/std：边界差异整体水平与波动；
- disagreement.mean_hd95_to_others：该模型与其余模型边界平均 HD95；
- disagreement.area_rel_to_group：面积相对群体均值的偏差。
分歧大时优先选 agreement 高且 mean_hd95_to_others 低的模型。"""

        judge_block = ""
        if self.radiomics_judge is not None:
            judge_block = """
你还将看到每个分割结果经 GT-trained radiomics 模型的分类判断（radiomics_judge 字段）。
该裁判用金标准 mask 训练，分类置信度可作为分割质量的间接信号：
分割越准确，radiomics 特征越接近训练分布，分类置信度越合理。
若多个模型的 radiomics_judge 分类结果分歧大，说明分割质量差异大，需重点分析。"""

        return f"""你是一个分割模型选择代理，根据多个模型的掩码质量指标选择最佳结果。
只输出一个JSON对象，首字符{{末字符}}，无前缀后缀或Markdown。

输出格式：{desc}
示例：{fmt}
{disagreement_block}
{judge_block}
决策优先级（从高到低）：
1) 模型间一致性：agreement_with_others 高更可靠；分歧大时倚重 agreement 高且 mean_hd95_to_others 低的候选；
2) radiomics 裁判（如有）：分类置信度合理的 mask 更可信；
3) 设备匹配：training_devices 与输入设备越接近越好；
4) 形态学：单连通、边界平滑、circularity 0.6-0.9、面积合理；
5) 数据集规模：dataset_size 大的模型泛化性更好；
6) 置信度：mean_confidence 较高为次要加分项。

reasoning 须引用关键数值（agreement、dice、hd95、area_cv、pairwise_hd95_mean、radiomics_judge 等）。"""

    def _generate_system_prompt_brief(self) -> str:
        if self.ensemble_enabled and self.ensemble_top_k > 1:
            fmt = (
                '{"selected_models":["m1","m2"], "weights":[0.6,0.4], '
                '"confidence":0.9}'
            )
            fields = (
                f'"selected_models"(长度{self.ensemble_top_k}), "weights"(和1), '
                '"confidence"(0-1)；勿输出 reasoning 字段'
            )
        else:
            fmt = '{"selected_model":"模型名", "confidence":0.9}'
            fields = '"selected_model", "confidence"(0-1)；勿输出 reasoning 字段'
        return f"""你是分割模型选择代理。根据 user 中 JSON 指标选最佳 mask。
禁止输出思考过程、Markdown 或任何 JSON 外的文字。第一个字符必须是 {{，最后一个必须是 }}。
只输出一行 JSON：{fields}。示例：{fmt}
优先级：agreement_with_others↑；分歧大时 mean_hd95_to_others↓；radiomics_judge 与 classification_anchor 一致者优先；形态学与设备匹配次之。"""

    def _seg_llm_system_append_runtime(self, classification_anchor: Optional[str]) -> str:
        if self.brief_routing_prompt:
            if classification_anchor:
                return f"\n分类锚点：{classification_anchor}。与锚点矛盾的 radiomics_judge 降权。"
            return "\n无分类锚点。以 radiomics 内部一致性与 agreement 为主。"
        return seg_llm_system_append(classification_anchor)

    # ========== 分歧度量 ==========

    @staticmethod
    def _compact_disagreement_fields(
        agreement_metrics: dict, n_models: int
    ) -> tuple[Optional[dict], Optional[list[dict]]]:
        if n_models < 2:
            return None, None
        vols = np.asarray(agreement_metrics.get("volumes") or [], dtype=np.float64)
        hdm = np.asarray(agreement_metrics.get("pairwise_hd95_matrix") or [], dtype=np.float64)
        if vols.size != n_models or hdm.shape != (n_models, n_models):
            return None, None
        vmean = float(vols.mean())
        per_model = []
        for i in range(n_models):
            others = hdm[i, [j for j in range(n_models) if j != i]]
            mhd = float(np.mean(others)) if others.size else 0.0
            area_rel = float((vols[i] - vmean) / vmean) if vmean > 1e-6 else 0.0
            per_model.append({"mean_hd95_to_others": round(mhd, 2), "area_rel_to_group": round(area_rel, 4)})
        group = {
            "area_cv": round(float(agreement_metrics.get("volume_cv", 0.0)), 4),
            "pairwise_hd95_mean": round(float(agreement_metrics.get("pairwise_hd95_mean", 0.0)), 2),
            "pairwise_hd95_std": round(float(agreement_metrics.get("pairwise_hd95_std", 0.0)), 2),
        }
        return group, per_model

    @staticmethod
    def _build_disagreement_prefix(g_val: dict, per_val, predictions, max_models=8) -> str:
        lines = [
            f"【分歧摘要】area_cv={g_val['area_cv']}, "
            f"pairwise_hd95_mean={g_val['pairwise_hd95_mean']}, "
            f"pairwise_hd95_std={g_val['pairwise_hd95_std']}。"
        ]
        if per_val is not None and len(per_val) == len(predictions) and len(predictions) <= max_models:
            segs = [
                f"{p.model_name}:{dc['mean_hd95_to_others']:.2f},{dc['area_rel_to_group']:.4f}"
                for p, dc in zip(predictions, per_val)
            ]
            lines.append("【各模型】mhd95,area_rel丨" + "丨".join(segs) + "。")
        return "\n".join(lines) + "\n\n"

    # ========== Radiomics 裁判 ==========

    def _run_judge(
        self,
        image: np.ndarray,
        predictions: list[SegModelOutput],
        image_name: Optional[str] = None,
    ) -> Optional[list[dict]]:
        """对每个 pred mask 跑 radiomics 裁判。Phase 7 注入 judge 后生效。"""
        if self.radiomics_judge is None:
            return None
        masks = [p.mask for p in predictions]
        try:
            return self.radiomics_judge.judge_batch(
                image, masks, image_name=image_name
            )
        except Exception as e:
            print(f"  ⚠️ radiomics 裁判失败: {e}")
            return None

    def _pre_filter_by_judge(
        self,
        predictions: list[SegModelOutput],
        judge_results: Optional[list[dict]],
    ) -> tuple[list[SegModelOutput], Optional[list[dict]], list[int]]:
        """
        阶段 1: 基于 radiomics 裁判输出预筛选（Python 规则，不用 LLM）。

        两条规则（阈值在 config segmentation.agent.pre_filter 中配置）：
        1. 特征向量离群: 与所有其他模型 cosine similarity < 阈值 → 异常
        2. 分类置信度离群: malignant_prob 与中位数偏差 > 阈值 → 异常

        两条规则都是模型间相对比较，不依赖 GT 训练集分布，
        直接反映"哪个模型和其他模型不一样"。

        安全阀: 保留至少 max(min_keep, N//2) 个模型；不足则取消全部过滤。

        Returns:
            (filtered_predictions, filtered_judge_results, removed_indices)
        """
        n = len(predictions)

        # 安全阀: 模型太少 / 无裁判结果 → 跳过
        if n <= 3 or not judge_results:
            return predictions, judge_results, []

        valid_mask = [jr.get("valid", False) if jr else False for jr in judge_results]
        if not any(valid_mask):
            return predictions, judge_results, []

        removed: set[int] = set()

        # 规则 1: 特征向量 cosine similarity 离群检测
        candidates = [
            (i, judge_results[i].get("feature_vector"))
            for i in range(n)
            if valid_mask[i] and judge_results[i].get("feature_vector") is not None
        ]
        if len(candidates) >= 4:
            fv_array = np.array([fv for _, fv in candidates], dtype=np.float64)
            norms = np.linalg.norm(fv_array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = fv_array / norms
            sim_matrix = normalized @ normalized.T

            for idx_in_list, (orig_idx, _) in enumerate(candidates):
                others = np.delete(sim_matrix[idx_in_list], idx_in_list)
                max_sim = float(np.max(others)) if len(others) > 0 else 0.0
                if max_sim < self.pf_cosine:
                    removed.add(orig_idx)

        # 规则 2: malignant_prob 离群检测
        prob_candidates = [
            (i, judge_results[i].get("malignant_prob", 0.5))
            for i in range(n)
            if i not in removed and valid_mask[i]
        ]
        if len(prob_candidates) >= 5:
            probs = [p for _, p in prob_candidates]
            median_prob = float(np.median(probs))
            for i, p in prob_candidates:
                if abs(p - median_prob) > self.pf_prob_outlier:
                    removed.add(i)

        # 安全阀: 保留数量下限，不足则取消全部过滤
        min_keep = max(self.pf_min_keep, n // 2)
        if n - len(removed) < min_keep:
            return predictions, judge_results, []

        if not removed:
            return predictions, judge_results, []

        kept = [i for i in range(n) if i not in removed]
        filtered_preds = [predictions[i] for i in kept]
        filtered_judge = [judge_results[i] for i in kept] if judge_results else None
        return filtered_preds, filtered_judge, sorted(removed)

    # ========== Prompt 构造 ==========

    def format_predictions_for_agent(
        self,
        predictions: list[SegModelOutput],
        quality_results: dict,
        judge_results: Optional[list[dict]] = None,
        classification_anchor: Optional[str] = None,
        input_device_info: Optional[list[str]] = None,
        input_data_info: Optional[dict] = None,
    ) -> str:
        data: dict[str, Any] = {
            "num_models": len(predictions),
            "input_device_info": input_device_info if input_device_info else "未知 (null)",
            "input_data_info": self._normalize_unknown(input_data_info or {}),
            "models": [],
        }

        am = quality_results.get("agreement_metrics") or {}
        n = len(predictions)
        group_unc, per_disagree = None, None
        if self.include_disagreement:
            group_unc, per_disagree = self._compact_disagreement_fields(am, n)
            if group_unc:
                data["group_uncertainty"] = group_unc

        for idx, pred in enumerate(predictions):
            q = quality_results["individual_quality"][idx]
            meta = pred.metadata or {}

            model_info: dict[str, Any] = {
                "model_name": pred.model_name,
                "training_devices": meta.get("training_data_devices", []),
                "mask_statistics": {
                    "area": q["area"],
                    "num_components": q["num_components"],
                    "circularity": round(q["circularity"], 2),
                    "smoothness": round(q["smoothness"], 2),
                    "solidity": round(q["solidity"], 2),
                },
                "agreement_with_others": round(am.get("average_agreement", [0])[idx], 2) if am.get("average_agreement") else 0.0,
            }
            if per_disagree and idx < len(per_disagree):
                model_info["disagreement"] = per_disagree[idx]

            # 数据集信息（只保留数据量，减少 token）
            di = meta.get("dataset_info", {})
            if di:
                model_info["dataset_size"] = di.get("dataset_size", 0)

            # 置信度
            if pred.confidence_map is not None and np.sum(pred.mask) > 0:
                model_info["mean_confidence"] = round(float(np.mean(pred.confidence_map[pred.mask > 0])), 2)

            # radiomics 裁判（新增）
            if judge_results and idx < len(judge_results) and judge_results[idx].get("valid", False):
                jr = judge_results[idx]
                rj: dict[str, Any] = {
                    "malignant_prob": round(jr["malignant_prob"], 3),
                    "confidence": round(jr["confidence"], 3),
                }
                if not self.brief_routing_prompt:
                    rj["top_features"] = jr.get("top_features", [])[:3]
                model_info["radiomics_judge"] = rj

            data["models"].append(model_info)

        # 分类锚点
        if self.brief_routing_prompt:
            data["classification_anchor"] = (
                classification_anchor if classification_anchor else None
            )
        elif classification_anchor:
            data["classification_anchor"] = {
                "class": classification_anchor,
                "source": "3个独立分类模型(无mask)一致判断",
                "usage": "排除Radiomics裁判分类与锚点矛盾的mask"
            }
        else:
            data["classification_anchor"] = {
                "class": None,
                "status": "无共识(独立分类模型判断不一致)",
                "implication": "以Radiomics裁判内部一致性和形态学为主要依据"
            }

        if not self.brief_routing_prompt:
            req: dict[str, Any] = {}
            if group_unc:
                req["follow_prefix"] = "reasoning 须含 area_cv 及所选模型 disagreement。"
            if self.radiomics_judge is not None and judge_results:
                if classification_anchor:
                    req["mention_judge"] = (
                        "reasoning 须提及 radiomics_judge 分类置信度，"
                        "并逐模型分析是否与分类锚点吻合。"
                    )
                else:
                    req["mention_judge"] = (
                        "reasoning 须提及 radiomics_judge 分类置信度，分析裁判的内部分歧。"
                    )
            if classification_anchor:
                req["use_anchor"] = "与锚点矛盾的Radiomics分类需在reasoning中明确指出并解释原因。"
            else:
                req["no_anchor_strategy"] = "无分类锚点时，优先级：1)裁判一致性 2)形态学 3)历史性能。"
            if req:
                data["reasoning_requirements"] = req

        from shared.metadata_prompt import filter_seg_routing_payload

        data = filter_seg_routing_payload(data, self.metadata_mode)
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _normalize_unknown(value: Any) -> Any:
        if value is None:
            return "未知 (null)"
        if isinstance(value, dict):
            return {k: SegmentationAgent._normalize_unknown(v) for k, v in value.items()}
        if isinstance(value, list):
            return ["未知 (空列表)"] if not value else [SegmentationAgent._normalize_unknown(v) for v in value]
        if isinstance(value, str) and not value.strip():
            return "未知 (空字符串)"
        return value

    # ========== JSON 解析 ==========

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        # markdown code block
        if "```json" in text:
            extracted = text.split("```json")[1].split("```")[0].strip()
            if extracted.startswith("{") and ("selected_model" in extracted or "selected_models" in extracted):
                return extracted
        elif "```" in text:
            for part in text.split("```"):
                p = part.strip()
                if p.startswith("{") and ("selected_model" in p or "selected_models" in p):
                    complete = SegmentationAgent._extract_complete_json(p)
                    if complete:
                        return complete

        text = re.sub(r"\*Thinking[^*]*\*", "", text, flags=re.DOTALL)
        text = re.sub(r"^#+\s+.*$", "", text, flags=re.MULTILINE)

        start = text.find("{")
        if start != -1:
            complete = SegmentationAgent._extract_complete_json_from_position(text, start)
            if complete and ("selected_model" in complete or "selected_models" in complete):
                return complete

        raise ValueError(f"无法提取 JSON。响应前500字: {text[:500]}")

    @staticmethod
    def _extract_complete_json(text: str) -> Optional[str]:
        start = text.find("{")
        return SegmentationAgent._extract_complete_json_from_position(text, start) if start != -1 else None

    @staticmethod
    def _extract_complete_json_from_position(text: str, start: int) -> Optional[str]:
        brace = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"' and not escape:
                in_str = not in_str
                continue
            if not in_str:
                if c == "{":
                    brace += 1
                elif c == "}":
                    brace -= 1
                    if brace == 0:
                        s = text[start : i + 1]
                        s = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", s)
                        if "selected_model" in s or "selected_models" in s:
                            return s.strip()
                        return None
        return None

    # ========== Ensemble ==========

    @staticmethod
    def _ensemble_probability_maps(
        prob_maps: list[np.ndarray], weights: list[float], method: str = "weighted_average"
    ) -> np.ndarray:
        total = sum(weights)
        weights = [w / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)

        target = prob_maps[0].shape
        normalized = [
            cv2.resize(pm.astype(np.float32), (target[1], target[0]), interpolation=cv2.INTER_LINEAR)
            if pm.shape != target
            else pm.astype(np.float32)
            for pm in prob_maps
        ]

        if method in ("weighted_average", "equal_weight"):
            result = np.zeros_like(normalized[0])
            for pm, w in zip(normalized, weights):
                result += pm * w
        elif method == "geometric_mean":
            result = np.ones_like(normalized[0])
            for pm, w in zip(normalized, weights):
                result *= np.power(np.clip(pm, 1e-6, 1.0), w)
        else:
            raise ValueError(f"未知 ensemble 方法: {method}")

        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _generate_mask_from_probability(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        mask = (prob_map > threshold).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest).astype(np.uint8)
        return mask

    def _emit_seg_routing_log(
        self,
        decision: SegAgentDecision,
        *,
        llm_invoked: bool,
        method: str,
        parse_ok: bool = True,
        error: Optional[str] = None,
        prompts: Optional[dict] = None,
    ) -> None:
        logger = self.routing_logger
        if logger is None or not logger.enabled:
            return
        ctx = self._seg_log_ctx or {}
        quality_results = ctx.get("quality_results")
        predictions = ctx.get("predictions") or []
        record: dict[str, Any] = {
            "image_name": ctx.get("image_name"),
            "task": "seg",
            "path": decision.path,
            "classification_anchor": decision.classification_anchor,
            "classification_consensus": bool(decision.classification_anchor),
            "metadata_mode": ctx.get("metadata_mode", self.metadata_mode),
            "disagreement_tier": seg_disagreement_tier(quality_results),
            "llm_invoked": llm_invoked,
            "method": method,
            "parse_ok": parse_ok,
            "error": error,
            "candidates_summary": summarize_seg_candidates(predictions, quality_results),
            "judge_summary": summarize_judge_results(ctx.get("judge_results")),
            "selected_model": decision.selected_model,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "is_ensemble": decision.is_ensemble,
        }
        record.update(llm_config_dict(self.llm_client))
        if logger.include_prompts and prompts:
            record["prompts"] = prompts
        logger.log(record)

    def _finish_seg_decision(
        self,
        decision: SegAgentDecision,
        *,
        llm_invoked: bool,
        method: str,
        parse_ok: bool = True,
        error: Optional[str] = None,
        prompts: Optional[dict] = None,
    ) -> SegAgentDecision:
        self._emit_seg_routing_log(
            decision,
            llm_invoked=llm_invoked,
            method=method,
            parse_ok=parse_ok,
            error=error,
            prompts=prompts,
        )
        return decision

    # ========== 核心选择方法 ==========

    def select_best_mask(
        self,
        image: np.ndarray,
        predictions: list[SegModelOutput],
        classification_anchor: Optional[str] = None,
        gt_mask: Optional[np.ndarray] = None,
        input_device_info: Optional[list[str]] = None,
        input_data_info: Optional[dict] = None,
        judge_results: Optional[list[dict]] = None,
        image_name: Optional[str] = None,
    ) -> SegAgentDecision:
        """
        选择最佳分割 mask。

        Args:
            image: (H, W, 3) RGB uint8 [0,255]。radiomics_judge 需要。
            predictions: 多个模型的预测结果。
            classification_anchor: 分类锚点 (Path A: "恶性"/"良性"; Path B: None)。
            gt_mask: 可选 GT mask（仅用于评估，不影响选择）。
            input_device_info: 输入设备信息。
            input_data_info: 输入数据元信息。
            judge_results: 外部预先跑好的 radiomics 裁判结果（与 predictions 等长）。
                传入则跳过内部 _run_judge，避免与 Phase 4 重复计算。
        """
        path = "A" if classification_anchor else "B"
        if not predictions:
            raise ValueError("没有预测结果")

        # radiomics 裁判（阶段 1 前置）
        # 若外部已传入 judge_results（Phase 3 统一跑过），则不重复计算
        if judge_results is None:
            judge_results = self._run_judge(image, predictions, image_name=image_name)

        # 阶段 1: 基于 radiomics 裁判预筛选（Python 规则，不用 LLM）
        predictions, judge_results, removed_indices = self._pre_filter_by_judge(
            predictions, judge_results
        )
        if removed_indices:
            print(f"  阶段1预筛选: 移除 {len(removed_indices)} 个模型 (索引 {removed_indices})，保留 {len(predictions)} 个")

        masks = [p.mask for p in predictions]
        model_names = [p.model_name for p in predictions]

        # 统一 mask 尺寸
        shapes = {m.shape[:2] for m in masks}
        if len(shapes) > 1:
            target = masks[0].shape[:2]
            masks = [
                cv2.resize(m.astype(np.uint8), (target[1], target[0]), interpolation=cv2.INTER_NEAREST)
                if m.shape[:2] != target
                else m
                for m in masks
            ]

        quality_results = self.quality_evaluator.evaluate_batch(masks, model_names)

        # 分歧度量
        g_val, per_val = None, None
        if self.include_disagreement:
            g_val, per_val = self._compact_disagreement_fields(
                quality_results.get("agreement_metrics") or {}, len(predictions)
            )
        prefix = ""

        # 构造 prompt
        formatted = self.format_predictions_for_agent(
            predictions, quality_results, judge_results,
            classification_anchor=classification_anchor,
            input_device_info=input_device_info, input_data_info=input_data_info
        )

        from shared.metadata_prompt import seg_user_prefix_lines

        norm_data = self._normalize_unknown(input_data_info or {})
        device_text, data_text = seg_user_prefix_lines(
            self.metadata_mode, input_device_info, norm_data if isinstance(norm_data, dict) else {}
        )

        if self.ensemble_enabled and self.ensemble_top_k > 1:
            question = f"请选择 Top {self.ensemble_top_k} 个最佳模型用于 ensemble。"
            fmt = '{"selected_models": ["模型1", "模型2"], "weights": [0.6, 0.4], "confidence": 0.95}'
        else:
            question = "哪个模型提供了最佳分割结果？"
            fmt = '{"selected_model": "模型名称", "confidence": 0.95}'

        if self.brief_routing_prompt:
            out_hint = f"只输出一行 JSON，勿包含 reasoning 字段。格式：{fmt}"
        else:
            out_hint = (
                "只输出纯JSON，单行，不要 markdown 代码块。"
                "reasoning 不超过 60 个中文字符，勿换行。格式："
                f"{fmt}"
            )

        user_prompt = f"""{prefix}{device_text}{data_text}
以下是 {len(predictions)} 个分割模型的输出及质量评估：

{formatted}

{question}
{out_hint}"""

        # 调用 LLM（锚点指令动态追加到 system_prompt）
        system_prompt = self.system_prompt + self._seg_llm_system_append_runtime(
            classification_anchor
        )

        prompts_bundle = {"system": system_prompt, "user": user_prompt}
        self._seg_log_ctx = {
            "image_name": image_name,
            "path": path,
            "classification_anchor": classification_anchor,
            "predictions": predictions,
            "quality_results": quality_results,
            "judge_results": judge_results,
            "metadata_mode": self.metadata_mode,
        }

        if not self.enable_agent:
            print(f"  Agent 已禁用，使用静态规则选择")
            return self._finish_seg_decision(
                self._fallback_selection(
                    predictions, gt_mask, quality_results, judge_results, classification_anchor
                ),
                llm_invoked=False,
                method="static_fallback",
            )

        from shared.llm_json import loads_routing_json

        parse_error: Optional[str] = None
        decision_data = None
        response_text = ""
        try:
            response_text = self.llm_client.chat(system_prompt, user_prompt)
            decision_data = loads_routing_json(self._extract_json_from_text, response_text)
        except Exception as e:
            parse_error = str(e)
            if self.brief_routing_prompt:
                retry_tail = "\n\n【重试】仅输出一行合法 JSON，勿含 reasoning 字段，以}结尾。"
            else:
                retry_tail = (
                    "\n\n【重试】上次 JSON 不完整。仅输出一行合法 JSON，reasoning≤40字，"
                    "字段齐全，以}}结尾。"
                )
            retry_user = user_prompt + retry_tail
            try:
                response_text = self.llm_client.chat(system_prompt, retry_user)
                decision_data = loads_routing_json(self._extract_json_from_text, response_text)
                parse_error = None
            except Exception as e2:
                parse_error = f"{parse_error}; retry: {e2}"

        if decision_data is None:
            print(f"  ✗ LLM 调用或解析失败: {parse_error}，使用降级选择")
            return self._finish_seg_decision(
                self._fallback_selection(
                    predictions, gt_mask, quality_results, judge_results, classification_anchor
                ),
                llm_invoked=True,
                method="llm_fallback",
                parse_ok=False,
                error=parse_error or "unknown",
                prompts=prompts_bundle,
            )

        if decision_data.pop("_salvaged", False):
            print("  ⚠️ JSON 已截断，使用 salvage 恢复 selected_model（建议增大 max_tokens）", flush=True)

        # 解析决策
        try:
            confidence = max(0.0, min(1.0, float(decision_data["confidence"])))
            reasoning = str(decision_data.get("reasoning", "")).strip()
            if self.brief_routing_prompt:
                reasoning = BRIEF_ROUTING_REASONING
            elif not reasoning:
                raise ValueError("reasoning 为空")

            is_ensemble = "selected_models" in decision_data and self.ensemble_enabled

            if is_ensemble:
                return self._finish_seg_decision(
                    self._handle_ensemble(
                        decision_data, predictions, quality_results, gt_mask, judge_results,
                        confidence, reasoning, classification_anchor
                    ),
                    llm_invoked=True,
                    method="llm_ensemble",
                    prompts=prompts_bundle,
                )
            return self._finish_seg_decision(
                self._handle_single(
                    decision_data, predictions, quality_results, gt_mask, judge_results,
                    confidence, reasoning, classification_anchor
                ),
                llm_invoked=True,
                method="llm_single",
                prompts=prompts_bundle,
            )
        except (KeyError, ValueError) as e:
            print(f"  ✗ 决策解析失败: {e}，使用降级选择")
            return self._finish_seg_decision(
                self._fallback_selection(
                    predictions, gt_mask, quality_results, judge_results, classification_anchor
                ),
                llm_invoked=True,
                method="llm_fallback",
                parse_ok=False,
                error=str(e),
                prompts=prompts_bundle,
            )

    def _handle_single(
        self, decision_data, predictions, quality_results, gt_mask, judge_results,
        confidence, reasoning, classification_anchor
    ) -> SegAgentDecision:
        name = str(decision_data.get("selected_model", "")).strip()
        idx = self._find_pred_index(predictions, name)
        if idx is None:
            raise ValueError(f"模型 '{name}' 不在预测列表中")

        pred = predictions[idx]
        selected_q = quality_results["individual_quality"][idx]
        agreement = quality_results["agreement_metrics"].get("average_agreement", [None])[idx]

        dice, hd95 = self._compute_metrics(pred.mask, gt_mask) if gt_mask is not None else (None, None)
        ece = self._compute_ece(pred, gt_mask) if gt_mask is not None else None

        jr = {predictions[i].model_name: judge_results[i] for i in range(len(predictions))} if judge_results else None

        return SegAgentDecision(
            selected_model=pred.model_name,
            selected_mask=pred.mask,
            confidence=confidence,
            reasoning=reasoning,
            all_predictions=[p.to_dict() for p in predictions],
            quality_metrics=selected_q,
            agreement_score=agreement,
            dice_score=dice,
            hd95_score=hd95,
            ece_metrics=ece,
            is_ensemble=False,
            judge_scores=jr,
            classification_anchor=classification_anchor,
            path="A" if classification_anchor else "B",
        )

    def _handle_ensemble(
        self, decision_data, predictions, quality_results, gt_mask, judge_results,
        confidence, reasoning, classification_anchor
    ) -> SegAgentDecision:
        names = decision_data["selected_models"][: self.ensemble_top_k]
        weights = [float(w) for w in decision_data.get("weights", [])[: len(names)]]
        if not weights:
            weights = [1.0 / len(names)] * len(names)
        total = sum(weights)
        weights = [w / total for w in weights] if total > 0 else [1.0 / len(names)] * len(names)

        selected_preds, indices = [], []
        for name in names:
            idx = self._find_pred_index(predictions, name)
            if idx is None:
                raise ValueError(f"模型 '{name}' 不在预测列表中")
            selected_preds.append(predictions[idx])
            indices.append(idx)

        prob_maps = [
            p.confidence_map if p.confidence_map is not None else p.mask.astype(np.float32)
            for p in selected_preds
        ]
        ensemble_map = self._ensemble_probability_maps(prob_maps, weights, self.ensemble_method)
        final_mask = self._generate_mask_from_probability(ensemble_map, self.ensemble_threshold)

        selected_q = quality_results["individual_quality"][indices[0]]
        agreement = quality_results["agreement_metrics"].get("average_agreement", [None])[indices[0]]
        dice, hd95 = self._compute_metrics(final_mask, gt_mask) if gt_mask is not None else (None, None)
        ece = None
        if gt_mask is not None:
            gt_r = self._resize_gt(gt_mask, final_mask.shape)
            ece = {"ece": compute_ece(ensemble_map, gt_r)} if gt_r is not None else None

        jr = {predictions[i].model_name: judge_results[i] for i in range(len(predictions))} if judge_results else None

        return SegAgentDecision(
            selected_model=names[0],
            selected_mask=final_mask,
            confidence=confidence,
            reasoning=reasoning,
            all_predictions=[p.to_dict() for p in predictions],
            quality_metrics=selected_q,
            agreement_score=agreement,
            dice_score=dice,
            hd95_score=hd95,
            ece_metrics=ece,
            selected_models=names,
            model_weights=weights,
            is_ensemble=True,
            judge_scores=jr,
            classification_anchor=classification_anchor,
            path="A" if classification_anchor else "B",
        )

    def _fallback_selection(
        self, predictions, gt_mask, quality_results, judge_results=None, classification_anchor=None
    ) -> SegAgentDecision:
        """降级选择：选一致性最高的模型。"""
        agreement = quality_results["agreement_metrics"].get("average_agreement")
        if agreement and len(agreement) > 0:
            if self.ensemble_enabled and self.ensemble_top_k > 1:
                top_k = min(self.ensemble_top_k, len(predictions))
                indices = np.argsort(agreement)[-top_k:][::-1].tolist()
            else:
                indices = [int(np.argmax(agreement))]
        else:
            indices = list(range(min(self.ensemble_top_k if self.ensemble_enabled else 1, len(predictions))))

        best_idx = indices[0]
        best_pred = predictions[best_idx]
        selected_q = quality_results["individual_quality"][best_idx]
        agree = agreement[best_idx] if agreement else None

        dice, hd95 = self._compute_metrics(best_pred.mask, gt_mask) if gt_mask is not None else (None, None)
        ece = self._compute_ece(best_pred, gt_mask) if gt_mask is not None else None

        jr = {predictions[i].model_name: judge_results[i] for i in range(len(predictions))} if judge_results else None

        reasoning = f"降级选择：一致性最高（IoU={agree:.2f}）" if agree else "降级选择：第一个模型"

        if self.ensemble_enabled and self.ensemble_top_k > 1 and len(indices) > 1:
            selected_preds = [predictions[i] for i in indices]
            names = [p.model_name for p in selected_preds]
            weights = [1.0 / len(names)] * len(names)
            prob_maps = [
                p.confidence_map if p.confidence_map is not None else p.mask.astype(np.float32)
                for p in selected_preds
            ]
            ensemble_map = self._ensemble_probability_maps(prob_maps, weights, self.ensemble_method)
            final_mask = self._generate_mask_from_probability(ensemble_map, self.ensemble_threshold)
            return SegAgentDecision(
                selected_model=names[0],
                selected_mask=final_mask,
                confidence=agree or 0.5,
                reasoning=f"降级选择：ensemble {len(names)} 个模型（平均IoU={agree:.2f}）" if agree else "降级选择",
                all_predictions=[p.to_dict() for p in predictions],
                quality_metrics=selected_q,
                agreement_score=agree,
                dice_score=dice,
                hd95_score=hd95,
                ece_metrics=ece,
                selected_models=names,
                model_weights=weights,
                is_ensemble=True,
                judge_scores=jr,
                classification_anchor=classification_anchor,
                path="A" if classification_anchor else "B",
            )

        return SegAgentDecision(
            selected_model=best_pred.model_name,
            selected_mask=best_pred.mask,
            confidence=agree or 0.5,
            reasoning=reasoning,
            all_predictions=[p.to_dict() for p in predictions],
            quality_metrics=selected_q,
            agreement_score=agree,
            dice_score=dice,
            hd95_score=hd95,
            ece_metrics=ece,
            is_ensemble=False,
            judge_scores=jr,
            classification_anchor=classification_anchor,
            path="A" if classification_anchor else "B",
        )

    # ========== 辅助方法 ==========

    @staticmethod
    def _find_pred_index(predictions: list[SegModelOutput], name: str) -> Optional[int]:
        if not name:
            return None
        nl = name.strip().lower()
        for i, p in enumerate(predictions):
            if p.model_name == name or p.model_name.strip().lower() == nl:
                return i
        return None

    @staticmethod
    def _compute_metrics(pred_mask, gt_mask):
        if pred_mask is None or gt_mask is None:
            return None, None
        try:
            gt_r = SegmentationAgent._resize_gt(gt_mask, pred_mask.shape)
            if gt_r is None:
                return None, None
            return compute_dice(pred_mask, gt_r), compute_hd95(pred_mask, gt_r)
        except Exception:
            return None, None

    @staticmethod
    def _resize_gt(gt_mask, target_shape):
        if gt_mask.shape == target_shape:
            return gt_mask
        try:
            return cv2.resize(
                gt_mask.astype(np.uint8),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        except Exception:
            return None

    @staticmethod
    def _compute_ece(pred: SegModelOutput, gt_mask):
        if pred is None or gt_mask is None or pred.confidence_map is None:
            return None
        try:
            gt_r = SegmentationAgent._resize_gt(gt_mask, pred.confidence_map.shape)
            if gt_r is None:
                return None
            return {"ece": compute_ece(pred.confidence_map, gt_r)}
        except Exception:
            return None
