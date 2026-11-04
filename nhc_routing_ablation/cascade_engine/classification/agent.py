"""
LLM 驱动的分类 Agent：从多个模型预测中选择最佳结果。

重写自 Classification_Agent/agent/classification_agent.py。
关键改动：
  - 用 shared/llm_client.LLMClient 代替内联 OpenAI 调用
  - 修复 __init__ 导出（原代码导出废弃的 GeminiAgent）
  - 保留 unanimous check / top_k soft voting / fallback 逻辑
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from shared.base_datasets_info import BASE_DATASETS_INFO, infer_device_match
from shared.llm_client import BRIEF_ROUTING_REASONING, LLMClient
from shared.routing_log import (
    RoutingLogger,
    cls_disagreement_tier,
    llm_config_dict,
    summarize_cls_candidates,
)

from .base_model import ClsModelOutput
from .soft_voting import (
    average_class_probabilities,
    winning_class_from_avg_probs,
    resolve_topk_model_outputs,
)

PATH_B_LLM_SYSTEM_PROMPT = """你是甲状腺超声诊断专家。独立分类模型对该图像判断不一致，你需要综合所有信号，自己做出最终的良恶性判断。

你将收到：
1. 独立分类模型预测（3个，无需mask，整图判断）
2. AutoGluon 基于分割筛选 mask 的 Radiomics 分类（有 ROI 引导）
3. 分割 Agent 选择该 mask 的推理

Markup:
- 独立模型置信度普遍低(<0.6) → 整图级特征不明显，ROI 级特征更可信
- AutoGluon 分类基于 ROI 特征 → 在 mask 靠谱的前提下更细腻
- 分割 Agent reasoning 解释了为什么选这个 mask

若所有信号都模糊，保守倾向恶性（宁可假阳性不可假阴性），但标注低置信度。

输出纯JSON（无 Markdown/代码块），首尾为{}：
{"selected_class": "恶性", "confidence": 0.75, "reasoning": "..."}
reasoning 须引用具体置信度数值。"""

PATH_B_LLM_SYSTEM_PROMPT_BRIEF = """甲状腺超声分类裁决：各独立分类模型未形成共识。
依据 user JSON 中的 independent_models、autogluon_based_on_selected_mask、segmentation_decision。
禁止思考过程与 Markdown。只输出一行 JSON：
{"selected_class":"恶性或良性", "confidence":0.0-1.0}
勿输出 reasoning 字段。信号均模糊时略倾向恶性但降低 confidence。"""

PATH_B_USER_PROMPT_EXAMPLE: dict[str, Any] = {
    "scenario": "独立分类模型无共识，需综合裁决",
    "independent_models": [
        {"model_name": "<name>", "top_class": "良性|恶性", "top_confidence": 0.0},
    ],
    "autogluon_based_on_selected_mask": {
        "top_class": "良性|恶性",
        "top_confidence": 0.0,
    },
    "segmentation_decision": {
        "selected_mask_from": "<seg_model_name>",
        "reasoning": "<segmentation agent reasoning>",
    },
    "question": "请综合以上信息，给出最终良恶性判断和置信度(0-1)。",
}


@dataclass
class ClsAgentDecision:
    """Agent 对最佳分类的选择结果。"""

    selected_model: str
    selected_class: str
    confidence: float
    reasoning: str
    all_predictions: list[dict[str, Any]]
    selected_models: Optional[list[str]] = None  # top_k soft voting 时
    method: str = "single"  # "single" | "soft_voting" | "unanimous" | "fallback"

    def to_dict(self) -> dict[str, Any]:
        result = {
            "selected_model": self.selected_model,
            "selected_class": self.selected_class,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "all_predictions": self.all_predictions,
            "method": self.method,
        }
        if self.selected_models:
            result["selected_models"] = self.selected_models
        return result


def _normalize_label(label) -> str:
    """将分类标签统一为中文「良性」/「恶性」。

    medsiglip 等模型输出 "0"/"1"，其他模型输出 "良性"/"恶性"，
    统一后再比较，避免字符串不一致导致的误判。
    """
    if isinstance(label, (int, float)):
        return "恶性" if int(label) == 1 else "良性"
    if isinstance(label, str):
        if label in ("恶性", "malignant", "M", "1", "1.0"):
            return "恶性"
        if label in ("良性", "benign", "B", "0", "0.0"):
            return "良性"
    return str(label)


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


class LLMClassificationAgent:
    """LLM 驱动的分类选择 Agent。"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[dict] = None,
        routing_logger: Optional[RoutingLogger] = None,
    ):
        self.llm_client = llm_client
        self.base_datasets_info = BASE_DATASETS_INFO
        self.routing_logger = routing_logger

        cfg = config or {}
        self.top_k = max(1, int(cfg.get("top_k", 1)))
        self.enable_agent = cfg.get("enable_agent", True)
        self.routing_mode = cfg.get("routing_mode")
        self.metadata_mode = cfg.get("metadata_mode", "full")
        self.brief_routing_prompt = bool(cfg.get("brief_routing_prompt", False))
        self._cls_log_ctx: dict[str, Any] = {}

        self.system_prompt_single = self._build_system_prompt_single()
        self.system_prompt_multi = self._build_system_prompt_multi()

    def _emit_cls_routing_log(
        self,
        decision: ClsAgentDecision,
        *,
        task: str,
        llm_invoked: bool,
        method: str,
        image_name: Optional[str] = None,
        path: Optional[str] = None,
        classification_consensus: Optional[bool] = None,
        predictions: Optional[list] = None,
        parse_ok: bool = True,
        error: Optional[str] = None,
        prompts: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> None:
        logger = self.routing_logger
        if logger is None or not logger.enabled:
            return
        preds = predictions or []
        record: dict[str, Any] = {
            "image_name": image_name or (self._cls_log_ctx or {}).get("image_name"),
            "task": task,
            "path": path,
            "classification_consensus": classification_consensus,
            "metadata_mode": self.metadata_mode,
            "disagreement_tier": cls_disagreement_tier(preds),
            "llm_invoked": llm_invoked,
            "method": method,
            "parse_ok": parse_ok,
            "error": error,
            "candidates_summary": summarize_cls_candidates(preds),
            "selected_model": decision.selected_model,
            "selected_class": decision.selected_class,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        }
        record.update(llm_config_dict(self.llm_client))
        if extra:
            record.update(extra)
        if logger.include_prompts and prompts:
            record["prompts"] = prompts
        logger.log(record)

    def _finish_cls_decision(
        self,
        decision: ClsAgentDecision,
        **log_kw,
    ) -> ClsAgentDecision:
        self._emit_cls_routing_log(decision, **log_kw)
        return decision

    def log_path_a_decision(
        self,
        image_name: str,
        anchor_class: str,
        indie_preds: list[ClsModelOutput],
        decision: ClsAgentDecision,
    ) -> None:
        """Path A：分类共识，未调用 LLM（由 cascade Phase 5 调用）。"""
        self._emit_cls_routing_log(
            decision,
            task="cls_path_a",
            llm_invoked=False,
            method="classification_consensus",
            image_name=image_name,
            path="A",
            classification_consensus=True,
            predictions=indie_preds,
        )

    def _build_system_prompt_single(self) -> str:
        if self.brief_routing_prompt:
            return """从 user JSON  predictions 中选最可信模型。
禁止思考/Markdown。只输出一行 JSON：selected_model, selected_class, confidence(0-1)。勿输出 reasoning 字段。"""
        return """你是甲状腺超声多模型预测整合专家，从若干模型输出中选最可信的一项。

【字段】top_confidence（或 top_confidence_calibrated）为主置信度。entropy 越大越不确定，margin_top2 越大越稳。num_models_same_class 为投票一致性。

【决策序】1) 主置信度高 2) training_devices 与输入设备匹配 3) entropy↓ margin↑ 4) dataset_size 大优先 5) 投票一致性高。

【输出】只输出纯JSON（无Markdown/思考/代码块），首尾为{}。字段：selected_model, selected_class, confidence, reasoning。"""

    def _build_system_prompt_multi(self) -> str:
        tk = self.top_k
        if self.brief_routing_prompt:
            return f"""你是甲状腺超声多模型预测整合专家，选出最值得信任的 {tk} 个模型（按信任度从高到低），用于对各类别概率取平均融合。

【字段】top_confidence（或 top_confidence_calibrated）为主置信度。entropy、margin_top2、num_models_same_class。

【决策序】综合判断哪 {tk} 个模型组合最可信，优先级与单选类似，需考虑组合互补性。

【输出】只输出纯JSON（无Markdown/思考），首尾为{{}}。字段：selected_models(长度恰好{tk}的字符串数组)。勿输出 reasoning、selected_class 或 confidence。"""
        return f"""你是甲状腺超声多模型预测整合专家，选出最值得信任的 {tk} 个模型（按信任度从高到低），用于对各类别概率取平均融合。

【字段】top_confidence（或 top_confidence_calibrated）为主置信度。entropy、margin_top2、num_models_same_class。

【决策序】综合判断哪 {tk} 个模型组合最可信，优先级与单选类似，需考虑组合互补性。

【输出】只输出纯JSON（无Markdown/思考），首尾为{{}}。字段：selected_models(长度恰好{tk}的字符串数组), reasoning。不要输出 selected_class 或 confidence。"""

    # ========== 一致性检查 ==========

    @staticmethod
    def _predictions_top_class_unanimous(predictions: list[ClsModelOutput]) -> bool:
        if not predictions:
            return False
        first = predictions[0].top_class
        return all(p.top_class == first for p in predictions)

    def _decision_unanimous(self, predictions: list[ClsModelOutput]) -> ClsAgentDecision:
        """各模型类别一致：不调 LLM，取 top_confidence 最高者。"""
        best = max(predictions, key=lambda p: p.top_confidence)
        reasoning = (
            f"所有模型均预测为「{best.top_class}」，决策一致，未调用大模型；"
            f"选取 top_confidence 最高的模型 {best.model_name}（{best.top_confidence:.4f}）。"
        )
        return ClsAgentDecision(
            selected_model=best.model_name,
            selected_class=best.top_class,
            confidence=float(best.top_confidence),
            reasoning=reasoning,
            all_predictions=[p.to_dict() for p in predictions],
            method="unanimous",
        )

    # ========== Prompt 构造 ==========

    def _build_compact_prediction_dicts(self, predictions: list[ClsModelOutput]) -> list[dict[str, Any]]:
        """构建 LLM 输入的紧凑预测 JSON。"""
        votes_per_class: dict[str, int] = {}
        for pred in predictions:
            votes_per_class[pred.top_class] = votes_per_class.get(pred.top_class, 0) + 1

        compact: list[dict[str, Any]] = []
        for pred in predictions:
            pd = pred.to_dict()
            meta = pd.get("metadata") or {}
            full_probs = pd.get("predictions", {}) or {}

            class_probs = list(full_probs.values())
            entropy = None
            margin_top2 = None
            if class_probs:
                total_p = sum(class_probs)
                if total_p > 0:
                    norm = [p / total_p for p in class_probs]
                    entropy = -sum(p * math.log(p + 1e-12, 2) for p in norm if p > 0)
                sorted_p = sorted(class_probs, reverse=True)
                margin_top2 = (sorted_p[0] - sorted_p[1]) if len(sorted_p) >= 2 else sorted_p[0]

            sorted_items = sorted(full_probs.items(), key=lambda x: x[1], reverse=True)
            top2 = [{k: float(v)} for k, v in sorted_items[:2]]

            cu = meta.get("classification_uncertainty", {}) or {}
            top_conf_cal = cu.get("top_confidence_calibrated")

            ds_info = meta.get("dataset_info", {}) or {}
            train_devices = meta.get("training_data_devices") or []

            compact.append({
                "model_name": pd.get("model_name"),
                "top_class": pd.get("top_class"),
                "top_confidence": pd.get("top_confidence"),
                "class_probabilities": top2,
                "metadata": {
                    **({"top_confidence_calibrated": top_conf_cal} if top_conf_cal is not None else {}),
                    "entropy": entropy,
                    "margin_top2": margin_top2,
                    "num_models_same_class": votes_per_class.get(pd.get("top_class"), 0),
                    "training_devices": train_devices,
                    "dataset_size": ds_info.get("dataset_size"),
                },
            })
        return compact

    def format_predictions_json(
        self,
        predictions: list[ClsModelOutput],
    ) -> str:
        """构造 LLM 输入 JSON。"""
        # 计算 vote_entropy（顶层，避免每个模型重复）
        votes_per_class: dict[str, int] = {}
        for pred in predictions:
            votes_per_class[pred.top_class] = votes_per_class.get(pred.top_class, 0) + 1
        total_models = len(predictions)
        vote_entropy = 0.0
        if total_models > 0:
            probs = [c / total_models for c in votes_per_class.values()]
            vote_entropy = -sum(p * math.log(p + 1e-12, 2) for p in probs if p > 0)

        from shared.metadata_prompt import filter_cls_routing_envelope

        data = {
            "num_models": total_models,
            "vote_entropy": round(vote_entropy, 3),
            "predictions": self._build_compact_prediction_dicts(predictions),
        }
        data = filter_cls_routing_envelope(data, self.metadata_mode)
        return json.dumps(data, ensure_ascii=False)

    # ========== JSON 解析 ==========

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """从 LLM 响应文本提取 JSON 对象。"""
        if text.startswith("*Thinking"):
            parts = text.split("\n\n")
            for i, part in enumerate(parts):
                if "{" in part:
                    text = "\n\n".join(parts[i:])
                    break

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            for part in text.split("```"):
                if "{" in part and "}" in part:
                    text = part.strip()
                    break

        start = text.find("{")
        if start == -1:
            return text

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
                        return s

        s = text[start:]
        return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", s)

    # ========== 核心选择方法 ==========

    def select_best_model(
        self,
        predictions: list[ClsModelOutput],
        input_device_info: Optional[list[str]] = None,
        input_data_info: Optional[dict] = None,
        image_name: Optional[str] = None,
    ) -> ClsAgentDecision:
        """
        选择最佳分类结果。

        Args:
            predictions: 多个模型的预测结果。
            input_device_info: 输入设备信息。
            input_data_info: 输入数据元信息。
        """
        if not predictions:
            raise ValueError("没有预测结果")

        self._cls_log_ctx = {"image_name": image_name, "predictions": predictions}

        # 一致时不调 LLM
        if self._predictions_top_class_unanimous(predictions):
            return self._finish_cls_decision(
                self._decision_unanimous(predictions),
                task="cls_standalone",
                llm_invoked=False,
                method="unanimous",
                image_name=image_name,
                path=None,
                classification_consensus=True,
                predictions=predictions,
            )

        # 不启用 agent 时直接 soft voting
        if not self.enable_agent:
            return self._finish_cls_decision(
                self._soft_voting_decision(predictions),
                task="cls_standalone",
                llm_invoked=False,
                method="soft_voting",
                image_name=image_name,
                path=None,
                classification_consensus=False,
                predictions=predictions,
            )

        # 构造 prompt
        formatted = self.format_predictions_json(predictions)
        from shared.metadata_prompt import cls_device_line

        device_line = cls_device_line(self.metadata_mode, input_device_info)

        if self.top_k > 1:
            sys_prompt = self.system_prompt_multi
            question = f"选出最值得信任的 {self.top_k} 个模型，严格只回复 JSON。"
        else:
            sys_prompt = self.system_prompt_single
            question = "选出最佳结果，严格只回复 JSON。"

        device_block = f"{device_line}\n" if device_line else ""
        user_prompt = f"""{sys_prompt}

{device_block}以下为 {len(predictions)} 个模型的预测(JSON)：

{formatted}

{question}"""

        prompts_bundle = {"system": sys_prompt, "user": user_prompt}

        # 调用 LLM
        try:
            response_text = self.llm_client.chat(sys_prompt, user_prompt)
            json_text = self._extract_json_from_text(response_text)
            decision_data = json.loads(json_text)
        except Exception as e:
            print(f"  ✗ LLM 调用或解析失败: {e}，使用降级选择")
            return self._finish_cls_decision(
                self._fallback_selection(predictions),
                task="cls_standalone",
                llm_invoked=True,
                method="llm_fallback",
                image_name=image_name,
                classification_consensus=False,
                predictions=predictions,
                parse_ok=False,
                error=str(e),
                prompts=prompts_bundle,
            )

        # 解析决策
        try:
            if self.top_k > 1:
                decision = self._decision_from_llm_topk(predictions, decision_data)
            else:
                decision = self._decision_from_llm_single(predictions, decision_data)
            return self._finish_cls_decision(
                decision,
                task="cls_standalone",
                llm_invoked=True,
                method="llm_single" if self.top_k <= 1 else "llm_topk",
                image_name=image_name,
                classification_consensus=False,
                predictions=predictions,
                prompts=prompts_bundle,
            )
        except (KeyError, ValueError) as e:
            print(f"  ✗ 决策解析失败: {e}，使用降级选择")
            return self._finish_cls_decision(
                self._fallback_selection(predictions),
                task="cls_standalone",
                llm_invoked=True,
                method="llm_fallback",
                image_name=image_name,
                classification_consensus=False,
                predictions=predictions,
                parse_ok=False,
                error=str(e),
                prompts=prompts_bundle,
            )

    def _decision_from_llm_single(
        self, predictions, decision_data
    ) -> ClsAgentDecision:
        """LLM 单选模式。"""
        for field in ["selected_model", "selected_class", "confidence"]:
            if field not in decision_data:
                raise ValueError(f"响应缺少 '{field}'")
        if "reasoning" not in decision_data:
            if not self.brief_routing_prompt:
                raise ValueError("响应缺少 'reasoning'")

        # 修正 confidence 为实际模型置信度
        conf_map = {p.model_name: p.top_confidence for p in predictions}
        selected = decision_data["selected_model"]
        if selected in conf_map:
            confidence = float(conf_map[selected])
        else:
            confidence = max(0.0, min(1.0, float(decision_data["confidence"])))

        reasoning = (
            BRIEF_ROUTING_REASONING
            if self.brief_routing_prompt
            else decision_data["reasoning"]
        )

        return ClsAgentDecision(
            selected_model=selected,
            selected_class=decision_data["selected_class"],
            confidence=confidence,
            reasoning=reasoning,
            all_predictions=[p.to_dict() for p in predictions],
            method="single",
        )

    def _decision_from_llm_topk(
        self, predictions, decision_data
    ) -> ClsAgentDecision:
        """LLM top_k 模式：soft voting。"""
        raw_names = decision_data.get("selected_models")
        if not isinstance(raw_names, list):
            raise ValueError("响应缺少有效的 'selected_models'")

        names = [str(x) for x in raw_names]
        subset = resolve_topk_model_outputs(predictions, names, self.top_k)
        avg_probs = average_class_probabilities(subset)
        cls, conf = winning_class_from_avg_probs(avg_probs)

        return ClsAgentDecision(
            selected_model="agent_topk_soft_voting",
            selected_class=cls,
            confidence=float(conf),
            reasoning=(
                BRIEF_ROUTING_REASONING
                if self.brief_routing_prompt
                else str(decision_data.get("reasoning", ""))
            ),
            all_predictions=[p.to_dict() for p in predictions],
            selected_models=[p.model_name for p in subset],
            method="soft_voting",
        )

    def _soft_voting_decision(
        self, predictions: list[ClsModelOutput]
    ) -> ClsAgentDecision:
        """不启用 agent 时的 soft voting。"""
        sorted_preds = sorted(predictions, key=lambda p: p.top_confidence, reverse=True)
        k = min(self.top_k, len(sorted_preds))
        subset = sorted_preds[:k]
        avg_probs = average_class_probabilities(subset)
        cls, conf = winning_class_from_avg_probs(avg_probs)

        return ClsAgentDecision(
            selected_model="soft_voting",
            selected_class=cls,
            confidence=float(conf),
            reasoning=f"未启用 agent，按 top_confidence 取前 {k} 个模型 soft voting",
            all_predictions=[p.to_dict() for p in predictions],
            selected_models=[p.model_name for p in subset],
            method="soft_voting",
        )

    def _fallback_selection(
        self, predictions: list[ClsModelOutput]
    ) -> ClsAgentDecision:
        """降级选择：top_k=1 选最高置信度；top_k>1 soft voting。"""
        if self.top_k <= 1:
            best = max(predictions, key=lambda p: p.top_confidence)
            agree = sum(1 for p in predictions if p.top_class == best.top_class and p.top_confidence > 0.7)
            reasoning = f"降级选择：最高置信度模型（{best.top_confidence:.2%}）"
            if agree >= 3:
                reasoning += f"，{agree} 个模型一致预测为 {best.top_class}"
            return ClsAgentDecision(
                selected_model=best.model_name,
                selected_class=best.top_class,
                confidence=best.top_confidence,
                reasoning=reasoning,
                all_predictions=[p.to_dict() for p in predictions],
                method="fallback",
            )

        sorted_preds = sorted(predictions, key=lambda p: p.top_confidence, reverse=True)
        k = min(self.top_k, len(sorted_preds))
        subset = sorted_preds[:k]
        avg_probs = average_class_probabilities(subset)
        cls, conf = winning_class_from_avg_probs(avg_probs)
        return ClsAgentDecision(
            selected_model="agent_topk_soft_voting",
            selected_class=cls,
            confidence=float(conf),
            reasoning=f"降级选择：前 {k} 个模型 soft voting",
            all_predictions=[p.to_dict() for p in predictions],
            selected_models=[p.model_name for p in subset],
            method="fallback",
        )

    def resolve_path_b(
        self,
        indie_predictions: list["ClsModelOutput"],
        autogluon_pred: Optional["ClsModelOutput"],
        seg_decision: Any,
        voting_weights: Optional[dict] = None,
        image_name: Optional[str] = None,
    ) -> "ClsAgentDecision":
        """
        Path B 分类裁决：独立分类模型无共识时的分类决策。

        将所有分类模型输出（含 AutoGluon）交给 LLM 进行分类决策。
        LLM 失败时降级到加权软投票（含 AutoGluon）。
        """
        if not indie_predictions:
            raise ValueError("没有独立分类模型预测")

        if not self.enable_agent:
            all_preds = list(indie_predictions)
            if autogluon_pred is not None and autogluon_pred.predictions:
                all_preds.append(autogluon_pred)
            sv_class, sv_conf = _weighted_soft_vote(all_preds, voting_weights)
            decision = ClsAgentDecision(
                selected_model="weighted_soft_vote",
                selected_class=sv_class,
                confidence=float(sv_conf),
                reasoning=f"分类 Agent 关闭，加权软投票={sv_class}({sv_conf:.2f})",
                all_predictions=[p.to_dict() for p in all_preds],
                method="path_b_static_softvote",
            )
            return self._finish_cls_decision(
                decision,
                task="cls_path_b",
                llm_invoked=False,
                method="path_b_static_softvote",
                image_name=image_name,
                path="B",
                classification_consensus=False,
                predictions=indie_predictions,
                extra=self._path_b_extra(autogluon_pred, seg_decision),
            )

        try:
            return self._resolve_path_b_with_llm(
                indie_predictions, autogluon_pred, seg_decision, image_name=image_name
            )
        except Exception as e:
            print(f"  ✗ Path B LLM 裁决失败: {e}")
            # 降级到加权软投票（含 AutoGluon）
            all_preds = list(indie_predictions)
            if autogluon_pred is not None and autogluon_pred.predictions:
                all_preds.append(autogluon_pred)
            sv_class, sv_conf = _weighted_soft_vote(all_preds, voting_weights)
            decision = ClsAgentDecision(
                selected_model="weighted_soft_vote",
                selected_class=sv_class,
                confidence=float(sv_conf),
                reasoning=f"LLM仲裁失败，降级加权软投票={sv_class}({sv_conf:.2f})",
                all_predictions=[p.to_dict() for p in all_preds],
                method="path_b_softvote",
            )
            return self._finish_cls_decision(
                decision,
                task="cls_path_b",
                llm_invoked=True,
                method="path_b_softvote",
                image_name=image_name,
                path="B",
                classification_consensus=False,
                predictions=indie_predictions,
                parse_ok=False,
                error=str(e),
                extra=self._path_b_extra(autogluon_pred, seg_decision),
            )

    @staticmethod
    def _decision_majority(predictions, majority) -> "ClsAgentDecision":
        """Path B 最终降级：输出多数派。"""
        best = max(predictions, key=lambda p: p.top_confidence)
        return ClsAgentDecision(
            selected_model="majority_fallback",
            selected_class=majority,
            confidence=float(best.top_confidence),
            reasoning=(
                f"独立模型无共识(多数派={majority})，AutoGluon也不确定，"
                f"退回到多数派投票。置信度=low。"
            ),
            all_predictions=[p.to_dict() for p in predictions],
            method="path_b_majority_fallback",
        )

    @staticmethod
    def _path_b_extra(autogluon_pred: Optional["ClsModelOutput"], seg_decision: Any) -> dict:
        seg_model = (
            seg_decision.selected_model
            if hasattr(seg_decision, "selected_model")
            else None
        )
        seg_reason = (
            seg_decision.reasoning if hasattr(seg_decision, "reasoning") else None
        )
        al = None
        if autogluon_pred is not None:
            al = {
                "top_class": autogluon_pred.top_class,
                "top_confidence": autogluon_pred.top_confidence,
            }
        return {
            "autogluon_summary": al,
            "segmentation_selected_model": seg_model,
            "segmentation_reasoning": seg_reason,
        }

    def _resolve_path_b_with_llm(
        self,
        indie_preds: list["ClsModelOutput"],
        autogluon_pred: Optional["ClsModelOutput"],
        seg_decision: Any,
        image_name: Optional[str] = None,
    ) -> "ClsAgentDecision":
        """Path B 最困难情况：调 LLM 做最终裁决。

        LLM 收到的信号：
        - 3 个独立模型各自的预测和置信度
        - AutoGluon 基于筛选 mask 的分类结果（如有）
        - 分割 Agent 选择该 mask 的理由
        """
        # 独立模型信息
        indie_info = []
        for p in indie_preds:
            pd = p.to_dict()
            indie_info.append({
                "model_name": pd.get("model_name"),
                "top_class": pd.get("top_class"),
                "top_confidence": pd.get("top_confidence"),
            })

        # AutoGluon 信息
        al_info = None
        if autogluon_pred is not None:
            al_info = {
                "top_class": autogluon_pred.top_class,
                "top_confidence": autogluon_pred.top_confidence,
            }

        # 分割 Agent reasoning
        seg_reason = (
            seg_decision.reasoning
            if hasattr(seg_decision, "reasoning")
            else str(seg_decision)
        )
        seg_model = (
            seg_decision.selected_model
            if hasattr(seg_decision, "selected_model")
            else "unknown"
        )

        seg_block: dict[str, Any] = {"selected_mask_from": seg_model}
        if not self.brief_routing_prompt:
            seg_block["reasoning"] = seg_reason

        prompt_data = {
            "scenario": "独立分类模型无共识，需综合裁决",
            "independent_models": indie_info,
            "autogluon_based_on_selected_mask": al_info,
            "segmentation_decision": seg_block,
            "question": "请综合以上信息，给出最终良恶性判断和置信度(0-1)。",
        }

        sys_prompt = (
            PATH_B_LLM_SYSTEM_PROMPT_BRIEF
            if self.brief_routing_prompt
            else PATH_B_LLM_SYSTEM_PROMPT
        )

        user_prompt = json.dumps(prompt_data, ensure_ascii=False)
        prompts_bundle = {"system": sys_prompt, "user": user_prompt}

        response_text = self.llm_client.chat(sys_prompt, user_prompt)
        json_text = self._extract_json_from_text(response_text)
        decision_data = json.loads(json_text)

        pb_reasoning = (
            BRIEF_ROUTING_REASONING
            if self.brief_routing_prompt
            else str(decision_data.get("reasoning", "")).strip()
        )

        decision = ClsAgentDecision(
            selected_model="agent_path_b_llm",
            selected_class=decision_data["selected_class"],
            confidence=float(decision_data["confidence"]),
            reasoning=pb_reasoning or str(decision_data.get("reasoning", "")),
            all_predictions=[p.to_dict() for p in indie_preds],
            method="path_b_llm",
        )
        return self._finish_cls_decision(
            decision,
            task="cls_path_b",
            llm_invoked=True,
            method="path_b_llm",
            image_name=image_name,
            path="B",
            classification_consensus=False,
            predictions=indie_preds,
            prompts=prompts_bundle,
            extra=self._path_b_extra(autogluon_pred, seg_decision),
        )

    def batch_select(
        self,
        batch_predictions: list[list[ClsModelOutput]],
        input_device_info: Optional[list[str]] = None,
        input_data_info: Optional[dict] = None,
    ) -> list[ClsAgentDecision]:
        """批处理：逐图调用 select_best_model。"""
        decisions = []
        for preds in batch_predictions:
            decisions.append(
                self.select_best_model(preds, input_device_info, input_data_info)
            )
        return decisions
