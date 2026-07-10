"""
LLM-based agent for selecting the best segmentation mask
Supports OpenAI-compatible API and local in-process transformers backends
"""

import json
import os
import time
import numpy as np
import cv2
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from openai import OpenAI
try:
    from openai import RateLimitError
except ImportError:
    RateLimitError = None  # older openai versions

from models.base_model import ModelOutput
from utils.quality_evaluator import SegmentationQualityEvaluator
from utils.metrics import compute_dice, compute_hd95, compute_ece


@dataclass
class AgentDecision:
    """
    Agent's decision on which model(s) has the best segmentation
    Supports both single model selection and ensemble (multiple models)
    """
    selected_model: str  # For backward compatibility, always contains the first selected model
    selected_mask: np.ndarray  # Final mask (from single model or ensemble)
    confidence: float
    reasoning: str
    all_predictions: List[Dict[str, Any]]
    quality_metrics: Optional[Dict[str, Any]] = None  # Individual quality metrics for selected mask
    agreement_score: Optional[float] = None  # Agreement score with other models
    # Performance metrics (only if GT is provided)
    dice_score: Optional[float] = None
    hd95_score: Optional[float] = None
    # ECE (from model metadata base_dataset_performance, if provided)
    # Structure: {dataset_name: ece_value}
    ece_metrics: Optional[Dict[str, float]] = None
    # Ensemble support
    selected_models: Optional[List[str]] = None  # List of selected models for ensemble
    model_weights: Optional[List[float]] = None  # Weights for each selected model
    is_ensemble: bool = False  # Whether this is an ensemble result
    
    def to_dict(self, include_mask: bool = False) -> Dict:
        """
        Convert to dictionary format
        
        Args:
            include_mask: Whether to include the full mask array (default: False for smaller output)
        """
        result = {
            'selected_model': self.selected_model,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'all_predictions': self.all_predictions,
            'quality_metrics': self.quality_metrics,
            'agreement_score': self.agreement_score,
            'is_ensemble': self.is_ensemble
        }
        
        # Add ensemble information if available
        if self.is_ensemble and self.selected_models is not None:
            result['selected_models'] = self.selected_models
            if self.model_weights is not None:
                result['model_weights'] = self.model_weights
        
        if include_mask:
            result['selected_mask'] = self.selected_mask.tolist()
        else:
            result['mask_shape'] = self.selected_mask.shape
            result['mask_area'] = int(np.sum(self.selected_mask))
        
        # Add performance metrics if available
        if self.dice_score is not None:
            result['dice_score'] = self.dice_score
        if self.hd95_score is not None:
            result['hd95_score'] = self.hd95_score
        if self.ece_metrics is not None:
            result['ece_metrics'] = self.ece_metrics
        
        return result
    
    def to_simplified_dict(self) -> Dict:
        """
        Convert to simplified dictionary format for saving
        Only includes: selected_model, confidence, reasoning, and simplified all_predictions
        """
        # Simplify all_predictions to only include mask_area and mean_confidence
        simplified_predictions = []
        for pred in self.all_predictions:
            simplified_pred = {
                'model_name': pred['model_name'],
                'mask_area': pred.get('mask_area', 0)
            }
            if pred.get('has_confidence_map', False):
                simplified_pred['mean_confidence'] = pred.get('mean_confidence', 0.0)
            simplified_predictions.append(simplified_pred)
        
        result = {
            'selected_model': self.selected_model,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'all_predictions': simplified_predictions,
            'is_ensemble': self.is_ensemble
        }
        
        # Add ensemble information if available
        if self.is_ensemble and self.selected_models is not None:
            result['selected_models'] = self.selected_models
            if self.model_weights is not None:
                result['model_weights'] = [float(w) for w in self.model_weights]
        
        # Add ECE metrics if available
        if self.ece_metrics:
            result['ece_metrics'] = self.ece_metrics
            ece_vals = list(self.ece_metrics.values())
            if ece_vals:
                result['ece_mean'] = float(np.mean(ece_vals))
        
        return result


class SegmentationAgent:
    """
    LLM-powered agent for selecting the best segmentation result
    Supports two backends:
    1) OpenAI-compatible chat completions API (e.g. 智谱 GLM, 阿里 DashScope)
    2) Local in-process GPT-OSS style model via transformers
    """
    
    def __init__(
        self,
        backend_type: str = "llm",
        api_key: Optional[str] = None,
        model_name: str = "qwen-plus-2025-07-14",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        base_url: Optional[str] = None,
        base_datasets_info: Optional[Dict[str, Any]] = None,
        ensemble_enabled: bool = True,
        ensemble_top_k: int = 2,
        ensemble_method: str = "weighted_average",
        ensemble_threshold: float = 0.5,
        max_retries: int = 3,
        include_disagreement_metrics_in_prompt: bool = True,
        local_model_path: Optional[str] = None,
        local_device_map: str = "auto",
        local_torch_dtype: str = "bfloat16",
        local_trust_remote_code: bool = True,
        local_max_new_tokens: Optional[int] = None,
    ):
        """
        Initialize the Segmentation agent.
        
        Args:
            backend_type: Backend type: "llm" or "local_gpt_oss"
            api_key: API key for "llm" backend. If None, uses DASHSCOPE_API_KEY env var.
            model_name: Model name (default: "qwen-plus-2025-07-14" for 阿里云百炼)
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens in response
            base_url: OpenAI-compatible API base URL. If None, uses 阿里云百炼默认地址.
            base_datasets_info: Information about base datasets (device distribution)
            ensemble_enabled: Whether to enable ensemble (default: True)
            ensemble_top_k: Number of top models to select for ensemble (default: 2)
            ensemble_method: Ensemble method: "weighted_average", "equal_weight", "geometric_mean" (default: "weighted_average")
            ensemble_threshold: Threshold for generating final mask from ensemble probability map (default: 0.5)
            max_retries: Max retries for API calls on rate limit (429) or transient errors (default: 3)
            include_disagreement_metrics_in_prompt: If True, add compact inter-model area CV / HD95
                summaries (no full matrices) to the user JSON to inform uncertainty-aware selection.
            local_model_path: Local model directory/path for "local_gpt_oss" backend
            local_device_map: device_map passed to transformers.from_pretrained in local mode
            local_torch_dtype: torch dtype name ("bfloat16"/"float16"/"float32") in local mode
            local_trust_remote_code: trust_remote_code for local tokenizer/model loading
            local_max_new_tokens: max_new_tokens for local generation (defaults to max_tokens)
        """
        normalized_backend = str(backend_type or "llm").strip().lower()
        if normalized_backend == "glm":
            normalized_backend = "llm"
        if normalized_backend not in {"llm", "local_gpt_oss"}:
            raise ValueError(
                f"Unsupported backend_type: {backend_type}. Use one of: llm, local_gpt_oss."
            )
        self.backend_type = normalized_backend

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.base_datasets_info = base_datasets_info or {}
        self.local_model_path = local_model_path
        self.local_device_map = local_device_map
        self.local_torch_dtype = str(local_torch_dtype or "bfloat16").strip().lower()
        self.local_trust_remote_code = bool(local_trust_remote_code)
        self.local_max_new_tokens = int(local_max_new_tokens) if local_max_new_tokens is not None else self.max_tokens
        self._local_model = None
        self._local_tokenizer = None
        self._local_harmony_encoding = None
        
        # OpenAI-compatible endpoint (default: 阿里云百炼 DashScope compatible-mode)
        _base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not _base_url.endswith("/"):
            _base_url = _base_url.rstrip("/")
        self.base_url = _base_url
        self.client = None
        if self.backend_type == "llm":
            if not self.api_key:
                raise ValueError("API key is required. Set api_key or DASHSCOPE_API_KEY (for 阿里云百炼).")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=_base_url,
            )
        else:
            if not self.local_model_path:
                raise ValueError("local_model_path is required when backend_type='local_gpt_oss'.")
        
        # Ensemble configuration
        self.ensemble_enabled = ensemble_enabled
        self.ensemble_top_k = ensemble_top_k
        self.ensemble_method = ensemble_method
        self.ensemble_threshold = ensemble_threshold
        self.max_retries = max(0, max_retries)
        self.include_disagreement_metrics_in_prompt = bool(include_disagreement_metrics_in_prompt)
        
        # Initialize quality evaluator
        self.quality_evaluator = SegmentationQualityEvaluator()
        
        # Generate system prompt based on ensemble configuration
        self.system_prompt = self._generate_system_prompt()

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str):
        import torch

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(str(dtype_name or "bfloat16").strip().lower(), torch.bfloat16)

    @staticmethod
    def _collect_harmony_channel_content(
        parsed_messages: List[Dict[str, Any]],
        preferred_channel: str = "final",
    ) -> str:
        """
        Collect content from parsed Harmony messages.
        Prefer the target channel (usually "final"); if missing, fallback to last non-empty content.
        """
        if not parsed_messages:
            return ""

        preferred = []
        preferred_key = str(preferred_channel or "final").strip().lower()
        for msg in parsed_messages:
            channel = str(msg.get("channel") or "").strip().lower()
            content = str(msg.get("content") or "").strip()
            if channel == preferred_key and content:
                preferred.append(content)
        if preferred:
            return "\n".join(preferred).strip()

        non_empty = [str(m.get("content") or "").strip() for m in parsed_messages]
        non_empty = [c for c in non_empty if c]
        if not non_empty:
            return ""
        return non_empty[-1]

    def _parse_harmony_messages_from_tokens(self, token_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Parse GPT-OSS Harmony channels from generated token ids.
        Mirrors TAge llm_manager StreamableParser usage, but keeps implementation local.
        """
        if not token_ids:
            return []
        try:
            from openai_harmony import HarmonyEncodingName, load_harmony_encoding, StreamableParser, Role
        except Exception:
            return []

        try:
            if self._local_harmony_encoding is None:
                self._local_harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            parser = StreamableParser(self._local_harmony_encoding, role=Role.ASSISTANT)
        except Exception:
            return []

        messages: List[Dict[str, Any]] = []
        current_message: Dict[str, Any] = {
            "role": None,
            "channel": None,
            "content": "",
            "recipient": None,
            "content_type": None,
        }

        for tok in token_ids:
            parser.process(int(tok))
            if (
                (parser.current_role != current_message["role"] or parser.current_channel != current_message["channel"])
                and current_message["content"]
            ):
                messages.append(current_message.copy())
                current_message = {
                    "role": parser.current_role,
                    "channel": parser.current_channel,
                    "content": parser.last_content_delta or "",
                    "recipient": parser.current_recipient,
                    "content_type": parser.current_content_type,
                }
            else:
                current_message["role"] = parser.current_role
                current_message["channel"] = parser.current_channel
                if parser.last_content_delta:
                    current_message["content"] += parser.last_content_delta
                current_message["recipient"] = parser.current_recipient
                current_message["content_type"] = parser.current_content_type

        if current_message["content"]:
            messages.append(current_message)
        return messages

    def _ensure_local_model_loaded(self) -> None:
        if self._local_model is not None and self._local_tokenizer is not None:
            return
        if not self.local_model_path:
            raise ValueError("local_model_path is required in local_gpt_oss mode.")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Local backend requires transformers and torch. Please install them in current environment."
            ) from e

        torch_dtype = self._resolve_torch_dtype(self.local_torch_dtype)
        self._local_tokenizer = AutoTokenizer.from_pretrained(
            self.local_model_path,
            trust_remote_code=self.local_trust_remote_code,
        )
        self._local_model = AutoModelForCausalLM.from_pretrained(
            self.local_model_path,
            trust_remote_code=self.local_trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=self.local_device_map,
        )
        self._local_model.eval()

    def _call_local_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        self._ensure_local_model_loaded()
        import torch

        tokenizer = self._local_tokenizer
        model = self._local_model
        if tokenizer is None or model is None:
            raise RuntimeError("Local model is not loaded.")

        # Preferred path for GPT-OSS: Harmony conversation rendering + channel parsing.
        harmony_response = self._call_local_chat_completion_harmony(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if harmony_response:
            return harmony_response

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                model_inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            except TypeError:
                chat_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = tokenizer(chat_text, return_tensors="pt")
        else:
            plain_prompt = "\n".join(
                [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
            ) + "\nassistant:"
            model_inputs = tokenizer(plain_prompt, return_tensors="pt")

        model_device = getattr(model, "device", None)
        if isinstance(model_inputs, dict):
            if model_device is not None:
                for k, v in list(model_inputs.items()):
                    if hasattr(v, "to"):
                        model_inputs[k] = v.to(model_device)
            input_ids = model_inputs["input_ids"]
            if "attention_mask" not in model_inputs:
                model_inputs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)
        else:
            input_ids = model_inputs
            if model_device is not None and hasattr(input_ids, "to"):
                input_ids = input_ids.to(model_device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        do_sample = float(temperature) > 0
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max(1, int(max_tokens)),
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id
            if getattr(tokenizer, "pad_token_id", None) is not None
            else getattr(tokenizer, "eos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)
            generation_kwargs["top_p"] = 0.9

        with torch.no_grad():
            if isinstance(model_inputs, dict):
                outputs = model.generate(**model_inputs, **generation_kwargs)
            else:
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)

        prompt_len = int(input_ids.shape[-1])
        generated_ids = outputs[0][prompt_len:]
        token_ids = generated_ids.tolist() if hasattr(generated_ids, "tolist") else [int(x) for x in generated_ids]

        # Prefer Harmony channel parsing for GPT-OSS outputs (analysis/commentary/final).
        parsed_messages = self._parse_harmony_messages_from_tokens(token_ids)
        if parsed_messages:
            final_content = self._collect_harmony_channel_content(parsed_messages, preferred_channel="final")
            if final_content:
                return final_content

        # Fallback to plain decode if parsing is unavailable.
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        lowered = response_text.lower()
        for prefix in ("analysis", "commentary", "final"):
            if lowered.startswith(prefix):
                response_text = response_text[len(prefix):].lstrip(" :\n\t")
                break
        return response_text

    def _call_local_chat_completion_harmony(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        GPT-OSS Harmony-native generation path (same family as TAge llm_manager).
        Returns final-channel content when available; empty string means caller should fallback.
        """
        if not messages:
            return ""
        try:
            from openai_harmony import (
                HarmonyEncodingName,
                load_harmony_encoding,
                Conversation,
                Message,
                Role,
                SystemContent,
                DeveloperContent,
                ReasoningEffort,
            )
        except Exception:
            return ""

        tokenizer = self._local_tokenizer
        model = self._local_model
        if tokenizer is None or model is None:
            return ""

        try:
            if self._local_harmony_encoding is None:
                self._local_harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            encoding = self._local_harmony_encoding
        except Exception:
            return ""

        # Convert OpenAI-style messages to Harmony-style (system + developer instructions + user request).
        dev_chunks: List[str] = []
        user_chunks: List[str] = []
        for msg in messages:
            role = str(msg.get("role") or "").strip().lower()
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            if role in {"system", "developer"}:
                dev_chunks.append(content)
            elif role == "user":
                user_chunks.append(content)
            else:
                user_chunks.append(content)
        if not user_chunks and dev_chunks:
            user_chunks.append(dev_chunks[-1])

        try:
            system_content = SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)
            harmony_messages = [Message.from_role_and_content(Role.SYSTEM, system_content)]

            if dev_chunks:
                developer_content = DeveloperContent.new().with_instructions("\n\n".join(dev_chunks))
                harmony_messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_content))
            if user_chunks:
                harmony_messages.append(Message.from_role_and_content(Role.USER, "\n\n".join(user_chunks)))

            conversation = Conversation.from_messages(harmony_messages)
            input_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        except Exception:
            return ""

        import torch

        input_tensor = torch.tensor([input_ids], device=model.device)
        attention_mask = torch.ones_like(input_tensor, dtype=torch.long)
        do_sample = float(temperature) > 0
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max(1, int(max_tokens)),
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)
            generation_kwargs["top_p"] = 0.9

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
            new_tokens = outputs[0][len(input_ids):].tolist()
        except Exception:
            return ""

        parsed_messages = self._parse_harmony_messages_from_tokens(new_tokens)
        if parsed_messages:
            final_content = self._collect_harmony_channel_content(parsed_messages, preferred_channel="final")
            if final_content:
                return final_content

        # Fallback: decode raw generated tokens as plain text.
        try:
            raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return raw_text
        except Exception:
            return ""

    def _call_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_json: bool = True,
    ) -> str:
        if self.backend_type == "local_gpt_oss":
            return self._call_local_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=self.local_max_new_tokens if max_tokens is None else max_tokens,
            )

        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            completion = self.client.chat.completions.create(
                **kwargs, extra_body={"enable_thinking": False}
            )
        except TypeError:
            kwargs.pop("response_format", None)
            completion = self.client.chat.completions.create(**kwargs)
        except Exception:
            completion = self.client.chat.completions.create(**kwargs)

        response_text = ""
        if completion.choices:
            choice0 = completion.choices[0]
            msg = choice0.message
            content = getattr(msg, "content", None)
            response_text = (content or "").strip()
            if not response_text:
                reasoning_content = getattr(msg, "reasoning_content", None)
                if reasoning_content:
                    response_text = str(reasoning_content).strip()

            finish_reason = getattr(choice0, "finish_reason", None)
            if (finish_reason or "").lower() == "length" and not response_text:
                raise ValueError("LLM 输出被截断 (finish_reason=length)，且未返回可解析文本")

        return response_text
    
    def _generate_system_prompt(self) -> str:
        """
        Generate system prompt based on ensemble configuration
        """
        if self.ensemble_enabled and self.ensemble_top_k > 1:
            # Ensemble mode: return multiple models
            model3_part = ', "模型3名称"' if self.ensemble_top_k >= 3 else ''
            weight3_part = ', 0.0' if self.ensemble_top_k >= 3 else ''
            output_format = f'''{{{{"selected_models": ["模型1名称", "模型2名称"{model3_part}], "weights": [0.6, 0.4{weight3_part}], "confidence": 0.95, "reasoning": "选择理由"}}}}'''
            output_description = (
                f'返回一个JSON对象，字段包括: '
                f'"selected_models" (长度为{self.ensemble_top_k}的模型名称数组，按质量降序)，'
                f'"weights" (对应每个模型的权重，和为1.0)，'
                f'"confidence" (0-1的决策置信度)，'
                f'"reasoning" (3-4句中文：先1-2句写核心依据含关键数值，再扩展1-2句补充形态学/设备/数据集等，约120-180字)。'
            )
        else:
            # Single model mode: return one model
            output_format = '''{"selected_model": "模型名称", "confidence": 0.95, "reasoning": "选择理由"}'''
            output_description = (
                '返回一个JSON对象，字段包括: '
                '"selected_model" (最佳模型名称)、'
                '"confidence" (0-1的决策置信度)、'
                '"reasoning" (3-4句中文：先1-2句写核心依据含关键数值，再扩展1-2句补充形态学/设备/数据集等，约120-180字)。'
            )
        
        disagreement_block = ""
        if self.include_disagreement_metrics_in_prompt:
            disagreement_block = """
当输入 JSON 含 group_uncertainty 与各模型的 disagreement 时（无完整矩阵，仅为标量摘要）：
- group_uncertainty.area_cv：各模型前景面积相对离散度，越大表示面积共识越差；
- group_uncertainty.pairwise_hd95_mean / pairwise_hd95_std：模型间边界差异（HD95）的整体水平与波动，越大表示边界分歧越大；
- disagreement.mean_hd95_to_others：该模型与其余模型边界的平均 HD95，越低越接近群体边界；
- disagreement.area_rel_to_group：该模型面积相对群体均值的相对偏差，绝对值大表示面积异常。
在 area_cv 或 pairwise_hd95_mean 较高时，优先选择 agreement_with_others 高且 mean_hd95_to_others 低的模型；mask_statistics.smoothness 是单掩码边界光滑度，与 mean_hd95_to_others（跨模型边界分歧）含义不同，二者可同时参考。
"""
        return f"""你是一个分割模型选择代理，需要根据多个模型的掩码质量指标选择最佳结果。
若用户消息开头有【分歧摘要】【各模型】两行，必须先阅读其中的面积 CV 与跨模型 HD95 指标，并在 reasoning 中原样包含相关数值。
只允许输出一个JSON对象，禁止输出思考过程、解释性文字或Markdown代码块；首字符必须是{{，末字符必须是}}，不要任何前缀或后缀。

输出格式说明：{output_description}
示例：{output_format}
{disagreement_block}
决策优先级（从高到低）：
1) 模型间一致性：agreement_with_others 高、IoU>0.7 的模型更可靠；若存在 group_uncertainty/disagreement，在整体分歧大时更倚重「agreement 高且 mean_hd95_to_others 低」的候选；
2) 设备匹配：训练设备与输入设备越接近越好，可参考 training_devices 和 input_device_info；
3) 形态学：掩码应尽量单连通，边界平滑，circularity 0.6-0.9、面积与长宽比合理；
4) 数据集信息：在 base_dataset_performance 中 dice 高、hd95 低 且 dataset_size 大 的模型更优；
5) 置信度：mean_confidence 较高时可作为次要加分项。

总体原则：优先考虑一致性和设备匹配，在形态学或性能明显异常时降低该模型的权重，不依赖模型名称本身做主观判断。
若输入 JSON 含 reasoning_requirements，则 reasoning 必须满足其中全部条目（缺任何一项视为未完成）；此时字数可至约150–220字。
reasoning 字段使用简洁中文：须引用关键数值（如 agreement_with_others、dice、hd95、dataset_size、circularity、group_uncertainty 中 area_cv 与 pairwise_hd95_mean/std、所选模型的 disagreement 等）；先1-2句概括决策依据，再扩展1-2句补充形态学、设备匹配或跨数据集对比等；禁止仅用「高/低」而不给具体数。"""

    def _compute_ece_metrics_from_prob_map(
        self,
        prob_map: np.ndarray,
        gt_mask: np.ndarray,
        n_bins: int = 15,
    ) -> Optional[Dict[str, float]]:
        """
        Compute per-image ECE from a probability map and GT mask.

        `prob_map` should represent foreground probability per pixel (ideally in [0, 1]).
        `gt_mask` should be binary (0/1).
        """
        try:
            if prob_map is None or gt_mask is None:
                return None
            if prob_map.shape != gt_mask.shape:
                prob_map_resized = cv2.resize(
                    prob_map.astype(np.float32),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                prob_map_resized = prob_map

            ece_val = compute_ece(prob_map_resized, gt_mask, n_bins=n_bins)
            return {"ece": float(ece_val)}
        except Exception:
            return None

    def _extract_ece_metrics_from_model_output(
        self,
        pred: ModelOutput,
        gt_mask: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        """
        Compute ECE using a model's confidence/probability map and GT mask.
        """
        if pred is None or gt_mask is None:
            return None
        prob_map = getattr(pred, "confidence_map", None)
        if prob_map is None:
            return None
        return self._compute_ece_metrics_from_prob_map(prob_map, gt_mask)

    def _ensemble_probability_maps(
        self,
        prob_maps: List[np.ndarray],
        weights: List[float],
        method: str = "weighted_average"
    ) -> np.ndarray:
        """
        Ensemble multiple probability maps
        
        Args:
            prob_maps: List of probability maps (H, W) with values in [0, 1]
            weights: List of weights for each probability map (should sum to 1.0)
            method: Ensemble method: "weighted_average", "equal_weight", "geometric_mean"
            
        Returns:
            Ensemble probability map (H, W)
        """
        if len(prob_maps) == 0:
            raise ValueError("No probability maps provided")
        
        if len(prob_maps) != len(weights):
            raise ValueError(f"Number of probability maps ({len(prob_maps)}) must match number of weights ({len(weights)})")
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total_weight for w in weights]
        
        # Ensure all probability maps have the same size
        target_shape = prob_maps[0].shape
        normalized_prob_maps = []
        for pm in prob_maps:
            if pm.shape != target_shape:
                pm_resized = cv2.resize(
                    pm.astype(np.float32),
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                normalized_prob_maps.append(pm_resized)
            else:
                normalized_prob_maps.append(pm.astype(np.float32))
        
        # Ensemble based on method
        if method == "weighted_average" or method == "equal_weight":
            ensemble_map = np.zeros_like(normalized_prob_maps[0])
            for pm, w in zip(normalized_prob_maps, weights):
                ensemble_map += pm * w
        
        elif method == "geometric_mean":
            # Use geometric mean: (prod(p^w))^(1/sum(w))
            # Clip to avoid numerical issues
            ensemble_map = np.ones_like(normalized_prob_maps[0])
            for pm, w in zip(normalized_prob_maps, weights):
                pm_clipped = np.clip(pm, 1e-6, 1.0)
                ensemble_map *= np.power(pm_clipped, w)
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        # Ensure values are in [0, 1]
        ensemble_map = np.clip(ensemble_map, 0.0, 1.0)
        
        return ensemble_map
    
    def _generate_mask_from_probability(
        self,
        prob_map: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Generate binary mask from probability map
        
        Args:
            prob_map: Probability map (H, W) with values in [0, 1]
            threshold: Threshold for binarization
            
        Returns:
            Binary mask (H, W) with values 0 or 1
        """
        mask = (prob_map > threshold).astype(np.uint8)
        
        # Post-processing: keep only the largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            # Find the largest component (excluding background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8)
        
        return mask
    
    @staticmethod
    def _compact_disagreement_fields(
        agreement_metrics: Dict[str, Any], n_models: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, float]]]]:
        """
        Build compact group + per-model disagreement stats (no pairwise matrices) for the LLM prompt.
        """
        if n_models < 2:
            return None, None
        vols = np.asarray(agreement_metrics.get("volumes") or [], dtype=np.float64)
        hdm = np.asarray(agreement_metrics.get("pairwise_hd95_matrix") or [], dtype=np.float64)
        if vols.size != n_models or hdm.shape != (n_models, n_models):
            return None, None
        vmean = float(vols.mean())
        per_model: List[Dict[str, float]] = []
        for i in range(n_models):
            idx_others = [j for j in range(n_models) if j != i]
            others = hdm[i, idx_others]
            mhd = float(np.mean(others)) if others.size else 0.0
            area_rel = float((vols[i] - vmean) / vmean) if vmean > 1e-6 else 0.0
            per_model.append(
                {
                    "mean_hd95_to_others": round(mhd, 2),
                    "area_rel_to_group": round(area_rel, 4),
                }
            )
        group = {
            "area_cv": round(float(agreement_metrics.get("volume_cv", 0.0)), 4),
            "pairwise_hd95_mean": round(float(agreement_metrics.get("pairwise_hd95_mean", 0.0)), 2),
            "pairwise_hd95_std": round(float(agreement_metrics.get("pairwise_hd95_std", 0.0)), 2),
        }
        return group, per_model
    
    @staticmethod
    def _build_disagreement_prefix_lines(
        g_val: Dict[str, Any],
        per_val: Optional[List[Dict[str, float]]],
        predictions: List[ModelOutput],
        max_models_in_compact_line: int = 8,
    ) -> str:
        """
        Very short lines placed at the top of the user message so the model sees CV / HD95 before the JSON.
        """
        lines = [
            f"【分歧摘要】area_cv={g_val['area_cv']}, pairwise_hd95_mean={g_val['pairwise_hd95_mean']}, "
            f"pairwise_hd95_std={g_val['pairwise_hd95_std']}。reasoning 必须逐字包含上述三个数值。"
        ]
        if (
            per_val is not None
            and len(per_val) == len(predictions)
            and len(predictions) <= max_models_in_compact_line
        ):
            segs = [
                f"{p.model_name}:{dc['mean_hd95_to_others']:.2f},{dc['area_rel_to_group']:.4f}"
                for p, dc in zip(predictions, per_val)
            ]
            lines.append(
                "【各模型】mhd95,area_rel丨" + "丨".join(segs) + "。reasoning 须含所选模型对应两项。"
            )
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _find_prediction_index(predictions: List[ModelOutput], name: str) -> Optional[int]:
        if not name:
            return None
        name_l = name.strip().lower()
        for idx, pred in enumerate(predictions):
            if pred.model_name == name or pred.model_name.strip().lower() == name_l:
                return idx
        return None

    def _reasoning_includes_disagreement_numbers(
        self,
        reasoning: str,
        g_val: Dict[str, Any],
        per_val: Optional[List[Dict[str, float]]],
        predictions: List[ModelOutput],
        primary_selected_name: str,
    ) -> bool:
        """Check whether reasoning cites global + per-model disagreement numbers (same format as prefix)."""
        r = reasoning.replace(" ", "").replace("，", ",")
        a = f"{float(g_val['area_cv']):.4f}"
        b = f"{float(g_val['pairwise_hd95_mean']):.2f}"
        c = f"{float(g_val['pairwise_hd95_std']):.2f}"
        if not (a in r and b in r and c in r):
            return False
        if per_val is None or len(predictions) > 8:
            return True
        idx = self._find_prediction_index(predictions, primary_selected_name)
        if idx is None or idx >= len(per_val):
            return True
        d1 = f"{float(per_val[idx]['mean_hd95_to_others']):.2f}"
        d2 = f"{float(per_val[idx]['area_rel_to_group']):.4f}"
        return d1 in r and d2 in r

    def _inject_disagreement_if_missing(
        self,
        reasoning: str,
        g_val: Optional[Dict[str, Any]],
        per_val: Optional[List[Dict[str, float]]],
        predictions: List[ModelOutput],
        primary_selected_name: str,
    ) -> str:
        """Append compact numeric suffix if the model omitted citations (auditability; no extra prompt tokens)."""
        if not g_val:
            return reasoning
        if self._reasoning_includes_disagreement_numbers(
            reasoning, g_val, per_val, predictions, primary_selected_name
        ):
            return reasoning
        tail = (
            f"【分歧补充】area_cv={g_val['area_cv']}, pairwise_hd95_mean={g_val['pairwise_hd95_mean']}, "
            f"pairwise_hd95_std={g_val['pairwise_hd95_std']}"
        )
        idx = self._find_prediction_index(predictions, primary_selected_name)
        if per_val is not None and idx is not None and idx < len(per_val):
            tail += (
                f"；所选模型mean_hd95_to_others={per_val[idx]['mean_hd95_to_others']}, "
                f"area_rel_to_group={per_val[idx]['area_rel_to_group']}"
            )
        base = reasoning.rstrip()
        return (base + " " + tail) if base else tail

    @staticmethod
    def _has_citable_input_data(info: Optional[Dict[str, Any]]) -> bool:
        """True if config.data has at least one non-empty / non-unknown field worth citing in reasoning."""

        def _walk(x: Any) -> bool:
            if x is None:
                return False
            if isinstance(x, bool):
                return True
            if isinstance(x, (int, float)):
                if isinstance(x, float) and np.isnan(x):
                    return False
                return True
            if isinstance(x, str):
                return bool(x.strip())
            if isinstance(x, list):
                return any(_walk(i) for i in x)
            if isinstance(x, dict):
                return any(_walk(v) for v in x.values())
            return False

        return _walk(info) if info else False

    @staticmethod
    def _has_citable_device_info(info: Optional[List[str]]) -> bool:
        if not info:
            return False
        return any(str(s).strip() for s in info)

    def format_predictions_for_agent(
        self,
        predictions: List[ModelOutput],
        quality_results: Dict[str, Any],
        input_device_info: Optional[List[str]] = None,
        input_data_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format predictions and quality metrics for the agent
        
        Args:
            predictions: List of ModelOutput from different models
            quality_results: Quality evaluation results
            input_device_info: Optional device information for input data
            
        Returns:
            Formatted JSON string
        """
        data = {
            "num_models": len(predictions),
            "input_device_info": input_device_info if input_device_info else "未知 (null)",
            "input_data_info": self._normalize_unknown_fields(input_data_info or {}),
            "base_datasets_info": self.base_datasets_info,
            "models": []
        }
        
        am = quality_results.get("agreement_metrics") or {}
        n_models = len(predictions)
        group_uncertainty: Optional[Dict[str, Any]] = None
        per_disagreement: Optional[List[Dict[str, float]]] = None
        if self.include_disagreement_metrics_in_prompt:
            group_uncertainty, per_disagreement = self._compact_disagreement_fields(am, n_models)
            if group_uncertainty is not None:
                data["group_uncertainty"] = group_uncertainty
        
        # Add each model's information
        for idx, pred in enumerate(predictions):
            quality = quality_results['individual_quality'][idx]
            metadata = pred.metadata or {}
            
            model_info = {
                "model_name": pred.model_name,
                "training_devices": metadata.get('training_data_devices', []),
                "mask_statistics": {
                    "area": quality['area'],
                    "is_single_component": quality['is_single_component'],
                    "num_components": quality['num_components'],
                    "circularity": round(quality['circularity'], 2),
                    "smoothness": round(quality['smoothness'], 2),
                    "compactness": round(quality['compactness'], 2),
                    "solidity": round(quality['solidity'], 2),
                    "aspect_ratio": round(quality['aspect_ratio'], 2)
                },
                "agreement_with_others": round(quality_results['agreement_metrics']['average_agreement'][idx], 2) if quality_results['agreement_metrics']['average_agreement'] else 0.0
            }
            if per_disagreement is not None and idx < len(per_disagreement):
                model_info["disagreement"] = per_disagreement[idx]
            
            # Add base dataset performance if available
            if 'base_dataset_performance' in metadata and metadata['base_dataset_performance']:
                model_info["base_dataset_performance"] = {}
                for dataset_name, perf in metadata['base_dataset_performance'].items():
                    model_info["base_dataset_performance"][dataset_name] = {
                        "dice": round(perf.get('dice', 0.0), 2),
                        "hd95": round(perf.get('hd95', float('inf')), 2)
                    }
            
            # Add dataset info if available
            if 'dataset_info' in metadata and metadata['dataset_info']:
                dataset_info = metadata['dataset_info']
                model_info["dataset_info"] = {
                    "training_dataset": dataset_info.get('training_dataset', ''),
                    "base_datasets": dataset_info.get('base_datasets', []),
                    "dataset_size": dataset_info.get('dataset_size', 0)
                }
            
            # Add confidence if available
            if pred.confidence_map is not None:
                mask_pixels = pred.mask > 0
                if np.sum(mask_pixels) > 0:
                    mean_conf = float(np.mean(pred.confidence_map[mask_pixels]))
                    model_info["mean_confidence"] = round(mean_conf, 2)
            
            data["models"].append(model_info)
        
        # Add overall agreement
        data["overall_agreement"] = round(quality_results['agreement_metrics']['overall_agreement'], 2)

        # Short checklist (详细分歧数值见 select_best_mask 首段【分歧摘要】，避免 JSON 重复长列表占 token)
        req: Dict[str, Any] = {}
        if group_uncertainty is not None:
            req["follow_prefix"] = (
                "对话首段【分歧摘要】【各模型】与下方 group_uncertainty/models[].disagreement 数值一致；"
                "reasoning 须含摘要三数及所选模型两项 disagreement。"
            )
        cite_input = self._has_citable_input_data(input_data_info) or self._has_citable_device_info(
            input_device_info
        )
        if cite_input:
            req["must_mention_input_context"] = (
                "reasoning 中用短语引用 input_device_info 或 input_data_info 至少一项非未知内容。"
            )
        if req:
            data["reasoning_requirements"] = req

        return json.dumps(data, indent=None, separators=(',', ':'), ensure_ascii=False)

    def _normalize_unknown_fields(self, value: Any) -> Any:
        """
        Recursively normalize null/empty input fields for prompt readability.

        By convention in config, null means unknown.
        """
        if value is None:
            return "未知 (null)"
        if isinstance(value, dict):
            return {k: self._normalize_unknown_fields(v) for k, v in value.items()}
        if isinstance(value, list):
            if len(value) == 0:
                return "未知 (空列表)"
            return [self._normalize_unknown_fields(v) for v in value]
        if isinstance(value, str) and value.strip() == "":
            return "未知 (空字符串)"
        return value
    
    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON object from text, handling markdown and other formats
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON string
            
        Raises:
            ValueError: If no valid JSON object can be extracted
        """
        import re
        
        # First, try to extract from markdown code blocks (highest priority)
        if "```json" in text:
            extracted = text.split("```json")[1].split("```")[0].strip()
            # Validate it looks like JSON
            if extracted.startswith('{') and '"selected_model"' in extracted:
                return extracted
        elif "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith('{') and '"selected_model"' in part:
                    # Try to extract complete JSON
                    json_match = self._extract_complete_json(part)
                    if json_match:
                        return json_match
        
        # Remove all markdown formatting and thinking process
        # Remove *Thinking...* blocks (more aggressive)
        text = re.sub(r'\*Thinking[^*]*\*', '', text, flags=re.DOTALL)
        
        # Remove markdown blockquotes (entire lines starting with >)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip markdown blockquotes, headers, and empty lines
            if stripped.startswith('>') or stripped.startswith('#') or (not cleaned_lines and not stripped):
                continue
            cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)
        
        # Remove markdown headers and other formatting
        text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\*\*.*?\*\*', '', text, flags=re.MULTILINE)
        
        # Strategy 1: Look for JSON object containing "selected_model" or "selected_models" field
        # This is the most reliable indicator
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"(?:selected_model|selected_models)"[^}]*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        for match in matches:
            candidate = match.group(0)
            # Try to extract complete JSON from this candidate
            complete_json = self._extract_complete_json(candidate)
            if complete_json:
                return complete_json
        
        # Strategy 2: Find all potential JSON objects and validate them
        # Look for pattern: {"selected_model": ...} or {"selected_models": ...}
        pattern = r'\{[^}]*"(?:selected_model|selected_models)"[^}]*\}'
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            # Expand to find complete JSON object
            start_pos = match.start()
            complete_json = self._extract_complete_json_from_position(text, start_pos)
            if complete_json:
                return complete_json
        
        # Strategy 3: Find first { and try to extract complete JSON
        start_idx = text.find('{')
        if start_idx != -1:
            complete_json = self._extract_complete_json_from_position(text, start_idx)
            if complete_json and ('"selected_model"' in complete_json or '"selected_models"' in complete_json):
                return complete_json
        
        # If nothing found, raise error
        raise ValueError(f"无法从响应中提取有效的JSON对象。响应内容: {text[:500]}")
    
    def _extract_complete_json(self, text: str) -> Optional[str]:
        """
        Extract complete JSON object from text starting at the first {
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Complete JSON string or None if not found
        """
        import re
        
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        return self._extract_complete_json_from_position(text, start_idx)
    
    def _extract_complete_json_from_position(self, text: str, start_pos: int) -> Optional[str]:
        """
        Extract complete JSON object from text starting at given position
        
        Args:
            text: Full text
            start_pos: Position of first {
            
        Returns:
            Complete JSON string or None if not found
        """
        import re
        
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start_pos, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[start_pos:i+1]
                        # Clean control characters
                        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                        # Remove any markdown formatting
                        json_str = re.sub(r'^\s*[>#*]+\s*', '', json_str, flags=re.MULTILINE)
                        # Validate it contains required fields
                        if '"selected_model"' in json_str or '"selected_models"' in json_str:
                            return json_str.strip()
                        return None
        
        return None
    
    def select_best_mask(
        self,
        predictions: List[ModelOutput],
        gt_mask: Optional[np.ndarray] = None,
        input_device_info: Optional[List[str]] = None,
        input_data_info: Optional[Dict[str, Any]] = None,
    ) -> AgentDecision:
        """
        Use LLM agent to select the best segmentation mask
        
        Args:
            predictions: List of ModelOutput from different models
            gt_mask: Optional ground truth mask for performance evaluation (does not affect decision)
            input_device_info: Optional list of device information for input data
            
        Returns:
            AgentDecision with the selected mask and reasoning
        """
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Extract masks
        masks = [pred.mask for pred in predictions]
        model_names = [pred.model_name for pred in predictions]
        
        # Check and normalize mask sizes for fair comparison
        # All masks should be the same size, but we normalize to ensure consistency
        mask_shapes = [mask.shape[:2] for mask in masks]
        unique_shapes = set(mask_shapes)
        
        if len(unique_shapes) > 1:
            # If masks have different sizes, normalize to the first mask's size
            target_shape = mask_shapes[0]
            print(f"⚠️  检测到不同尺寸的mask，统一到 {target_shape} 进行对比")
            normalized_masks = []
            for i, (mask, name) in enumerate(zip(masks, model_names)):
                if mask.shape[:2] != target_shape:
                    normalized_mask = cv2.resize(
                        mask.astype(np.uint8),
                        (target_shape[1], target_shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(mask.dtype)
                    normalized_masks.append(normalized_mask)
                    print(f"    模型 {name}: {mask.shape[:2]} -> {target_shape}")
                else:
                    normalized_masks.append(mask)
            masks = normalized_masks
        
        # Evaluate quality
        quality_results = self.quality_evaluator.evaluate_batch(masks, model_names)

        g_val: Optional[Dict[str, Any]] = None
        per_val: Optional[List[Dict[str, float]]] = None
        if self.include_disagreement_metrics_in_prompt:
            g_val, per_val = self._compact_disagreement_fields(
                quality_results.get("agreement_metrics") or {}, len(predictions)
            )
        disagreement_prefix = ""
        if self.include_disagreement_metrics_in_prompt and g_val is not None:
            disagreement_prefix = self._build_disagreement_prefix_lines(g_val, per_val, predictions)
        
        # Format for agent
        formatted_data = self.format_predictions_for_agent(
            predictions, 
            quality_results,
            input_device_info,
            input_data_info
        )
        
        # Device info text
        device_info_text = ""
        if input_device_info:
            device_info_text = f"\n**输入数据设备信息**: {', '.join(input_device_info)}\n"
        else:
            device_info_text = "\n**输入数据设备信息**: 未知 (null)\n"

        normalized_input_data_info = self._normalize_unknown_fields(input_data_info or {})
        input_data_text = (
            "\n**输入数据元信息（来自 config.data，null 表示未知）**:\n"
            f"{json.dumps(normalized_input_data_info, ensure_ascii=False)}\n"
        )
        
        # Construct prompt based on ensemble mode
        if self.ensemble_enabled and self.ensemble_top_k > 1:
            prompt_question = f"基于上述信息，请选择Top {self.ensemble_top_k}个最佳模型用于ensemble融合。"
            if self.ensemble_top_k == 2:
                prompt_format = '{"selected_models": ["模型1", "模型2"], "weights": [0.6, 0.4], "confidence": 0.95, "reasoning": "3-4句含数值，约120-180字"}'
            else:
                prompt_format = '{"selected_models": ["模型1", "模型2", "模型3"], "weights": [0.5, 0.3, 0.2], "confidence": 0.95, "reasoning": "3-4句含数值，约120-180字"}'
        else:
            prompt_question = "基于上述信息，哪个模型提供了最佳的分割结果？"
            prompt_format = '{"selected_model": "模型名称", "confidence": 0.95, "reasoning": "3-4句含数值，约120-180字"}'

        req_line = ""
        if self.include_disagreement_metrics_in_prompt:
            try:
                parsed = json.loads(formatted_data)
                rr = parsed.get("reasoning_requirements") if isinstance(parsed, dict) else None
                if isinstance(rr, dict) and rr:
                    parts: List[str] = ["上文 JSON 含 reasoning_requirements 时，reasoning 必须逐条满足。"]
                    if rr.get("follow_prefix"):
                        parts.append("须落实首段【分歧摘要】三数与【各模型】中对应项；")
                    if rr.get("must_mention_input_context"):
                        parts.append("须按 must_mention_input_context 引用输入侧信息。")
                    req_line = "\n【重要】" + " ".join(parts) + "\n"
            except json.JSONDecodeError:
                pass

        user_prompt = f"""{disagreement_prefix}{device_info_text}{input_data_text}
以下是来自 {len(predictions)} 个不同分割模型的输出掩码及其质量评估（结构化JSON，字段含模型名称、形态学指标、一致性、跨模型分歧摘要、训练数据集性能等）：

{formatted_data}

{prompt_question}
{req_line}
只输出纯JSON、无前缀后缀，reasoning用中文写若干句（若含 reasoning_requirements 则约150–220字，否则约120–180字，均须含具体数值）。格式示例：{prompt_format}"""
        if self.backend_type == "local_gpt_oss":
            user_prompt += "\n请把最终JSON放在 final 通道；analysis/commentary 通道不要输出最终JSON。"
        
        response_text = None
        try:
            # Call selected backend, with retry on API rate limits (for llm mode)
            last_error = None
            for attempt in range(self.max_retries + 1):
                try:
                    response_text = self._call_chat_completion(
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_json=True,
                    )
                    if not response_text:
                        print("✗ LLM 后端返回空内容，使用降级选择")
                        return self._fallback_selection(predictions, gt_mask, quality_results)
                    last_error = None
                    break
                except Exception as e:
                    is_rate_limit = (
                        self.backend_type == "llm"
                        and (
                            (RateLimitError is not None and isinstance(e, RateLimitError))
                            or "429" in str(e)
                            or "速率限制" in str(e)
                        )
                    )
                    if is_rate_limit:
                        last_error = e
                        if attempt < self.max_retries:
                            wait_sec = min(60, (2 ** attempt) + 1)
                            print(f"⚠️  速率限制(429)，{wait_sec}s 后重试 ({attempt + 1}/{self.max_retries})...")
                            time.sleep(wait_sec)
                        else:
                            raise
                    else:
                        raise
            if last_error is not None:
                raise last_error
            
            # Extract JSON
            try:
                json_text = self._extract_json_from_text(response_text)
            except ValueError as extract_error:
                # Attempt one "repair" roundtrip: ask the model to convert its own output to strict JSON.
                repaired_json_text: Optional[str] = None
                if response_text and len(response_text) > 0:
                    repair_prompt = (
                        "把下面这段内容转换为【严格的、可解析的纯JSON对象】。\n"
                        "要求：只输出JSON本体（首字符{末字符}），不要Markdown/解释/代码块。\n"
                        'JSON字段必须包含：selected_model 或 selected_models、confidence、reasoning。\n\n'
                        f"原始内容：\n{response_text}"
                    )
                    try:
                        repair_text = self._call_chat_completion(
                            messages=[
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": repair_prompt},
                            ],
                            temperature=0,
                            max_tokens=min(512, self.max_tokens),
                            response_json=True,
                        )
                        repaired_json_text = self._extract_json_from_text(repair_text)
                    except Exception:
                        repaired_json_text = None

                if repaired_json_text:
                    json_text = repaired_json_text
                else:
                    # Failed to extract JSON from response (and repair attempt failed)
                    print(f"✗ 无法从响应中提取JSON: {extract_error}")
                    if response_text:
                        print(f"   原始响应内容 (前500字符): {response_text[:500]}")
                    print("   使用降级选择（选择一致性最高的掩码）")
                    return self._fallback_selection(predictions, gt_mask, quality_results)
            
            # Try to parse JSON, with better error reporting
            try:
                decision_data = json.loads(json_text)
            except json.JSONDecodeError as parse_error:
                # Log the extracted text for debugging
                print(f"⚠️  提取的JSON文本无法解析:")
                print(f"   前200字符: {json_text[:200]}")
                print(f"   解析错误位置: line {parse_error.lineno}, col {parse_error.colno}")
                raise
            
            # Validate required fields
            is_ensemble_mode = "selected_models" in decision_data
            if not is_ensemble_mode and "selected_model" not in decision_data:
                raise ValueError("响应中缺少 'selected_model' 或 'selected_models' 字段")
            if "confidence" not in decision_data:
                raise ValueError("响应中缺少 'confidence' 字段")
            if "reasoning" not in decision_data:
                raise ValueError("响应中缺少 'reasoning' 字段")
            
            # Validate and normalize confidence
            try:
                confidence_value = float(decision_data["confidence"])
                # Clamp confidence to [0, 1] range
                confidence_value = max(0.0, min(1.0, confidence_value))
            except (ValueError, TypeError):
                raise ValueError(f"confidence 值无效: {decision_data['confidence']}，必须是 0-1 之间的数字")
            
            # Validate reasoning is not empty
            reasoning_text = str(decision_data["reasoning"]).strip()
            if not reasoning_text:
                raise ValueError("reasoning 字段不能为空")
            
            # Handle ensemble mode or single model mode
            if is_ensemble_mode and self.ensemble_enabled:
                # Ensemble mode: multiple models
                selected_model_names = decision_data["selected_models"]
                if not isinstance(selected_model_names, list) or len(selected_model_names) == 0:
                    raise ValueError("selected_models 必须是一个非空数组")
                
                # Limit to top_k
                selected_model_names = selected_model_names[:self.ensemble_top_k]
                
                # Get weights (if provided, otherwise use equal weights)
                if "weights" in decision_data and decision_data["weights"]:
                    weights = [float(w) for w in decision_data["weights"][:len(selected_model_names)]]
                    # Normalize weights
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]
                    else:
                        weights = [1.0 / len(weights)] * len(weights)
                else:
                    weights = [1.0 / len(selected_model_names)] * len(selected_model_names)
                
                # Find selected predictions
                selected_preds = []
                selected_indices = []
                for model_name in selected_model_names:
                    found = False
                    for idx, pred in enumerate(predictions):
                        if pred.model_name == model_name or pred.model_name.strip().lower() == model_name.strip().lower():
                            selected_preds.append(pred)
                            selected_indices.append(idx)
                            found = True
                            break
                    if not found:
                        available_models = [pred.model_name for pred in predictions]
                        raise ValueError(f"选择的模型 '{model_name}' 不在预测列表中。可用模型: {available_models}")
                
                # Check if all selected models have confidence maps
                prob_maps = []
                for pred in selected_preds:
                    if pred.confidence_map is None:
                        print(f"⚠️  警告: 模型 {pred.model_name} 没有概率图，将使用二值掩码生成概率图")
                        # Create a simple probability map from mask
                        prob_map = pred.mask.astype(np.float32)
                        prob_maps.append(prob_map)
                    else:
                        prob_maps.append(pred.confidence_map)
                
                # Ensemble probability maps
                ensemble_prob_map = self._ensemble_probability_maps(
                    prob_maps, 
                    weights, 
                    method=self.ensemble_method
                )
                
                # Generate final mask from ensemble probability map
                final_mask = self._generate_mask_from_probability(
                    ensemble_prob_map,
                    threshold=self.ensemble_threshold
                )
                
                # Use first selected model for quality metrics (for backward compatibility)
                selected_model_name = selected_model_names[0]
                selected_idx = selected_indices[0]
                selected_quality = quality_results['individual_quality'][selected_idx]
                agreement_score = quality_results['agreement_metrics']['average_agreement'][selected_idx] if quality_results['agreement_metrics']['average_agreement'] else None
                
                # Calculate performance metrics if GT is provided
                dice_score = None
                hd95_score = None
                if gt_mask is not None:
                    try:
                        if final_mask.shape != gt_mask.shape:
                            gt_mask_resized = cv2.resize(
                                gt_mask.astype(np.uint8),
                                (final_mask.shape[1], final_mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(gt_mask.dtype)
                            dice_score = compute_dice(final_mask, gt_mask_resized)
                            hd95_score = compute_hd95(final_mask, gt_mask_resized)
                        else:
                            dice_score = compute_dice(final_mask, gt_mask)
                            hd95_score = compute_hd95(final_mask, gt_mask)
                    except Exception as e:
                        print(f"⚠️  计算性能指标失败: {e}")
                
                # Create AgentDecision with ensemble information
                ece_metrics = None
                if gt_mask is not None:
                    try:
                        gt_mask_for_ece = gt_mask
                        if final_mask.shape != gt_mask.shape:
                            gt_mask_for_ece = cv2.resize(
                                gt_mask.astype(np.uint8),
                                (final_mask.shape[1], final_mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(gt_mask.dtype)
                        ece_metrics = self._compute_ece_metrics_from_prob_map(ensemble_prob_map, gt_mask_for_ece)
                    except Exception:
                        ece_metrics = None
                reasoning_text = self._inject_disagreement_if_missing(
                    reasoning_text, g_val, per_val, predictions, selected_model_name
                )
                decision = AgentDecision(
                    selected_model=selected_model_name,
                    selected_mask=final_mask,
                    confidence=confidence_value,
                    reasoning=reasoning_text,
                    all_predictions=[pred.to_dict() for pred in predictions],
                    quality_metrics=selected_quality,
                    agreement_score=agreement_score,
                    dice_score=dice_score,
                    hd95_score=hd95_score,
                    ece_metrics=ece_metrics,
                    selected_models=selected_model_names,
                    model_weights=weights,
                    is_ensemble=True
                )
                
                print(f"✓ Ensemble模式: 融合了 {len(selected_model_names)} 个模型")
                print(f"  模型: {', '.join(selected_model_names)}")
                print(f"  权重: {[f'{w:.3f}' for w in weights]}")
                
            else:
                # Single model mode
                selected_model_name = str(decision_data.get("selected_model", decision_data.get("selected_models", [""])[0] if "selected_models" in decision_data else "")).strip()
                selected_pred = None
                selected_idx = None
                
                # Try exact match first
                for idx, pred in enumerate(predictions):
                    if pred.model_name == selected_model_name:
                        selected_pred = pred
                        selected_idx = idx
                        break
                
                # If not found, try case-insensitive and whitespace-tolerant matching
                if selected_pred is None:
                    for idx, pred in enumerate(predictions):
                        pred_name_normalized = pred.model_name.strip().lower()
                        selected_name_normalized = selected_model_name.lower()
                        if pred_name_normalized == selected_name_normalized:
                            selected_pred = pred
                            selected_idx = idx
                            selected_model_name = pred.model_name
                            break
                
                if selected_pred is None:
                    available_models = [pred.model_name for pred in predictions]
                    raise ValueError(f"选择的模型 '{selected_model_name}' 不在预测列表中。可用模型: {available_models}")
                
                # Get quality metrics for selected mask
                selected_quality = quality_results['individual_quality'][selected_idx]
                agreement_score = quality_results['agreement_metrics']['average_agreement'][selected_idx] if quality_results['agreement_metrics']['average_agreement'] else None
                
                # Calculate performance metrics if GT is provided
                dice_score = None
                hd95_score = None
                if gt_mask is not None:
                    try:
                        pred_mask = selected_pred.mask
                        if pred_mask.shape != gt_mask.shape:
                            gt_mask_resized = cv2.resize(
                                gt_mask.astype(np.uint8),
                                (pred_mask.shape[1], pred_mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(gt_mask.dtype)
                            dice_score = compute_dice(pred_mask, gt_mask_resized)
                            hd95_score = compute_hd95(pred_mask, gt_mask_resized)
                        else:
                            dice_score = compute_dice(pred_mask, gt_mask)
                            hd95_score = compute_hd95(pred_mask, gt_mask)
                    except Exception as e:
                        print(f"⚠️  计算性能指标失败: {e}")
                
                # Create AgentDecision
                ece_metrics = None
                if gt_mask is not None:
                    try:
                        ece_metrics = self._extract_ece_metrics_from_model_output(selected_pred, gt_mask) if selected_pred else None
                    except Exception:
                        ece_metrics = None
                reasoning_text = self._inject_disagreement_if_missing(
                    reasoning_text, g_val, per_val, predictions, selected_model_name
                )
                decision = AgentDecision(
                    selected_model=selected_model_name,
                    selected_mask=selected_pred.mask,
                    confidence=confidence_value,
                    reasoning=reasoning_text,
                    all_predictions=[pred.to_dict() for pred in predictions],
                    quality_metrics=selected_quality,
                    agreement_score=agreement_score,
                    dice_score=dice_score,
                    hd95_score=hd95_score,
                    ece_metrics=ece_metrics,
                    is_ensemble=False
                )
            
            return decision
            
        except (json.JSONDecodeError, ValueError) as e:
            # ValueError: failed to extract JSON, JSONDecodeError: failed to parse extracted JSON
            error_type = "提取" if isinstance(e, ValueError) else "解析"
            print(f"✗ 无法{error_type} LLM 响应为 JSON: {e}")
            if response_text:
                print(f"   原始响应内容 (前500字符): {response_text[:500]}")
            print("   使用降级选择（选择一致性最高的掩码）")
            return self._fallback_selection(predictions, gt_mask, quality_results)
        
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "速率限制" in err_msg:
                print(f"✗ LLM 后端速率限制或调用失败，已达最大重试次数: {e}")
            else:
                print(f"✗ 调用 LLM 后端失败: {e}")
                import traceback
                traceback.print_exc()
            if response_text:
                print(f"   响应内容 (前500字符): {response_text[:500]}")
            print("   使用降级选择（选择一致性最高的掩码）")
            return self._fallback_selection(predictions, gt_mask, quality_results)
    
    def _fallback_selection(
        self, 
        predictions: List[ModelOutput],
        gt_mask: Optional[np.ndarray],
        quality_results: Dict[str, Any]
    ) -> AgentDecision:
        """
        Fallback method to select best mask when LLM backend fails or is unavailable
        Selects the mask with highest agreement score
        
        Args:
            predictions: List of ModelOutput
            gt_mask: Optional ground truth mask
            quality_results: Quality evaluation results
            
        Returns:
            AgentDecision with highest agreement mask
        """
        # Get agreement scores
        agreement_scores = quality_results['agreement_metrics']['average_agreement']
        
        if agreement_scores is None or len(agreement_scores) == 0:
            # If no agreement scores, use first model(s)
            if self.ensemble_enabled and self.ensemble_top_k > 1:
                best_indices = list(range(min(self.ensemble_top_k, len(predictions))))
            else:
                best_indices = [0]
        else:
            # Select top_k models with highest agreement
            if self.ensemble_enabled and self.ensemble_top_k > 1:
                top_k = min(self.ensemble_top_k, len(predictions))
                best_indices = np.argsort(agreement_scores)[-top_k:][::-1].tolist()
            else:
                best_indices = [int(np.argmax(agreement_scores))]
        
        best_idx = best_indices[0]
        best_pred = predictions[best_idx]
        selected_quality = quality_results['individual_quality'][best_idx]
        agreement_score = agreement_scores[best_idx] if agreement_scores is not None else None
        
        # Calculate performance metrics if GT is provided
        dice_score = None
        hd95_score = None
        if gt_mask is not None:
            try:
                # Ensure GT mask and predicted mask have the same size
                pred_mask = best_pred.mask
                if pred_mask.shape != gt_mask.shape:
                    # Resize GT mask to match predicted mask size
                    import cv2
                    gt_mask_resized = cv2.resize(
                        gt_mask.astype(np.uint8),
                        (pred_mask.shape[1], pred_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(gt_mask.dtype)
                    dice_score = compute_dice(pred_mask, gt_mask_resized)
                    hd95_score = compute_hd95(pred_mask, gt_mask_resized)
                else:
                    dice_score = compute_dice(pred_mask, gt_mask)
                    hd95_score = compute_hd95(pred_mask, gt_mask)
            except Exception as e:
                print(f"⚠️  计算性能指标失败: {e}")
        
        # Handle ensemble mode in fallback
        if self.ensemble_enabled and self.ensemble_top_k > 1 and len(predictions) > 1:
            # Use the already selected best_indices
            selected_preds = [predictions[idx] for idx in best_indices]
            selected_model_names = [pred.model_name for pred in selected_preds]
            
            # Use equal weights for fallback
            weights = [1.0 / len(selected_preds)] * len(selected_preds)
            
            # Get probability maps
            prob_maps = []
            for pred in selected_preds:
                if pred.confidence_map is None:
                    prob_map = pred.mask.astype(np.float32)
                    prob_maps.append(prob_map)
                else:
                    prob_maps.append(pred.confidence_map)
            
            # Ensemble probability maps
            ensemble_prob_map = self._ensemble_probability_maps(
                prob_maps,
                weights,
                method=self.ensemble_method
            )
            
            # Generate final mask
            final_mask = self._generate_mask_from_probability(
                ensemble_prob_map,
                threshold=self.ensemble_threshold
            )
            
            # Format reasoning
            if agreement_score is not None:
                reasoning = f"降级选择：LLM后端调用失败，选择一致性得分最高的{len(selected_preds)}个模型进行ensemble融合（平均IoU: {agreement_score:.2f}）"
            else:
                reasoning = f"降级选择：LLM后端调用失败，选择前{len(selected_preds)}个模型进行ensemble融合"
            
            ece_metrics = None
            if gt_mask is not None:
                try:
                    gt_mask_for_ece = gt_mask
                    if ensemble_prob_map.shape != gt_mask.shape:
                        gt_mask_for_ece = cv2.resize(
                            gt_mask.astype(np.uint8),
                            (ensemble_prob_map.shape[1], ensemble_prob_map.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(gt_mask.dtype)
                    ece_metrics = self._compute_ece_metrics_from_prob_map(ensemble_prob_map, gt_mask_for_ece)
                except Exception:
                    ece_metrics = None
            
            return AgentDecision(
                selected_model=selected_model_names[0],
                selected_mask=final_mask,
                confidence=agreement_score if agreement_score else 0.5,
                reasoning=reasoning,
                all_predictions=[pred.to_dict() for pred in predictions],
                quality_metrics=selected_quality,
                agreement_score=agreement_score,
                dice_score=dice_score,
                hd95_score=hd95_score,
                ece_metrics=ece_metrics,
                selected_models=selected_model_names,
                model_weights=weights,
                is_ensemble=True
            )
        else:
            # Single model mode
            # Format reasoning with safe handling of None agreement_score
            if agreement_score is not None:
                reasoning = f"降级选择：选择与其他模型一致性最高的掩码（平均IoU: {agreement_score:.2f}），该掩码形态学质量良好"
            else:
                reasoning = "降级选择：选择第一个模型（无法计算一致性得分），该掩码形态学质量良好"
            
            ece_metrics = None
            if gt_mask is not None:
                try:
                    ece_metrics = self._extract_ece_metrics_from_model_output(best_pred, gt_mask) if best_pred else None
                except Exception:
                    ece_metrics = None
            
            return AgentDecision(
                selected_model=best_pred.model_name,
                selected_mask=best_pred.mask,
                confidence=agreement_score if agreement_score else 0.5,
                reasoning=reasoning,
                all_predictions=[pred.to_dict() for pred in predictions],
                quality_metrics=selected_quality,
                agreement_score=agreement_score,
                dice_score=dice_score,
                hd95_score=hd95_score,
                ece_metrics=ece_metrics,
                is_ensemble=False
            )
    
    def select_best_masks_batch(
        self,
        batch_predictions: List[List[ModelOutput]],
        gt_masks: Optional[List[Optional[np.ndarray]]] = None,
        input_device_info: Optional[List[str]] = None,
        input_data_info: Optional[Dict[str, Any]] = None,
        batch_size: int = 10
    ) -> List[AgentDecision]:
        """
        Batch select best masks for multiple images to reduce token usage
        
        Args:
            batch_predictions: List of predictions for each image (each is a list of ModelOutput)
            gt_masks: Optional list of ground truth masks (one per image, can be None)
            input_device_info: Optional device information for input data
            batch_size: Number of images to process in each batch (default: 10)
            
        Returns:
            List of AgentDecision, one for each image
        """
        if gt_masks is None:
            gt_masks = [None] * len(batch_predictions)
        
        all_decisions = []
        
        # Process in batches
        for batch_start in range(0, len(batch_predictions), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_predictions))
            batch_indices = list(range(batch_start, batch_end))
            
            print(f"  处理批量 {batch_start//batch_size + 1} (图像 {batch_start+1}-{batch_end}/{len(batch_predictions)})...")
            
            # Process each image in the batch sequentially
            for idx in batch_indices:
                predictions = batch_predictions[idx]
                gt_mask = gt_masks[idx] if idx < len(gt_masks) else None
                
                try:
                    decision = self.select_best_mask(
                        predictions,
                        gt_mask=gt_mask,
                        input_device_info=input_device_info,
                        input_data_info=input_data_info
                    )
                    all_decisions.append(decision)
                except (TimeoutError, KeyboardInterrupt) as e:
                    # For timeout or interrupt, use fallback immediately
                    print(f"  ⚠️  图像 {idx+1} API 超时或中断，使用降级选择")
                    if predictions:
                        masks = [pred.mask for pred in predictions]
                        model_names = [pred.model_name for pred in predictions]
                        quality_results = self.quality_evaluator.evaluate_batch(masks, model_names)
                        fallback_decision = self._fallback_selection(
                            predictions,
                            gt_mask,
                            quality_results
                        )
                        all_decisions.append(fallback_decision)
                    else:
                        raise
                except Exception as e:
                    error_str = str(e)
                    # Check if it's a provider error that we should handle quickly
                    is_provider_error = 'provider' in error_str.lower() or 'Error from provider' in error_str
                    if is_provider_error:
                        print(f"  ⚠️  图像 {idx+1} provider 错误，使用降级选择: {error_str[:100]}")
                    else:
                        print(f"  ✗ 处理图像 {idx+1} 失败: {e}")
                    
                    # Create a fallback decision
                    if predictions:
                        # Use fallback selection
                        masks = [pred.mask for pred in predictions]
                        model_names = [pred.model_name for pred in predictions]
                        quality_results = self.quality_evaluator.evaluate_batch(masks, model_names)
                        fallback_decision = self._fallback_selection(
                            predictions,
                            gt_mask,
                            quality_results
                        )
                        all_decisions.append(fallback_decision)
                    else:
                        raise
                
                # Add small delay between requests to avoid rate limiting
                # Only delay if not the last image in batch
                if idx < batch_end - 1:
                    import time
                    time.sleep(0.5)  # 0.5 second delay between requests
        
        return all_decisions
