"""
RadiomicsJudge：GT-trained AutoGluon radiomics 模型作为分割质量裁判。

对每个 pred mask，输出分类置信度 + 特征摘要 + 马氏距离，
作为分割 Agent LLM 选择 mask 的独立信号。

待做文档任务1核心：用 GT 分割结果训练的 radiomics 模型衡量 pred mask 可信度。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from .feature_extractor import RadiomicsFeatureExtractor
from .feature_summary import FeatureSummarizer


class RadiomicsJudge:
    """
    GT-trained radiomics 裁判。

    用 GT mask 训练的 AutoGluon 模型评估 pred mask 的可信度。
    直觉：分割越准确，radiomics 特征越接近训练分布，分类置信度越合理。
    """

    def __init__(
        self,
        model_dir: str,
        top_k_features: int = 5,
        shap_reference_path: Optional[str] = None,
        radiomics_params: Optional[str] = None,
    ):
        """
        Args:
            model_dir: GT-trained AutoGluon TabularPredictor 模型目录。
            top_k_features: 返回的 top 特征数。
            shap_reference_path: SHAP 参考文件路径（特征重要性 + 训练集统计量）。
            radiomics_params: pyradiomics YAML 参数文件路径。
                              None 时使用 RadiomicsFeatureExtractor 默认路径
                              （与 pyradiomics_train/radiomics_2d.yaml 一致的配置）。
        """
        self.model_dir = model_dir
        self._predictor = None
        self._extractor = RadiomicsFeatureExtractor(params_path=radiomics_params)
        self._summarizer = FeatureSummarizer(top_k_features, shap_reference_path)
        self._feature_names: Optional[list[str]] = None

    def _ensure_loaded(self) -> None:
        """延迟加载 AutoGluon 模型。"""
        if self._predictor is not None:
            return

        path = Path(self.model_dir)
        if not path.exists():
            raise FileNotFoundError(f"裁判模型目录不存在: {path}")

        from autogluon.tabular import TabularPredictor

        self._predictor = TabularPredictor.load(str(path), require_py_version_match=False)
        try:
            self._feature_names = list(self._predictor.feature_metadata_in.get_features())
        except Exception:
            self._feature_names = None

    def judge(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        image_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        评估单个 pred mask 的可信度。

        Args:
            image: (H, W, 3) RGB uint8 [0,255]。
            mask: (H, W) uint8 [0,255] 或 0/1。

        Returns:
            {
                'valid': bool,
                'predicted_class': int,       # 0=benign, 1=malignant
                'malignant_prob': float,
                'confidence': float,          # max(prob, 1-prob)
                'top_features': [...],        # SHAP top-K
            }
            空 mask 返回 {'valid': False, 'reason': 'empty mask'}。
        """
        self._ensure_loaded()

        # 提取特征
        features = self._extractor.extract(image, mask, image_name=image_name)
        if not features:
            return {"valid": False, "reason": "empty mask or feature extraction failed"}

        # AutoGluon 推理
        import pandas as pd

        df = pd.DataFrame([features])
        if self._feature_names:
            for col in self._feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df.reindex(columns=self._feature_names, fill_value=0.0)

        # as_multiclass=True 确保返回所有类别的概率列（与训练侧 infer.py 一致）
        pred_proba = self._predictor.predict_proba(
            df, as_pandas=True, as_multiclass=True
        )

        # 获取恶性概率
        class_labels = list(self._predictor.class_labels)
        proba_row = pred_proba.iloc[0]

        if len(class_labels) == 2:
            # 二分类：找恶性标签（1 / "1" / "malignant" / "M"）
            malignant_label = None
            for cl in class_labels:
                s = str(cl).lower()
                if s in ("1", "malignant", "m"):
                    malignant_label = cl
                    break
            if malignant_label is None:
                malignant_label = class_labels[-1]
            malignant_prob = float(proba_row[malignant_label])
        else:
            # 多分类：取最后一个类别（约定为最高风险等级）
            malignant_prob = float(proba_row[class_labels[-1]])

        # 特征摘要
        summary = self._summarizer.summarize(features)

        # 完整特征向量（用于阶段 1 预筛选的特征相似性计算）
        feature_vector = df.iloc[0].tolist() if self._feature_names else None

        return {
            "valid": True,
            "predicted_class": int(malignant_prob > 0.5),
            "malignant_prob": malignant_prob,
            "confidence": float(max(malignant_prob, 1 - malignant_prob)),
            "top_features": summary["top_features"],
            "feature_count": summary["feature_count"],
            "feature_vector": feature_vector,
        }

    def judge_batch(
        self,
        image: np.ndarray,
        masks: list[np.ndarray],
        image_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        对同一张图的多个 pred mask 批量裁判。

        Args:
            image: (H, W, 3) RGB uint8。
            masks: 多个 pred mask 的列表。

        Returns:
            每个 mask 的裁判结果列表。
        """
        return [self.judge(image, m, image_name=image_name) for m in masks]
