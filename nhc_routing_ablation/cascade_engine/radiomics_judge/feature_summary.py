"""
特征摘要：SHAP top-K 特征。

将 100+ 维 radiomics 特征压缩为 LLM 可读的摘要，
控制 prompt token 长度。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class FeatureSummarizer:
    """把高维 radiomics 特征压缩成 LLM 可读摘要。"""

    def __init__(
        self,
        top_k: int = 5,
        shap_reference_path: Optional[str] = None,
    ):
        """
        Args:
            top_k: 返回的 top 特征数。
            shap_reference_path: SHAP 参考文件路径（JSON），含：
                - feature_importance: {feature_name: shap_value}
        """
        self.top_k = top_k
        self._shap_importance: Optional[dict[str, float]] = None

        if shap_reference_path:
            self.load_shap_reference(shap_reference_path)

    def load_shap_reference(self, path: str) -> None:
        """加载 SHAP 特征重要性。"""
        p = Path(path)
        if not p.exists():
            print(f"  ⚠️ SHAP 参考文件不存在: {path}")
            return

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠️ SHAP 参考文件加载失败: {e}")
            return

        self._shap_importance = data.get("feature_importance", {})

    def summarize(self, features: dict[str, float]) -> dict[str, Any]:
        """
        生成特征摘要。

        Args:
            features: {feature_name: value}，来自 RadiomicsFeatureExtractor.extract。

        Returns:
            {
                'top_features': [{name, value, shap, direction}, ...],
                'feature_count': int,
            }
        """
        if not features:
            return {"top_features": [], "feature_count": 0}

        top_features = self._get_top_features(features)

        return {
            "top_features": top_features,
            "feature_count": len(features),
        }

    def _get_top_features(self, features: dict[str, float]) -> list[dict[str, Any]]:
        """按 SHAP 重要性选 top_k 特征；无 SHAP 时按特征值绝对值排序。"""
        if self._shap_importance:
            # 按 SHAP 重要性排序
            ranked = sorted(
                features.items(),
                key=lambda x: abs(self._shap_importance.get(x[0], 0)),
                reverse=True,
            )
        else:
            # 降级：按特征值绝对值排序
            ranked = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)

        result = []
        for name, value in ranked[: self.top_k]:
            shap_val = self._shap_importance.get(name, 0.0) if self._shap_importance else 0.0
            direction = "malignant" if shap_val > 0 else ("benign" if shap_val < 0 else "neutral")
            result.append({
                "name": name,
                "value": round(float(value), 4),
                "shap": round(float(shap_val), 4),
                "direction": direction,
            })
        return result
