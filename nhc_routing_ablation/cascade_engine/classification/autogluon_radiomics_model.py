"""
AutoGluon Radiomics 分类模型。

重写自 Classification_Agent/models/autogluon_radiomics_model.py。
requires_mask = True：输入 image + mask，用 pyradiomics 提特征 → AutoGluon 推理。

特征提取复用 radiomics_judge.RadiomicsFeatureExtractor，
确保与 pyradiomics_train 训练侧配置完全一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base_model import BaseClassificationModel, ClsModelOutput
from .model_factory import register_cls_model
from radiomics_judge.feature_extractor import RadiomicsFeatureExtractor


BINARY_CLASS_NAMES = ["良性", "恶性"]


@register_cls_model("autogluon_radiomics")
class AutoGluonRadiomicsModel(BaseClassificationModel):
    """pyradiomics 特征 + AutoGluon TabularPredictor 分类。"""

    def __init__(
        self,
        model_name: str,
        model_dir: str,
        base_dataset_performance: Optional[dict] = None,
        dataset_info: Optional[dict] = None,
        device: str = "cpu",
        radiomics_params: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_path=model_dir,
            use_tirads=False,
            base_dataset_performance=base_dataset_performance,
            dataset_info=dataset_info,
            device=device,
            **kwargs,
        )
        self.model_dir = model_dir
        self._predictor = None
        self._feature_names: Optional[list[str]] = None
        # 复用 RadiomicsFeatureExtractor，确保特征提取与训练侧 + RadiomicsJudge 一致
        self._extractor = RadiomicsFeatureExtractor(params_path=radiomics_params)

    @property
    def requires_mask(self) -> bool:
        return True

    def load_model(self) -> None:
        """延迟加载 AutoGluon TabularPredictor。"""
        print(f"  加载 AutoGluon 模型 {self.model_name} 从 {self.model_dir}")

        path = Path(self.model_dir)
        if not path.exists():
            raise FileNotFoundError(f"模型目录不存在: {path}")

        from autogluon.tabular import TabularPredictor

        self._predictor = TabularPredictor.load(
            str(path), require_py_version_match=False
        )
        self.is_loaded = True

        # 获取特征名（从训练数据推断）
        try:
            self._feature_names = list(
                self._predictor.feature_metadata_in.get_features()
            )
        except Exception:
            self._feature_names = None

        print(f"    ✓ AutoGluon 加载成功")

    def validate_inputs(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        if image is None:
            raise ValueError("image 不能为 None")
        if mask is None:
            raise ValueError(f"{self.model_name} 需要 mask 输入")

    def predict(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> ClsModelOutput:
        """radiomics 特征提取 → AutoGluon 推理。"""
        if not self.is_loaded:
            self.load_model()

        self.validate_inputs(image, mask)
        features = self._extractor.extract(image, mask)

        if not features:
            # 空 mask 或提取失败
            return ClsModelOutput(
                model_name=self.model_name,
                predictions={"良性": 0.5, "恶性": 0.5},
                top_class="良性",
                top_confidence=0.5,
                requires_mask=True,
                metadata={**self.get_info(), "feature_extraction_failed": True},
            )

        import pandas as pd

        df = pd.DataFrame([features])
        # 对齐特征列
        if self._feature_names:
            for col in self._feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df.reindex(columns=self._feature_names, fill_value=0.0)

        # as_multiclass=True 确保返回所有类别的概率列（与训练侧 infer.py 一致）
        pred_proba = self._predictor.predict_proba(
            df, as_pandas=True, as_multiclass=True
        )

        # 获取类别标签和概率
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
            p_mal = float(proba_row[malignant_label])
            predictions = {"良性": 1.0 - p_mal, "恶性": p_mal}
        else:
            # 多分类：直接映射标签 → 概率
            predictions = {
                str(cl): float(proba_row[cl]) for cl in class_labels
            }

        top_class = max(predictions, key=predictions.get)

        return ClsModelOutput(
            model_name=self.model_name,
            predictions=predictions,
            top_class=top_class,
            top_confidence=predictions[top_class],
            requires_mask=True,
            metadata=self.get_info(),
        )
