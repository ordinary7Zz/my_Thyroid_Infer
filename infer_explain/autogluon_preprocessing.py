from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


def get_preprocessed_features(
    model: Any,
    X: pd.DataFrame,
    model_name: str,
    predictor: Optional[Any] = None,
) -> Optional[pd.DataFrame]:
    """Get preprocessed features from an AutoGluon model."""
    if predictor is not None:
        try:
            if hasattr(predictor, "_learner") and hasattr(predictor._learner, "feature_generator"):
                X_processed = predictor._learner.feature_generator.transform(X)
                if isinstance(X_processed, pd.DataFrame):
                    return X_processed
                if isinstance(X_processed, np.ndarray):
                    feature_names = None
                    if hasattr(predictor._learner.feature_generator, "feature_metadata_in"):
                        try:
                            feature_metadata = predictor._learner.feature_generator.feature_metadata_in
                            if hasattr(feature_metadata, "get_features"):
                                feature_names = feature_metadata.get_features()
                        except Exception:
                            pass

                    if feature_names is None or len(feature_names) != X_processed.shape[1]:
                        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

                    return pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        except Exception as e:
            print(f"    feature_generator.transform failed: {e}")

    if hasattr(model, "models"):
        try:
            models_list = model.models
            if models_list and len(models_list) > 0:
                fold_name = models_list[0]
                if isinstance(fold_name, str) and predictor is not None:
                    from autogluon.common.loaders import load_pkl

                    model_path = os.path.join(predictor.path, "models", model_name, fold_name, "model.pkl")
                    if os.path.exists(model_path):
                        fold_model = load_pkl.load(path=model_path)
                        if hasattr(fold_model, "_preprocess"):
                            try:
                                X_processed = fold_model._preprocess(X, fit=False)
                                if isinstance(X_processed, pd.DataFrame):
                                    return X_processed
                                if isinstance(X_processed, np.ndarray):
                                    feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                                    return pd.DataFrame(X_processed, columns=feature_names, index=X.index)
                            except Exception as e:
                                print(f"    fold_model._preprocess failed: {e}")
        except Exception as e:
            print(f"    Error accessing fold models: {e}")

    if hasattr(model, "_preprocess"):
        try:
            X_processed = model._preprocess(X, fit=False)
            if isinstance(X_processed, pd.DataFrame):
                return X_processed
            if isinstance(X_processed, np.ndarray):
                feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                return pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        except Exception as e:
            print(f"    model._preprocess failed: {e}")

    if hasattr(model, "preprocess"):
        try:
            X_processed = model.preprocess(X)
            if isinstance(X_processed, pd.DataFrame):
                return X_processed
            if isinstance(X_processed, np.ndarray):
                feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                return pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        except Exception as e:
            print(f"    model.preprocess failed: {e}")

    return None


def get_tree_model_from_bag(model: Any, model_name: str, predictor: Any) -> Optional[Tuple[Any, Any]]:
    """Extract a tree model and preprocessing function from a BAG model."""
    if not hasattr(model, "models"):
        return None

    try:
        models_list = model.models
        if not models_list or len(models_list) == 0:
            return None

        fold_name = models_list[0]
        if not isinstance(fold_name, str):
            return None

        from autogluon.common.loaders import load_pkl

        model_path = os.path.join(predictor.path, "models", model_name, fold_name, "model.pkl")
        if not os.path.exists(model_path):
            return None

        fold_model = load_pkl.load(path=model_path)

        tree_model = fold_model
        while tree_model is not None:
            if hasattr(tree_model, "_model") and tree_model._model is not None:
                tree_model = tree_model._model
            elif hasattr(tree_model, "model") and tree_model.model is not None:
                tree_model = tree_model.model
            else:
                break

        model_type_str = str(type(tree_model))
        if "lightgbm" in model_type_str.lower() or "xgboost" in model_type_str.lower() or "catboost" in model_type_str.lower():
            def preprocess_func(X: pd.DataFrame):
                return get_preprocessed_features(fold_model, X, model_name, predictor)

            return (tree_model, preprocess_func)
    except Exception as e:
        print(f"    Error extracting tree model from BAG: {e}")

    return None
