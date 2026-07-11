from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from autogluon_introspection import (
    is_bag_model,
    is_tree_model,
    load_autogluon_model,
)
from autogluon_preprocessing import get_tree_model_from_bag


def compute_shap_tree_bag(
    model: Any,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    model_name: str,
    predictor: Any,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for BAG tree models."""
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required. Install with: pip install shap")

    print(f"  Computing SHAP for BAG model {model_name}...")

    tree_result = get_tree_model_from_bag(model, model_name, predictor)

    if tree_result is None:
        print(f"  Could not extract tree model, using KernelExplainer...")
        return compute_shap_kernel(model, X_background, X_explain, model_name)

    tree_model, preprocess_func = tree_result
    print(f"  Extracted tree model type: {type(tree_model)}")

    print(f"  Getting preprocessed features using predictor's feature_generator...")
    X_background_processed = preprocess_func(X_background)
    X_explain_processed = preprocess_func(X_explain)

    if X_background_processed is None or X_explain_processed is None:
        print(f"  Could not get preprocessed features, using KernelExplainer...")
        return compute_shap_kernel(model, X_background, X_explain, model_name)

    print(f"  Preprocessed features: {X_background_processed.shape[1]} (from {X_background.shape[1]} original)")

    if hasattr(tree_model, "num_feature"):
        expected_features = tree_model.num_feature()
        actual_features = X_explain_processed.shape[1]

        if actual_features != expected_features:
            print(f"  Feature mismatch: data has {actual_features}, model expects {expected_features}")
            print(f"  This suggests preprocessing didn't work correctly. Using KernelExplainer...")
            return compute_shap_kernel(model, X_background, X_explain, model_name)
        else:
            print(f"  Feature count verified: {actual_features} features match model expectations")

    print(f"  Using TreeExplainer with {X_explain_processed.shape[1]} features...")

    X_background_array = X_background_processed.values if isinstance(X_background_processed, pd.DataFrame) else X_background_processed
    X_explain_array = X_explain_processed.values if isinstance(X_explain_processed, pd.DataFrame) else X_explain_processed

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_explain_array, X_background_array)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if not isinstance(shap_values, np.ndarray):
        shap_values = np.array(shap_values)

    if isinstance(X_explain_processed, pd.DataFrame):
        feature_names = X_explain_processed.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

    if len(feature_names) != shap_values.shape[1]:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_explain.index)

    return shap_values, shap_df


def compute_shap_kernel(
    model: Any,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    model_name: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values using KernelExplainer."""
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required. Install with: pip install shap")

    print(f"  Using KernelExplainer for {model_name}...")
    print(f"    Warning: KernelExplainer is slow for large datasets. Using {len(X_background)} background samples.")

    def model_wrapper(X):
        X_df = pd.DataFrame(X, columns=X_background.columns, index=range(len(X)))
        try:
            proba = model.predict_proba(X_df)
            if isinstance(proba, pd.DataFrame):
                if 1 in proba.columns:
                    return proba[1].values
                return proba.iloc[:, -1].values
            elif isinstance(proba, np.ndarray):
                if proba.ndim == 2 and proba.shape[1] > 1:
                    return proba[:, 1]
                return proba.flatten()
            else:
                return np.array(proba).flatten()
        except Exception as e:
            print(f"    Error in model_wrapper: {e}")
            raise

    explainer = shap.KernelExplainer(model_wrapper, X_background.values)
    shap_values = explainer.shap_values(X_explain.values, nsamples=100)

    feature_names = X_explain.columns.tolist()
    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_explain.index)

    return shap_values, shap_df


def compute_shap_for_model(
    predictor: Any,
    model_name: str,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    skip_neural_net: bool = False,
) -> Optional[Tuple[np.ndarray, pd.DataFrame]]:
    """Compute SHAP values for a single model."""
    if skip_neural_net and "NeuralNet" in model_name:
        print(f"  Skipping {model_name} (neural network, --skip_neural_net enabled)")
        return None

    try:
        model = load_autogluon_model(predictor, model_name)
        print(f"  Loaded model type: {type(model)}")
    except Exception as e:
        print(f"  Warning: Could not load model {model_name}: {e}")
        import traceback

        traceback.print_exc()
        return None

    try:
        if is_bag_model(model_name) and is_tree_model(model_name):
            return compute_shap_tree_bag(model, X_background, X_explain, model_name, predictor)
        elif is_tree_model(model_name):
            print(f"  Non-BAG tree model detected, using KernelExplainer for safety...")
            return compute_shap_kernel(model, X_background, X_explain, model_name)
        else:
            return compute_shap_kernel(model, X_background, X_explain, model_name)
    except Exception as e:
        print(f"  Error computing SHAP for {model_name}: {e}")
        import traceback

        traceback.print_exc()
        return None
