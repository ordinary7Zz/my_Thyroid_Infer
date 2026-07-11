from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional


def extract_ensemble_weights_from_log(log_path: str) -> Optional[Dict[str, float]]:
    """Extract ensemble weights from predictor_log.txt."""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = r"Ensemble Weights:\s*\{([^}]+)\}"
        match = re.search(pattern, content)
        if not match:
            return None

        weights_str = match.group(1)
        weights: Dict[str, float] = {}
        for item in weights_str.split(","):
            item = item.strip()
            if not item:
                continue
            model_match = re.search(r"'([^']+)':\s*([\d.]+)", item)
            if model_match:
                model_name = model_match.group(1)
                weight = float(model_match.group(2))
                weights[model_name] = weight

        return weights if weights else None
    except Exception as e:
        print(f"Warning: Failed to extract ensemble weights from log: {e}")
        return None


def load_autogluon_model(predictor: Any, model_name: str) -> Any:
    """Load a model from AutoGluon using the available predictor APIs."""
    model = None

    if hasattr(predictor, "_trainer") and hasattr(predictor._trainer, "load_model"):
        try:
            model = predictor._trainer.load_model(model_name)
        except Exception:
            pass

    if model is None and hasattr(predictor, "model_info"):
        try:
            model_info_attr = predictor.model_info
            model_info = None
            if callable(model_info_attr):
                try:
                    model_info = model_info_attr()
                except TypeError:
                    try:
                        model_info = model_info_attr(model=model_name)
                    except Exception:
                        model_info = None
            else:
                model_info = model_info_attr

            selected_model_info = None
            if isinstance(model_info, dict):
                if "path" in model_info:
                    selected_model_info = model_info
                elif model_name in model_info:
                    selected_model_info = model_info[model_name]

            if isinstance(selected_model_info, dict) and "path" in selected_model_info:
                from autogluon.common.loaders import load_pkl

                model_path = selected_model_info["path"]
                if not os.path.isabs(model_path):
                    model_path = os.path.join(predictor.path, model_path)
                model = load_pkl.load(path=model_path)
        except Exception:
            pass

    if model is None:
        raise AttributeError(f"Could not load model {model_name} using any available method")

    return model


def get_main_models(
    predictor: Any,
    model_dir: str,
    main_models: Optional[List[str]] = None,
) -> List[str]:
    """Get list of main models from ensemble or use provided list."""
    if main_models is not None:
        return main_models

    log_path = os.path.join(model_dir, "logs", "predictor_log.txt")
    weights = extract_ensemble_weights_from_log(log_path)
    if weights:
        print(f"Found ensemble weights from log: {weights}")
        return list(weights.keys())

    try:
        leaderboard = predictor.leaderboard(silent=True)
        best_model_name = leaderboard.iloc[0]["model"]
        print(f"Best model: {best_model_name}")

        try:
            if hasattr(predictor, "_trainer") and hasattr(predictor._trainer, "load_model"):
                ensemble_model = predictor._trainer.load_model(best_model_name)
                if hasattr(ensemble_model, "model_names"):
                    return ensemble_model.model_names
        except Exception:
            pass

        print("Warning: Could not extract ensemble sub-models. Using top 5 non-ensemble models.")
        non_ensemble = leaderboard[~leaderboard["model"].str.contains("Ensemble", case=False)]
        return non_ensemble.head(5)["model"].tolist()
    except Exception as e:
        print(f"Warning: Could not determine main models automatically: {e}")
        print("Using default: top models from leaderboard")
        leaderboard = predictor.leaderboard(silent=True)
        return leaderboard.head(5)["model"].tolist()


def is_tree_model(model_name: str) -> bool:
    """Check if model is a tree-based model."""
    tree_keywords = [
        "LightGBM",
        "XGBoost",
        "CatBoost",
        "RandomForest",
        "ExtraTrees",
    ]
    return any(kw in model_name for kw in tree_keywords)


def is_bag_model(model_name: str) -> bool:
    """Check if model is a BAG model."""
    return "_BAG_" in model_name
