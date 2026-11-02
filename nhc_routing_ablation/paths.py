"""Workspace paths for NHC routing ablation (no LLM strategies 1–6 and 8)."""

from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
INFER_ROOT = PKG_DIR.parent
WORKSPACE_ROOT = INFER_ROOT.parent
CASCADE_ROOT = WORKSPACE_ROOT / "ThyroidCascadeAgent"

CONFIG_DIR = PKG_DIR / "config"
RESULTS_ROOT = INFER_ROOT / "results" / "nhc_routing_ablation"

# Reply letter R12/R13: strategies (1)–(6) + (8); (7) LLM is N/A on NHC.
STRATEGY_ROWS = (
    {"id": 1, "label": "Best single (1)", "kind": "static", "mode": "best_single"},
    {"id": 2, "label": "Soft voting (2)", "kind": "static", "mode": "soft_voting"},
    {"id": 3, "label": "Majority (3)", "kind": "static", "mode": "majority"},
    {"id": 4, "label": "Static rule (4)", "kind": "static", "mode": "static_rule"},
    {"id": 5, "label": "Max-confidence (5)", "kind": "static", "mode": "max_confidence"},
    {"id": 6, "label": "Learned router (6)", "kind": "learned", "mode": "learned"},
    {"id": 8, "label": "Oracle (8)", "kind": "static", "mode": "oracle"},
)

STATIC_MODES = tuple(s["mode"] for s in STRATEGY_ROWS if s["kind"] == "static")


def resolve_cascade_path(rel_or_abs: str | Path) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (CASCADE_ROOT / p).resolve()


def resolve_infer_path(rel_or_abs: str | Path) -> Path:
    """Resolve paths relative to my_Thyroid_Infer root (``./infer_*``, ``./pretrained_models/*``)."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p.resolve()
    text = str(rel_or_abs)
    if text.startswith("./"):
        return (INFER_ROOT / text[2:]).resolve()
    return (INFER_ROOT / p).resolve()


_WEIGHT_PATH_KEYS = frozenset(
    {
        "model_path",
        "sam_checkpoint",
        "pretrained_model_path",
        "model_dir",
        "infer_root",
        "checkpoint",
    }
)


def _resolve_infer_weight_strings(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _WEIGHT_PATH_KEYS and isinstance(v, str) and v:
                if v.startswith("./infer_") or v.startswith("./pretrained_models/"):
                    out[k] = str(resolve_infer_path(v))
                else:
                    out[k] = v
            elif k == "radiomics_params" and isinstance(v, str) and v.startswith("./"):
                out[k] = str(resolve_infer_path(v))
            else:
                out[k] = _resolve_infer_weight_strings(v)
        return out
    if isinstance(obj, list):
        return [_resolve_infer_weight_strings(x) for x in obj]
    return obj


def materialize_cascade_config_for_infer(src: Path, dest: Path) -> Path:
    """Write a Cascade-facing yaml with infer-relative weight paths expanded to absolute paths."""
    import yaml

    with open(src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _resolve_infer_weight_strings(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return dest.resolve()


def b3_config_for_profile(profile: str) -> Path:
    p = CONFIG_DIR / f"{profile}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"Missing profile config: {p}")
    return p


def cascade_phase12_config_for_profile(profile: str) -> Path:
    p = CONFIG_DIR / f"cascade_phase12_{profile}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"Missing phase-1–2 config: {p}")
    return p
