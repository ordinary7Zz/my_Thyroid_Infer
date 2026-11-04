"""Workspace paths for NHC routing ablation (no LLM strategies 1–6 and 8)."""

from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
INFER_ROOT = PKG_DIR.parent
WORKSPACE_ROOT = INFER_ROOT.parent
NHC_PKG = PKG_DIR
RESULTS_ROOT = INFER_ROOT / "results" / "nhc_routing_ablation"
CONFIG_DIR = PKG_DIR / "config"

# B3/B7 内置引擎目录 ``cascade_engine/``（见 ``VENDOR_SOURCE.txt``）。
ARTIFACTS_ROOT = PKG_DIR / "artifacts"
ENGINE_ROOT = PKG_DIR / "cascade_engine"


def get_engine_root() -> Path:
    """NHC 七策略 + Phase1–2 子进程的工作目录（内含 run_pipeline.py、pipeline/ 等）。"""
    env = os.environ.get("NHC_CASCADE_ENGINE_ROOT", "").strip()
    if env:
        p = Path(env)
        return p.resolve() if p.is_absolute() else (INFER_ROOT / env).resolve()
    return ENGINE_ROOT.resolve()


def get_cascade_root() -> Path:
    """兼容旧名；等同 :func:`get_engine_root`。"""
    return get_engine_root()


def __getattr__(name: str):
    if name == "CASCADE_ROOT":
        return get_engine_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    """相对路径：``artifacts/...`` → ``nhc_routing_ablation/artifacts/``；否则相对 ``cascade_engine/``。"""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p.resolve()
    text = str(rel_or_abs).replace("\\", "/")
    if text.startswith("artifacts/"):
        cand = (ARTIFACTS_ROOT / text[len("artifacts/") :]).resolve()
        if cand.is_file() or cand.is_dir():
            return cand
    under = (ARTIFACTS_ROOT / p).resolve()
    if under.is_file() or under.is_dir():
        return under
    return (get_engine_root() / p).resolve()


def resolve_nhc_path(rel_or_abs: str | Path) -> Path:
    """相对路径：相对 ``my_Thyroid_Infer`` 根（结果目录、Phase1–2 缓存等）。"""
    return resolve_infer_path(rel_or_abs)


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
            elif k in ("seg_artifact", "cls_artifact") and isinstance(v, str) and v:
                out[k] = str(resolve_cascade_path(v))
            elif k == "radiomics_params" and isinstance(v, str) and v.startswith("./"):
                out[k] = str(resolve_infer_path(v))
            else:
                out[k] = _resolve_infer_weight_strings(v)
        return out
    if isinstance(obj, list):
        return [_resolve_infer_weight_strings(x) for x in obj]
    return obj


def materialize_cascade_config_for_infer(src: Path, dest: Path) -> Path:
    """Write engine-facing yaml with infer-relative weight paths expanded to absolute paths."""
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


ROOT_INFER_CONFIG = INFER_ROOT / "config.yaml"

ANON_MAPPING_FILENAME = "name_mapping.json"


def load_root_infer_config() -> dict:
    import yaml

    if not ROOT_INFER_CONFIG.is_file():
        raise FileNotFoundError(f"Missing root config: {ROOT_INFER_CONFIG}")
    with open(ROOT_INFER_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def processed_dir_from_root_config() -> Path:
    """NHC 预处理输出目录（``prepare.output_dir``；环境变量 ``NHC_PROCESSED_DIR`` 可覆盖）。"""
    env = os.environ.get("NHC_PROCESSED_DIR", "").strip()
    if env:
        return resolve_infer_path(env)
    cfg = load_root_infer_config()
    prep = cfg.get("prepare") or {}
    out = prep.get("output_dir") or "datasets/processed"
    return resolve_infer_path(out)


def processed_run_config_path() -> Path:
    """``pipeline.py`` Step 2 生成、供 ``run_all.py`` 使用的配置。"""
    return processed_dir_from_root_config() / "config.yaml"


def _resolve_infer_data_path(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (INFER_ROOT / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def pipeline_data_from_processed_run_config(run_cfg: dict) -> dict[str, str]:
    """将 ``datasets/processed/config.yaml`` 映射为 ``pipeline.data`` 字段。"""
    datasets = run_cfg.get("datasets") or {}
    labels = run_cfg.get("labels") or {}
    image = datasets.get("nodule_images") or datasets.get("binary_images")
    masks = datasets.get("nodule_masks") or datasets.get("binary_masks")
    label_file = labels.get("binary_json")
    if not image or not masks or not label_file:
        raise ValueError(
            "processed config.yaml must set datasets.nodule_images, "
            "datasets.nodule_masks, labels.binary_json"
        )
    return {
        "image_input": _resolve_infer_data_path(image),
        "gt_mask_dir": _resolve_infer_data_path(masks),
        "label_file": _resolve_infer_data_path(label_file),
        "label_key": labels.get("binary_field") or "malignancy",
    }


def apply_processed_dataset_to_config(
    cfg: dict,
    *,
    preserve_keys: frozenset[str] | None = None,
) -> tuple[dict, Path, Path]:
    """用 ``prepare.output_dir/config.yaml`` 覆盖 ``pipeline.data`` 中的路径字段。"""
    preserve = preserve_keys or frozenset({"start_index", "max_images", "device_info", "input_data_info"})
    proc_dir = processed_dir_from_root_config()
    run_cfg_path = processed_run_config_path()
    if not run_cfg_path.is_file():
        raise FileNotFoundError(
            f"未找到 pipeline 生成配置: {run_cfg_path}\n"
            f"请确认已运行 pipeline.py，且 config.yaml 中 prepare.output_dir 指向该目录（当前: {proc_dir}）"
        )
    import yaml

    with open(run_cfg_path, "r", encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f) or {}
    patch = pipeline_data_from_processed_run_config(run_cfg)
    pipe = cfg.setdefault("pipeline", {})
    data = pipe.setdefault("data", {})
    saved = {k: data[k] for k in preserve if k in data}
    for key, val in patch.items():
        data[key] = val
    data.update(saved)
    return cfg, proc_dir, run_cfg_path


class NhcDatasetConfigError(Exception):
    """``config.yaml`` / 预处理目录 / pipeline 生成配置或数据路径无效。"""


def _format_dataset_config_help(*, use_processed: bool) -> str:
    lines = [
        "请检查 my_Thyroid_Infer 根目录 config.yaml：",
        "  · prepare.input_dir / prepare.output_dir 是否正确",
    ]
    if use_processed:
        lines.append("  · 是否已在仓库根执行: python pipeline.py")
        lines.append("  · prepare.output_dir 下是否存在 config.yaml 及 images / nodule_masks / labels.json")
    lines.append(f"  · 配置文件路径: {ROOT_INFER_CONFIG}")
    return "\n".join(lines)


def _assert_data_path(path: Path, *, role: str, expect: str) -> None:
    if expect == "dir" and not path.is_dir():
        raise NhcDatasetConfigError(f"  · {role} 不是有效目录: {path}")
    if expect == "file" and not path.is_file():
        raise NhcDatasetConfigError(f"  · {role} 不是有效文件: {path}")
    if expect == "dir_or_file" and not (path.is_dir() or path.is_file()):
        raise NhcDatasetConfigError(f"  · {role} 不存在: {path}")


def validate_resolved_pipeline_data(data: dict[str, str], *, source_label: str) -> None:
    """校验 ``pipeline.data`` 已解析后的 image / mask / label 路径。"""
    missing = [
        k
        for k in ("image_input", "gt_mask_dir", "label_file")
        if not (isinstance(data.get(k), str) and str(data.get(k)).strip())
    ]
    if missing:
        raise NhcDatasetConfigError(
            f"数据集配置不完整（来源: {source_label}），缺少: {', '.join(missing)}"
        )

    checks = (
        ("image_input", "dir_or_file", "image_input（结节图）"),
        ("gt_mask_dir", "dir", "gt_mask_dir（结节 mask）"),
        ("label_file", "file", "label_file"),
    )
    problems: list[str] = []
    for key, expect, role in checks:
        try:
            _assert_data_path(Path(data[key]), role=role, expect=expect)
        except NhcDatasetConfigError as e:
            problems.append(str(e))

    if problems:
        raise NhcDatasetConfigError(
            f"数据集路径无效（来源: {source_label}）:\n" + "\n".join(problems)
        )


def validate_nhc_dataset_setup(*, use_processed: bool, b3_cfg: dict | None = None) -> None:
    """
    启动前校验：根 config.yaml、pipeline 生成 config、磁盘上的数据路径。
    失败时抛出 :class:`NhcDatasetConfigError`（含排查提示）。
    """
    if not ROOT_INFER_CONFIG.is_file():
        raise NhcDatasetConfigError(
            f"未找到仓库根配置文件:\n  {ROOT_INFER_CONFIG}\n\n{_format_dataset_config_help(use_processed=use_processed)}"
        )
    import yaml

    try:
        root_cfg = load_root_infer_config()
    except yaml.YAMLError as e:
        raise NhcDatasetConfigError(
            f"config.yaml 解析失败:\n  {ROOT_INFER_CONFIG}\n  {e}\n\n{_format_dataset_config_help(use_processed=use_processed)}"
        ) from e

    if use_processed:
        prep = root_cfg.get("prepare") or {}
        if not prep.get("output_dir") and not os.environ.get("NHC_PROCESSED_DIR", "").strip():
            raise NhcDatasetConfigError(
                f"config.yaml 未设置 prepare.output_dir（也可用环境变量 NHC_PROCESSED_DIR）。\n\n"
                f"{_format_dataset_config_help(use_processed=True)}"
            )
        try:
            proc_dir = processed_dir_from_root_config()
        except FileNotFoundError as e:
            raise NhcDatasetConfigError(f"{e}\n\n{_format_dataset_config_help(use_processed=True)}") from e

        if not proc_dir.is_dir():
            raise NhcDatasetConfigError(
                f"预处理输出目录不存在:\n  {proc_dir}\n\n{_format_dataset_config_help(use_processed=True)}"
            )

        run_cfg_path = processed_run_config_path()
        if not run_cfg_path.is_file():
            raise NhcDatasetConfigError(
                f"未找到 pipeline 生成的配置文件:\n  {run_cfg_path}\n"
                f"对应 prepare.output_dir: {proc_dir}\n\n{_format_dataset_config_help(use_processed=True)}"
            )
        import yaml as _yaml

        try:
            with open(run_cfg_path, "r", encoding="utf-8") as f:
                run_cfg = _yaml.safe_load(f) or {}
        except _yaml.YAMLError as e:
            raise NhcDatasetConfigError(
                f"预处理 config.yaml 解析失败:\n  {run_cfg_path}\n  {e}\n\n"
                f"{_format_dataset_config_help(use_processed=True)}"
            ) from e

        try:
            data = pipeline_data_from_processed_run_config(run_cfg)
        except ValueError as e:
            raise NhcDatasetConfigError(
                f"{run_cfg_path} 字段不完整:\n  {e}\n\n{_format_dataset_config_help(use_processed=True)}"
            ) from e

        validate_resolved_pipeline_data(data, source_label=str(run_cfg_path))
        return

    if b3_cfg is None:
        return
    data_raw = (b3_cfg.get("pipeline") or {}).get("data") or {}
    resolved = {}
    for key in ("image_input", "gt_mask_dir", "label_file"):
        raw = data_raw.get(key)
        if isinstance(raw, str) and raw.strip():
            resolved[key] = _resolve_infer_data_path(raw)
    if data_raw.get("label_key"):
        resolved["label_key"] = data_raw["label_key"]
    try:
        validate_resolved_pipeline_data(resolved, source_label="profile yaml pipeline.data")
    except NhcDatasetConfigError as e:
        raise NhcDatasetConfigError(f"{e}\n\n{_format_dataset_config_help(use_processed=False)}") from e


def find_anonymization_mapping_file(*, use_processed: bool, cfg: dict) -> Path | None:
    """Locate ``name_mapping.json`` (prepare_data 原名→匿名名) beside the active dataset."""
    seen: set[Path] = set()
    candidates: list[Path] = []

    def _add(p: Path) -> None:
        p = p.resolve()
        if p not in seen:
            seen.add(p)
            candidates.append(p)

    if use_processed:
        try:
            _add(processed_dir_from_root_config() / ANON_MAPPING_FILENAME)
        except FileNotFoundError:
            pass

    data = (cfg.get("pipeline") or {}).get("data") or {}
    for key in ("label_file", "image_input", "gt_mask_dir"):
        raw = data.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        parent = Path(_resolve_infer_data_path(raw)).parent
        _add(parent / ANON_MAPPING_FILENAME)

    if not use_processed:
        try:
            _add(processed_dir_from_root_config() / ANON_MAPPING_FILENAME)
        except FileNotFoundError:
            pass

    for path in candidates:
        if path.is_file():
            return path
    return None


def copy_anonymization_mapping_to_output(
    output_root: Path,
    *,
    use_processed: bool,
    cfg: dict,
) -> Path | None:
    """Copy dataset ``name_mapping.json`` into this run's ``output_root``."""
    import shutil

    src = find_anonymization_mapping_file(use_processed=use_processed, cfg=cfg)
    if src is None:
        print(
            f"⚠ 未找到 {ANON_MAPPING_FILENAME}（已查预处理目录与 pipeline.data 同级目录），跳过复制",
            flush=True,
        )
        return None
    output_root.mkdir(parents=True, exist_ok=True)
    dest = output_root / ANON_MAPPING_FILENAME
    shutil.copy2(src, dest)
    print(f"✓ 匿名映射已复制: {src} → {dest}", flush=True)
    return dest
