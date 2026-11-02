#!/usr/bin/env python3
"""
run_all 扩展版：在原有任务基础上，为良恶性二分类增加 RepViT、LSNet、ResNet-50。

不修改 run_all.py；通过导入并 patch TASKS / _load_config 实现。
推理入口统一在 infer_repvit / infer_lsnet / infer_resnet50（与其它 infer_* 目录一致）。

用法与 run_all.py 相同，例如:
  python run_all_extended.py --mode finetune --config config.yaml --tasks binary
  python run_all_extended.py --tasks nodule binary --models repvit lsnet resnet50
  python run_all_extended.py --list

配置（在 config.yaml 中可选；未写时使用各 infer_*/paths.py 默认权重）:

  weights:
    binary:
      repvit: ...
      lsnet: ...
      resnet50: ...

  extended_binary:
    repvit_model: repvit_m1_0
    test_cls_dir: /path/to/TN3K/test_cls
    test_cls_name: tn3k
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXTRA_BINARY_MODELS = ("repvit", "lsnet", "resnet50")


def _load_infer_paths(model_key: str):
    spec = importlib.util.spec_from_file_location(
        f"infer_{model_key}_paths",
        ROOT / f"infer_{model_key}" / "paths.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _default_weights() -> dict[str, str]:
    out = {}
    for key in EXTRA_BINARY_MODELS:
        paths = _load_infer_paths(key)
        out[key] = str(paths.DEFAULT_CHECKPOINT)
    return out


DEFAULT_WEIGHTS = _default_weights()

INFER_BINARY_SCRIPT = {
    "repvit": ROOT / "infer_repvit" / "infer_binary.py",
    "lsnet": ROOT / "infer_lsnet" / "infer_binary.py",
    "resnet50": ROOT / "infer_resnet50" / "infer_binary.py",
}

INFER_CWD = {
    "repvit": ROOT / "infer_repvit",
    "lsnet": ROOT / "infer_lsnet",
    "resnet50": ROOT / "infer_resnet50",
}


def _load_run_all():
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_all"] = mod
    spec.loader.exec_module(mod)
    return mod


def _infer_test_cls_dir(binary_images: str) -> str | None:
    p = Path(binary_images)
    if p.name == "images":
        cand = p.parent / "test_cls"
        if cand.is_dir():
            return str(cand.resolve())
    return None


def _merge_b2_config(cfg: dict) -> dict:
    weights = cfg.setdefault("weights", {})
    binary_w = weights.setdefault("binary", {})
    for key, path in DEFAULT_WEIGHTS.items():
        if not binary_w.get(key):
            binary_w[key] = path
    cfg.setdefault("extended_binary", {})
    return cfg


def _out_dir(ra, task: str, model: str) -> str:
    return ra._resolve(os.path.join(ra.CONFIG["output_root"], task, model))


def _build_extra_binary_cmds(ra):
    """RepViT / LSNet / ResNet-50：调用 infer_<model>/infer_binary.py。"""
    cmds = []
    if ra.MODE == "pretrained":
        return cmds

    ext = ra.CONFIG.get("extended_binary", {})
    w = ra.CONFIG.get("weights", {}).get("binary", {})
    nb = str(ra.CONFIG["n_bootstrap"])
    ds = ra.CONFIG["datasets"]
    lb = ra.CONFIG["labels"]

    test_cls = ext.get("test_cls_dir") or _infer_test_cls_dir(ds.get("binary_images", ""))
    test_name = ext.get("test_cls_name") or (
        Path(test_cls).name if test_cls else "test_cls"
    )

    if w.get("repvit") and test_cls and Path(test_cls).is_dir():
        out_dir = _out_dir(ra, "binary", "repvit")
        os.makedirs(out_dir, exist_ok=True)
        cmds.append(
            (
                "repvit",
                [
                    sys.executable,
                    str(INFER_BINARY_SCRIPT["repvit"]),
                    "--model",
                    ext.get("repvit_model", "repvit_m1_0"),
                    "--checkpoint",
                    w["repvit"],
                    "--test-dirs",
                    test_cls,
                    "--test-names",
                    test_name,
                    "--output-dir",
                    out_dir,
                    "--device",
                    ra.CONFIG.get("device", "cuda"),
                    "--bootstrap-iters",
                    nb,
                ],
            )
        )
    elif w.get("repvit"):
        print("  [repvit] 跳过: 未配置 extended_binary.test_cls_dir 且无法从 binary_images 推断 test_cls")

    if w.get("lsnet") and test_cls and Path(test_cls).is_dir():
        out_dir = _out_dir(ra, "binary", "lsnet")
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, f"metrics_{ra._RUN_TIMESTAMP}.log")
        cmds.append(
            (
                "lsnet",
                [
                    sys.executable,
                    str(INFER_BINARY_SCRIPT["lsnet"]),
                    "--checkpoint",
                    w["lsnet"],
                    "--test-dirs",
                    test_cls,
                    "--names",
                    test_name,
                    "--output-dir",
                    out_dir,
                    "--output-prefix",
                    "lsnet",
                    "--device",
                    ra.CONFIG.get("device", "cuda"),
                    "--log-file",
                    log_path,
                    "--bootstrap-samples",
                    nb,
                ],
            )
        )
    elif w.get("lsnet"):
        print("  [lsnet] 跳过: 缺少 test_cls 目录")

    if w.get("resnet50"):
        img = ds.get("binary_images")
        label_json = lb.get("binary_json")
        task_key = lb.get("binary_field", "malignancy")
        if img and label_json and Path(label_json).is_file():
            out_dir = _out_dir(ra, "binary", "resnet50")
            os.makedirs(out_dir, exist_ok=True)
            cmds.append(
                (
                    "resnet50",
                    [
                        sys.executable,
                        str(INFER_BINARY_SCRIPT["resnet50"]),
                        "--image_dir",
                        img,
                        "--label_json",
                        label_json,
                        "--task_key",
                        task_key,
                        "--checkpoint_path",
                        w["resnet50"],
                        "--output_dir",
                        out_dir,
                        "--n_boot",
                        nb,
                    ],
                )
            )
        else:
            print("  [resnet50] 跳过: binary_images / binary_json 未配置或标签文件不存在")

    return cmds


def _patch_run_all(ra):
    original_binary = ra._cls_binary

    def _cls_binary_extended():
        return list(original_binary()) + _build_extra_binary_cmds(ra)

    ra.TASKS["binary"] = {
        **ra.TASKS["binary"],
        "builder": _cls_binary_extended,
        "models": list(ra.TASKS["binary"]["models"]) + list(EXTRA_BINARY_MODELS),
    }

    for name in EXTRA_BINARY_MODELS:
        ra._MODEL_PRETRAINED_DEPS[name] = []

    _script_cwd = {str(INFER_BINARY_SCRIPT[k]): str(INFER_CWD[k]) for k in EXTRA_BINARY_MODELS}
    _orig_run_command = ra.run_command

    def _run_command_with_cwd(cmd, cwd=None):
        if cwd is None and len(cmd) >= 2:
            script = cmd[1]
            cwd = _script_cwd.get(script)
        return _orig_run_command(cmd, cwd=cwd)

    ra.run_command = _run_command_with_cwd

    original_load = ra._load_config

    def _load_config_extended(path):
        return _merge_b2_config(original_load(path))

    ra._load_config = _load_config_extended


def main():
    ra = _load_run_all()
    _patch_run_all(ra)
    print(
        "run_all_extended: 已加载 run_all + "
        "infer_repvit / infer_lsnet / infer_resnet50（binary 任务）\n"
    )
    ra.main()


if __name__ == "__main__":
    main()
