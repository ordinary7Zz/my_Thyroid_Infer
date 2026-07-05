#!/usr/bin/env python3
"""
统一推理脚本
============
一键运行全部四个任务的推理，覆盖所有模型：

  1. 腺体分割 (gland)     — dinov3_unet, medsam2, medsegx, transunet, ultrafedfm
  2. 结节分割 (nodule)    — dinov3_unet, medsam2, medsegx, transunet, ultrafedfm
  3. 良恶性二分类 (binary) — biomedclip, medsiglip, ultrafedfm, dinov3_unet_multitask
  4. TIRADS五分类 (tirads) — biomedclip, medsiglip, ultrafedfm, dinov3_unet_multitask

用法:
  python run_all.py                                              # 运行全部四个任务
  python run_all.py --tasks gland nodule                          # 只运行分割任务
  python run_all.py --tasks binary tirads                         # 只运行分类任务
  python run_all.py --tasks gland --models dinov3_unet medsam2    # 指定模型
  python run_all.py --dry_run                                     # 只打印命令不执行
  python run_all.py --config path/to/config.yaml                  # 指定配置文件

配置:
  所有路径在 config.yaml 中集中管理，其他代码也可读写该文件来修改配置。
  默认使用脚本同目录下的 config.yaml，可用 --config 指定其他路径。
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ============================================================================
# 路径常量与配置加载
# ============================================================================

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

# 运行时由 main() 从 YAML 文件加载填充；模块级保留空 dict 以便函数引用
CONFIG: dict = {}

# CONFIG 必须包含的顶层字段（用于加载时的最小校验）
_REQUIRED_CONFIG_KEYS = (
    "datasets", "labels", "weights", "pretrained",
    "output_root", "device", "n_bootstrap",
)


def _load_config(path):
    """从 YAML 文件加载配置。

    参数:
        path: 配置文件路径（字符串或 Path）

    返回:
        解析后的配置 dict

    退出码:
        1 — 配置文件不存在 / PyYAML 未安装 / 缺少必需字段
    """
    p = Path(path)
    if not p.is_file():
        print(f"  [错误] 配置文件不存在: {p}")
        print(f"         默认配置位于: {DEFAULT_CONFIG_PATH}")
        print(f"         可用 --config 指定其他路径。")
        sys.exit(1)

    try:
        import yaml
    except ImportError:
        print("  [错误] 未安装 PyYAML，请运行: pip install pyyaml")
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        print(f"  [错误] 配置文件格式错误: 顶层应为映射，实际为 {type(cfg).__name__}")
        sys.exit(1)

    missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        print(f"  [错误] 配置文件缺少必需字段: {', '.join(missing)}")
        sys.exit(1)

    return cfg


# ============================================================================
# 任务定义：每个任务下各模型的命令构建器
# ============================================================================


def _resolve(path):
    """将路径解析为绝对路径。
    相对路径以脚本所在目录 (ROOT) 为基准，绝对路径保持不变。
    这样无论从哪个目录运行脚本，路径行为都一致。
    """
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return str(p.resolve())


def _out(task, model, filename=""):
    """构建输出路径: results/<task>/<model>/<filename>"""
    p = os.path.join(CONFIG["output_root"], task, model)
    if filename:
        p = os.path.join(p, filename)
    return _resolve(p)


# ---------- 分割：腺体 ----------

def _seg_gland():
    img = CONFIG["datasets"]["gland_images"]
    gt  = CONFIG["datasets"]["gland_masks"]
    w   = CONFIG["weights"]["gland"]
    pt  = CONFIG["pretrained"]
    dev = CONFIG["device"]
    cmds = []

    # dinov3_unet
    cmds.append(("dinov3_unet", [
        sys.executable, str(ROOT / "infer_dinov3_unet" / "infer.py"),
        "--checkpoint", w["dinov3_unet"],
        "--input_dir", img,
        "--gt_dir", gt,
        "--log_dir", _out("gland", "dinov3_unet"),
    ]))

    # medsam2
    cmds.append(("medsam2", [
        sys.executable, str(ROOT / "infer_medsam2" / "infer.py"),
        "--image_dir", img,
        "--checkpoint", w["medsam2"],
        "--gt_dir", gt,
        "--config", os.path.basename(pt["medsam2_config"]),
        "--log_dir", _out("gland", "medsam2"),
    ]))

    # medsegx
    cmds.append(("medsegx", [
        sys.executable, str(ROOT / "infer_medsegx" / "inference.py"),
        "--input_dir", img,
        "--gt_dir", gt,
        "--task_name", "US_GlndThyroid",
        "--checkpoint", pt["medsegx_sam_dir"],
        "--model_weight", w["medsegx"],
        "--log_file", _out("gland", "medsegx", "metrics.log"),
    ]))

    # transunet
    cmds.append(("transunet", [
        sys.executable, str(ROOT / "infer_transunet" / "infer.py"),
        "--ckpt", w["transunet"],
        "--img_dir", img,
        "--gt_dir", gt,
        "--log", _out("gland", "transunet", "metrics.log"),
    ]))

    # ultrafedfm (segment)
    cmds.append(("ultrafedfm", [
        sys.executable, str(ROOT / "infer_ultrafedfm" / "segment.py"),
        "--data_path", img,
        "--resume", w["ultrafedfm"],
        "--gt_dir", gt,
        "--output_log", _out("gland", "ultrafedfm", "metrics.log"),
    ]))

    return cmds


# ---------- 分割：结节 ----------

def _seg_nodule():
    img = CONFIG["datasets"]["nodule_images"]
    gt  = CONFIG["datasets"]["nodule_masks"]
    w   = CONFIG["weights"]["nodule"]
    pt  = CONFIG["pretrained"]
    cmds = []

    cmds.append(("dinov3_unet", [
        sys.executable, str(ROOT / "infer_dinov3_unet" / "infer.py"),
        "--checkpoint", w["dinov3_unet"],
        "--input_dir", img,
        "--gt_dir", gt,
        "--log_dir", _out("nodule", "dinov3_unet"),
    ]))

    cmds.append(("medsam2", [
        sys.executable, str(ROOT / "infer_medsam2" / "infer.py"),
        "--image_dir", img,
        "--checkpoint", w["medsam2"],
        "--gt_dir", gt,
        "--config", os.path.basename(pt["medsam2_config"]),
        "--log_dir", _out("nodule", "medsam2"),
    ]))

    cmds.append(("medsegx", [
        sys.executable, str(ROOT / "infer_medsegx" / "inference.py"),
        "--input_dir", img,
        "--gt_dir", gt,
        "--task_name", "US_ThyroidNodule",
        "--checkpoint", pt["medsegx_sam_dir"],
        "--model_weight", w["medsegx"],
        "--log_file", _out("nodule", "medsegx", "metrics.log"),
    ]))

    cmds.append(("transunet", [
        sys.executable, str(ROOT / "infer_transunet" / "infer.py"),
        "--ckpt", w["transunet"],
        "--img_dir", img,
        "--gt_dir", gt,
        "--log", _out("nodule", "transunet", "metrics.log"),
    ]))

    cmds.append(("ultrafedfm", [
        sys.executable, str(ROOT / "infer_ultrafedfm" / "segment.py"),
        "--data_path", img,
        "--resume", w["ultrafedfm"],
        "--gt_dir", gt,
        "--output_log", _out("nodule", "ultrafedfm", "metrics.log"),
    ]))

    return cmds


# ---------- 分类：良恶性二分类 ----------

def _cls_binary():
    img   = CONFIG["datasets"]["binary_images"]
    label = CONFIG["labels"]["binary_json"]
    field = CONFIG["labels"]["binary_field"]
    w     = CONFIG["weights"]["binary"]
    pt    = CONFIG["pretrained"]
    nb    = str(CONFIG["n_bootstrap"])
    cmds = []

    # biomedclip
    cmds.append(("biomedclip", [
        sys.executable, str(ROOT / "infer_biomedclip" / "infer.py"),
        "--ckpt", w["biomedclip"],
        "--model_dir", pt["biomedclip_dir"],
        "--folder", img,
        "--num_classes", "2",
        "--class_names", "0", "1",
        "--label_json", label,
        "--label_field", field,
        "--output", _out("binary", "biomedclip", "predictions.csv"),
        "--eval_output", _out("binary", "biomedclip", "metrics.log"),
        "--n_bootstrap", nb,
    ]))

    # medsiglip
    cmds.append(("medsiglip", [
        sys.executable, str(ROOT / "infer_medsiglip" / "inference.py"),
        "--checkpoint", w["medsiglip"],
        "--model_path", pt["medsiglip_dir"],
        "--input", img,
        "--output", _out("binary", "medsiglip", "predictions.csv"),
        "--label_file", label,
        "--label_field", field,
        "--metrics_output", _out("binary", "medsiglip", "metrics.log"),
        "--n_bootstrap", nb,
    ]))

    # ultrafedfm (classify)
    cmds.append(("ultrafedfm", [
        sys.executable, str(ROOT / "infer_ultrafedfm" / "classify.py"),
        "--data_path", img,
        "--resume", w["ultrafedfm"],
        "--nb_classes", "2",
        "--label_file", label,
        "--label_field", field,
        "--output_csv", _out("binary", "ultrafedfm", "predictions.csv"),
        "--output_log", _out("binary", "ultrafedfm", "metrics.log"),
        "--n_bootstrap", nb,
    ]))

    # dinov3_unet_multitask
    cmds.append(("dinov3_unet_multitask", [
        sys.executable, str(ROOT / "infer_dinov3_unet_multitask" / "infer_classification.py"),
        "--image_dir", img,
        "--checkpoint", w["dinov3_unet_multitask"],
        "--num_classes", "2",
        "--output", _out("binary", "dinov3_unet_multitask", "predictions.csv"),
        "--label_file", label,
        "--label_field", field,
        "--log_file", _out("binary", "dinov3_unet_multitask", "metrics.log"),
        "--n_boot", nb,
    ]))

    return cmds


# ---------- 分类：TIRADS 五分类 ----------

def _cls_tirads():
    img   = CONFIG["datasets"]["tirads_images"]
    label = CONFIG["labels"]["tirads_json"]
    field = CONFIG["labels"]["tirads_field"]
    w     = CONFIG["weights"]["tirads"]
    pt    = CONFIG["pretrained"]
    nb    = str(CONFIG["n_bootstrap"])
    cmds = []

    # biomedclip
    cmds.append(("biomedclip", [
        sys.executable, str(ROOT / "infer_biomedclip" / "infer.py"),
        "--ckpt", w["biomedclip"],
        "--model_dir", pt["biomedclip_dir"],
        "--folder", img,
        "--num_classes", "5",
        "--class_names", "1", "2", "3", "4", "5",
        "--label_json", label,
        "--label_field", field,
        "--output", _out("tirads", "biomedclip", "predictions.csv"),
        "--eval_output", _out("tirads", "biomedclip", "metrics.log"),
        "--n_bootstrap", nb,
    ]))

    # medsiglip
    cmds.append(("medsiglip", [
        sys.executable, str(ROOT / "infer_medsiglip" / "inference.py"),
        "--checkpoint", w["medsiglip"],
        "--model_path", pt["medsiglip_dir"],
        "--input", img,
        "--output", _out("tirads", "medsiglip", "predictions.csv"),
        "--label_file", label,
        "--label_field", field,
        "--metrics_output", _out("tirads", "medsiglip", "metrics.log"),
        "--n_bootstrap", nb,
    ]))

    # ultrafedfm (classify)
    cmds.append(("ultrafedfm", [
        sys.executable, str(ROOT / "infer_ultrafedfm" / "classify.py"),
        "--data_path", img,
        "--resume", w["ultrafedfm"],
        "--nb_classes", "5",
        "--label_file", label,
        "--label_field", field,
        "--output_csv", _out("tirads", "ultrafedfm", "predictions.csv"),
        "--output_log", _out("tirads", "ultrafedfm", "metrics.log"),
        "--n_bootstrap", nb,
    ]))

    # dinov3_unet_multitask
    cmds.append(("dinov3_unet_multitask", [
        sys.executable, str(ROOT / "infer_dinov3_unet_multitask" / "infer_classification.py"),
        "--image_dir", img,
        "--checkpoint", w["dinov3_unet_multitask"],
        "--num_classes", "5",
        "--output", _out("tirads", "dinov3_unet_multitask", "predictions.csv"),
        "--label_file", label,
        "--label_field", field,
        "--log_file", _out("tirads", "dinov3_unet_multitask", "metrics.log"),
        "--n_boot", nb,
    ]))

    return cmds


# ============================================================================
# 任务注册表
# ============================================================================

TASKS = {
    "gland":  {"desc": "腺体分割",      "builder": _seg_gland,  "models": ["dinov3_unet", "medsam2", "medsegx", "transunet", "ultrafedfm"]},
    "nodule": {"desc": "结节分割",      "builder": _seg_nodule, "models": ["dinov3_unet", "medsam2", "medsegx", "transunet", "ultrafedfm"]},
    "binary": {"desc": "良恶性二分类",  "builder": _cls_binary, "models": ["biomedclip", "medsiglip", "ultrafedfm", "dinov3_unet_multitask"]},
    "tirads": {"desc": "TIRADS五分类",  "builder": _cls_tirads, "models": ["biomedclip", "medsiglip", "ultrafedfm", "dinov3_unet_multitask"]},
}


# ============================================================================
# 运行逻辑
# ============================================================================

def run_command(cmd, cwd=None):
    """运行单条命令，实时打印输出。返回 (returncode, elapsed)。"""
    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0
    return proc.returncode, elapsed


# 不需要解析的命令行 flag（后面紧跟的值是路径，需要解析）
_PATH_FLAGS = {
    "--ckpt", "--checkpoint", "--resume", "--model_weight",
    "--input_dir", "--image_dir", "--img_dir", "--data_path", "--folder", "--input",
    "--gt_dir", "--log_dir", "--log_file", "--output_dir",
    "--output", "--output_csv", "--output_log", "--eval_output", "--log",
    "--model_dir", "--model_path", "--label_json", "--label_file",
}


def _resolve_cmd_paths(cmd):
    """将命令中所有路径类参数解析为绝对路径（相对路径以脚本所在目录为基准）。

    遍历命令列表，当遇到 _PATH_FLAGS 中的 flag 时，将其后一个参数解析为绝对路径。
    其余参数（如 --num_classes, --device, task_name 等）保持不变。
    """
    resolved = list(cmd)
    for i, arg in enumerate(cmd):
        if arg in _PATH_FLAGS and i + 1 < len(cmd):
            resolved[i + 1] = _resolve(cmd[i + 1])
    return resolved


# ============================================================================
# 预检查：运行前统一校验 CONFIG 中所有路径
# ============================================================================

# 每个模型依赖的预训练骨干 (key, 类型: "dir" | "file")
_MODEL_PRETRAINED_DEPS = {
    "biomedclip":            [("biomedclip_dir",  "dir")],
    "medsiglip":             [("medsiglip_dir",   "dir")],
    "medsegx":               [("medsegx_sam_dir", "dir")],
    "medsam2":               [("medsam2_config",  "file")],
    "transunet":             [],
    "ultrafedfm":            [],
    "dinov3_unet":           [],
    "dinov3_unet_multitask": [],
}

# 每个任务依赖的数据集目录和标签文件 (CONFIG key, 描述)
_TASK_DEPS = {
    "gland": {
        "dirs":  [("gland_images", "datasets.gland_images"),
                  ("gland_masks",  "datasets.gland_masks")],
        "files": [],
    },
    "nodule": {
        "dirs":  [("nodule_images", "datasets.nodule_images"),
                  ("nodule_masks",  "datasets.nodule_masks")],
        "files": [],
    },
    "binary": {
        "dirs":  [("binary_images", "datasets.binary_images")],
        "files": [("binary_json",   "labels.binary_json")],
    },
    "tirads": {
        "dirs":  [("tirads_images", "datasets.tirads_images")],
        "files": [("tirads_json",   "labels.tirads_json")],
    },
}


def preflight_check(tasks, models_filter=None):
    """运行前统一检查 CONFIG 中所需路径/文件是否存在。

    只检查 tasks 中涉及的、且未被 models_filter 排除的模型所依赖的路径，
    避免运行到一半才发现路径缺失。

    参数:
        tasks:          要运行的任务 id 列表
        models_filter:   模型筛选列表（None 表示不筛选）

    返回:
        missing: 缺失项列表，每项为 (类型, 描述, 原始路径, 解析路径)
    """
    missing = []
    pt = CONFIG["pretrained"]
    ds = CONFIG["datasets"]
    lb = CONFIG["labels"]
    wt = CONFIG["weights"]

    def _check_dir(path, desc):
        p = _resolve(path)
        if not os.path.isdir(p):
            missing.append(("目录", desc, path, p))

    def _check_file(path, desc):
        p = _resolve(path)
        if not os.path.isfile(p):
            missing.append(("文件", desc, path, p))

    # 输出根目录
    _check_dir(CONFIG["output_root"], "output_root")

    for task_id in tasks:
        dep = _TASK_DEPS[task_id]
        # 数据集目录
        for key, desc in dep["dirs"]:
            _check_dir(ds[key], f"{task_id}/{desc}")
        # 标签文件
        for key, desc in dep["files"]:
            _check_file(lb[key], f"{task_id}/{desc}")
        # 权重 + 预训练模型
        for model_name, weight_path in wt[task_id].items():
            if models_filter and model_name not in models_filter:
                continue
            _check_file(weight_path, f"{task_id}/weights.{model_name}")
            for pt_key, pt_type in _MODEL_PRETRAINED_DEPS.get(model_name, []):
                desc = f"{task_id}/pretrained.{pt_key} ({model_name})"
                if pt_type == "dir":
                    _check_dir(pt[pt_key], desc)
                else:
                    # medsam2 的 config 参数：run_all 传裸文件名，
                    # 由 infer_medsam2/infer.py 在 <脚本目录>/sam2/configs/ 下查找。
                    # 预检查也按相同方式查找。
                    if model_name == "medsam2":
                        fname = os.path.basename(pt[pt_key])
                        candidates = [
                            os.path.join(ROOT, "infer_medsam2", "sam2", "configs", fname),
                            os.path.join(ROOT, "infer_medsam2", "sam2", fname),
                            os.path.join(ROOT, "infer_medsam2", fname),
                        ]
                        found = any(os.path.isfile(c) for c in candidates)
                        if not found:
                            missing.append(("文件", desc, pt[pt_key],
                                            " 或 ".join(candidates)))
                    else:
                        _check_file(pt[pt_key], desc)

    return missing


def main():
    parser = argparse.ArgumentParser(
        description="统一推理脚本 — 一键运行全部四个任务的所有模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_all.py                                     # 运行全部
  python run_all.py --tasks gland nodule                 # 只运行分割
  python run_all.py --tasks binary tirads               # 只运行分类
  python run_all.py --tasks gland --models dinov3_unet  # 指定模型
  python run_all.py --dry_run                            # 只打印命令不执行
  python run_all.py --list                               # 列出所有任务和模型
  python run_all.py --config my_config.yaml              # 指定配置文件
        """,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help=f"配置文件路径（默认: {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        choices=list(TASKS.keys()),
                        help="要运行的任务（默认全部）")
    parser.add_argument("--models", nargs="+", default=None,
                        help="只运行指定的模型（跨任务筛选）")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印命令不执行")
    parser.add_argument("--list", action="store_true",
                        help="列出所有任务和模型后退出")
    args = parser.parse_args()

    # 加载配置文件（必须在使用 CONFIG 之前完成）
    global CONFIG
    CONFIG = _load_config(args.config)

    if args.list:
        print("可用任务和模型:\n")
        for task_id, task in TASKS.items():
            print(f"  {task_id:8s} — {task['desc']}")
            for m in task["models"]:
                print(f"             ├─ {m}")
        return

    # 汇总统计
    total = 0
    success = 0
    failed = 0
    skipped = 0
    results_log = []

    print("=" * 70)
    print("  统一推理脚本")
    print("=" * 70)
    print(f"  配置文件: {args.config}")
    print(f"  任务: {', '.join(args.tasks)}")
    if args.models:
        print(f"  模型筛选: {', '.join(args.models)}")
    print(f"  输出目录: {CONFIG['output_root']}")
    print(f"  设备: {CONFIG['device']}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70)

    # ---- 预检查：CONFIG 中所有路径 ----
    print("\n  预检查 CONFIG 路径...")
    missing = preflight_check(args.tasks, args.models)
    if missing:
        print(f"\n  发现 {len(missing)} 项缺失:")
        for kind, desc, orig, resolved in missing:
            print(f"    [{kind}] {desc}")
            print(f"           配置: {orig}")
            print(f"           解析: {resolved}")
        print(f"\n  请修正 CONFIG 后重试。")
        sys.exit(1)
    print("  预检查通过\n")

    for task_id in args.tasks:
        task = TASKS[task_id]
        all_cmds = task["builder"]()

        # 模型筛选
        if args.models:
            all_cmds = [(m, c) for m, c in all_cmds if m in args.models]

        if not all_cmds:
            continue

        print(f"\n{'─' * 70}")
        print(f"  任务: {task['desc']} ({task_id})")
        print(f"  模型数: {len(all_cmds)}")
        print(f"{'─' * 70}")

        for model_name, cmd in all_cmds:
            total += 1
            print(f"\n  >>> [{task_id}/{model_name}]")

            # 将命令中所有路径参数解析为绝对路径（相对路径以脚本所在目录为基准）
            cmd = _resolve_cmd_paths(cmd)

            # 确保输出目录存在
            out_dir = _resolve(os.path.join(CONFIG["output_root"], task_id, model_name))
            os.makedirs(out_dir, exist_ok=True)

            if args.dry_run:
                print(f"  [DRY RUN] {' '.join(cmd)}")
                skipped += 1
                results_log.append((task_id, model_name, "SKIP", 0))
                continue

            # 路径检查已由 preflight_check() 在运行前统一完成，此处直接执行
            rc, elapsed = run_command(cmd, cwd=str(ROOT))
            status = "OK" if rc == 0 else f"FAIL (rc={rc})"
            if rc == 0:
                success += 1
            else:
                failed += 1
            results_log.append((task_id, model_name, status, elapsed))
            print(f"  [{status}] 耗时 {elapsed:.1f}s")

    # 汇总报告
    print(f"\n{'=' * 70}")
    print("  汇总报告")
    print(f"{'=' * 70}")
    print(f"  总计: {total}  成功: {success}  失败: {failed}  跳过: {skipped}")
    print(f"{'─' * 70}")
    print(f"  {'任务':10s} {'模型':25s} {'状态':20s} {'耗时':>8s}")
    print(f"{'─' * 70}")
    for task_id, model_name, status, elapsed in results_log:
        print(f"  {task_id:10s} {model_name:25s} {status:20s} {elapsed:7.1f}s")
    print(f"{'=' * 70}")

    # 保存汇总到文件
    summary_path = os.path.join(CONFIG["output_root"], "summary.log")
    os.makedirs(CONFIG["output_root"], exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"统一推理汇总报告\n")
        f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"任务: {', '.join(args.tasks)}\n")
        f.write(f"总计: {total}  成功: {success}  失败: {failed}  跳过: {skipped}\n")
        f.write("-" * 70 + "\n")
        for task_id, model_name, status, elapsed in results_log:
            f.write(f"  {task_id:10s} {model_name:25s} {status:20s} {elapsed:7.1f}s\n")
    print(f"\n  汇总报告已保存至: {summary_path}")


if __name__ == "__main__":
    main()
