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
  python run_all.py                    # 运行全部四个任务
  python run_all.py --tasks gland nodule           # 只运行分割任务
  python run_all.py --tasks binary tirads          # 只运行分类任务
  python run_all.py --tasks gland --models dinov3_unet medsam2  # 指定模型
  python run_all.py --dry_run          # 只打印命令不执行

配置:
  所有路径在下方 CONFIG 字典中集中管理，修改后即可使用。
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ============================================================================
# 配置区：修改这里的路径即可
# ============================================================================

CONFIG = {
    # --- 数据集路径 ---
    "datasets": {
        "gland_images": "./datasets/TGVideo_PNG/test/image",
        "gland_masks":  "./TGVideo_PNG/test/mask",
        "nodule_images": "./datasets/TN3K/test/images",
        "nodule_masks":  "./datasets/TN3K/test/masks",
        "binary_images": "./datasets/TN3K/test/images",
        "tirads_images": "./datasets/Cine-Clip/test/images",
    },

    # --- 标签文件 ---
    "labels": {
        "binary_json":   "./datasets/TN3K/test/TN3K_test_label.json",
        "binary_field":  "malignancy",
        "tirads_json":   "./datasets/Cine-Clip/test/Cine-Clip_test_label.json",
        "tirads_field":  "tirads",
    },

    # --- 模型权重 ---
    "weights": {
        # 分割 — 腺体
        "gland": {
            "dinov3_unet": "./infer_dinov3_unet/checkpoints/gland/dino_unet_train_TGVideo_epoch_30.pth",
            "medsam2":     "./infer_medsam2/checkpoints/MedSAM2_TG_Video/checkpoint_10.pt",
            "medsegx":     "./infer_medsegx/checkpoints/TG_Video/checkpoint_epoch_29.pth",
            "transunet":   "./infer_transunet/checkpoints/TG_Video/epoch_49.pth",
            "ultrafedfm":  "/mnt/wangbd8/workspace/ThyroidAgent/UltraFedFM/my_pth/gland_seg/epoch_bestDice.pth",
        },
        # 分割 — 结节
        "nodule": {
            "dinov3_unet": "/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/train_Nodule/train_dataset_4/dino_unet_train_dataset_4_epoch_50.pth",
            "medsam2":     "/mnt/wangbd8/workspace/ThyroidAgent/MedSAM2/my_finetune/MedSAM2_Noudle_FullBox/checkpoints/checkpoint_5.pt",
            "medsegx":     "/mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/MedSegX/finetune/cross_site/US_ThyroidNodule/NoduleData/model_best.pth",
            "transunet":   "/mnt/wangbd8/workspace/ThyroidAgent/TransUNet/my_model/Nodule/epoch_49.pth",
            "ultrafedfm":  "/mnt/wangbd8/workspace/ThyroidAgent/UltraFedFM/my_pth/nodule_seg/epoch_bestDice.pth",
        },
        # 分类 — 良恶性二分类
        "binary": {
            "biomedclip":            "/mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_BiomedCLIP/output/BM/best_model.pth",
            "medsiglip":             "/mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_MedSigLIP/checkpoints/binary_cls/best_model.pt",
            "ultrafedfm":            "/mnt/wangbd8/workspace/ThyroidAgent/UltraFedFM/output_dir/dataset_3_cls_experiment/checkpoint-best_auroc.pth",
            "dinov3_unet_multitask": "/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/multitask/best_model.pth",
        },
        # 分类 — TIRADS 五分类
        "tirads": {
            "biomedclip":            "/mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_BiomedCLIP/output/TIRADS/best_model.pth",
            "medsiglip":             "/mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_MedSigLIP/checkpoints/multi_cls/best_model.pt",
            "ultrafedfm":            "/mnt/wangbd8/workspace/ThyroidAgent/UltraFedFM/output_dir/Cine-Clip_TIRADS/checkpoint-best_auroc.pth",
            "dinov3_unet_multitask": "/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/multitask/best_model.pth",
        },
    },

    # --- 预训练骨干模型目录 ---
    "pretrained": {
        "biomedclip_dir": "/mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_BiomedCLIP/pretrained_models/biomedclip",
        "medsiglip_dir":  "/mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_MedSigLIP/pretrained/medsiglip-448",
        "medsegx_sam_dir": "/mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/SAM",
        "medsam2_config":  "sam2.1_hiera_t512.yaml",
    },

    # --- 输出根目录 ---
    "output_root": "./results",

    # --- 通用参数 ---
    "device": "cuda",
    "n_bootstrap": 2000,
}

# ============================================================================
# 任务定义：每个任务下各模型的命令构建器
# ============================================================================

ROOT = Path(__file__).resolve().parent


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
        "--config", pt["medsam2_config"],
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
        "--config", pt["medsam2_config"],
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
        """,
    )
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
    print(f"  任务: {', '.join(args.tasks)}")
    if args.models:
        print(f"  模型筛选: {', '.join(args.models)}")
    print(f"  输出目录: {CONFIG['output_root']}")
    print(f"  设备: {CONFIG['device']}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70)

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

            # 检查权重文件是否存在
            weight_idx = None
            for i, arg in enumerate(cmd):
                if arg in ("--ckpt", "--checkpoint", "--resume", "--model_weight"):
                    weight_idx = i + 1
                    break
            if weight_idx is not None and not os.path.isfile(cmd[weight_idx]):
                print(f"  [SKIP] 权重文件不存在: {cmd[weight_idx]}")
                skipped += 1
                results_log.append((task_id, model_name, "SKIP (权重不存在)", 0))
                continue

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
