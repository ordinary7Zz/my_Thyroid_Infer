#!/usr/bin/env python3
"""
串联 Pipeline 入口：分割筛选 → 分类筛选。

待做文档任务1+2的完整实现：
  1. 分割 Agent 用 radiomics 裁判评估 pred mask 可信度
  2. 筛选后的 mask 喂给分类 Agent
用法: python run_pipeline.py [--config config/config.yaml]
"""

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path

from shared.sklearn_warnings import configure_sklearn_pickle_warnings

configure_sklearn_pickle_warnings()

import yaml

from shared.image_io import ImageIO
from shared.llm_client import LLMClient
from segmentation import (
    SegModelRegistry,
    SegmentationQualityEvaluator,
    SegmentationAgent,
)
from segmentation.model_factory import build_seg_model
from classification import (
    ClsModelRegistry,
    LLMClassificationAgent,
)
from classification.model_factory import build_cls_model
from radiomics_judge import RadiomicsJudge
from pipeline import CascadePipeline
from shared.routing_log import build_routing_logger_from_config


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_seg_registry(seg_cfg: dict, device: str = "cuda") -> SegModelRegistry:
    registry = SegModelRegistry()
    for m in seg_cfg.get("models", []):
        m = dict(m)
        m["device"] = device
        model = build_seg_model(m)
        model.load_model()
        registry.register_model(model)
    return registry


def build_cls_registry(cls_cfg: dict, device: str = "cuda") -> ClsModelRegistry:
    registry = ClsModelRegistry()
    for m in cls_cfg.get("models", []):
        m = dict(m)
        m["device"] = "cpu" if m.get("type") == "autogluon_radiomics" else device
        model = build_cls_model(m)
        model.load_model()
        registry.register_model(model)
    return registry


def main():
    parser = argparse.ArgumentParser(description="串联 Pipeline: 分割筛选 → 分类筛选")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stop-after-phase2",
        action="store_true",
        help="仅跑 Phase 1–2 写 intermediate，跳过 Phase 3–5（无 LLM 路由）",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="B6：只对 N 张图跑 pipeline")
    parser.add_argument("--sample-seed", type=int, default=None, help="与 --sample-mode random 联用")
    parser.add_argument(
        "--sample-mode",
        choices=("first", "random"),
        default=None,
        help="子集选取方式；默认 first（或 yaml pipeline.data.stability_sample）",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    llm_cfg = config["shared"]["agent_llm"]
    seg_cfg = config["segmentation"]
    cls_cfg = config["classification"]
    judge_cfg = config.get("radiomics_judge", {})
    pipe_cfg = config.get("pipeline", {})
    data_cfg = pipe_cfg.get("data", {})

    # 阶段开关
    run_segmentation = pipe_cfg.get("run_segmentation", True)
    run_classification = pipe_cfg.get("run_classification", True)
    if not run_segmentation and not run_classification:
        print("⚠️ 分割与分类均已关闭，无可执行阶段，退出")
        return

    stop_after_phase2 = args.stop_after_phase2 or pipe_cfg.get("stop_after_phase2", False)

    # 检查是否需要 LLM（阶段关闭时对应 Agent 也不需要）
    enable_seg_agent = run_segmentation and seg_cfg.get("agent", {}).get("enable_agent", True)
    enable_cls_agent = run_classification and cls_cfg.get("agent", {}).get("enable_agent", True)
    need_llm = (enable_seg_agent or enable_cls_agent) and not stop_after_phase2
    if stop_after_phase2:
        print("✓ stop_after_phase2：本 run 仅 Phase 1–2，不调用 LLM")

    # LLM 客户端
    if need_llm:
        api_key = os.getenv(llm_cfg["api_key_env"], "")
        if not api_key:
            print("⚠️ 警告: API key 未设置（环境变量 %s）" % llm_cfg["api_key_env"])
        llm_client = LLMClient(
            api_key=api_key,
            base_url=llm_cfg["base_url"],
            model_name=llm_cfg["model_name"],
            temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
        )
    else:
        print("✓ 分割和分类 Agent 均已禁用（或对应阶段已关闭），跳过 LLM 客户端初始化")
        llm_client = None

    # Radiomics 裁判（仅在分割阶段开启且会被用到时加载：seg Agent 或分类阶段）
    judge = None
    need_judge = run_segmentation and (enable_seg_agent or run_classification)
    if judge_cfg.get("enabled", False) and need_judge:
        try:
            judge = RadiomicsJudge(
                model_dir=judge_cfg["model_dir"],
                top_k_features=judge_cfg.get("top_k_features", 5),
                shap_reference_path=judge_cfg.get("shap_reference_path"),
                radiomics_params=judge_cfg.get("radiomics_params"),
            )
            print("✓ Radiomics 裁判已加载")
        except Exception as e:
            print(f"⚠️ Radiomics 裁判加载失败: {e}")

    # 构建组件（不再预加载所有模型，由 pipeline 按需加载/卸载）
    print("\n构建 Pipeline（模型将在推理时按需加载/卸载）...")

    output_dir_final = pipe_cfg.get("output", {}).get("output_dir", "output/pipeline_run")
    routing_logger = build_routing_logger_from_config(config, output_dir_final)
    if routing_logger.enabled and not stop_after_phase2:
        print(f"✓ 路由审计日志: {routing_logger.path}")
    elif routing_logger.enabled and stop_after_phase2:
        print("  routing_log: 跳过（stop_after_phase2，本 run 无 LLM 路由）")

    seg_agent = SegmentationAgent(
        llm_client=llm_client,
        quality_evaluator=SegmentationQualityEvaluator(),
        radiomics_judge=judge,
        config=seg_cfg.get("agent", {}),
        routing_logger=routing_logger,
    )
    cls_agent = LLMClassificationAgent(
        llm_client=llm_client,
        config=cls_cfg.get("agent", {}),
        routing_logger=routing_logger,
    )

    # 构建 Pipeline（不预加载模型，由 run_batch 按需加载）
    pipeline = CascadePipeline(
        seg_agent=seg_agent,
        cls_agent=cls_agent,
        seg_registry=None,
        cls_registry=None,
        image_io=ImageIO(),
        config=config,
        device=args.device,
    )

    from pipeline.revision.b6.case_selection import (
        MANIFEST_NAME,
        select_stability_cases,
        write_manifest,
    )

    stab_cfg = data_cfg.get("stability_sample") or {}
    sample_size = args.sample_size if args.sample_size is not None else stab_cfg.get("n")
    sample_seed = (
        args.sample_seed if args.sample_seed is not None else stab_cfg.get("seed")
    )
    sample_mode = args.sample_mode or stab_cfg.get("mode", "first")
    image_paths_subset = None
    if sample_size is not None:
        mode_eff = "random" if sample_seed is not None else sample_mode
        image_paths_subset = select_stability_cases(
            Path(data_cfg["image_input"]),
            int(sample_size),
            start_index=data_cfg.get("start_index", 0),
            seed=sample_seed,
            mode=mode_eff,
        )
        sample_mode = mode_eff
        write_manifest(
            Path(output_dir_final) / MANIFEST_NAME,
            image_paths_subset,
            {
                "n": len(image_paths_subset),
                "seed": sample_seed,
                "mode": sample_mode,
                "source": "run_pipeline",
            },
        )
        print(
            f"✓ B6 子集: {len(image_paths_subset)} 张 "
            f"(mode={sample_mode}, seed={sample_seed}) → {MANIFEST_NAME}"
        )

    # 运行
    results = pipeline.run_batch(
        image_dir=data_cfg["image_input"],
        output_dir=output_dir_final,
        gt_mask_dir=data_cfg.get("gt_mask_dir"),
        label_file=data_cfg.get("label_file"),
        label_key=data_cfg.get("label_key", "malignancy"),
        input_device_info=data_cfg.get("device_info"),
        input_data_info=data_cfg.get("input_data_info"),
        start_index=data_cfg.get("start_index", 0),
        max_images=data_cfg.get("max_images") if image_paths_subset is None else None,
        image_paths=image_paths_subset,
    )

    n_batch = getattr(pipeline, "last_batch_image_count", None)
    if stop_after_phase2:
        n = n_batch if n_batch is not None else len(results)
        print(
            f"\nPipeline 完成: Phase 1–2 覆盖 {n} 张图像"
            "（stop_after_phase2；Phase 3–5 未运行，results 为空列表）"
        )
        print("✓ stop_after_phase2：跳过 evaluate / compare_models")
        return

    print(f"\nPipeline 完成: {len(results)} 张图像")

    # 评估指标
    from pipeline.evaluate import evaluate_pipeline
    evaluate_pipeline(
        results,
        output_dir_final,
        image_io=ImageIO(),
        gt_mask_dir=data_cfg.get("gt_mask_dir"),
    )

    # 对比各独立模型与 pipeline 的性能（需要 gt_mask_dir 和 label_file，且分割与分类阶段均开启）
    label_file = data_cfg.get("label_file")
    gt_mask_dir = data_cfg.get("gt_mask_dir")
    if gt_mask_dir and label_file and run_segmentation and run_classification:
        from pipeline.compare_models import compare_models
        compare_models(
            output_dir=output_dir_final,
            gt_mask_dir=gt_mask_dir,
            label_file=label_file,
            label_key=data_cfg.get("label_key", "malignancy"),
        )
    else:
        print("\n⚠️ 缺少 gt_mask_dir/label_file 或分割/分类阶段未全开，跳过模型对比")


if __name__ == "__main__":
    main()
