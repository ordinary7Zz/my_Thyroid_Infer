#!/usr/bin/env bash
# NHC 主编排（无 LLM 策略 7）:
#   1. 七策略 B3 路由消融（R12/R13）
#   2. 七策略中心性能 + 结节/分类异质性 → _center_analysis/<mode>/ + heterogeneity_by_strategy.csv
#   3. 腺体分割（单 TGVideo 权重，不跑七策略）→ _center_analysis/gland/
#      B7：per_image_gland、中心 gland Dice、异质性 gland_dice（I²/τ²）
#      开发阶段 pred 可用 nodule_masks 占位；NHC 正式跑时改为 TGVideo 推理输出目录
#
# 全量 NHC（PROFILE=nhc_misd_tus，默认）:
#   从 my_Thyroid_Infer/config.yaml → prepare.output_dir → processed/config.yaml
# 产物：results/nhc_routing_ablation/<profile>/ 含 _summary、_center_analysis/（含 gland/）
# 跑完后自动打 zip（同级 <profile>.zip），再删 Phase1–2 seg/cls 与各策略 intermediate/；终端提示拷贝 zip
#   SKIP_RESULT_ZIP=1                 结束时不压缩结果目录
#   NHC_ABLATION_KEEP_CACHE=1         不删缓存（等同 --keep-cache）
#   SKIP_CENTER_ANALYSIS=1  只跳过七策略中心块（腺体仍默认跑，除非 SKIP_GLAND=1）
#   SKIP_GLAND=1            跳过腺体 B7 单路径（兼容旧名 SKIP_GLAND_SMOKE=1）
#   GLAND_REFERENCE_MODE=best_single  腺体目录拷贝 results/seg 用的参考 B3 子目录
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != *thyroid_infer* ]]; then
  echo "建议先: conda activate /mnt/wangbd8/workspace/envs/thyroid_infer"
fi

PROFILE="${PROFILE:-nhc_misd_tus}"
if [[ -n "${NHC_PROCESSED_DIR:-}" ]]; then export NHC_PROCESSED_DIR; fi
if [[ -n "${NHC_ABLATION_OUTPUT_ROOT:-}" ]]; then export NHC_ABLATION_OUTPUT_ROOT; fi
EXTRA=()
if [[ "${SKIP_PHASE12:-0}" == "1" ]]; then EXTRA+=(--skip-phase12); fi
if [[ "${SKIP_STATIC:-0}" == "1" ]]; then EXTRA+=(--skip-static); fi
if [[ "${SKIP_LEARNED:-0}" == "1" ]]; then EXTRA+=(--skip-learned); fi
if [[ "${USE_PROCESSED_CONFIG:-}" == "0" ]]; then EXTRA+=(--no-use-processed-config); fi
if [[ "${USE_PROCESSED_CONFIG:-}" == "1" ]]; then EXTRA+=(--use-processed-config); fi
if [[ "${SKIP_CENTER_ANALYSIS:-0}" == "1" ]]; then EXTRA+=(--skip-center-analysis); fi
if [[ "${SKIP_GLAND:-${SKIP_GLAND_SMOKE:-0}}" == "1" ]]; then EXTRA+=(--skip-gland); fi
if [[ "${CENTER_ANALYSIS_ONLY:-0}" == "1" ]]; then EXTRA+=(--center-analysis-only); fi
if [[ "${NHC_ABLATION_KEEP_CACHE:-0}" == "1" ]]; then EXTRA+=(--keep-cache); fi
if [[ -n "${G3_CSV:-}" ]]; then EXTRA+=(--g3-csv "$G3_CSV"); fi
GLAND_REF="${GLAND_REFERENCE_MODE:-${GLAND_SMOKE_REFERENCE_MODE:-}}"
if [[ -n "$GLAND_REF" ]]; then
  EXTRA+=(--gland-reference-mode "$GLAND_REF")
fi

exec python run_nhc_ablation.py --profile "$PROFILE" "${EXTRA[@]}" "$@"
