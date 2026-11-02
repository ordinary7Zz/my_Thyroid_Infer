#!/usr/bin/env bash
# B2 对比基线：全部模型 × cohorts.yaml 中全部队列
#   - 分割: transunet, medsam2, medsegx, ultrafedfm
#   - 分类: biomedclip, medsiglip, ultrafedfm, repvit, lsnet, resnet50
# 推理 → results/b2/<cohort>/ → b2_exports/ →（可选）ThyroidCascadeAgent cluster CI
#
# 默认跳过 results/b2 里已有输出的 model×cohort（--skip-existing）。
#
# 用法:
#   cd /mnt/wangbd8/workspace/ThyroidAgent/my_Thyroid_Infer
#   chmod +x baseline_export/run_b2_all.sh
#   export CUDA_VISIBLE_DEVICES=0
#   ./baseline_export/run_b2_all.sh
#
# 环境变量:
#   CUDA_VISIBLE_DEVICES   GPU（默认 0）
#   COHORTS                逗号分隔，默认 yaml 全部 8 队列
#   MODELS                 逗号分隔 run_all 模型 key，默认 seg+cls 全表
#   SKIP_EXISTING=0        强制重跑已有结果
#   SKIP_INFER=1           只 convert + CI（不推理）
#   SKIP_CI=1              不跑 run_b2_exports_all.sh
#   FORCE_CONVERT=1        推理阶段也带 --skip-existing 时仍全量 convert（默认每次 convert 全 cohort）
#   PYTHON                 python 解释器

set -euo pipefail

INFER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${INFER_ROOT}/.." && pwd)"
CASCADE_ROOT="${WORKSPACE_ROOT}/ThyroidCascadeAgent"
LOG_DIR="${INFER_ROOT}/results/b2"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/b2_all_${STAMP}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo " B2 all baselines (my_Thyroid_Infer)"
echo " started:  $(date -Iseconds)"
echo " infer:    ${INFER_ROOT}"
echo " cohorts:  ${COHORTS:-ALL (see baseline_export/cohorts.yaml)}"
echo " models:   ${MODELS:-ALL seg+cls in yaml}"
echo " skip_existing: ${SKIP_EXISTING:-1}"
echo " log:      ${LOG_FILE}"
echo "======================================================================"

cd "${INFER_ROOT}"

INFER_ARGS=()
[[ -n "${COHORTS:-}" ]] && INFER_ARGS+=(--cohorts "${COHORTS}")
[[ -n "${MODELS:-}" ]] && INFER_ARGS+=(--models "${MODELS}")

if [[ "${SKIP_INFER:-0}" == "1" ]]; then
  echo "Phase 1: SKIP_INFER → convert only"
  INFER_ARGS+=(--skip-infer)
else
  echo "Phase 1: run_all_extended inference (nodule + binary only)"
  if [[ "${SKIP_EXISTING:-1}" != "0" ]]; then
    INFER_ARGS+=(--skip-existing)
  fi
fi

"${PYTHON}" baseline_export/run_infer_batch.py "${INFER_ARGS[@]}"

if [[ "${SKIP_CI:-0}" != "1" ]]; then
  echo ""
  echo "Phase 2: cluster-bootstrap CI"
  cd "${CASCADE_ROOT}"
  bash scripts/run_b2_exports_all.sh
  echo "Index: ${CASCADE_ROOT}/docs/revision/b2/tables/BASELINES_INDEX.json"
else
  echo "SKIP_CI=1 → 跳过 CI"
fi

echo ""
echo "finished: $(date -Iseconds)"
echo "b2_exports: ${WORKSPACE_ROOT}/b2_exports"
echo "log: ${LOG_FILE}"
