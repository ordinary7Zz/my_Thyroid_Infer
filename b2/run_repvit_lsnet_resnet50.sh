#!/usr/bin/env bash
# 一次性：RepViT + LSNet + ResNet-50 在所有分类 cohort 上推理 → b2_exports →（可选）cluster CI
#
# 不跑 BiomedCLIP / MedSigLIP / UltraFedFM 等 run_all 已有模型。
#
# 用法:
#   cd /mnt/wangbd8/workspace/ThyroidAgent/my_Thyroid_Infer
#   chmod +x b2/run_repvit_lsnet_resnet50.sh
#   export CUDA_VISIBLE_DEVICES=0
#   ./b2/run_repvit_lsnet_resnet50.sh
#
# 可选环境变量:
#   COHORTS          逗号分隔，默认 tn3k,tn5k,thyroidxl,zjh8k,ddti_cls
#   SKIP_INFER=1     仅 convert + CI（已有 results/b2/<cohort>/binary/{repvit,lsnet,resnet50})
#   SKIP_CI=1        不跑 ThyroidCascadeAgent/run_b2_exports_all.sh
#   PYTHON           默认 python

set -euo pipefail

INFER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${INFER_ROOT}/.." && pwd)"
CASCADE_ROOT="${WORKSPACE_ROOT}/ThyroidCascadeAgent"
LOG_DIR="${INFER_ROOT}/results/b2"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/repvit_lsnet_resnet50_${STAMP}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"
COHORTS="${COHORTS:-tn3k,tn5k,thyroidxl,zjh8k,ddti_cls}"
MODELS="repvit,lsnet,resnet50"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo " RepViT + LSNet + ResNet-50 (classification cohorts)"
echo " started: $(date -Iseconds)"
echo " infer:   ${INFER_ROOT}"
echo " cohorts: ${COHORTS}"
echo " GPU:     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " log:     ${LOG_FILE}"
echo "======================================================================"

cd "${INFER_ROOT}"

INFER_ARGS=(--cohorts "${COHORTS}" --models "${MODELS}")

if [[ "${SKIP_INFER:-0}" == "1" ]]; then
  echo "SKIP_INFER=1 → 仅导出 b2_exports"
  INFER_ARGS+=(--skip-infer)
else
  echo "Phase 1: run_all_b2 (binary, models=${MODELS})"
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
echo "log: ${LOG_FILE}"
