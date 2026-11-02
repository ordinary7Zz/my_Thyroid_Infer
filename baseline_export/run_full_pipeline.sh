#!/usr/bin/env bash
# B2 基线：一次性跑完 b2_cohorts.yaml 中全部队列（my_Thyroid_Infer / run_all）
# 并写入 b2_exports/，再在 ThyroidCascadeAgent 计算 cluster-bootstrap CI。
#
# RepViT / LSNet / ResNet-50 由 run_all_extended + infer_repvit|lsnet|resnet50 提供。
#
# 用法:
#   cd /mnt/wangbd8/workspace/ThyroidAgent/my_Thyroid_Infer
#   chmod +x b2/run_b2_full_pipeline.sh
#   export CUDA_VISIBLE_DEVICES=0
#   ./b2/run_b2_full_pipeline.sh
#
# 可选环境变量:
#   CUDA_VISIBLE_DEVICES   GPU（默认 0）
#   COHORTS                逗号分隔 cohort id，默认跑 yaml 中全部 8 个
#   SKIP_INFER=1           跳过推理，仅从已有 results/b2 转 b2_exports + 算 CI
#   PYTHON                 Python 解释器（默认当前 PATH 的 python）
#
# 日志:
#   results/b2/full_pipeline_<时间戳>.log

set -euo pipefail

INFER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${INFER_ROOT}/.." && pwd)"
CASCADE_ROOT="${WORKSPACE_ROOT}/ThyroidCascadeAgent"
B2_CONFIG="${INFER_ROOT}/baseline_export/cohorts.yaml"
LOG_DIR="${INFER_ROOT}/results/b2"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/full_pipeline_${STAMP}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"

mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo " B2 full pipeline"
echo " started:  $(date -Iseconds)"
echo " infer:    ${INFER_ROOT}"
echo " cascade:  ${CASCADE_ROOT}"
echo " GPU:      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " log:      ${LOG_FILE}"
echo "======================================================================"

if [[ ! -f "${B2_CONFIG}" ]]; then
  echo "[error] missing ${B2_CONFIG}" >&2
  exit 1
fi

if [[ ! -d "${CASCADE_ROOT}/scripts" ]]; then
  echo "[error] missing ThyroidCascadeAgent at ${CASCADE_ROOT}" >&2
  exit 1
fi

cd "${INFER_ROOT}"

INFER_ARGS=()
if [[ -n "${COHORTS:-}" ]]; then
  echo "cohorts (subset): ${COHORTS}"
  INFER_ARGS+=(--cohorts "${COHORTS}")
else
  echo "cohorts: ALL in b2_cohorts.yaml (pktn, tn3k, tn5k, thyroidxl, zjh8k, ddti_seg, ddti_cls, rjh7k)"
fi

if [[ "${SKIP_INFER:-0}" == "1" ]]; then
  echo "Phase 1/2: SKIP_INFER=1 → convert only → b2_exports"
  INFER_ARGS+=(--skip-infer)
else
  echo "Phase 1/2: inference + b2_exports (run_all_extended --mode finetune)"
  echo "  仅任务: nodule(结节分割) + binary(良恶性); 不含腺体 gland / TIRADS tirads"
  if [[ "${SKIP_EXISTING:-1}" != "0" ]]; then
    INFER_ARGS+=(--skip-existing)
  fi
fi
"${PYTHON}" baseline_export/run_infer_batch.py "${INFER_ARGS[@]}"

echo ""
echo "Phase 2/2: cluster-bootstrap CI (run_b2_exports_all.sh)"
cd "${CASCADE_ROOT}"
bash scripts/run_b2_exports_all.sh

INDEX="${CASCADE_ROOT}/docs/revision/b2/tables/BASELINES_INDEX.json"
echo ""
echo "======================================================================"
echo " finished: $(date -Iseconds)"
echo " BASELINES_INDEX: ${INDEX}"
echo " b2_exports root: ${WORKSPACE_ROOT}/b2_exports"
echo "======================================================================"

if [[ -f "${INDEX}" ]]; then
  echo ""
  echo "--- exports indexed ($( "${PYTHON}" -c "import json; print(len(json.load(open('${INDEX}'))))" )) ---"
  "${PYTHON}" -c "
import json
from pathlib import Path
idx = json.loads(Path('${INDEX}').read_text())
for row in idx:
    print(f\"  {row['model_id']:12s} {row['cohort_id']:12s} n={row['n_records']}\")
"
else
  echo "[warn] index not found" >&2
  exit 1
fi

echo ""
echo "Full log: ${LOG_FILE}"
