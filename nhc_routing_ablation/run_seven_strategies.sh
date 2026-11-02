#!/usr/bin/env bash
# NHC routing ablation: seven strategies (no LLM). Requires GPU + thyroid_infer env.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != *thyroid_infer* ]]; then
  echo "建议先: conda activate /mnt/wangbd8/workspace/envs/thyroid_infer"
fi

PROFILE="${PROFILE:-nhc_misd_tus}"
EXTRA=()
if [[ "${SKIP_PHASE12:-0}" == "1" ]]; then EXTRA+=(--skip-phase12); fi
if [[ "${SKIP_STATIC:-0}" == "1" ]]; then EXTRA+=(--skip-static); fi
if [[ "${SKIP_LEARNED:-0}" == "1" ]]; then EXTRA+=(--skip-learned); fi

exec python run_seven_strategies.py --profile "$PROFILE" "${EXTRA[@]}" "$@"
