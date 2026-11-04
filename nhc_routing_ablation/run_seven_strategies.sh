#!/usr/bin/env bash
# 兼容入口：请优先使用 run_nhc_ablation.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/run_nhc_ablation.sh" "$@"
