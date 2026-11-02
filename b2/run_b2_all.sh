#!/usr/bin/env bash
# 兼容入口 → baseline_export/run_b2_all.sh
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/baseline_export/run_b2_all.sh" "$@"
