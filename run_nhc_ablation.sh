#!/usr/bin/env bash
# 转发到 nhc_routing_ablation/run_nhc_ablation.sh（B3 七策略 + 中心分析 + 腺体 B7 单路径）
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/nhc_routing_ablation/run_nhc_ablation.sh" "$@"
