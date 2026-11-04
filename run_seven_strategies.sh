#!/usr/bin/env bash
# 转发到 nhc_routing_ablation/run_seven_strategies.sh
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/nhc_routing_ablation/run_seven_strategies.sh" "$@"
