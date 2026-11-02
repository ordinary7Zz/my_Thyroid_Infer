#!/usr/bin/env bash
# 转发到 nhc_routing_ablation/run_seven_strategies.sh
exec "$(cd "$(dirname "$0")" && pwd)/nhc_routing_ablation/run_seven_strategies.sh" "$@"
