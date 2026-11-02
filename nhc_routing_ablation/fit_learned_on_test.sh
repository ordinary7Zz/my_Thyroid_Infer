#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python fit_learned_on_test.py --cohort tn5k "$@"
