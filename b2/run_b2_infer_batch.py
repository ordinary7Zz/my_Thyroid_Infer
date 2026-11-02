#!/usr/bin/env python3
"""Compatibility shim: use baseline_export/run_infer_batch.py (configs under baseline_export/generated_configs/)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent.parent / "baseline_export" / "run_infer_batch.py"

if __name__ == "__main__":
    sys.argv[0] = str(_TARGET)
    runpy.run_path(str(_TARGET), run_name="__main__")
