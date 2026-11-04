# `pipeline/revision/` — 返修库代码（B3–B8）

产品主路径 **`run_pipeline.py` → `pipeline.cascade`** 仅依赖根目录 **核心三文件**。

**Import 约定**：`from pipeline.revision.b8.calibration import …`（不再使用根目录 `pipeline/calibration.py`）。

| 目录 | 模块 |
|------|------|
| **b3/** | `routing_ablation`, `routing_methods`, `static_b3_bundle`, `seg_routing_offline`, `ablation_metrics`, `ablation_summary`, `intermediate_manifest` |
| **b4/** | `b4_summary`, `b4_pathb_summary` |
| **b5/** | `analyze_routing_bias` |
| **b6/** | `routing_stability` |
| **b7/** | `nhc_heterogeneity`, `meta_regression` |
| **b8/** | `calibration`, `abstention`, `plot_calibration`, `plot_abstention`, `backfill_uncertainty` |

CLI：[`../../scripts/revision/README.md`](../../scripts/revision/README.md)

`best_single` 专家名解析见 [`../../routing/best_single_expert.py`](../../routing/best_single_expert.py)（主 `cascade` 与 B3 共用）。
