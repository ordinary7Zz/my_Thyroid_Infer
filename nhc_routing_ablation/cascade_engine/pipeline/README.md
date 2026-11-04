# `pipeline/` 模块索引

主推理入口：仓库根目录 **`run_pipeline.py`** → **`cascade.py`**。  
返修 CLI：**`scripts/revision/`**（见 [`../scripts/README_REVISION.md`](../scripts/README_REVISION.md)）。

## 层级

| 层级 | 路径 | 说明 |
|------|------|------|
| **核心** | `cascade.py`, `evaluate.py`, `compare_models.py` | 产品级联与评估 |
| **返修** | [`revision/b3/`](revision/b3/) … [`revision/b8/`](revision/b8/) | B 包专用库；import 形如 `pipeline.revision.b8.abstention` |

详见 [`revision/README.md`](revision/README.md)。

## 核心文件

| 模块 | 作用 |
|------|------|
| [`cascade.py`](cascade.py) | Phase 1–5 级联 |
| [`evaluate.py`](evaluate.py) | Dice/AUROC 等；cluster bootstrap CI |
| [`compare_models.py`](compare_models.py) | Pipeline vs 独立模型对比 |

## 返修模块目录

| B | 目录 |
|---|------|
| B3 | `revision/b3/` — 路由消融、static bundle |
| B4 | `revision/b4/` — 组件消融表 |
| B5 | `revision/b5/` — 路由偏倚 |
| B6 | `revision/b6/` — `routing_stability` |
| B7 | `revision/b7/` — NHC 异质性 |
| B8 | `revision/b8/` — 校准与拒答 |

## 依赖简图

```text
run_pipeline.py → cascade.py → evaluate.py
run_routing_stability.py → revision/b6/routing_stability.py

cascade.py → routing/best_single_expert.get_best_single_names
```
