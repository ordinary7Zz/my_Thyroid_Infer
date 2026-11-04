# NHC 七策略：内置引擎说明

## 结论

- **`my_Thyroid_Infer` 主流程**（`pipeline.py` / `run_all.py` / 各 `infer_*`）与 **`nhc_routing_ablation/`（七策略）** 是同一仓库内的两套入口。
- 七策略所需的 **Phase1–2、B3 路由、B7 中心分析** Python 代码都在  
  **`nhc_routing_ablation/cascade_engine/`**  
  策略 (6) 的 joblib 在 **`nhc_routing_ablation/artifacts/b3/tn5k/`**。
- 私有机同事 **只需 clone / pull `my_Thyroid_Infer`**，按 [`NHC_QUICKSTART_ZH.md`](NHC_QUICKSTART_ZH.md) 跑通；细节见 [`NHC_OPERATOR_GUIDE_ZH.md`](NHC_OPERATOR_GUIDE_ZH.md)。

## 编排如何调用引擎

| 组件 | 作用 |
|------|------|
| `paths.get_engine_root()` | 指向 `cascade_engine/`（调试可用 `NHC_CASCADE_ENGINE_ROOT` 覆盖，一般不必） |
| `run_nhc_ablation.py` | 子进程在 `engine_root` 下执行 `run_pipeline.py`、`run_b3_static_bundle.py`、`run_learned_b3.py` |
| `center_analysis.py` | 从 `engine_root` 导入 `pipeline.revision.b7.*` 做中心性能与异质性 |

## 目录速查

```text
my_Thyroid_Infer/
├── run_nhc_ablation.sh          # 根入口（转发到 nhc_routing_ablation/）
├── nhc_routing_ablation/
│   ├── cascade_engine/          # 内置引擎（含 run_pipeline.py）
│   ├── artifacts/b3/tn5k/       # learned_router_*.joblib
│   ├── config/                  # nhc_misd_tus.yaml 等
│   └── run_nhc_ablation.py
└── results/nhc_routing_ablation/
    ├── _intermediate/           # Phase1–2 缓存
    └── nhc_misd_tus/            # 七策略结果（默认）
```

引擎快照对应的源码 commit（仅维护者更新引擎时用）：`cascade_engine/VENDOR_SOURCE.txt`。
