# NHC 七策略路由消融（无 LLM）

在 **NHC 私有数据**上跑回信 **R12/R13**（`tab_routing_ablation_seg` / `tab_routing_ablation_cls`）所需的 **7 条策略**，不含策略 **(7) Full ThyroidXAgent（LLM）**。

| 策略 | 输出子目录 | 说明 |
|------|------------|------|
| (1) Best single | `best_single/` | 静态 bundle |
| (2) Soft voting | `soft_voting/` | |
| (3) Majority | `majority/` | |
| (4) Static rule | `static_rule/` | |
| (5) Max-confidence | `max_confidence/` | |
| (6) Learned router | `learned/` | 单独 `run_learned_b3` |
| (8) Oracle | `oracle/` | 上界 |
| **(7) LLM** | — | **NHC 禁止，不跑** |

实现复用 **`ThyroidCascadeAgent`** 的 B3（`run_b3_static_bundle.py`、`run_learned_b3.py`）；本目录只提供 **配置 + 编排入口**，便于在托管 NHC 的机器从 `my_Thyroid_Infer` 一键启动。

## 环境

- Conda：`thyroid_infer`（见 `ThyroidCascadeAgent/docs/revision/b7/NHC_GPU_AND_WEIGHTS_ZH.md`）
-  sibling 仓库：`../ThyroidCascadeAgent`
- 数据政策：**无** Qwen/GPT/Gemini API（与 `docs/revision/nhc_sample/MODEL_POLICY_ZH.md` 一致）

## 专家池（`config/EXPERT_POOL.md`）

| 分割 5 | `all_datasets`, `TN3K`, `ThyroidXL`, `PKTN_comprehensive`, `TN5K` |
| 分类 4 | `all_datasets`, `TN3K`, `TN5K`, `ThyroidXL` |
| Radiomics | AutoGluon `dataset_3`（yaml 内 `radiomics_judge`） |

策略 (6) 的 joblib 路径在 `artifacts/b3/tn5k/`，表示 **在 TN5K validation 上 fit 的路由器**，专家名与上表相同，**不是**只用 TN5K 两个 checkpoint。

## 运行

**全量 NHC（8721 图）**

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/my_Thyroid_Infer/nhc_routing_ablation
bash run_seven_strategies.sh
# 或
python run_seven_strategies.py --profile nhc_misd_tus
```

**10 图样板**

```bash
# 1) 用 EXPERT_POOL 5+4 重跑 Phase 1–2（旧 cache 仅 2 专家时会失败）
cd ../ThyroidCascadeAgent
conda activate .../thyroid_infer
rm -rf output/b7/nhc_cascade_sample/intermediate
python run_pipeline.py --config ../my_Thyroid_Infer/nhc_routing_ablation/config/cascade_phase12_nhc_prepared_sample.yaml --stop-after-phase2

# 2) 七策略
cd ../my_Thyroid_Infer/nhc_routing_ablation
PROFILE=nhc_prepared_sample bash run_seven_strategies.sh
```

可选环境变量：`SKIP_PHASE12=1`（已有 Phase 1–2 缓存）、`SKIP_STATIC=1`、`SKIP_LEARNED=1`。

## 产出位置

| Profile | 默认输出根目录 |
|---------|----------------|
| `nhc_misd_tus` | `my_Thyroid_Infer/results/nhc_routing_ablation/nhc_misd_tus/` |
| `nhc_prepared_sample` | `my_Thyroid_Infer/results/nhc_routing_ablation/nhc_prepared_sample/` |

每个策略目录含 `results.json`、`b3_metrics.json` 等；汇总表在 **`_summary/b3_routing_table.csv`**（由 B3 `summarize` 生成）。

Phase 1–2 共享缓存（相对 Cascade 根目录）：

- 全量：`ThyroidCascadeAgent/output/NHC-MISD-TUS/pipeline_run/intermediate`
- 样板：`ThyroidCascadeAgent/output/b7/nhc_cascade_sample/intermediate`

运行结束会写 **`_run_manifest.json`**（策略编号 ↔ 子目录映射）。

## 填 LaTeX 表

1. 从各策略 `b3_metrics.json` 或 `_summary/b3_routing_table.csv` 取 Dice/AUROC 与 CI。  
2. **手改** `ThyroidCascadeAgent/docs/revision/暂定的修改和回复计划/正式回复/latex/generated/routing_ablation_tables_*.tex` 中 NHC 行（或新增 NHC cohort 块）。  
3. 策略 (6) 依赖 `artifacts/b3/tn5k/learned_router_*.joblib`（开发队列拟合，**不在 NHC 上 fit**）。

## Learned（策略 6）：test 拟合试跑（→ 合并 val）

稿面目标是 **OpenThyroid 合并 validation 拟合、外测 frozen**。仓库主表仍是 **各队列 validation**（`artifacts/b3/tn5k/` 等）。

**试跑（在 TN5K 测试集 intermediate 上 fit，不用 validation）**：

```bash
# 前置：ThyroidCascadeAgent 已有 output/TN5K/pipeline_run/intermediate
cd nhc_routing_ablation
python fit_learned_on_test.py --cohort tn5k
# 产出：ThyroidCascadeAgent/artifacts/b3/learned_fit_tn5k_test/*.joblib
```

将 `config/nhc_*.yaml` 的 `learned_router` 改为 `artifacts/b3/learned_fit_tn5k_test/...` 后再跑 `run_seven_strategies.sh`。

⚠️ Test 上 fit 不能在同一 test 上当作无偏评估；NHC 外测可用此 joblib 试跑七策略。定稿请改 **合并 validation** + `fit_learned_router.py --fit-split val`。

Cascade 侧新增：`--fit-split test` / yaml `learned_fit_split: test`（`scripts/revision/b3/fit_learned_router.py`）。

---

## 与 B7 / R19–R21 的区别

- **本目录**：多策略消融 → **R12/R13**  
- **`run_pipeline.py` + `run_b7_nhc.sh`**：单一非 LLM 部署口径 → **R19–R21** 中心表

二者都要跑时，可共用同一份 Phase 1–2 `intermediate`（专家池一致即可）。
