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

七策略与中心分析代码在 **`cascade_engine/`**；本目录提供 **配置 + 编排**，在 NHC 私有机上从 `my_Thyroid_Infer` 一键运行。

## 环境

- Conda：`thyroid_infer`（`thyroid_infer_env.tar.gz` 下载见 [`docs/NHC_QUICKSTART_ZH.md`](docs/NHC_QUICKSTART_ZH.md)）
- **NHC 七策略**：内置引擎 **`nhc_routing_ablation/cascade_engine/`**，见 [`docs/NHC_ENGINE_ZH.md`](docs/NHC_ENGINE_ZH.md)
- **快速开始**：[`docs/NHC_QUICKSTART_ZH.md`](docs/NHC_QUICKSTART_ZH.md)（能联网 / 离线两种路径、托管地址、**R2.6 / shijie**）
- **私有机操作说明**：[`docs/NHC_OPERATOR_GUIDE_ZH.md`](docs/NHC_OPERATOR_GUIDE_ZH.md)（配置细节、产物与故障排查）
- 数据政策：**无** LLM API

## 专家池（`config/EXPERT_POOL.md`）

| 分割 5 | `all_datasets`, `TN3K`, `ThyroidXL`, `PKTN_comprehensive`, `TN5K` |
| 分类 4 | `all_datasets`, `TN3K`, `TN5K`, `ThyroidXL` |
| Radiomics | AutoGluon `dataset_3`（yaml 内 `radiomics_judge`） |

策略 (6) 的 joblib 在 **`nhc_routing_ablation/artifacts/b3/tn5k/`**（TN5K validation 拟合），专家名与上表相同。

## 预处理 NHC 数据从哪来（`run_seven_strategies` 默认行为）

全量 profile **`nhc_misd_tus`** 不会使用 yaml 里写死的 `/mnt/.../NHC-MISD-TUS/...`，而是：

1. 读 **`my_Thyroid_Infer/config.yaml`** → `prepare.output_dir`（默认 `./datasets/processed`）
2. 读 **`<output_dir>/config.yaml`**（同事已跑 **`pipeline.py`** 时自动生成）
3. 将其中的 `datasets.nodule_*` / `labels.binary_json` 写入运行配置里的 **`pipeline.data`**（`image_input` / `gt_mask_dir` / `label_file`）

若第 2 步文件不存在，脚本会报错并提示先跑 `pipeline.py` 或修正 `prepare.output_dir`。

关闭该逻辑（仍用 yaml 内路径）：`python run_seven_strategies.py --no-use-processed-config`

## 与 B7 / R18–R20 的关系

- **`run_nhc_ablation.sh`** 已包含七策略 B3（**R12/R13**）与 **`_center_analysis/`**（**R18–R20**），一般 **无需** 其它单独 B7 脚本。

---

## 运行

**主编排（推荐）**

```bash
cd my_Thyroid_Infer/nhc_routing_ablation
bash run_nhc_ablation.sh
# 兼容旧名: bash run_seven_strategies.sh
```

跳过中心分析（只要 R12/R13；**仍会跑腺体单路径 B7**，除非 `SKIP_GLAND=1`）：

```bash
SKIP_CENTER_ANALYSIS=1 bash run_nhc_ablation.sh
```

只补算七策略中心表 + 腺体（已有 B3 结果）：

```bash
CENTER_ANALYSIS_ONLY=1 bash run_nhc_ablation.sh
```

只测腺体 B7 链路（跳过七策略中心块）：

```bash
CENTER_ANALYSIS_ONLY=1 SKIP_CENTER_ANALYSIS=1 bash run_nhc_ablation.sh
```

跳过腺体（七策略中心 + B3 仍跑）：

```bash
SKIP_GLAND=1 bash run_nhc_ablation.sh
```

**全量 NHC（8721 图）**

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/my_Thyroid_Infer/nhc_routing_ablation
bash run_nhc_ablation.sh
# 或
python run_nhc_ablation.py --profile nhc_misd_tus
```

**10 图样板**

```bash
# 1) 重跑 Phase 1–2（在 nhc_routing_ablation 目录，引擎已内置）
rm -rf ../results/nhc_routing_ablation/_intermediate/nhc_prepared_sample/pipeline_run/intermediate
python run_nhc_ablation.py --profile nhc_prepared_sample --skip-static --skip-learned --skip-center-analysis

# 2) 七策略
PROFILE=nhc_prepared_sample bash run_seven_strategies.sh
```

可选环境变量：`SKIP_PHASE12=1`（已有 Phase 1–2 缓存）、`SKIP_STATIC=1`、`SKIP_LEARNED=1`。

## 产出位置

| Profile | 默认输出根目录 |
|---------|----------------|
| `nhc_misd_tus` | `my_Thyroid_Infer/results/nhc_routing_ablation/nhc_misd_tus/` |
| `nhc_prepared_sample` | `my_Thyroid_Infer/results/nhc_routing_ablation/nhc_prepared_sample/` |

每个策略目录含 `results.json`、`b3_metrics.json` 等；汇总表在 **`_summary/b3_routing_table.csv`**（由 B3 `summarize` 生成）。

Phase 1–2 共享缓存（在 **my_Thyroid_Infer** 内）：

- 全量：`results/nhc_routing_ablation/_intermediate/nhc_misd_tus/pipeline_run/intermediate/`
- 样板：`results/nhc_routing_ablation/_intermediate/nhc_prepared_sample/...`（见对应 yaml）

运行结束会写 **`_run_manifest.json`**（策略编号 ↔ 子目录映射）。

## 填 LaTeX 表

1. 从各策略 `b3_metrics.json` 或 `_summary/b3_routing_table.csv` 取 Dice/AUROC 与 CI。  
2. 将 NHC 行写入稿面 **`routing_ablation_tables_*.tex`**（或维护者提供的 generated 目录）。  
3. 策略 (6) 使用 `nhc_routing_ablation/artifacts/b3/tn5k/learned_router_*.joblib`（**不在 NHC 上 fit**）。

## Learned（策略 6）：test 拟合试跑（→ 合并 val）

稿面目标是 **OpenThyroid 合并 validation 拟合、外测 frozen**。仓库主表仍是 **各队列 validation**（`artifacts/b3/tn5k/` 等）。

**试跑（在 TN5K 测试集 intermediate 上 fit，不用 validation）**：

```bash
# 前置：维护者已准备 TN5K Phase1–2 intermediate（NHC 同事通常跳过本节）
cd nhc_routing_ablation
python fit_learned_on_test.py --cohort tn5k
# 产出：nhc_routing_ablation/artifacts/b3/learned_fit_tn5k_test/*.joblib（若已配置）
```

将 `config/nhc_*.yaml` 的 `learned_router` 改为 `artifacts/b3/learned_fit_tn5k_test/...` 后再跑 `run_nhc_ablation.sh`。

⚠️ Test 上 fit 不能在同一 test 上当作无偏评估。定稿请用 **合并 validation** 拟合（`cascade_engine/scripts/revision/b3/fit_learned_router.py`）。

---

## `_center_analysis/` 产物（七策略 + 腺体单路径）

| 路径 | 内容 |
|------|------|
| `heterogeneity_by_strategy.csv` | 七策略 × `nodule_dice` / `classification_score`；另有一行 **`routing_mode=gland`**、`metric=gland_dice`（单权重，非 B3 策略） |
| `gland/` | 腺体 B7（单权重）：`per_image_gland_metrics.jsonl`、`nhc_center_performance.csv`、`nhc_heterogeneity_summary.csv`、`b7_report.json` |
| `<mode>/` | 各路由策略中心性能 + 结节/分类异质性 |
| `center_performance_by_strategy.csv` | 七策略宏平均 Dice / AUROC |
| `center_analysis_summary_wide.csv` | 上两项合并宽表 |
| `<mode>/b7_report.json` | 与 B7 同结构的中心表 + 异质性 |
| `<mode>/nhc_center_performance.csv` | 该策略各中心 Dice/AUROC |
| `<mode>/nhc_heterogeneity_summary.csv` | 该策略 *I*² / τ² |
| `center_descriptive_from_labels.json` | labels 统计的中心图像数/恶性比例（**非 G3**） |

可选环境变量：`G3_CSV=...` merge 院方元数据；`SKIP_CENTER_ANALYSIS=1` 仅跳过七策略中心块；`SKIP_GLAND=1` 跳过腺体；`GLAND_REFERENCE_MODE=best_single`。（旧名 `SKIP_GLAND_SMOKE` / `GLAND_SMOKE_REFERENCE_MODE` 仍兼容）
