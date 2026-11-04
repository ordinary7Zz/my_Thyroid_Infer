# NHC 私有机操作说明（my_Thyroid_Infer 单仓七策略）

面向：**已在机器上有一份 `my_Thyroid_Infer` 的同事**，需要补齐配置细节、权重路径、运行变体与回信表对应关系。

**先跑通流程**：[`NHC_QUICKSTART_ZH.md`](NHC_QUICKSTART_ZH.md)（能联网 / 不能联网两种路径、托管地址、**R2.6 需 shijie 回复**）。

---

## 1. 只需本仓库

| 能力 | 在仓库内的位置 |
|------|----------------|
| Phase 1–2 专家前向缓存 | `nhc_routing_ablation/cascade_engine/run_pipeline.py` |
| 七策略 B3（(1)–(6)(8)） | `cascade_engine/scripts/revision/b3/run_b3_static_bundle.py`、`run_learned_b3.py` |
| 中心性能 / 异质性 | `cascade_engine/pipeline/revision/b7/`（由 `run_nhc_ablation.py` 调用） |
| 策略 (6) 路由器 | `nhc_routing_ablation/artifacts/b3/tn5k/learned_router_{seg,cls}.joblib` |
| 一键入口 | 仓库根目录 **`bash run_nhc_ablation.sh`** |

**不需要**第二个代码仓库、**不需要**与 infer 同级的其它项目目录、**不需要**额外环境变量指向外部代码路径。

架构速查：[`NHC_ENGINE_ZH.md`](NHC_ENGINE_ZH.md)

### 1.1 更新代码（GitHub）

```bash
cd /path/to/my_Thyroid_Infer
git fetch origin
git pull origin main    # 默认分支若为 master 则改为 master
```

远程：`https://github.com/ordinary7Zz/my_Thyroid_Infer.git`

`git pull` 会更新 `cascade_engine/` 与仓库内的 learned joblib（若已纳入版本管理）。**`.pth` 权重不在 git 中**，见第 3 节或 [`NHC_QUICKSTART_ZH.md`](NHC_QUICKSTART_ZH.md)。

---

## 2. 目录与 Profile

| Profile | 用途 | 默认结果目录 |
|---------|------|----------------|
| `nhc_misd_tus` | 全量 NHC-MISD-TUS（默认） | `results/nhc_routing_ablation/nhc_misd_tus/` |
| `nhc_prepared_sample` | 小样调试（约 10 图） | `results/nhc_routing_ablation/nhc_prepared_sample/` |

Phase 1–2 **共享缓存**（可复跑、可 `SKIP_PHASE12=1`）：

```text
results/nhc_routing_ablation/_intermediate/<profile>/pipeline_run/intermediate/
  seg/   # 各专家分割 npz
  cls/   # 各专家分类 npz
```

配置：`nhc_routing_ablation/config/nhc_misd_tus.yaml`、`cascade_phase12_nhc_misd_tus.yaml` 等。

---

## 3. 权重：DINO + AutoGluon（含腺体 TGVideo）

NHC 七策略与腺体 B7 共用同一套下载（**无需单独下腺体**）：

| 模块 | 路径（相对 `my_Thyroid_Infer` 根） |
|------|-------------------------------------|
| 结节分割 ×5 | `infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_*.pth`（名单见 `nhc_routing_ablation/config/EXPERT_POOL.md`） |
| 良恶性分类 ×4 | `infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_*.pth` |
| 腺体 TGVideo（B7） | `infer_dinov3_unet/checkpoints/TGVideo/dino_unet_train_TGVideo_epoch_30.pth` |
| static_rule 用 Radiomics | `infer_autogluon/checkpoints/binary/autogluon_model_20260204_140346/` + `infer_autogluon/radiomics_2d.yaml` |

**不需要**为 NHC 下载：`infer_lsnet/`、`infer_repvit/`、`infer_resnet50/` 等（除非另跑 `run_all.py` 九模型）。

### 3.1 推荐：一键脚本（魔搭，国内）

在 **`my_Thyroid_Infer` 根目录**：

```bash
# 需已 ms login 或 MODELSCOPE_API_TOKEN
bash nhc_routing_ablation/scripts/download_nhc_b3_weights.sh
```

走 HuggingFace（可设镜像）：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内可选
hf auth login
WEIGHT_SOURCE=huggingface bash nhc_routing_ablation/scripts/download_nhc_b3_weights.sh
```

### 3.2 手动：HuggingFace 按文件下载

仓库：<https://huggingface.co/instincts7Zz/my_Thyroid_infer>（`--repo-type model`）

```bash
cd /path/to/my_Thyroid_Infer
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com   # 可选

hf download instincts7Zz/my_Thyroid_infer \
  infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_all_datasets_epoch_40.pth \
  infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_TN3K_epoch_50.pth \
  infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_ThyroidXL_epoch_50.pth \
  infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_PKTN_comprehensive_epoch_50.pth \
  infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_TN5K_epoch_50.pth \
  infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_all_datasets_epoch_40.pth \
  infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_TN3K_epoch_50.pth \
  infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_TN5K_epoch_50.pth \
  infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_ThyroidXL_epoch_50.pth \
  infer_dinov3_unet/checkpoints/TGVideo/dino_unet_train_TGVideo_epoch_30.pth \
  infer_autogluon/radiomics_2d.yaml \
  --repo-type model --local-dir .

hf download instincts7Zz/my_Thyroid_infer \
  infer_autogluon/checkpoints/binary/autogluon_model_20260204_140346 \
  --repo-type model --local-dir .
```

### 3.3 手动：ModelScope

仓库：<https://www.modelscope.cn/models/instincts/my_Thyroid_infer>

```bash
pip install modelscope
ms login --token YOUR_TOKEN   # 或 export MODELSCOPE_API_TOKEN=...
cd /path/to/my_Thyroid_Infer
bash nhc_routing_ablation/scripts/download_nhc_b3_weights.sh
```

---

## 4. 环境与 GPU

**推荐（与分发包一致）**：`thyroid_infer` 的 conda-pack 包为仓库根 **`thyroid_infer_env.tar.gz`**。已解压且 **`~/envs/thyroid_infer` 可用则无需再下**；若 tar 或环境目录已删除，**下载命令**见 [`NHC_QUICKSTART_ZH.md`](NHC_QUICKSTART_ZH.md)「下载 thyroid_infer_env.tar.gz」；解压步骤见同文档「已有仓库」或快速开始 §3。

之后进入仓库跑 NHC 前：

```bash
export CUDA_VISIBLE_DEVICES=0   # 按机器改
source ~/envs/thyroid_infer/bin/activate
```

（Linux x86_64；打包环境用 `source .../bin/activate`，不是 `conda activate`。若自行 `conda create -n thyroid_infer` 安装依赖，则可用 `conda activate thyroid_infer`，见主 `README.md`。）

**NHC 数据与模型政策（运行时必须遵守）**：

- **不调用** Qwen / GPT / Gemini 等 LLM API；配置中 `enable_agent: false`
- **不跑** 策略 **(7) Full ThyroidXAgent（LLM）**；回信表中该格写 N/A
- 专家为固定 DINO 池 +（static_rule）Radiomics；策略 (6) 使用仓库内 **frozen** joblib，**不在 NHC 上重新 fit 路由器**

---

## 5. NHC 数据准备

1. 原始数据目录写入 **`my_Thyroid_Infer/config.yaml`** → `prepare.input_dir`
2. 运行预处理（生成 `datasets/processed/` 及 **`datasets/processed/config.yaml`**）：

```bash
cd /path/to/my_Thyroid_Infer
python pipeline.py
# 或按主 README / pipeline.py 使用 --skip-* 仅跑 prepare
```

3. 确认存在：

- `datasets/processed/images/`、`nodule_masks/`、`gland_masks/`、`labels.json`
- `datasets/processed/name_mapping.json`（匿名映射，**中心分析必需**）
- `datasets/processed/config.yaml`

`prepare.output_dir` 默认 `./datasets/processed`。若改路径，设置环境变量 `NHC_PROCESSED_DIR` 或改 `config.yaml`，并与同事约定一致。

**全量 profile `nhc_misd_tus` 默认行为**：`run_nhc_ablation` 会读上述 `prepare.output_dir/config.yaml`，自动覆盖 yaml 里写死的 `pipeline.data` 路径（**不要**依赖开发机 `/mnt/...` 绝对路径）。

---

## 6. 运行七策略 + 中心分析 + 腺体 B7

### 6.1 全量（推荐）

```bash
cd /path/to/my_Thyroid_Infer
bash run_nhc_ablation.sh
```

等价：

```bash
cd nhc_routing_ablation && bash run_nhc_ablation.sh
# 或
python run_nhc_ablation.py --profile nhc_misd_tus
```

### 6.2 常用变体

| 场景 | 命令 |
|------|------|
| 已有 Phase1–2 缓存 | `SKIP_PHASE12=1 bash run_nhc_ablation.sh` |
| 只补中心表（已有 B3 目录） | `CENTER_ANALYSIS_ONLY=1 bash run_nhc_ablation.sh` |
| 只要 R12/R13，不要中心块 | `SKIP_CENTER_ANALYSIS=1 bash run_nhc_ablation.sh` |
| 不要策略 (6) learned | `SKIP_LEARNED=1 bash run_nhc_ablation.sh` |
| 不要腺体 B7 | `SKIP_GLAND=1 bash run_nhc_ablation.sh` |
| 小样 profile | `PROFILE=nhc_prepared_sample bash run_nhc_ablation.sh` |

### 6.3 环境变量一览

| 变量 | 含义 |
|------|------|
| `SKIP_PHASE12=1` | 跳过 Phase1–2（要求 `_intermediate/.../intermediate` 已有 seg+cls） |
| `SKIP_STATIC=1` / `SKIP_LEARNED=1` | 跳过静态 bundle / learned |
| `SKIP_CENTER_ANALYSIS=1` | 跳过七策略 `_center_analysis` |
| `CENTER_ANALYSIS_ONLY=1` | 只跑中心分析（跳过 B3 阶段） |
| `SKIP_GLAND=1` | 跳过 `_center_analysis/gland/` |
| `NHC_PROCESSED_DIR` | 覆盖 `config.yaml` 的 `prepare.output_dir` |
| `NHC_ABLATION_OUTPUT_ROOT` | 覆盖 B3 结果根目录（默认见 profile yaml） |
| `NHC_CASCADE_ENGINE_ROOT` | 仅调试：替换内置 `cascade_engine/` 路径（一般不设） |

### 6.4 产物（默认全量）

根目录：`results/nhc_routing_ablation/nhc_misd_tus/`

| 内容 | 路径 |
|------|------|
| 七策略 B3 | `best_single/`、`soft_voting/`、…、`oracle/`、`learned/` |
| 汇总表 R12/R13 | `_summary/b3_routing_table.csv` |
| 运行元信息 | `_run_manifest.json` |
| 匿名映射（若复制成功） | `name_mapping.json` |
| 中心性能 / 异质性 R18–R20 | `_center_analysis/<mode>/`、`heterogeneity_by_strategy.csv` |
| 腺体单路径 B7 | `_center_analysis/gland/` |

脚本结束会打印 **整目录 scp/打包** 提示。**只需下载上述结果根目录**；Phase1–2 缓存在 `_intermediate/`，通常留在私有机。

**换 cohort 或增删图像后**：须重跑 Phase1–2。删除：

```text
results/nhc_routing_ablation/_intermediate/<profile>/pipeline_run/intermediate/seg
results/nhc_routing_ablation/_intermediate/<profile>/pipeline_run/intermediate/cls
```

（只删 `seg` 不重跑 `cls` 会导致新图无分类缓存。）

---

## 7. 自检命令

```bash
cd /path/to/my_Thyroid_Infer

# 内置引擎 + learned joblib
test -f nhc_routing_ablation/cascade_engine/run_pipeline.py && echo OK engine
test -f nhc_routing_ablation/artifacts/b3/tn5k/learned_router_seg.joblib && echo OK learned

# DINO 权重（5+4）+ 腺体 TGVideo
ls infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_*.pth
ls infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_*.pth
test -f infer_dinov3_unet/checkpoints/TGVideo/dino_unet_train_TGVideo_epoch_30.pth && echo OK tgvideo

# 预处理
test -f datasets/processed/config.yaml && echo OK processed
test -f datasets/processed/name_mapping.json && echo OK name_mapping

# 编排 CLI
python nhc_routing_ablation/run_nhc_ablation.py --help >/dev/null && echo OK run_nhc_ablation
```

---

## 8. 与回信表的关系（简要）

| 回信 | 数据来源 |
|------|----------|
| **R12/R13** 路由消融 | `_summary/b3_routing_table.csv`、各策略 `b3_metrics.json` |
| **R18–R20** 中心 / 异质性 | `_center_analysis/` 下 CSV、`b7_report.json`、`heterogeneity_by_strategy.csv` |
| **策略 (7) LLM** | **不跑**；表中 N/A |
| **G3** 描述性元数据 | 可选 `--g3-csv`；与 center manifest 合并 |

LaTeX 同步仍可能需手改或脚本导出；策略编号见 `nhc_routing_ablation/config/README_STRATEGIES.md`。

---

## 9. 代码与模型托管

| 用途 | 仓库 | 说明 |
|------|------|------|
| **代码** | [GitHub ordinary7Zz/my_Thyroid_Infer](https://github.com/ordinary7Zz/my_Thyroid_Infer) | `git clone` / `git pull`；含 `nhc_routing_ablation/cascade_engine/` |
| **权重（NHC 子集）** | 同上 HF / 魔搭，用 §3 脚本按文件拉 | 勿用 `git pull` 拉 `.pth` |
| **整包镜像（可选）** | [HF instincts7Zz/my_Thyroid_infer](https://huggingface.co/instincts7Zz/my_Thyroid_infer) | 含代码快照 + 全权重 + `thyroid_infer_env.tar.gz` 等 |
| **整包镜像（可选）** | [魔搭 instincts/my_Thyroid_infer](https://www.modelscope.cn/models/instincts/my_Thyroid_infer) | 与 HF 同步 |

权重在 `.gitignore` 中；**更新 NHC 用权重请用 §3 或 [`NHC_QUICKSTART_ZH.md`](NHC_QUICKSTART_ZH.md)**。

---

## 10. 故障排查（简）

| 现象 | 常见原因 |
|------|----------|
| 未找到 `datasets/processed/config.yaml` | 未跑 `pipeline.py` 或 `prepare.output_dir` 不对 |
| Phase1–2 后 B3 行数少于图像数 | 只清了 `intermediate/seg` 未清 `cls`；两目录都删后重跑 Phase1–2 |
| 中心 AUROC 某中心为 NE | 该中心仅单类别（样本过小）；全量 cohort 或见 `scripts/verify_center_auroc_estimable.py` |
| 缺少 `cascade_engine/run_pipeline.py` | 仓库不完整，重新 `git pull` 或联系维护者 |
| learned 报错缺 joblib | 检查 `nhc_routing_ablation/artifacts/b3/tn5k/` 两文件是否存在 |

更细说明：`nhc_routing_ablation/README.md`、[`NHC_QUICKSTART_ZH.md`](NHC_QUICKSTART_ZH.md)。
