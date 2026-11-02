# B2 基线批跑（复用 `run_all.py`）

将 **my_Thyroid_Infer** 中 TransUNet / MedSAM2 / MedSegX / UltraFedFM（分割）与 BiomedCLIP / MedSigLIP / UltraFedFM（分类）在外测队列上的结果导出到工作区 **`b2_exports/`**，再于 **ThyroidCascadeAgent** 计算 cluster CI。

分类另三个基线通过 **`infer_repvit/`、`infer_lsnet/`、`infer_resnet50/`** 接入 **`run_all_extended.py`**（不修改 `run_all.py`）；B2 批跑默认调用 `run_all_extended.py`。

## 1. 一键：全部模型 × 全部队列（推荐）

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/my_Thyroid_Infer
export CUDA_VISIBLE_DEVICES=0
./baseline_export/run_b2_all.sh
# 或: ./b2/run_b2_all.sh
```

默认 **`--skip-existing`**：`results/b2/<cohort>/` 里已有输出的 model 不再推理，只补缺失项。强制全量重跑：`SKIP_EXISTING=0 ./baseline_export/run_b2_all.sh`

## 1b. 旧入口（同逻辑，日志名 full_pipeline_*）

```bash
./baseline_export/run_full_pipeline.sh
```

日志：`results/b2/full_pipeline_<时间戳>.log`。CI 表在 `ThyroidCascadeAgent/docs/revision/b2/tables/`。

## 2. 分步：推理 + 转换

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/my_Thyroid_Infer

# 全部 B2 队列（耗时大，需 GPU）
python baseline_export/run_infer_batch.py

# 单队列试跑
python baseline_export/run_infer_batch.py --cohorts zjh8k

# 只看将执行的 run_all 命令
python baseline_export/run_infer_batch.py --cohorts tn3k --dry-run

# 已有 results/b2/<cohort>/，只转 b2_exports
python baseline_export/run_infer_batch.py --cohorts ddti_seg --skip-infer
```

队列与路径见 `baseline_export/cohorts.yaml`（与 `ThyroidCascadeAgent/docs/revision/b2/cohort_tasks.yaml` 对齐，不含 nhc）。

**注意**：使用 `--mode finetune`（脚本内固定），输出在 `my_Thyroid_Infer/results/b2/<cohort_id>/`。

## 3. 仅算 CI（已有 b2_exports）

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/ThyroidCascadeAgent
bash scripts/run_b2_exports_all.sh
```

产出：`docs/revision/b2/tables/baseline_<Model>_<cohort>_b2.json` 与 `BASELINES_INDEX.json`。

## 4. 从旧结果目录转换（可选）

若结果不在 `results/b2/<cohort>/`，可指定路径：

```bash
python baseline_export/convert_to_exports.py \
  --cohort-id ddti_seg \
  --results-root ../results/nodule_one_dataset/DDTI_seg
```

（需配合 `--cohorts ddti_seg` 读取 manifest 元数据。）

## 5. DDTI 分类

`ddti_cls` 默认使用 `DDTI_Classification/all/`（与 pipeline `DDTI_cls` 一致）。若稿面改为 54 图 test 子集，请改 `b2_cohorts.yaml` 中 `ddti_cls.classification` 路径后再跑。
