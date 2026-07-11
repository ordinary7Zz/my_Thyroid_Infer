# 可解释性分析推理（infer_explain）

甲状腺超声图像分割与分类模型的可解释性分析工具集，包含两类方法：

| 方法 | 任务 | 入口脚本 | 原理 |
|------|------|----------|------|
| **SHAP** | 分类可解释性 | `shap_analyze.py` | Shapley 值：量化每个 radiomics 特征对 AutoGluon 预测的贡献 |
| **GradCAM** | 分割可解释性 | `gradcam_seg.py` | 梯度加权类激活映射：定位 DINOv3-UNet 分割决策的关注区域 |

---

## 目录结构

```
infer_explain/
├── shap_analyze.py              # SHAP 批量分析入口（beeswarm + waterfall + 特征重要性）
├── shap_analyze_single.py       # SHAP 单图/指定样本分析入口
├── gradcam_seg.py               # GradCAM 热力图生成入口
├── model.py                     # DINOv3_S_UNet 模型定义（与 infer_dinov3_unet 共用）
│
├── autogluon_introspection.py   # AutoGluon 模型内省（ensemble 权重提取、模型加载）
├── autogluon_preprocessing.py   # AutoGluon 特征预处理（BAG 模型树提取）
├── shap_compute.py              # SHAP 值计算（TreeExplainer / KernelExplainer）
├── shap_local_plots.py          # SHAP 局部图（waterfall + compact bar）
├── plotting_utils.py            # 绘图工具（CJK 字体、beeswarm、waterfall 保存）
│
├── requirements.txt             # 额外依赖
└── README.md                    # 本文件
```

---

## SHAP 分析（分类可解释性）

### 原理

SHAP（SHapley Additive exPlanations）基于博弈论 Shapley 值，为每个特征计算对模型预测的边际贡献。

针对 AutoGluon TabularPredictor 的集成模型：
- 树模型（LightGBM / XGBoost / CatBoost）使用 `TreeExplainer`（快速）
- 其他模型使用 `KernelExplainer`（通用但较慢）
- 自动提取 ensemble 权重，计算加权集成 SHAP 值

### 输入

| 数据 | 说明 |
|------|------|
| **AutoGluon 模型目录** | 包含 `predictor.pkl` 和 `logs/predictor_log.txt` |
| **训练特征 CSV** | radiomics 特征表，含 `label` 列和 `filename`/`image_path` 列 |

### 批量分析

```bash
python infer_explain/shap_analyze.py \
    --model_dir /path/to/autogluon_model \
    --train_csv /path/to/radiomics_features.csv \
    --output_dir ./results/explain/shap \
    --label label \
    --background_samples 100 \
    --explain_samples 500 \
    --skip_neural_net \
    --plot_beeswarm_for LightGBM_BAG_L1 WeightedEnsemble_L3 \
    --plot_waterfall \
    --waterfall_samples 3 \
    --top_features 20
```

### 单图分析

为指定样本生成 waterfall 图和 compact SHAP bar 图：

```bash
# 单个样本
python infer_explain/shap_analyze_single.py \
    --model_dir /path/to/autogluon_model \
    --train_csv /path/to/radiomics_features.csv \
    --filename case_001.png \
    --output_dir ./results/explain/shap_single

# 批量样本（从文件列表）
python infer_explain/shap_analyze_single.py \
    --model_dir /path/to/autogluon_model \
    --train_csv /path/to/radiomics_features.csv \
    --filename_list /path/to/filenames.txt \
    --output_dir ./results/explain/shap_single
```

### 输出结构

```
<output_dir>/
├── <ModelName>_shap_values.csv        # 样本级 SHAP 值
├── <ModelName>_feature_importance.csv # 特征重要性（mean |SHAP|）
├── WeightedEnsemble_L3_shap_values.csv # 集成加权 SHAP 值
├── shap_analysis_summary.txt          # 分析摘要
├── beeswarm/                          # beeswarm 图
│   └── <ModelName>_beeswarm.png/.svg
└── waterfall/                         # waterfall 图
    ├── <ModelName>_waterfall_best_1.png
    ├── <ModelName>_waterfall_worst_1.png
    └── waterfall_sample_images.csv
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | （必填） | AutoGluon 模型目录 |
| `--train_csv` | （必填） | 训练特征 CSV |
| `--label` | `label` | 标签列名 |
| `--output_dir` | `<model_dir>/shap_analysis` | 输出目录 |
| `--background_samples` | 100 | SHAP 背景样本数 |
| `--explain_samples` | 500 | 解释样本数 |
| `--skip_neural_net` | False | 跳过神经网络模型 |
| `--main_models` | 自动检测 | 指定分析的模型列表 |
| `--plot_beeswarm_for` | 无 | 生成 beeswarm 图的模型名 |
| `--plot_waterfall` | False | 生成 waterfall 图 |
| `--waterfall_samples` | 3 | 每类 waterfall 样本数 |
| `--top_features` | 5 | 显示的 top 特征数 |
| `--sample_filename` | 无 | 为指定文件名生成 waterfall |

---

## GradCAM 可视化（分割可解释性）

### 原理

Grad-CAM（Gradient-weighted Class Activation Mapping）通过目标层的梯度加权和，生成热力图，定位模型决策的关注区域。

针对 DINOv3-UNet 分割模型：
- 通过 forward/backward hook 获取目标层的特征图和梯度
- 全局平均池化梯度作为通道权重，加权求和得到 CAM
- 支持 GT mask 引导的目标区域 Grad-CAM

### 使用

```bash
# 单张图像
python infer_explain/gradcam_seg.py \
    --checkpoint /path/to/dino_unet.pth \
    --image_path /path/to/image.png \
    --mask_path /path/to/mask.png \
    --output_dir ./results/explain/gradcam

# 批量目录
python infer_explain/gradcam_seg.py \
    --checkpoint /path/to/dino_unet.pth \
    --image_dir /path/to/images/ \
    --mask_dir /path/to/masks/ \
    --output_dir ./results/explain/gradcam \
    --output_type all \
    --target_layer reduce4 \
    --alpha 0.45
```

### 输出结构

```
<output_dir>/
├── original/        # 原图（resize 后）
├── overlay/         # GradCAM 热力图叠加原图
├── overlay_gt/      # 热力图叠加 + GT mask 轮廓
├── original_gt/     # 原图 + GT mask 轮廓
└── gradcam_map/     # 纯 GradCAM 彩色热力图
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | （必填） | DINOv3-UNet 权重文件 |
| `--image_path` | — | 单图模式：图像路径 |
| `--image_dir` | — | 批量模式：图像目录 |
| `--mask_path` | — | 单图模式：mask 路径 |
| `--mask_dir` | — | 批量模式：mask 目录 |
| `--output_dir` | `./gradcam_single_image_out` | 输出目录 |
| `--output_type` | `all` | 输出类型：all/original/overlay/overlay_gt/original_gt/gradcam_map |
| `--img_size` | 224 | 输入尺寸 |
| `--target_layer` | `reduce4` | GradCAM 目标层 |
| `--alpha` | 0.45 | 热力图不透明度 |
| `--smooth_sigma_ratio` | 0.02 | 高斯平滑核比例 |
| `--gamma` | 1.0 | Gamma 变换指数 |
| `--saturation_scale` | 1.3 | 饱和度增强倍数 |

---

## 独立运行

可解释性分析工具独立运行，不通过 `run_all.py` 统一调度。直接使用各入口脚本：

```bash
# SHAP 批量分析
python infer_explain/shap_analyze.py \
    --model_dir /path/to/autogluon_model \
    --train_csv /path/to/radiomics_features.csv \
    --output_dir ./results/explain/shap

# SHAP 单图分析
python infer_explain/shap_analyze_single.py \
    --model_dir /path/to/autogluon_model \
    --train_csv /path/to/radiomics_features.csv \
    --filename case_001.png \
    --output_dir ./results/explain/shap_single

# GradCAM 可视化
python infer_explain/gradcam_seg.py \
    --checkpoint /path/to/dino_unet.pth \
    --image_dir /path/to/images/ \
    --mask_dir /path/to/masks/ \
    --output_dir ./results/explain/gradcam
```
