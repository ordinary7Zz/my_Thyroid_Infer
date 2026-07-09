# AutoGluon 分类推理模块

基于 PyRadiomics 特征 + AutoGluon TabularPredictor 的甲状腺结节分类推理。

## 与其它分类模型的区别

其它分类模型（BiomedCLIP、MedSigLIP 等）直接对图像像素进行推理。
AutoGluon 模型的输入是 **radiomics 特征**，需要先从 (image, mask) 对中提取特征，
再用 TabularPredictor 进行推理。

**推理流程**: `图像 + 掩码 → radiomics 特征提取 → AutoGluon 推理 → 分类结果`

## 用法

```bash
# 二分类推理 + 评估
python infer.py \
    --image_dir /path/to/images/ \
    --mask_dir /path/to/nodule_masks/ \
    --model_dir /path/to/autogluon_binary_model/ \
    --num_classes 2 \
    --class_names benign malignant \
    --label_json /path/to/labels.json \
    --label_field malignancy \
    --output results.csv \
    --eval_output metrics.log

# TIRADS 五分类推理 + 评估
python infer.py \
    --image_dir /path/to/images/ \
    --mask_dir /path/to/nodule_masks/ \
    --model_dir /path/to/autogluon_tirads_model/ \
    --num_classes 5 \
    --class_names 1 2 3 4 5 \
    --label_json /path/to/labels.json \
    --label_field tirads \
    --output results.csv \
    --eval_output metrics.log
```

## 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--image_dir` | ✓ | 待推理的图片目录 |
| `--mask_dir` | ✓ | 结节 ROI 掩码目录 |
| `--model_dir` | ✓ | AutoGluon 模型目录（含 `predictor.pkl`） |
| `--num_classes` | ✓ | 类别数（2 或 5） |
| `--class_names` | ✓ | 类别名称列表 |
| `--label_json` | | 标签 JSON（提供后计算指标） |
| `--label_field` | | 标签字段名（如 `malignancy`, `tirads`） |
| `--output` | | 分类结果 CSV 路径 |
| `--eval_output` | | 指标日志 `.log` 路径 |
| `--radiomics_params` | | PyRadiomics YAML（默认使用同目录 `radiomics_2d.yaml`） |
| `--n_bootstrap` | | Bootstrap 迭代次数（默认 2000） |

## 依赖

```bash
pip install autogluon pyradiomics SimpleITK PyYAML
```
