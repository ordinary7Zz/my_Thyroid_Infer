# infer_resnet50

Thyroid 良恶性二分类（ResNet-50），与 `run_all_extended.py` 对接。

| 文件 | 说明 |
|------|------|
| `infer_binary.py` | **入口** |
| `infer_resnet50_cls_auroc_json.py` | 按 label JSON 推理 + AUROC JSON |
| `cls_dataset.py` / `model.py` | 数据集与 `build_model`（推理 `pretrained=False`） |
| `utils/classification_bootstrap.py` | Bootstrap 指标 |
| `checkpoints/resnet50_thyroid_cls_epoch_50.pth` | 默认权重（本地副本） |

```bash
cd infer_resnet50
python infer_binary.py --image_dir /path/images --label_json /path/labels.json \
  --task_key malignancy --checkpoint_path checkpoints/resnet50_thyroid_cls_epoch_50.pth \
  --output_dir /tmp/out
```
