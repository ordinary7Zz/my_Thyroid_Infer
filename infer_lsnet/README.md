# infer_lsnet

Thyroid 良恶性二分类（LSNet-t），与 `run_all_extended.py` 对接。

| 文件 | 说明 |
|------|------|
| `infer_binary.py` | **入口** |
| `eval_thyroid_auroc_json.py` | ImageFolder 评估 + JSON |
| `model/` | LSNet 结构 |
| `checkpoints/checkpoint_best.pth` | 默认权重（本地副本） |

```bash
cd infer_lsnet
python infer_binary.py --checkpoint checkpoints/checkpoint_best.pth \
  --test-dirs /path/to/test_cls --names tn3k --output-dir /tmp/out --output-prefix lsnet
```
