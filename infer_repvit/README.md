# infer_repvit

Thyroid 良恶性二分类（RepViT），与 `run_all_extended.py` 的 `binary` 任务对接。

| 文件 | 说明 |
|------|------|
| `infer_binary.py` | **入口**（`run_all` 调用） |
| `infer_multi_to_json.py` | 多数据集 ImageFolder 推理 + JSON 导出 |
| `eval_multi_test.py` | 指标与 bootstrap |
| `model/` | RepViT 结构（timm 注册） |
| `checkpoints/.../checkpoint_best.pth` | 默认 BM 权重（本地副本） |

```bash
cd infer_repvit
python infer_binary.py --model repvit_m1_0 --checkpoint checkpoints/BM/repvit_m1_0_224_0.025_0.001_0.25/checkpoint_best.pth \
  --test-dirs /path/to/test_cls --test-names tn3k --output-dir /tmp/out
```
