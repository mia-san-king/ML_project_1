# 简单基线

本目录收纳不使用 GeoCLIP 可训练投影和位置塔的 B0/B2 基线。

- `scripts/run_b0.py`：用训练坐标在 LV95 中的中位数作常量预测。
- `lib/knn.py`：在已有 CLIP 图像特征中按余弦相似度寻找最近训练样本。
- `scripts/run_knn.py`：独立运行 B2，返回最近训练图像的坐标并导出指标。

从仓库根目录运行，`P2_DATA` 指向现有数据目录：

```bash
python models/baselines/scripts/run_b0.py
python models/baselines/scripts/run_knn.py --device cpu
```

B2 使用共享的特征缓存与划分读取器 [`lib/feature_tables.py`](../../lib/feature_tables.py)，输出仍为 `$P2_DATA/results/preds/B2.parquet` 和 `$P2_DATA/results/metrics/B2_summary.json`。它也可通过 [`models/geoclip/scripts/run_baselines.py`](../geoclip/scripts/run_baselines.py) 的 `--which b2` 或 `--which all` 调度。B1/B3 的 GeoCLIP 检索代码位于 [geoclip](../geoclip/README.md)。
