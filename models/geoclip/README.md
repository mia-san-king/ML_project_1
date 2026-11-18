# 原始 GeoCLIP 实验

本目录收纳原有 GeoCLIP 模型操作、检索与微调实验，保留原来的训练目标、超参数、实验名称和结果路径。

- `lib/geoclip_ops.py`：位置编码、相似度检索、区域先验和原有训练循环。
- `lib/baseline_infer.py`：GeoCLIP 投影与候选坐标缓存，以及预测导出。
- `scripts/`：B1/B3 检索、A1 区域先验搜索、A3/A4 微调、可靠性分析及启动脚本。
- `configs/`：原有 A3/A4 实验配置。

命令均从仓库根目录运行，先设置 `P2_DATA` 指向现有数据目录：

```bash
python models/geoclip/scripts/run_baselines.py --which all --device cuda
python models/geoclip/scripts/search_a1.py --device cuda
python models/geoclip/scripts/run_a1.py --device cuda
python models/geoclip/scripts/train.py --config models/geoclip/configs/A4_tau5.yaml --device cuda
python models/geoclip/scripts/check_geoclip_ops.py
```

`run_baselines.py` 仍统一调度 B1/B2/B3；B2 实现在 [baselines](../baselines/README.md)。共享缓存表和指标 JSON 工具位于 [`lib/feature_tables.py`](../../lib/feature_tables.py)。时间条件双塔位于相邻的 `models/geoclip_time/`，项目方法说明统一保留在根目录 `doc/`。
