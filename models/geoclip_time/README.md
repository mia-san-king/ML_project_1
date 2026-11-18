# 带拍摄年代的 GeoCLIP

本目录集中保存时间条件双塔模型及无时间对照实验。模型原理、数据要求和运行命令见[完整说明](../../doc/time_geoclip.md)；命令均在仓库根目录执行。

- `lib/`：日期与特征对齐、模型、损失、检索和训练流程。
- `scripts/`：`train_time.py` 训练与数据检查，`predict_time.py` 导出预测。
- `configs/`：T0 无时间对照和 T1 双塔时间融合配置。
- `tests/`：合成数据与小型网络的行为验证。

```bash
python -m pytest -q models/geoclip_time/tests
```

共享数据仍位于 `P2_DATA`，划分文件位于仓库的 `data/splits/`；checkpoint 和预测产物路径保持不变。
