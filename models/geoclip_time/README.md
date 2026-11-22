# 带拍摄年代的 GeoCLIP

本目录集中保存时间条件双塔模型及无时间对照实验。模型原理、数据要求和运行命令见[完整说明](../../doc/time_geoclip.md)；命令均在仓库根目录执行。

- `lib/`：日期与特征对齐、模型、损失、检索和训练流程。
- `scripts/`：`train_time.py` 训练与数据检查，`predict_time.py` 导出网格预测，`rerank_references.py` 运行 T3 及其纯视觉对照。
- `configs/`：T0 无时间对照、T1 双塔残差融合、T1 仅图片侧融合和 T2 双塔直接拼接时间的配置。
- `tests/`：合成数据与小型网络的行为验证。

```bash
python -m pytest -q models/geoclip_time/tests
```

共享数据仍位于 `P2_DATA`，划分文件位于仓库的 `data/splits/`；checkpoint 和预测产物路径保持不变。

T0、双塔 T1 和 T3 的结果见[实验结果表](../../report/geoclip_time_baselines.md)。

### T3（Yang_Zhang）｜T0 图片编码器检索照片，再按年份重排

在仓库根目录执行下面的命令，`T0_CHECKPOINT` 指向训练好的无时间 T0 权重，`P2_DATA` 指向 SMAPshot 数据目录。

```bash
python -m models.geoclip_time.scripts.rerank_references \
  --checkpoint "$T0_CHECKPOINT" --data-root "$P2_DATA" \
  --split-csv data/splits/main.csv.gz \
  --output-dir "$P2_DATA/results/T3_photo_rerank"
```

脚本冻结 T0 的图片投影，从训练照片中按余弦相似度取前 50 张。它在完整验证集上选择年份权重，然后为每张查询照片保存纯视觉和年份重排两种预测，以及各自选中的训练照片 ID 和坐标。`summary.json` 汇总距离指标，`val_gamma_search.json` 记录权重选择过程，`predictions.parquet` 保存逐张预测，`method=year` 对应 T3，`method=visual` 对应纯视觉对照。

### T1 位置检索与档案年份重排

每条候选记录来自一张训练照片，含坐标和拍摄日期区间。第一轮用冻结的 T1 对查询图片和候选坐标打分。图片端与位置端都使用查询日期区间的中点、半宽和有效标记，取余弦得分最高的 50 条记录。第二轮使用这些候选各自的档案年份，按与 T3 相同的规则重排。`location` 表示直接选 T1 第一名，`year` 表示从同一批候选中按年份重排后选第一名。两种方法都输出胜出记录的坐标。

在仓库根目录执行，`T1_CHECKPOINT` 指向已经训练好的双塔残差 T1 权重。

```bash
python -m models.geoclip_time.scripts.rerank_locations \
  --checkpoint "$T1_CHECKPOINT" --data-root "$P2_DATA" \
  --split-csv data/splits/main.csv.gz \
  --output-dir "$P2_DATA/results/t1_location_rerank" --device cpu
```
