# Project 2 实验归档

每个日期目录对应一次实验，保留脚本、配置、来源清单和汇总指标。方法、模型编号和指标解释见[项目交接](../../doc/project2_handoff.md)。

[2026 年 11 月 22 日的本地模型实验](2026-11-22-local-models/README.md)汇总了 T1、T3、T4、T5、T6 的分区定位精度、扩散增强对照、置信度校准和 T6 逐图选择的得失。[五模型精度总表](2026-11-22-local-models/model_accuracy.csv)

原版 T3 的[前 50 名诊断](2026-11-21-t3-top50-coverage/summary.json)可直接在 GitHub 阅读。要复查某一张图，可在 DHLAB 的实验目录 `runs/2026-11-21-t3-top50-coverage/` 读取对应测试集的 Parquet。每行含查询照片 ID、按视觉相似度排列的 50 个参考照片 ID、每个候选到真值的距离，以及从 1 开始计数的年份重排名次。原版 T3 的逐图预测在 `runs/2026-11-18-geoclip-t0-photo-rerank/results/predictions.parquet`，其中 `method=year` 表示年份重排第一名。

归档中的服务器路径对应原实验环境。在另一台机器重新训练时，从仓库根目录运行 `models/` 下的入口，并指定当地的 `--data-root` 和 `--split-csv`。图像、CLIP 特征、checkpoint 和逐图预测按项目约定存放在数据盘。扩散生成的图像和 SQLite 清单位于原增强实验的 `synthetic_store/`。
