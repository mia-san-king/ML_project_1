# 瑞士历史照片定位：GeoCLIP 时间对照

两组模型使用相同的 176,308 张 SMAPshot 训练照片、冻结的 CLIP 图片特征和瑞士候选坐标网格。T0 只读取图片。T1 在图片和候选位置两侧都使用查询照片的日期区间，缺失日期的图片保留。

| 划分 | 图片数 | T0 中位误差 (km) | T1 中位误差 (km) | T0 ≤25 km | T1 ≤25 km |
|---|---:|---:|---:|---:|---:|
| val | 29,529 | 29.85 | 28.45 | 44.45% | 46.58% |
| test_in | 47,593 | 35.23 | 33.61 | 40.40% | 42.00% |
| test_city | 3,377 | 87.81 | 91.63 | 8.02% | 9.27% |
| test_region | 13,039 | 91.91 | 86.63 | 3.47% | 2.10% |

本次使用 seed 0，各训练 15 轮。两组在相同的 256 张验证图片上选择权重，T0 选中第 8 轮，T1 选中第 12 轮。表中使用全部有缓存的验证、测试图片。T1 在常规测试集上的中位误差下降，但在 Bern 和 Ticino 留出集上，两项指标的变化方向不同。

复现时使用 [T0](../models/geoclip_time/configs/T0_no_time.yaml) 和 [T1](../models/geoclip_time/configs/T1_dual_time.yaml) 的配置副本，将 `allow_partial_features` 设为 `true`、`query_chunk` 设为 `64`、`gallery_chunk` 设为 `2048`。训练划分中有 230 张照片缺少缓存。预测时传入 `--allow-partial-features`。训练与预测命令见[实验说明](../doc/time_geoclip.md)。
