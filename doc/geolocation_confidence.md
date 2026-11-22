# 每张照片的 25 公里命中概率

`GeoConfidence` 接收调用者指定的数值列，一行对应一张照片。每份参数对应一个已冻结的定位模型版本、一份输入定义和一个距离阈值。四个定位器独立运行和校准。定位器可以返回不同的字段。用于监督的标签统一为预测坐标到真实坐标的大圆距离是否小于等于阈值。

## 本地 8,192 张增强图版本的输入

| 定位器 | 这一版校准器读取的特征 | 来源及含义 |
| --- | --- | --- |
| T1 | `nearest_train_km`, `log1p_train_5km`, `log1p_train_25km`, `has_date` | 逐图文件只保存预测坐标。根据预测点和真实训练图库计算最近距离、5 和 25 公里内的照片数量，另取查询照片是否有可用拍摄日期。 |
| T3 | `t3_visual_score`, `t3_year_penalty`, `has_date` | 胜出训练照片的视觉相似度、年份惩罚、查询日期是否可用。年份惩罚为零可能表示年份一致，也可能表示日期缺失。 |
| T4 | `t4_was_projected`, `nearest_train_km`, `log1p_train_5km`, `log1p_train_25km`, `has_date` | 回归结果是否因越界而被调整到瑞士边界，以及最终预测点周围的图库分布和查询日期是否可用。 |
| T5 | `t5_visual_rank`, `t5_t3_score`, `t5_location_cosine` | 对最终胜出候选取原始视觉名次（1 为第一名）、T3 年份调整后的分数和 T1 位置编码器的余弦相似度。 |

两种照片数量特征使用 `log(1 + 照片数)`。图库含 176,308 张真实训练照片，同坐标的真实照片分别计数。扩散变体不重复计数。T3、T5 直接返回图库中的坐标，它们到最近训练照片的距离在全部样本上都是零。

## 接在定位器后面

下面的 `result` 是项目现有 `models.t5_reranker.predict_photo.predict_photo` 返回的一张照片的结果。运行时只需要 T5 的输出和为这一版本保存的校准器。

```python
import pandas as pd
from lib.confidence import GeoConfidence

cal = GeoConfidence.load("data/local_runs/confidence_models/T5.joblib")
x = pd.DataFrame([{
    "t5_visual_rank": result["visual_rank"],
    "t5_t3_score": result["t3_score"],
    "t5_location_cosine": result["location_cosine"],
}])
result["p_within_25km"] = float(cal.predict_proba(x)[0])
```

T1、T4 的图库特征可通过 `lib.confidence_features.GalleryFeatureExtractor(train_latlon).transform(predicted_latlon, has_date)` 批量计算，两类坐标都是依次按纬度、经度排列的 WGS84 度数。日期掩码沿用模型现有的拍摄日期区间解析规则。T4 再补入推断时产生的 `t4_was_projected`。预测阶段的输入不需要查询照片的真实位置。

新定位器可声明自己的列名，单独训练参数。下面各输入按同一批验证照片的行顺序对齐。每张照片使用冻结定位器的预测，在验证阶段才计算其定位误差。

```python
cal = GeoConfidence(feature_names=["retrieval_margin"], threshold_km=25)
cal.fit(val_features, val_errors_km, groups=val_blocks, query_ids=val_photo_ids)
cal.save("my_locator_25km.joblib")
```

这次实验先用 32 个验证地理块（23,563 张）拟合小型梯度提升树，再用另外 15 个验证地理块（5,966 张）把树的输出映射为概率。三个测试区的样本只用于检验。部分定位器在开发时已用验证集选择超参数。整个流程可从仓库根目录复算，第一条命令中的路径指向同版本 T1 的预测文件。

```text
python -m scripts.confidence.build_inputs --t1 <T1预测文件.parquet>
python -m scripts.confidence.evaluate
```

## 如何解读本次实测

下表按照片汇总，每格先给平均预测概率，再给 25 公里内实际命中率。普通留出地理块有 47,593 张照片，伯尔尼留出区有 3,377 张，提契诺留出区有 13,039 张。

| 模型 | 普通留出地理块 | 伯尔尼 | 提契诺 |
| --- | ---: | ---: | ---: |
| T1 | 42.3% / 42.4% | 39.5% / 8.5% | 43.1% / 2.9% |
| T3 | 47.9% / 45.4% | 41.0% / 5.5% | 36.1% / 3.1% |
| T4 | 21.3% / 21.1% | 23.2% / 10.0% | 17.4% / 20.7% |
| T5 | 52.2% / 48.2% | 48.4% / 7.0% | 48.2% / 3.0% |

平均概率可能掩盖逐张排序的问题。在普通地理块上，T1 的 AUROC 为 0.432，T3 为 0.691，T4 为 0.705，T5 为 0.646。AUROC 衡量随机选一张命中照片和一张未命中照片时，模型给命中照片更高置信度的频率，平分按一半计。T4 在提契诺仅为 0.142。T5 在两个专门留出的区域显著高估了命中概率。使用这些文件时应结合目标地区的独立验证结果。

输出在 `data/local_runs/confidence_models/`。四份 `.joblib` 保存特征与参数，`predictions.parquet` 保存按照片的坐标、概率、阶段和评估误差，`metrics.csv` 与 `comparison.json` 包含分地区指标、常数基线以及概率分箱结果。换定位器权重、图库、特征定义或距离阈值时应重新验证相应校准器。保存的本地参数使用 scikit-learn 1.7.2，其他环境可用上面的命令重新拟合。
