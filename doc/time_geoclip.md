# 带拍摄年代的 GeoCLIP

这组实验研究拍摄年代如何影响瑞士历史照片定位。T0 使用图像和经纬度，T1 和 T2 的图像和位置编码器还接收年代。三个模型使用相同的对比损失。

## 模型与训练目标

已有缓存是冻结 CLIP 产生的 **768 维、未归一化、进入 GeoCLIP 图像 MLP 之前**的特征。所有配置复用这些缓存以及 GeoCLIP 的预训练图像投影和位置编码器。双侧 T1 在两端加入小型残差 MLP 融合时间信息。输出归一化后比较余弦相似度。

\[
s_{ij}=\exp(\alpha)\,\langle F(I_i,t_i),G(g_j,t_i)\rangle.
\]

**同一查询的所有候选位置都使用查询的时间 `t_i`，不用候选照片自己的 `t_j`。** 例如定位一张 1900 年照片时，比较的是“伯尔尼，1900”和“洛桑，1900”。这避免仅靠年份相同就识别正样本，但不能排除年代与采集地区相关带来的地理先验。

训练使用图片到位置的单向对比损失；坐标完全相同的候选作为多个正样本处理。可选的 `soft_tau_km` 使用地理距离软目标，默认配置先使用硬目标。训练只用 `train`，checkpoint 由 `val` 的定位误差选择；测试集不参与选择。

空间支持域由实际候选库定义。现有 `grid_1km.npy` 的生成规则是瑞士边界加 5 km 缓冲区，并不是完整经纬度矩形。损失只对采样的有限候选归一化，不是整个平面上连续概率密度的截断；时间固定为查询时间，不在时间区间上搜索或积分。输出特征的单位球面归一化与地理范围限制是两件不同的事。

时间融合使位置向量依赖年代。推理会缓存时间无关的位置基底，再按查询年代融合并分块比较；不能复用旧实验最终的 `gallery_*_emb.npy` 作为带时间的位置向量。

## 日期处理

时间来自 `date_shot_min` 和 `date_shot_max`，不是上传或地理标注日期。当前只使用年份，输入三个数：

1. 区间年份中点，默认以 1950 为中心、100 年为尺度。
2. 区间年份半宽，除以同样的尺度。
3. 日期区间是否有效的标记。

例如 1890–1900 对应中点 1895、半宽 5。这表达“年代不确定”，不意味着实际拍摄于 1895。年内月份和日期不作为模型输入；日期先用于检查上下界顺序。

缺失、无法解析、只有一个有效端点、上下界颠倒或年份超出配置范围的记录使用未知标记，保留图片，不自动补造拍摄年份。数据检查会报告日期质量。默认范围为 1800–2026，可在首次训练前调整；时间预处理配置随 checkpoint 保存，预测沿用它。

完整日期先验证时间先后，再提取其记录的日历年份；两个带时区端点按实际时刻比较，两个无时区端点按原样比较。一个端点带时区、另一个不带时区时，保守标为无法解析。仅有年份的端点按年份比较，不假定是 1 月 1 日。有效性范围并不代表把全部图片过滤成一个严格年代窗口。

## 数据与环境

在已有 DHLAB `ml-p2` 环境和本分支工作目录运行。数据沿用 `P2_DATA` 下的现有文件：

```text
meta/smapshot.parquet
features/clip_vitl14.npy
features/clip_vitl14_ids.npy
geo/grid_1km.npy
```

划分文件是仓库的 `data/splits/main.csv.gz`。数据入口按 ID 对齐特征、坐标、日期和 split，拒绝重复或无法对应的 ID，并报告每个 split 的缓存覆盖率。训练和验证默认要求缓存完整；如确实要先跑部分缓存，显式设置配置中的 `allow_partial_features: true`，两套实验必须使用同一份缓存。

首次先检查数据，不创建模型、不训练：

```bash
export P2_DATA=/rcp-scratch/students/yijliu/ml-project2/data
python models/geoclip_time/scripts/train_time.py --config models/geoclip_time/configs/T1_dual_time.yaml --check-data
```

可用 `--data-root` 和 `--split-csv` 覆盖路径。缺少缓存时，使用项目已有的 `scripts/cache_features.py`；时间模型不需要重新提取图片特征。训练仅实例化 GeoCLIP 的图像 MLP 和位置编码器，不构建完整 CLIP 图像／文本网络。

## 训练与预测

下列配置使用相同的训练划分、损失、训练轮数和种子。

| 配置 | 用途 |
| --- | --- |
| `models/geoclip_time/configs/T0_no_time.yaml` | 不使用时间的对照 |
| `models/geoclip_time/configs/T1_dual_time.yaml` | 两塔通过残差网络结合时间 |
| `models/geoclip_time/configs/T1_image_only.yaml` | 仅图片侧保留 T1 的时间残差网络，位置编码器继续训练 |
| `models/geoclip_time/configs/T2_direct_time.yaml` | 把时间直接加入两塔的第一层 |

本目录的 `T0_no_time` 是无日期的 GeoCLIP。队友的 [T0 检索实验](https://github.com/mia-san-king/ML_project_1/blob/f019e23583dc236a80ea28dcbdb1b2584d561a19/scripts/run_knn_time.py) 按视觉相似度取前 50 张训练照片，再按年份差重排。小组报告中可分别标为“无时间 GeoCLIP”和“图像近邻＋年份重排”。

本分支的 **T3（Yang_Zhang）** 冻结 T0 的图片编码器，从训练照片中检索视觉上最像的 50 张，按拍摄年份重排后输出胜出照片的坐标。跨分支汇报时用分支名区分同名实验。`project_2_Yijun_Liu` 的 T3 在查询图片侧加入日期残差，并微调图片投影和位置编码器。[本分支 T3 运行说明](../models/geoclip_time/README.md) · [队友的 T3 配置](https://github.com/mia-san-king/ML_project_1/blob/f019e23583dc236a80ea28dcbdb1b2584d561a19/configs/T3.yaml)

T2 图像侧把三个日期特征拼到 768 维 CLIP 特征之后。位置侧在每个尺度的 512 维空间 Fourier 特征之后拼入同一个查询日期。两个输入层都复制 GeoCLIP 的预训练权重，并把新增日期输入的权重置零，随后联合训练。推理时，日期向量完全相同的照片共享该日期下的候选位置向量。T1 和 T2 的参数量不同，结果比较反映两套完整结构的表现。

GPU 使用前按实验室规则检查 `df -h /rcp-scratch` 和 `nvidia-smi`，选择空闲卡，不用 GPU 3，最多两块卡。设置 `CUDA_VISIBLE_DEVICES` 后，训练命令里的 `cuda` 指该可见设备。长任务放进 `screen`，缓存仍放在 scratch：

```bash
export HF_HOME=/rcp-scratch/students/yijliu/hf_cache
export TORCH_HOME=/rcp-scratch/students/yijliu/hf_cache/torch
export TMPDIR=/rcp-scratch/students/yijliu/tmp

python models/geoclip_time/scripts/train_time.py --config models/geoclip_time/configs/T0_no_time.yaml --device cuda
python models/geoclip_time/scripts/train_time.py --config models/geoclip_time/configs/T1_dual_time.yaml --device cuda
```

训练保存最佳验证 checkpoint 到 `$P2_DATA/ckpt/<run_name>.pt`，日志和数据审计位于 `$P2_DATA/results/metrics/<run_name>_train_log.json`。同名产物默认拒绝覆盖；更改实验时优先设置新的 `run_name`。`--overwrite-run` 只在明确要替换这个实验时使用，不是恢复训练。

训练完成后单独导出预测：

```bash
python models/geoclip_time/scripts/predict_time.py \
  --checkpoint "$P2_DATA/ckpt/T1_dual_time.pt" \
  --data-root "$P2_DATA" \
  --splits val test_in test_city test_region \
  --device cuda
```

预测表保留项目已有的 `id, split, lat_pred, lon_pred` 接口，保存为 parquet，并输出定位误差指标。每次预测使用照片元数据中的时间；checkpoint 包含恢复模型所需的权重与结构配置。

若请求的评估 split 缺少部分缓存，预测默认报错。仅在明确接受不完整评估时加 `--allow-partial-features`，覆盖率会写入指标文件；此时不要把结果当作完整测试集成绩。

默认 batch 和负样本数低于旧 A3/A4 实验，便于先跑通新增的时间分支。因此应该先比较本目录的 T0 与 T1，不能把差异直接全部归因于时间、再与旧 A3/A4 数字比较。正式报告还应检查不同年代、日期缺失与不同留出区域的表现，并增加“只看年代”的对照，区分地理先验与视觉判别的贡献。

T1 同时增加了时间输入和融合层容量。若要单独归因于年代信息，还应在相同 T1 结构上增加打乱年代或屏蔽年代的对照；这轮代码尚未提供该实验配置。

## T4：已知拍摄日期，直接回归坐标

T4 和 T1 读取同一份冻结的 768 维 CLIP 图片特征，以及查询照片的三个日期特征（年份中点、区间半宽、日期有效标记）。T4 将它们拼成 771 维向量，通过一个带偏置的线性层输出两个坐标。日期缺失时三个日期特征均为零，照片仍参与训练和评估。

线性层可以写作 `W_image × 图片特征 + W_date × 日期特征 + b`。因此，在国界投影之前，把同样的日期改动应用于两张不同的图片，两张预测坐标会移动相同的向量。

坐标使用瑞士 LV95 投影。网络输出相对 LV95 原点 (2,600,000 m, 1,200,000 m) 的东向和北向偏移，单位为千米。在所有有缓存的 `train` 照片上，用最小二乘法直接求解线性层的 1,544 个参数。平方损失在两个方向上按相同的距离尺度计算。预测落在瑞士国界外时，取国界上最近的一点，再转为经纬度计算大圆距离误差。

从仓库根目录运行，输出位置选用新目录。最后一条命令按照片 ID 与原始 T1 网格检索结果配对。

```bash
python models/clip_regression/scripts/train.py --data-root "$P2_DATA" --output "$RUN/T4"
python models/clip_regression/scripts/compare_t1.py --t4 "$RUN/T4/predictions.parquet" \
  --t1 "$P2_DATA/results/preds/zhang7_T1_dual_time_available_s0_20261118.parquet" \
  --output "$RUN/T4/comparison.json"
```

## 扩散增强实验的训练数据

本实验在真实 `train` 照片中抽取原图，用 SDXL 图生图生成同一视角的变体。每个合成样本在独立目录的 `synthetic.sqlite` 中有一条记录，图像文件在同目录的 `images/` 下。可按 `is_diffusion = 1 AND status = 'accepted'` 取出通过结构筛查的变体。表中的 `parent_id` 指向真实训练照片，`synthetic_id` 是该 ID 的相反数。变体继承原图的坐标和日期区间，表中同时记录生成模型版本、参数、种子和文件指纹。

`scripts/augmentation/generate.py` 写入独立图库。`build_dataset.py` 用原版 CLIP 提取合成图特征，并把合成行追加到一个新的训练视图。原验证和测试照片继续按原 ID 评估。`train_models.py` 复用各模型的训练设置。T3 重新训练 T0 图片编码器，再对每个原图 ID 取原图与变体的最高相似度，从不同的原图中选 50 个候选并按年份重排。T1 仍然检索瑞士网格，T4 仍然直接回归坐标。

## 本地验证

项目运行环境之外，测试额外需要 `pytest`。在仓库根目录执行：

```bash
python -m pytest -q models/geoclip_time/tests
python models/geoclip_time/scripts/train_time.py --help
python models/geoclip_time/scripts/predict_time.py --help
```

测试使用小型真实 PyTorch 网络和合成元数据检查日期／ID 对齐、查询时间条件化、梯度、分块检索、训练保存与恢复，不依赖 SMAPshot 图片或服务器 GPU。合成测试通过不代表真实历史图片上的精度已得到验证。
