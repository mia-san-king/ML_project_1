# 瑞士历史照片定位项目交接

实验结果截至 **2026-11-21**；仓库交接布局更新于 **2026-11-22**。本文记录已核实的方法、结果和解释边界，供后续在 Windows 或其他环境继续讨论。正文不包含服务器账户、认证信息或个人绝对路径。

本文位于仓库的 `doc/project2_handoff.md`。文中的代码路径相对于仓库根目录，链接相对于本文；历史结果统一归档为 `report/experiments/<原日期目录>/<原相对路径>`，迁移到其他主机无需保留原来的个人目录布局。代码、报告和归档应作为同一个仓库版本使用。Git 保存代码、固定划分、小型指标和说明；原始图片、完整特征缓存、逐图 Parquet、checkpoint 等大文件保留在 DHLAB 数据存储中。结果归档不等于完整重训数据或模型包。

## 1. 项目现在解决什么问题

- 将 GeoCLIP 等方法适配到瑞士 SMAPshot 历史照片定位，预测拍摄位置，并研究已知拍摄年代能否帮助定位。
- 新照片通常能提供年份或年代区间；当前主线把它作为输入，不要求从照片预测年份。日期未知的照片仍保留。
- 比较常规空间块测试与整个城市、州的留出测试。细粒度定位仍是目标，不能把“25 km 内命中”当作已经准确识别建筑或街道。
- 已有真实数据实验：T0/T1/T2、训练照片检索与年份重排 T3、线性坐标回归 T4，以及 512 张扩散变体的增强实验。
- 当前主要结论是方法间存在取舍：年份重排帮助常规测试；线性回归在常规测试较差、两个地区留出集较好；局部连续坐标优化与本轮小规模扩散增强尚未显示稳定收益。

持续更新的结果说明见[项目实验报告](../report/geoclip_time_baselines.md)，模型定义见[时间 GeoCLIP 说明](../doc/time_geoclip.md)。

## 2. 数据、划分与日期语义

### 数据与划分

元数据快照有 **355,281** 条 validated 记录；已有 **308,672** 条 CLIP 特征，其中包括不进入主实验的 nadir 图。冻结 CLIP ViT-L/14，每图缓存 **768 维、进入 GeoCLIP 图像 MLP 前的特征**。通过 ID 关联图片特征、元数据与划分，不依赖元数据行号。

| 划分 | 规则 | 原划分数 | 主实验可用数 |
|---|---|---:|---:|
| train | 留出地区与缓冲区之外，按 LV95 的 10×10 km 块分配 | 176,538 | 176,308 |
| val | 同一剩余区域的独立空间块 | 29,532 | 29,529 |
| test_in | 同一剩余区域的独立空间块 | 47,609 | 47,593 |
| test_city | Bern 中心 10 km 范围 | 3,378 | 3,377 |
| test_region | Ticino 州界内 | 13,040 | 13,039 |

Bern 的 10–20 km 环带、Ticino 外侧 10 km 缓冲带不进入主训练和测试。nadir 单独标记，不进入上述主表。空间块不重叠不代表块边界两侧照片一定相距 10 km；地区留出可以与训练照片具有相同 block 编号，因此不能只用 block 编号判断地区泄漏。

四个评估划分共 **93,538 张**，三个测试集共 **64,009 张**。T0/T1/T2 使用同一训练划分，并在同一固定 **256 张 val** 上选择 checkpoint；完整 val 指标使用全部 29,529 张。主实验使用 `split` 列，不使用 CSV 中的 `random_split` 列。缓存不完整已显式接受并记录 coverage，不能称为原划分全量覆盖。

证据：[基础实验比较](../report/experiments/2026-11-18-geoclip-time/run/comparison.json)、[数据审计](../report/experiments/2026-11-18-geoclip-time/run/data_audit.json)、代码 `data/splits/main.csv.gz`、`lib/splits.py`、`models/geoclip_time/lib/temporal_data.py`。

### 日期边界和未知值

使用 `date_shot_min` / `date_shot_max`，不是上传、入库或地理标注日期。T1/T2/T4 当前取两个端点的日历年份，输入：

`[(年份中点−1950)/100, 年份区间半宽/100, 有效标记]`

默认有效年份范围 **1800–2026**。1890–1900 表示中点 1895、半宽 5，不表示真实拍摄于 1895。月日用于检查完整日期先后，不作为模型特征；纯年份端点不强行解释为 1 月 1 日。

缺失、单侧日期、无法解析、上下界颠倒或超范围的记录使用全零未知时间向量，保留照片。日期处理配置随 checkpoint 保存，预测复用同一配置。

**T3 有历史兼容差异**：它沿用队友的日期中点规则，含日序近似小数年份，范围为 1800–2030；要求两端可解析，但旧规则未拒绝颠倒区间。查询或候选任一日期未知时，该候选的年份扣分为零。这可能使缺日期候选获得相对优势，不能把“缺失”理解为中性、随机缺失。不能声称所有方法的日期有效性规则完全相同。

## 3. 模型编号，尤其是两个不同的 T3

下文无特别说明时，编号指 `Yang_Zhang` 分支的方法。

| 名称 | 输入与方法 | 输出范围 |
|---|---|---|
| T0 | 冻结 CLIP 特征，经 GeoCLIP 图片投影；GPS 位置编码器共同微调，不使用时间 | 瑞士 1 km 候选网格 |
| T1 | `F(图片,查询时间)` 与 `G(GPS,查询时间)`，两塔通过非线性残差网络融合时间 | 同一网格 |
| T2 | 时间直接拼入图片 MLP 及各尺度位置编码的输入层，新增时间权重初始为零 | 同一网格 |
| T3 | 冻结训练好的 T0 图片投影，检索训练照片的视觉 top-50，再按年份差重排 | 胜出训练照片已有的坐标 |
| T4 | `[768维CLIP特征,3维日期] → 带偏置的线性层 → LV95东、北坐标`，用 QR 最小二乘求解 | 连续坐标，境外预测投影到瑞士国界 |

T0/T1/T2 为图片到位置的单向对比学习；坐标完全相同的候选视为多个正样本。正式基准使用 seed 0、15 epochs、batch 32、128 个采样负坐标，无 GPS jitter、无距离软目标。T0/T1 最佳轮次分别为 8、12。另有 `T1_image_only.yaml` 配置，本文不据未列出的正式结果下结论。

T1 每个查询的所有候选都使用同一查询时间：

`s(i,j) = exp(logit_scale) × cos(F(I_i,t_i), G(g_j,t_i))`

候选照片自己的年份不进入这一步。这样避免仅凭年份相等找到正样本，但不能排除年代与采集地区相关的先验。它学的是已知年代条件下定位，不是辨认“同一位置的不同年份”。

空间支持由**瑞士边界加 5 km 缓冲区内的 1 km 网格**定义，不是完整经纬度矩形。训练损失只对当批采样的离散正、负候选归一化；推理在网格中比较，查询年份始终固定，不搜索或积分年份。这不等于在整个平面上建立连续概率密度、再截断到瑞士范围重新归一化；详见[空间支持说明](../doc/time_geoclip.md)。

T3 的第二阶段：

`score = visual_cosine − γ × min(abs(query_year−reference_year)/50, 1)`

γ 只在完整 val 上选，原版与增强版均选出 **1.0**。原版图库是 176,308 张真实训练照片。历史结果目录名仍含 `t0-photo-rerank`，其 `method=year` 才是现在称呼的 T3；`method=visual` 是同一图库、同一候选列表的视觉对照。

T4 当前只有 **1,544 个参数**，设计矩阵秩 772。它预测两个位置坐标，不预测时间。线性拼接意味着投影国界前，同一个年份改变量对所有图片造成相同坐标平移，不具有 T1 那样的非线性图像—时间交互。

### 队友分支的编号不可直接混用

`project_2_Yijun_Liu` 的 T0 是原始 CLIP top-50 加年份重排；T1 是只训练图片侧 TimeBranch；T2 是再加 30% 日期 dropout；T3 是继续微调图片 MLP 和位置编码器。其日期只加在图片查询侧，GPS 编码器不接收日期：`normalize(image_emb + α × time_emb)`。

队友 T3 从 `A4_tau5` 初始化，再训 5 epochs，batch 512、负坐标 4096、距离软目标尺度 5 km。它与我们的 T1/T3 不是同条件消融。可引用队友 commit `f019e23583dc236a80ea28dcbdb1b2584d561a19` 的 `configs/T3.yaml`、`lib/geoclip_ops.py`、`lib/time_feats.py`；三 seed 配置都从同一个 A4 初始化。队友报告 test_in 三 seed 中位误差均值 25.23±0.61 km、@25km 均值 49.77%；这里 ± 是 seed 间标准差，不是置信区间。

## 4. 原版正式结果与统一指标

大圆距离使用 `R=6371.0088 km`。`@25km` 指误差 **≤25 km** 的照片百分比；百分点与相对百分比不同。中位数、均值和命中率可能给出不同排序，应一起查看；配对净命中变化为“新增命中−失去命中”。

每格为 **中位误差 km / @25km**：

| 划分 | T0 网格 | T1 网格 | T2 网格 | T3 照片＋年份 | T4 回归 |
|---|---:|---:|---:|---:|---:|
| val | 29.85 / 44.45% | 28.45 / 46.58% | 28.26 / 46.70% | 24.55 / 50.47% | 43.50 / 25.87% |
| test_in | 35.23 / 40.40% | 33.61 / 42.00% | 33.98 / 41.33% | 29.11 / 45.82% | 48.30 / 21.04% |
| Bern | 87.81 / 8.02% | 91.63 / 9.27% | 91.30 / 8.68% | 93.29 / 6.40% | 61.08 / 10.54% |
| Ticino | 91.91 / 3.47% | 86.63 / 2.10% | 88.61 / 2.80% | 82.19 / 3.36% | 47.45 / 20.40% |

证据：[T0/T1](../report/experiments/2026-11-18-geoclip-time/run/comparison.json)、[T2](../report/experiments/2026-11-18-geoclip-t2/comparison.json)、[T3](../report/experiments/2026-11-18-geoclip-t0-photo-rerank/results/summary.json)、[T4 对 T1 的逐 ID 比较](../report/experiments/2026-11-20-clip-regression-t4/comparison.json)。

T4 相对 T1 在 test_in 净少 9,977 次 25 km 命中；Bern 净多 43 次，Ticino 净多 2,386 次。回归预测更集中，不能把两个地区的收益单独归因于日期，或推广为所有未知地区都会改善。

### T1 的另一种候选库：位置检索后再按档案年份重排

冻结 T1，用训练照片的坐标作候选；先使用查询时间打分，再读候选自己的年份重排。它与 T3 的第一阶段不同，也不等于原 T1 网格预测。

test_in 中位误差 **33.57→32.04 km**，@25km **42.05%→43.47%**，净增 676 次命中；93 个 block、1,000 次配对 bootstrap 的中位降幅 95% 区间 **[0.65,2.13] km**。20 次档案年份置换检查固定缺失模式，真实年份好于置换；它支持馆藏日期具有预测信息，不能区分真实历史变化与年代—地点采集先验。缺日期候选零扣分的偏置仍需注意。

证据：[位置候选比较](../report/experiments/2026-11-18-geoclip-t1-location-rerank/verified_comparison.json)、[配对分析](../report/experiments/2026-11-18-geoclip-t1-location-rerank/paired_test_in.json)、[年份置换](../report/experiments/2026-11-18-geoclip-t1-location-rerank/shuffle_diagnostic_cpu.json)。GPU 主实验与 CPU 候选复算有少量近似并列排序差异，置换应与同一 CPU 对照比较。

## 5. 连续坐标优化：目前收益很小

冻结 T1、固定查询日期，从训练照片坐标候选的空间第一名出发，沿坐标梯度最大化余弦相似度。它优化输入坐标，不训练权重，也不是反推年份。

最初 8 次尝试采用归一化梯度方向，步长为 250、250、250、250、125、62.5、31.25、15.625 米；投影到起点周围约 1 km 圆盘，仅接受境内且提高分数的点。不是 8 次必定成功的更新，也不是普通 SGD 或 Gauss–Newton。境内检查只约束新接受的点，未保证初始点在国界内。

- 全 val：中位误差 **28.339→28.277 km**，@25km 仅净增 **1 张**。
- 同一 2,048 张 val 样本：重复上述步长周期，比较 1/8/32/128 次尝试及 1/5 km 半径。第 8→128 次的中位误差下降约 **13 米**，两半径的 @25km 均仍为 961 张。
- 按 45 个 block 做 1,000 次配对 bootstrap，8→128 步中位降幅：1 km 半径 **[−49.5,+32.7] 米**，5 km 半径 **[−49.5,+49.7] 米**，均包含零。

分数提高不保证定位误差减小。本轮只在 val 上探索，不能作为测试集改进结论。初值不是原网格预测，不能把两个 baseline 混用。

证据：[8 步验证](../report/experiments/2026-11-19-geoclip-t1-coordinate-refinement-pilot/validation_no_rerank.json)、[多步轨迹](../report/experiments/2026-11-19-geoclip-t1-iterative-refinement/summary.json)、[block bootstrap](../report/experiments/2026-11-19-geoclip-t1-iterative-refinement/block_bootstrap.json)。

## 6. SDXL 扩散增强：512 张变体的已完成实验

从可用真实 train 中按 seed 0 选 **512 张**，每图生成一张 SDXL 1.0 图生图变体。强度 **0.18**，调度器配置 **30 步**；低强度图生图不代表实际执行全部 30 个去噪步。保持原视角、构图和地标是生成目标；自动结构筛查通过不等于地理内容绝对保持，另有小样本人工查看。

512 张均 accepted，训练量 **176,308→176,820**（约 +0.29%）；变体继承父图 GPS、日期和 train 归属，用负 ID 区分，原图与变体按父 ID 成组。评估仍为原 93,538 张真实图，256 个 checkpoint 选择 ID 不变。

T1 与构建 T3 的 T0 从相同预训练基础重新训练 15 epochs。T1 最佳轮次 **12→11**，T0 **8→3**；增加样本同时改变了批次轨迹和更新次数。增强 T3 对每个父图取原图/变体最高相似度，再取 50 个不同父图，避免重复父图占掉候选槽位；图库独立坐标没有因此增加。T4 重新求解最小二乘。

| 模型 | test_in 中位误差，原→增强 | test_in @25km，原→增强 | 命中净变化 | 增强后 Bern / Ticino 中位误差 |
|---|---:|---:|---:|---:|
| T1 | 33.61→33.46 km | 42.002%→41.872% | −62 | 93.81 / 99.14 km |
| T3 | 29.11→29.27 km | 45.818%→45.696% | −58 | 93.26 / 81.68 km |
| T4 | 48.30→48.32 km | 21.039%→21.037% | −1 | 61.06 / 47.35 km |

T1 的 Bern @25km 由 9.27% 降到 3.35%。全四 split 的结果见[增强比较](../report/experiments/2026-11-20-diffusion-augmentation/comparison.json)。

test_in 以同一组 93 个地理块联合配对抽样 1,000 次、seed 0。定义中位误差改善为增强前减增强后，其 95% 区间：T1 **[−3.00,2.13] km**、T3 **[−0.61,0.28] km**、T4 **[−0.08,0.05] km**；@25km 改善区间也都包含零。它们反映当前测试块抽样不确定性，不含多次训练的随机性。

本轮未显示一致、稳定收益；不能据此判定所有扩散增强无效。证据：[配对区间](../report/experiments/2026-11-20-diffusion-augmentation/paired_bootstrap.json)、[生成统计](../report/experiments/2026-11-20-diffusion-augmentation/synthetic_store/generation_summary.json)、[图像检查](../report/experiments/2026-11-20-diffusion-augmentation/image-qa.json)、[增强构建记录](../report/experiments/2026-11-20-diffusion-augmentation/augmented/build_manifest.json)。

## 7. T3 图库覆盖与 top-50 诊断：已完成

图库覆盖依赖不能直接叫作经典过拟合。T3 只能返回训练照片已有坐标；地区内部没有候选不等于周围 25 km 内也没有候选。整库最近坐标 oracle 使用真实 GPS 作诊断，是可达上界，不是可部署的定位算法。

整库有 176,308 张照片、156,064 个不同坐标。已完成的三个测试集统计：

| 测试集 | n | 整库内存在 ≤25 km 坐标 | 视觉第一名命中 | 年份第一名命中 | 年份新增 / 失去命中 |
|---|---:|---:|---:|---:|---:|
| test_in | 47,593 | 47,593 | 18,480 | 21,806 | 6,132 / 2,806 |
| Bern | 3,377 | 3,377 | 225 | 216 | 72 / 81 |
| Ticino | 13,039 | 6,858 | 306 | 438 | 256 / 124 |
| 合计 | 64,009 | 57,828 | 19,011 | 22,460 | 6,460 / 3,011 |

年份重排净增 **3,449** 次命中；6,181 张在整库层面不可达，均来自 Ticino。Bern 的整库 oracle 可达率为 100%，因此 Bern 的低命中不能全部归因于没有训练候选。证据：[整库覆盖](../report/experiments/2026-11-18-geoclip-t0-photo-rerank/reference_coverage.json)；视觉/年份命中与 gains/losses 已由原实验的 `results/predictions.parquet` 逐 ID 配对计算。该逐图文件保留在数据存储中，仓库归档保存其汇总指标。

### top-50 里已有近位置，却未被最终选中的情况

本轮沿用原版 T3 的已训练 T0 图片投影、176,308 张训练照片和 γ=1。视觉第一名与年份重排第一名已逐 ID 核对，与原版预测一致。分析入口为 [audit_top50.py](../report/experiments/2026-11-21-t3-top50-coverage/audit_top50.py)，正式结果见归档中的 [summary.json](../report/experiments/2026-11-21-t3-top50-coverage/summary.json)。

下表的“近位置”统一指距真实位置 **≤25 km**。判断候选里是否有近位置时使用真实坐标，仅用于事后诊断。

| 测试集 | 照片数 | top-50 中至少有一个近位置 | 年份重排最终命中 | 有近位置但最终没选中 |
|---|---:|---:|---:|---:|
| test_in | 47,593 | 43,977 | 21,806 | 22,171 |
| Bern | 3,377 | 1,676 | 216 | 1,460 |
| Ticino | 13,039 | 2,933 | 438 | 2,495 |
| 合计 | 64,009 | 48,586 | 22,460 | 26,126 |

三个测试集共有 **41,549 张**最终未命中，可分成三种情况：

1. **6,181 张：整库都没有近位置。** 都来自 Ticino，当前只能输出训练照片坐标的方法受到图库覆盖限制。
2. **9,242 张：整库有近位置，但视觉检索没把它带进 top-50。** 其中 test_in 3,616 张、Bern 1,701 张、Ticino 3,925 张。
3. **26,126 张：top-50 已经包含近位置，但最后的打分没有选中它。** 说明候选内选择仍有很大差距；这不代表模型事先知道哪个候选正确。

三项相加为 `6,181 + 9,242 + 26,126 = 41,549`。因此，“top-50 没有近位置”的 15,423 张不能全部解释成图库缺乏覆盖：其中 9,242 张是在视觉检索阶段漏掉了整库已有的近位置。

同样，26,126 张候选内选择失败**不等于年份重排造成了这么多退步**。相对视觉第一名，年份重排仍新增 6,460 次命中、失去 3,011 次，净增 3,449 次。该分解定位了覆盖、视觉检索和最终选择三个环节的问题，不单独证明参数过拟合或年份的因果作用。

## 8. 代码与结果在哪里，如何接续

### 分支和工作副本状态

- `Yang_Zhang`：本组继续开发和交接的分支，本次整理范围包括模型实现、实验报告及 `report/experiments/` 结果归档。接续时确认该分支实际可获取的版本，并记录使用的 commit；历史提交 **a87763e** 包含 T2 与图片单侧时间融合，不应固定把它当作之后的最新版本。
- `project_2_Yijun_Liu`：队友实现。本文对队友方法编号的核对基于 **f019e23**；方法差异见第 3 节。
- `codex/yang-geoclip-time` 的 **4a65124** 是较早实现，目录布局也较早，不应当成最新全部成果。
- 本文记录交接内容，不代替提交或推送完成状态的核验；新主机应检查实际取得的代码与归档清单。后续修改在个人可写 checkout 中进行，不依赖或修改其他原始工作副本。

### 项目内相对路径

| 路径 | 用途 |
|---|---|
| `models/geoclip_time/lib/temporal_data.py` | ID 对齐、时间编码、覆盖率 |
| `models/geoclip_time/lib/temporal_model.py` | T1 时间残差双塔 |
| `models/geoclip_time/lib/temporal_ops.py` | 对比损失、候选分块检索 |
| `models/geoclip_time/lib/temporal_training.py` | train-only 训练、val 选择、恢复与预测 |
| `models/geoclip_time/configs/` | T0、双侧/单侧 T1、T2 配置 |
| `models/geoclip_time/scripts/rerank_references.py` | T3 照片检索、年份重排 |
| `models/geoclip_time/scripts/rerank_locations.py` | T1 坐标候选、档案年份重排 |
| `models/clip_regression/` | T4 线性回归、训练、比较和测试 |
| `scripts/augmentation/` | 生成、构建增强集、训练、父图归并检索 |
| `data/splits/main.csv.gz` | 固定原始划分 |
| `lib/`、`scripts/` | 共用地理、指标、缓存等基础工具 |
| `doc/`、`notebooks/`、`report/` | 方法文档、探索和结果说明 |
| `report/experiments/<日期目录>/` | 已归档的汇总、分析脚本及该次实验提供的产物 |

数据存储目录约定为 `meta/smapshot.parquet`、`features/clip_vitl14.npy`、`features/clip_vitl14_ids.npy`、`geo/grid_1km.npy`，这些行内路径不是 Git 文件清单。正式网格预测通常只含 `id,split,lat_pred,lon_pred`；照片检索产物另含 `method,neighbor_id,error_km` 等，同一 ID 有两种方法，必须先筛选方法。T3 取 `method=year`。

### 已核实的入口

以下在仓库根目录运行；尖括号表示需替换的资源位置，不是现成的数据路径。

```text
python models/geoclip_time/scripts/train_time.py --config models/geoclip_time/configs/T1_dual_time.yaml --data-root "<data_root>" --split-csv data/splits/main.csv.gz --check-data
python models/geoclip_time/scripts/train_time.py --config models/geoclip_time/configs/T1_dual_time.yaml --data-root "<data_root>" --split-csv data/splits/main.csv.gz --device cuda
python models/geoclip_time/scripts/predict_time.py --checkpoint "<checkpoint>" --data-root "<data_root>" --split-csv data/splits/main.csv.gz --splits val test_in test_city test_region --allow-partial-features --device cpu
python -m models.geoclip_time.scripts.rerank_references --checkpoint "<T0_checkpoint>" --data-root "<data_root>" --split-csv data/splits/main.csv.gz --output-dir "<new_output>"
python -m models.geoclip_time.scripts.rerank_locations --checkpoint "<T1_checkpoint>" --data-root "<data_root>" --split-csv data/splits/main.csv.gz --output-dir "<new_output>" --device cpu
python models/clip_regression/scripts/train.py --data-root "<data_root>" --split-csv data/splits/main.csv.gz --output "<new_output>"
```

数据根目录可通过 `P2_DATA` 或相应 CLI 参数配置；正式 T0/T1/T2 采用接受不完整缓存的配置副本，不能直接假设仓库默认配置等于正式配置。新实验应使用新 run 名和输出目录，保留原产物。

### 跨主机与 Windows 接续

1. 先取得 `Yang_Zhang` 分支的实际版本，在仓库内阅读本文、`report/geoclip_time_baselines.md` 和 `report/experiments/`，无需复原旧主机的个人目录。记录当前 commit 和所用实验配置。
2. 仅查看 Markdown/JSON 不需要模型环境；读取 Parquet 需要 pandas 及相应引擎，例如 pyarrow。训练、地理转换和推理还需要项目相应依赖与数据，`requirements-lock.txt` 来自既有环境，不保证直接适配 Windows。
3. 重跑时给出本机的数据根目录、划分文件、checkpoint 和新输出目录。尤其恢复旧 checkpoint 时，显式传入 `--data-root` 与 `--split-csv`，避免使用其中记录的原执行环境路径；时间预处理仍沿用 checkpoint 保存的规则。
4. `report/experiments/` 中的历史运行器和 manifest 用于追溯，重跑优先使用上面的项目参数化入口。逐图 Parquet 和 checkpoint 保留在 DHLAB，文中的行内产物路径不表示文件已随 Git 提供；需要逐图复算或模型推理时，另行准备有权使用的数据和权重。
5. 增强生成脚本使用 `fcntl` 文件锁，当前应视为 Linux/WSL 流程；已有结果分析可以与 GPU 生成分开进行。

测试入口为 `python -m pytest -q models/geoclip_time/tests models/clip_regression/tests scripts/augmentation/tests`。

## 9. 早期几何讨论、阅读清单和未来问题

早期考虑将 GeoCLIP 的 Equal Earth 平面位置输入替换为 **S² 与标准黎曼度量**。CP¹ 与 S² 是同一个底层流形；常见 Fubini–Study 度量与单位球面的标准度量对应时有常数尺度。CP¹ 完全可以定义实值函数，实值输出不是排除它的数学理由。把 xyz 输入 MLP 也不自动构成内蕴球面建模。由于当前仅研究瑞士局部区域，几何替换暂时搁置；尚无本项目 S²/CP¹ 实验结果。

已逐篇讨论的阅读清单（编号保留版本，具体内容以论文原文为准）：

| 论文 | arXiv |
|---|---|
| GeoCLIP | [2309.16020](https://arxiv.org/abs/2309.16020) |
| StreetCLIP | [2302.00275](https://arxiv.org/abs/2302.00275) |
| PIGEON | [2307.05845v6](https://arxiv.org/abs/2307.05845v6) |
| OpenStreetView-5M | [2404.18873v1](https://arxiv.org/abs/2404.18873v1) |
| Around the World in 80 Timesteps / PLONK | [2412.06781v1](https://arxiv.org/abs/2412.06781v1) |
| GT-Loc | [2507.10473v2](https://arxiv.org/abs/2507.10473v2) |
| GeoToken | [2511.01082v1](https://arxiv.org/abs/2511.01082v1) |

PLONK 的生成对象是位置；本项目的 SDXL 增强生成的是训练图片，二者不是同一个用途。GeoToken 的 S2 cell 层级离散编码也不等于 S² 上的黎曼学习。

已完成事项与后续问题：

- [x] 完成本轮 top-50 覆盖分解并逐 ID 核对：整库不可达 6,181 张、检索漏入 top-50 9,242 张、候选内最终选择失败 26,126 张，见 [summary.json](../report/experiments/2026-11-21-t3-top50-coverage/summary.json)。
- [ ] 检验时间作用：同一冻结 T1 使用真实/打乱查询时间时，两塔都同步替换；同容量的无信息时间对照、只看年代的地理先验对照仍有价值。已有收益不等于从图像识别了年份。
- [ ] 对 T4 增加图片单独、日期单独等对照，研究连续回归的集中化与留出集表现，不仅比较 @25km。
- [ ] 多 seed 与更严格的地区验证；现有 block bootstrap 不覆盖训练随机性。测试集已经被多轮探索性分析查看，后续报告应如实说明，避免再当作完全未触碰的最终盲测。
- [ ] 如继续扩散增强，应先检查地理内容保持、增强比例与常规增强对照。512 张全过自动筛查不证明标签绝对不变，也不能据本轮结果否定所有增强方案。
- [ ] 连续坐标优化若继续研究，应区分分数优化与真实定位精度，并与相同初值、候选支持和计算预算比较。

当前回归想法已落地为 T4；尚未实现的联合时间预测、球面位置模型等应明确作为新问题，而非已有模型能力。
