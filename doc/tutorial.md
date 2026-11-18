# CS-433 Project 2 ：瑞士历史照片的细粒度图像定位（个人从零实现版）

- **【已知】**：已经测出来或已经核实过的事实。
- **【假设】**：待实验验证的预期，不能当结果写进报告。
- **【需确认】**：还没核实的信息。

队友的结果只作为**背景和参照**（§2）。本教程里的数据、划分、代码和实验全部由你自己从零完成。

**工作地点**：只在 DHLAB 上做。代码在 `/rcp-scratch/students/yijliu/ml-project2/ML_project_1`，数据在 `/rcp-scratch/students/yijliu/ml-project2/data`，不进 Git。

---

## 0. 一页速览

| 项目 | 内容 |
| --- | --- |
| 截止 | **2026-12-17（周四）16:00**。今天 11-14，约 4.5 周 |
| 交付物 | 最多 4 页 LaTeX 报告；伦理段落 200–400 词（不计入 4 页）；AI 使用附录；GitHub 链接（指向最终 commit） |
| 核心目标 | 把全球图像定位模型 **GeoCLIP** 适配到**瑞士历史照片的细粒度定位**（主看 ≤1 / 5 / 25 km 命中率和中位误差） |
| 可选目标 1 | 对每个预测给出**可信的可靠度**：预测 $P(\text{误差}\le T)$，并且要校准好 |
| 可选目标 2 | 在**训练中从未出现的地区**上测试泛化：留出一整个城市和一整个州（建议 Ticino） |
| 数据 | SMAPshot 公开 API，已核实有 355,281 张 validated 图（§5.2） |
| 算力 | 全部在 DHLAB 上做（代码、EDA、元数据、下图、训练、作图、报告）。GPU 为 Quadro RTX 8000，最多 2 块，不用 GPU 3 |
| 关键工程决定 | ① **在下载图像之前**就把留出地区和空间划分定死；② CLIP 主干冻结，所以每张图只过一次 CLIP，把特征缓存下来，之后所有实验都在特征上跑（几分钟一次） |

**一句话故事线**：GeoCLIP 在瑞士历史照片上 25 km 内只对 10–18%（队友测的），而且不知道自己什么时候错。我们用瑞士候选库、瑞士负样本微调和距离平滑损失把它变成"瑞士专家"，再证明：(i) 收益在空间不相交的测试集上依然成立，甚至在没见过的州上也成立；(ii) 模型能告诉我们哪些预测可以信。

---

## 1. 官方要求（对照评分标准）

| 官方要求 | 对你的含义 |
| --- | --- |
| 4 页报告，参考文献和伦理可放额外页 | 只讲一条线：基线 → 适配 →（可选）可靠度 →（可选）泛化 |
| 评分看 *solid baselines*：从很简单的基线开始，展示每个改动的增益 | 所有方法在**同一组测试集**上比较；一定要有常数预测和 kNN 这种最朴素的基线 |
| 可复现：超参数、预处理、交叉验证细节都写清 | 固定种子；划分文件进 Git；每个实验一个配置文件；README 里写清每张表怎么复现 |
| 创新性：说清"解决当前模型的什么具体问题" | 具体问题：全球模型在瑞士尺度上分辨率不够，而且对新地区的泛化和自身可靠度都未知 |
| 伦理段落 200–400 词 | 建议写"识别出风险"的版本（§8.3） |
| AI 工具允许，但要在附录说明 | 从现在开始记录用 AI 做了什么 |
| 代码超过 100 MB 不能放 GitHub | 图像和特征不进仓库；只提交 ID、划分文件和脚本 |
| 提交链接 `https://github.com/<user>/<repo>/tree/<commit-hash>` | **【需确认】** 课程是按组提交的。你单独做的版本怎么提交（单独仓库、组仓库里的独立目录，还是合并进组报告），要和组里、助教确认 |

---

## 2. 队友结果（只作背景和参照）

这些数字用的测试集各不相同，而且很可能是**随机划分**。它们只用来：① 估计量级；② 在报告的 Related Work / Motivation 里说明"已知 GeoCLIP 在这类数据上不行"；③ 提醒我们哪些坑要避开。**不要**和你自己的表格放在一起直接比较。

| 来源 | 设置 | 结果【已知】 |
| --- | --- | --- |
| Week 1 | GeoCLIP 零样本，1000 张 | 中位 125 km，平均 671 km；≤25 km 18.2%；约 5.7% 的图被预测到 2500 km 以外 |
| Week 1 | 加瑞士区域先验 $s-\lambda U(g)$ | 中位 69 km，平均 92 km |
| Week 1 | 550/150/300 小规模微调 | 原始 79 km → 只训 MLP 27.7 km → MLP 加位置编码器 25.0 km；≤25 km 从 28% 到 50% |
| M2 | 5 个预训练模型，9103 张 | GeoCLIP 中位 206 km（≤25 km 10.2%）；GT-Loc 184 km；PLONK-YFCC 108.6 km（≤25 km 14.4%，最强） |

**从中学到的教训**（直接影响你的设计）：

1. **测试集必须统一**。同一个 GeoCLIP 在三个集合上的中位误差是 125 / 79 / 206 km，差将近 3 倍。
2. **随机划分会泄漏**。SMAPshot 里同一地点有很多张照片，随机划分时测试图的"邻居"就在训练集里，微调收益会被高估。你要用空间不相交的划分，再加一个完全没见过的地区（§5.5）。
3. **区域先验很有效，但它的收益有多少只是"别跑出瑞士"**？需要常数预测和 kNN 基线来回答。
4. **微调后平均误差仍有 306 km**，说明还有预测落在瑞士外。所以微调必须配合瑞士候选库一起用。
5. **数据量**：队友只用了 1000 张。你用几万到几十万张，这本身就是一个实质性的改进。

---

## 3. 背景知识速成

### 3.1 任务与指标

**任务**：输入一张照片，输出拍摄位置 $(\text{lat},\text{lon})$。

**距离**用球面大圆距离（haversine），$R=6371$ km：

$$
d=2R\arcsin\sqrt{\sin^2\tfrac{\Delta\phi}{2}+\cos\phi_1\cos\phi_2\sin^2\tfrac{\Delta\lambda}{2}}
$$

**主指标**（瑞士尺度）：

- ≤1 / 5 / 10 / 25 / 50 km 的命中率。
- 中位误差、平均误差、P90。

附录保留标准五级（≤1 / 25 / 200 / 750 / 2500 km），方便和论文比较。

**可靠度指标**：

- AUROC：区分"误差 ≤ T"和"误差 > T"的能力。
- Brier score、ECE：概率是否校准。
- 风险–覆盖曲线：只回答最有把握的 x% 图片时，中位误差和 ≤25 km 命中率是多少。

### 3.2 方法家族

| 类别 | 做法 | 代表 |
| --- | --- | --- |
| 检索：图像→图像 | 在带坐标的参考图库里找最像的图 | kNN 基线、Img2Loc |
| 检索：图像→GPS | 图像嵌入和 GPS 嵌入做对比学习，推理时在 GPS 候选库里找最像的坐标 | **GeoCLIP**、GT-Loc |
| 分类（geocell） | 把地图切成格子当类别 | PlaNet、PIGEON |
| 生成 | 在地球表面做扩散或流匹配，输出位置的概率分布 | PLONK |
| 混合 | 分类加检索细化、层次 token、检索增强的大模型、重排序 | PIGEON、GeoToken、Img2Loc、GeoRanker |

### 3.3 论文清单：读什么，借什么

| 论文 | 一句话 | 借什么 |
| --- | --- | --- |
| **GeoCLIP**（NeurIPS 2023）`papers/neurips-2023-1b57aadd.pdf` | 图像和 GPS 两个编码器做 CLIP 式对比学习，在 GPS 候选库上检索 | 骨干；候选库可换；GPS 队列；噪声增强 |
| **PIGEON**（CVPR 2024） | 语义 geocell、haversine 平滑标签、top-K 内的检索细化 | 平滑软目标（A4）、geocell 细化（A5，可选） |
| **OSV-5M**（CVPR 2024） | 大规模街景基准，严格的训练/测试地理分离 | 空间划分的论证方式 |
| **PLONK**（CVPR 2025） | 黎曼流匹配，输出位置分布，可衡量"可定位性" | 零样本强基线（可选）；可靠度的相关工作 |
| **GT-Loc**（ICCV 2025） | 图像、时间、位置联合嵌入，软目标度量学习 | 软目标的另一种写法（相关工作） |
| **Img2Loc**（SIGIR 2024） | CLIP 检索近邻坐标，再交给多模态大模型 | 支撑 kNN 基线 |
| **GeoRanker**（NeurIPS 2025） | 先检索候选，再做距离感知的重排序 | 相关工作 |
| **AmsterTime**（ICPR 2022） | 历史档案照片和现代街景之间的严重域偏移 | 论证历史照片的域差距 |
| **SMAPshot**（Produit et al. 2018） | 志愿者通过 3D 虚拟地球给历史照片做位姿标注 | 数据来源和研究动机 |
| **StreetCLIP**（arXiv 2302.00275）`papers/arxiv-2302.00275.pdf` | 用地理 caption 继续预训练 CLIP，做文本提示式零样本定位 | 相关工作 |

阅读顺序：GeoCLIP 的 Fig. 2 和 §3（1.5 小时）→ PIGEON 的 loss 和 geocell 两节（1 小时）→ OSV-5M 的数据划分部分（20 分钟）→ PLONK 的 Fig. 1（10 分钟）。其余论文只读摘要。

### 3.4 GeoCLIP 的内部结构（你要动的地方）

- **图像侧**：CLIP ViT-L/14（冻结）输出 768 维，再接可训练的 MLP（768→768→512）。
- **位置侧**：坐标先做 Equal Earth 投影，再用多尺度随机傅里叶特征（RFF），每个尺度一个 MLP，输出相加，得到 512 维。
- **打分**：两边 L2 归一化，相似度 = `exp(logit_scale) × 余弦相似度`。
- **训练**：batch 内 InfoNCE，外加一个 GPS 队列提供额外负样本；GPS 加高斯噪声。
- **推理**：在约 10 万个全球 GPS 点（MP-16）组成的候选库上取 top-k。

官方仓库里的属性名如下。第一次在 DHLAB 上用时请先 `print(model)` 核对：

| 属性 | 含义 |
| --- | --- |
| `model.image_encoder.CLIP` | 冻结的 CLIP（HuggingFace `CLIPModel`） |
| `model.image_encoder.image_processor` | CLIP 的图像预处理 |
| `model.image_encoder.mlp` | 可训练的图像侧 MLP |
| `model.location_encoder` | 位置编码器，输入 `(N, 2)` 的 `[lat, lon]`（度） |
| `model.gps_gallery` | 全球候选库 `(100000, 2)` |
| `model.logit_scale` | 温度（取 `exp` 后使用） |

**四个旋钮**，对应 §5 的 A1–A4：

1. 打分：加区域先验（A1）。
2. 候选库：换成瑞士网格（A2）。
3. 可训练部分：MLP、位置编码器（A3）。
4. 损失和负样本：瑞士境内的负样本，以及 haversine 平滑的软目标（A4）。

---

## 4. 研究问题与假设

**主研究问题**

> 能否把一个全球图像定位模型（GeoCLIP）适配到**瑞士历史照片的细粒度定位**？适配后的收益在**空间不相交**、甚至**完全没见过的地区**上是否依然成立？模型能否对每个预测给出**可信的可靠度**？

建议标题：*Where, and How Sure? Adapting GeoCLIP to Fine-Grained Geolocation of Historical Swiss Photographs*

| 编号 | 子问题 | 假设 | 验证方式 |
| --- | --- | --- | --- |
| **RQ1 适配（核心）** | 哪种适配最有效？各自贡献多少？ | H1a：瑞士候选库加区域先验，不训练就能大幅降低中位误差（主要靠"别出瑞士"）。H1b：瑞士负样本微调带来 ≤5 / 25 km 上的最大增益。H1c：平滑软目标主要改善大误差（P90）和训练稳定性 | 消融表 E0–E8（§5.13），在空间测试集 $T_\text{in}$ 上 |
| **RQ1′ 诊断** | 剩余误差来自哪里：候选库不够密，还是排序错了？ | 在 25 km 尺度上，主要是排序错了 | 比较实际误差和"oracle 误差"（候选库里离真值最近的点） |
| **RQ2 可靠度（可选）** | 能否只用模型输出预测 $P(e\le 1/5/25\text{ km})$？ | H2：学到的可靠度模型比直接用 top-1 概率的 AUROC 更高；只回答最有把握的 30% 时，≤25 km 命中率至少翻倍 | AUROC、ECE、风险–覆盖曲线（§5.11） |
| **RQ3 泛化（可选）** | 在训练中没出现过的城市和州上，适配是否还有效？可靠度是否还校准？ | H3a：kNN 和"训练坐标候选库"在没见过的地区彻底失效；用 GPS 编码器加瑞士网格的方法仍明显优于零样本。H3b：没见过的地区上的增益比 $T_\text{in}$ 小，可靠度模型会**过度自信** | 对比 $T_\text{in}$、$T_\text{city}$、$T_\text{region}$ 三列（§5.12） |

**为什么值得做**（Introduction 用）：

- 档案馆里有大量没有位置的历史照片，SMAPshot 的志愿者标注跟不上。
- 自动定位能给志愿者一个"初始位置"。但只有在模型**知道自己何时不准**、并且**对新地区也有效**时才真正可用。

---

## 5. 方法论（详细）：每一步在哪里做、怎么做

### 5.0 总览

```text
全部在 DHLAB
/rcp-scratch/students/yijliu/ml-project2/ML_project_1
S1 环境与仓库骨架
S2 拉元数据 → $P2_DATA/meta/smapshot.parquet
S3 EDA：看分布、选留出地区
S4 清洗 + 划分 → data/splits/main.csv.gz
S5 下载图像（screen，可断点续传）
S6 CLIP 特征缓存（GPU，一次）
S7 评估库（metrics、bootstrap）
S8  基线 B0–B4
S9  零训练适配 A1–A2（先验、瑞士候选库）
S10 微调 A3–A4（瑞士负样本、软目标）
S11（可选）可靠度模型
S13 汇总表格、作图、写报告（`test_city` / `test_region` 在最终评估时填主表第三、四列）
```

数据、特征、checkpoint 不进 Git。代码、划分文件和小的指标 JSON 进 Git。`P2_DATA=/rcp-scratch/students/yijliu/ml-project2/data`。

---

### S1. 环境与仓库骨架【已完成】

环境是 `/rcp-scratch/students/yijliu/.conda/envs/ml-p2`（torch 2.15.0+cu132，geoclip 1.3.0）。GPU 1 上 `predict` 对一张 Aproz 缩略图返回了 3 个坐标：前两名约 294 km（意大利北部），第三名距 Aproz 约 5.5 km。依赖锁在 `requirements-lock.txt`。`P2_DATA` 写在 `~/.bashrc`。路径代码在 `lib/paths.py`。

---

### S2. 拉取 SMAPshot 元数据【已完成】

`scripts/fetch_metadata.py` 调用 `GET /api/v1/images?limit=1000&offset=N&state[]=validated`（`state=validated` 会 400）。结果在 `$P2_DATA/meta/smapshot.parquet`：355,281 行，`id` 唯一，经纬度全部非空。`view_type`：lowOblique 146,929，terrestrial 100,999，highOblique 60,500，nadir 40,531，空 6,322。

坐标是相机位置，不是画面主体，所以结果要按 `view_type` 分开报。`title` 里常有地名，不能当模型输入。

---

### S3. 探索性分析（EDA）与选留出地区【已完成】

[`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb)。收藏名称在 `$P2_DATA/meta/collections.json`。

选择标准：留出集至少 2,000 张，并且 terrestrial 和航拍都要有；留出州要和训练区的语言、地形不同；留出城市不能占掉太大一块训练集。

据此选定 **Ticino**（14,467 张，terrestrial 8,994；阿尔卑斯以南、意大利语，另有 Fonds Perrochet）和 **Bern**（10 km 内 3,985 张，terrestrial 1,455）。Zürich 10 km 内 32,128 张，太大。Basel 有 3,633 张，但 terrestrial 只有 176。4Dbrowser 瑞士框内只有约 685 个现代全景点，不能当瑞士留出城市。

---

### S4. 清洗与划分【已完成】

代码在 `lib/splits.py`，入口 `scripts/make_splits.py`。文件 `data/splits/main.csv.gz`。Fig. 1 在 `report/figures/fig1_split_map.png`。

清洗：

1. 缺坐标：0 张。
2. 只留瑞士国界向外 5 km 以内，去掉里约、Vorarlberg 等。境外 31,138 张（另有 nadir 274），标成 `outside`，不下载。
3. `view_type == nadir` 不进主实验，标签加 `nadir_` 前缀，共 40,531 张。
4. 近重复（CLIP 余弦 > 0.97 且距离 < 50 m）留到 S6，现在还没删。

划分（seed 0）：

| 集合 | 规则 | 非 nadir 张数 |
| --- | --- | --- |
| `test_region` | Ticino 州界内 | 13,040 |
| `test_city` | Bern（46.9480, 7.4474）10 km 内 | 3,378 |
| `buffer` | Ticino 向外 10 km，以及 Bern 的 10–20 km 环带；训练和测试都不用 | 13,515 |
| train / val / `test_in` | 其余点按 10 km × 10 km 的 LV95 块随机分 70% / 10% / 20% | 176,538 / 29,532 / 47,609 |

块划分所以比例不会正好是 70/10/20。`random_split` 列是同一批「其余点」上的纯随机对照，用来看空间泄漏。从此不改这个文件。超参数只在 `val` 上选。5 km 和 20 km 块的检查放附录。文件还没 git commit。

---

### S5. 下载图像【DHLAB，已完成】

`scripts/download_images.py`，间隔 0.5 s，IIIF 统一 `/full/!512,512/`；已有文件跳过。日志 `$P2_DATA/download.log` 末尾：`finished seen=308947 saved_this_run=308670`；磁盘 **308,672** 张，约 **6.7 GB**。队列含 nadir，不含 `outside` / `buffer`。

**数据概览（发给组内对齐用）**：

| 项 | 内容 |
| --- | --- |
| 数据 `P2_DATA` | `/rcp-scratch/students/yijliu/ml-project2/data` |
| 划分 | Ticino 全州 + Bern 10 km 留出；`data/splits/main.csv.gz`（规则同 S4） |
| 图片 | `$P2_DATA/images/{id//1000}/{id}.jpg` |
| GT | `$P2_DATA/meta/smapshot.parquet`（355,281 validated；按 `id` 对齐） |

| 集合 | 规则 | 张数（非 nadir） |
| --- | --- | --- |
| `train` | 10 km LV95 块内 70% | 176,538 |
| `val` | 同块 10%（调参只用 val） | 29,532 |
| `test_in` | 同块 20% | 47,609 |
| `test_region` | 整州 Ticino（训练未见） | 13,040 |
| `test_city` | Bern 市中心 (46.9480°, 7.4474°) 10 km 内 | 3,378 |
| `buffer` | Ticino/Bern 缓冲带，不下载 | 13,515 |
| `outside` | 瑞士国界外，不下载 | 31,138 |
| `nadir` | 带 `nadir_` 前缀，不进主表 | 40,531 |

---

### S6. 缓存 CLIP 特征【DHLAB GPU，已完成】

CLIP 冻结；每张已下载图存 **768 维 fp16**（与 `geoclip` `ImageEncoder` 进 `mlp` 之前一致，走 `get_image_features`）。实现：`lib/features.py`、`scripts/cache_features.py`（分片可续跑，默认 `workers=0` 避免 NFS 上 DataLoader 临时目录问题）。

| 项 | 结果 |
| --- | --- |
| 输出 | `$P2_DATA/features/clip_vitl14.npy`（308,672×768，~453 MB）、`clip_vitl14_ids.npy`、`clip_vitl14_meta.json` |
| 分片 | `features/clip_vitl14_shards/`（中断后可重跑，已存在 shard 会跳过） |
| 日志 | `$P2_DATA/features/cache.log`（tmux 会话 `feat`） |
| 校验 | 首 batch 与 GeoCLIP 前向 `atol=1e-3`；无 NaN；`failed_ids=0` |

长任务示例：`scripts/run_cache_features_tmux.sh`（先 `nvidia-smi` 选空闲卡，勿用 GPU 3）。

**去重**（尚未做）：50 m 内余弦 > 0.97 的近重复，在 split 加 `dup` 列——等 B2/kNN 前再做即可。

---

### S7. 评估库【已完成】

`lib/metrics.py`：`summarize`（含 1/5/25 km 等命中率）、按 `block` 整块 bootstrap 的 `bootstrap_ci`、`evaluate`（按 split 和 `view_type`）。预测表统一为 `id, split, lat_pred, lon_pred`，存在 `$P2_DATA/results/preds/`。自测：`scripts/check_metrics.py`（洛桑–苏黎世约 173.46 km）。

---

### S8. 基线【B0–B3 已完成】

| 编号 | 基线 | 做法 | `val` 结果 |
| --- | --- | --- | --- |
| B0 | 常数预测 | `scripts/run_b0.py` | n=29,532；中位 **59.5 km**，≤25 km **10.9%** |
| B1 | GeoCLIP 零样本 | MP-16 候选库；`scripts/run_baselines.py` | n=29,529；中位 **248 km**，≤25 km **14.6%** |
| B2 | kNN 检索 | 非 nadir `train` top-1（768 维 CLIP） | n=29,529；中位 **37.6 km**，≤25 km **38.5%** |
| B3 | GeoCLIP + 瑞士网格 | `grid_1km.npy`，权重同 B1 | n=29,529；中位 **62.1 km**，≤25 km **20.8%** |
| B4（可选） | PLONK-YFCC | 官方权重 | 未做 |

**落盘（避免重跑）**：`features/clip_mlp512.npy`；`gallery_mp16_*`、`gallery_grid1km_*`；`preds/B{1,2,3}.parquet`；B1/B3 的 `B*_topk.npz`（top-20，供可靠度）；`metrics/B*_summary.json`（含各 split）。入口 `scripts/run_baselines.py`（`--which b1|b2|b3|all`，已有文件则跳过）。

**要点**：B1 全球库在瑞士上仍偏大；B2 在 `val` 很强；B3 换网格后中位优于 B0，仍弱于 B2。微调与 A1/A2 应超过 B2（至少在 `val` 上）。

---

### S9. 零训练适配 A1–A2【已完成】

**A1（区域先验）**：$s=s_\text{GeoCLIP}-\lambda U(g)$；`scripts/search_a1.py` 在 `val` 上网格搜 **λ∈{1,3,10,30}、scale_km∈{25,50,100}** → `metrics/a1_search.json`。**最优 λ=30、scale_km=25**；`scripts/run_a1.py` → `preds/A1.parquet`、`A1_topk.npz`、`metrics/A1_summary.json`。**val**：中位 **67.5 km**，≤25 km **24.1%**（B1 248 km / 14.6%）；平均误差由 ~1653 km 降到 ~85 km（少飞出瑞士）。

**A2（瑞士网格）**：即 **B3**（`grid_1km.npy`），**val** 中位 **62.1 km**，≤25 km **20.8%**。

**val 消融（零训练）**：`metrics/s9_val_ablation.json`

| 行 | 中位 km | ≤25 km |
| --- | --- | --- |
| B1 全球 | 248 | 14.6% |
| **A1 + 先验** | 67.5 | 24.1% |
| A2 网格 (=B3) | 62.1 | 20.8% |
| B2 kNN | 37.6 | 38.5% |

**仍 optional**：G-train / G-grid+train（S14 E9）。**Oracle**：`oracle_val.json`（网格 oracle 0.39 km vs B3 62 km → 排序瓶颈）。

**入口**：`scripts/run_s9_tmux.sh`（`search_a1` → `run_a1` → `s9_ablation`）；推理复用 `lib/baseline_infer.py`。

---

### S10. 微调 A3–A4【已完成首轮；推理用 1 km 网格】

非 nadir `train`（176,308）上读缓存 **768 维**特征；`scripts/train.py` + yaml；`lib/geoclip_ops.geoclip_train`；15 epoch，batch 512，每 epoch 在 **5k val 子集 + 1 km 网格** 上记中位误差并保留最佳 `ckpt/`。结束后对 **全 `val`/test** 在 1 km 网格上导出 `preds/`、`metrics/`。长任务：`scripts/run_s10_tmux.sh` → `results/s10.log`。

| 编号 | 训练 | 负样本 | `val` 中位 | ≤25 km |
| --- | --- | --- | --- | --- |
| A3a | 只 MLP | 全球 | 33.4 km | 42.5% |
| A3b | 只 MLP | 瑞士网格 | 28.9 km | 46.2% |
| A3c | MLP + 位置编码器 | 瑞士网格 | 29.9 km | 45.4% |
| **A4_tau5** | MLP + 位置编码器 | 瑞士网格，τ=5 km 软目标 | **27.7 km** | **47.3%** |

相对 **B2**（37.6 km / 38.5%）与 **B3**（62.1 km），微调后在 `val` 上更好。

**A4_tau5 三种子（seed 0/1/2）**，1 km 网格推理；汇总 `metrics/A4_tau5_multiseed.json`：

| split | 中位 km（3 seed 平均 ± std） | ≤25 km 平均 |
| --- | --- | --- |
| val | 26.6 ± 1.0 | 48.4% |
| test_in | 32.0 ± 1.9 | 43.2% |
| test_city | 93.3 ± 0.6 | 8.1% |
| test_region | 79.8 ± 0.9 | 3.7% |

配置：`A4_tau5`（seed 0）、`A4_tau5_s1`、`A4_tau5_s2`；`scripts/summarize_a4_multiseed.py`。

---

### S11. 预测可靠度【已完成】

对 A4_tau5（seed 0）估计 $\hat p_T=P(e\le T)$，$T\in\{1,5,25\}$ km。标签是真值误差是否不超过 T。只在 `val` 上拟合：train 上的预测过拟合，置信度会偏高。

`scripts/export_topk.py` 在 1 km 网格上导出 top-20，`$P2_DATA/results/preds/A4_tau5_topk.npz`（93,538 行：val / test_in / test_city / test_region）。特征：`p1`、`margin`、`entropy_k`、`spread_k`（候选点 `features/gallery_grid1km_gps.npy`）、B2 的 `knn_sim`、A4 与 B2 坐标的 `knn_agree`、`view_type` one-hot。宽高比这次没读 jpg。

`lib/reliability.py`：`HistGradientBoostingClassifier`，按 `block` 做 5 折 `GroupKFold` 在深度 {2, 4, 6} 里选，再 `CalibratedClassifierCV(method="isotonic")` 在全 val 上重训。对照是只对 `p1` 做 isotonic。入口 `scripts/run_reliability.py`（CPU）。指标 `$P2_DATA/results/metrics/reliability.json`。T=25 km 的图：`report/figures/risk_coverage.png`（test_in，模型 vs p1）、`calibration_test_in.png`、`calibration_test_region.png`。

**test_in AUROC**（模型 / p1 isotonic）：

| T | 模型 | p1 |
| --- | --- | --- |
| 1 km | 0.752 | 0.706 |
| 5 km | 0.706 | 0.684 |
| 25 km | 0.687 | 0.634 |

三个阈值上模型都高于 p1。T=25 km 时，只保留模型最有把握的一小部分，≤25 km 命中率能到 80% 以上；全覆盖时回到约 41%。

**没见过的地区，校准变差。** T=25 km 的 ECE：test_in 上模型 0.052（p1 0.058），test_region 上模型 0.354（p1 0.416）。val 里 ≤25 km 约占 47%，Ticino 只有 3.8%，所以概率整体偏高；校准图几乎是一条贴在横轴上的线。test_region 的 AUROC 也只剩 0.525，和 p1 的 0.522 差不多。`test_city` 和 `test_region` 在 1 km、5 km 上命中数为 0，AUROC 没有定义。

---

### S12. 拍摄时间（T 系列）【已完成】

**目的**：把 SMAPshot 的 `date_shot_min` / `date_shot_max` 接入 GeoCLIP 查询侧，和 A4 对比。

**特征**：`lib/time_feats.py` 生成 13 维时间向量（对齐 `clip_vitl14_ids`），落盘 `$P2_DATA/features/time_feats.npy`。含中点年份归一化、多周期 sin/cos、`log1p(跨度天数)`、`has_date`；缺日期用可学习 unknown 向量。

**模型**：`lib/geoclip_ops.TimeBranch`，残差融合 `q = normalize(img_emb + α·time_emb)`，从 `A4_tau5.pt` 初始化；1 km 网格不变。

| ID | 做法 |
| --- | --- |
| **T0** | B2 的 top-50 视觉近邻，按 `sim - γ·|Δyear|/50` 重排（val 上 γ=1.0） |
| **T1** | 只训 TimeBranch，冻结 MLP 与位置编码器，5 epoch |
| **T2** | T1 + 训练时 30% time dropout |
| **T3** | T2 + 与 A4 同样微调 MLP 与位置编码器（**val 上最优**） |

入口：`scripts/run_knn_time.py`（T0）、`scripts/train.py` + `configs/T{1,2,3}.yaml`、`scripts/eval_time.py`、`scripts/summarize_time_multiseed.py`（T3 三 seed）。

**val（T3 vs A4）**：中位 **21.5 ± 0.5 km** vs 27.7 km；≤25 km **54.0%** vs 47.3%。T0 仅 val：**26.9 km / 48.0%**（无训练）。

**test_in（T3 三 seed 平均）**：中位 **25.2 ± 0.6 km** vs A4 **32.0 ± 1.9 km**；≤25 km **49.8%** vs **43.2%**。达标：分布内增益超过 A4 种子波动。

**留出集**：`test_city` 中位 ~93.7 km、≤25 km ~8.2%（与 A4 接近）。`test_region` 中位 ~97 km（略差于 A4 79.8 km），≤25 km ~4.2%（略高于 A4 3.7%）。Ticino 年代分布偏 1920–30 年代，时间分支容易学到训练区年代先验。

**对照**：`eval_time.py` 打乱年份后，test_in 中位升至 **33.9 km**（真日期 **25.1 km**），说明增益确实用到日期。test_region 打乱后 **80.2 km**（真日期 **89.6 km**），Ticino 上日期与地点先验混杂，不能当可靠特征。

---

### S13. 汇总、作图与复现打包【DHLAB】

结果已经在 `$P2_DATA/results/`。在 `notebooks/02_figures.ipynb` 里出图：

| 图 | 内容 |
| --- | --- |
| Fig. 1 | 划分地图（train / val / `test_in` / 留出城市 / 留出州 / 缓冲带） |
| Fig. 2 | 误差 CDF：B0、B1、B2、A2、A4 |
| Fig. 3 | `test_in` 上的误差地图（LV95），或者"误差 vs 到最近训练图的距离" |
| Fig. 4（可选） | 风险–覆盖曲线 + 校准图 |
| Table 1 | 主结果：方法 × 三个测试集 |
| Table 2 | 消融：候选库、负样本、损失、可训练部分 |
| Table 3（可选） | 可靠度 |

**复现要求**：

- README 里每张表对应一条命令，例如 `python -m scripts.eval --preds A4_tau5 --splits test_in,test_city,test_region`。
- 固定种子，记录 commit hash。
- 附录写清所有超参数。

---

### S14. 实验矩阵（最终要填的表）

| ID | 方法 | 训练 | 候选库 | 回答 |
| --- | --- | --- | --- | --- |
| E0 | B0 常数预测 | – | – | 下限 |
| E1 | B1 GeoCLIP 零样本 | – | G-global | 起点 |
| E2 | B2 kNN（top-1 / top-k 中位点） | – | train 图像 | 检索门槛 |
| E3 | A1 区域先验 | – | G-global | 复现 Week 1 |
| E4 | A2 瑞士网格 | – | G-grid500m | 候选库的作用 |
| E5 | A3a MLP + 全球负样本 | MLP | G-grid500m | 原版微调 |
| E6 | A3b MLP + 瑞士负样本 | MLP | G-grid500m | 负样本的作用 |
| E7 | A3c + 位置编码器 | MLP + 位置编码器 | G-grid500m | 位置编码器的作用 |
| **E8** | **A4 软目标（最终模型）** | MLP + 位置编码器 | G-grid500m | 损失的作用 |
| E9 | E8 换 G-train | – | G-train | 泛化：候选库的影响 |
| E10 | E8 用随机划分训练和测试 | 同 E8 | G-grid500m | 空间泄漏有多大 |
| E11（可选） | PLONK-YFCC 零样本 | – | – | 外部强基线 |
| E12（可选） | E8 + 可靠度模型 | – | – | RQ2 |
| E13（可选） | E8 跨国家（Dresden） | – | G-global | 专家化的代价 |

E0–E10 在 `val` 上完成选择，然后在 `test_in`、`test_city`、`test_region` 上**各跑一次**。

---

## 6. 预期结果（假设，用来判断实验是否正常）

| 结果 | 预期量级【假设】 | 如果不符合 |
| --- | --- | --- |
| B0 常数预测 | 中位约 50–80 km | 检查坐标有没有反、过滤是否正确 |
| B1 零样本 | 中位 100–200 km；≤25 km 约 10–20% | 检查特征取法和 `assert` |
| B2 kNN | `test_in` 上可能和 A2 相当甚至更好；`test_region` 上明显变差 | 如果在 `test_in` 上 kNN 极强，说明空间块太小（泄漏），试 20 km 块 |
| A1 / A2 | 中位降到 40–70 km；平均误差大幅下降（不再出瑞士） | — |
| A3–A4 | 中位约 15–30 km；≤25 km 约 45–60%（队友在随机划分上是 50%；空间划分会更难） | 先查学习率，再查负样本 |
| A4 vs A3c | P90 下降；中位相近 | 调大 τ |
| E10 随机划分 | 比 E8 好很多 | 差距就是泄漏量，可以写进报告 |
| `test_region` | A4 增益保留一部分；kNN 崩溃 | 如果 A4 比 B1 还差，说明遗忘了，降低位置编码器学习率或者只训 MLP |
| 可靠度 | AUROC（T=25 km）约 0.75–0.85；覆盖 30% 时 ≤25 km 命中率翻倍 | 加 `spread_k` 和 `knn_agree` 特征 |
| 分 `view_type` | terrestrial 比倾斜航拍准 | 写进 Discussion |

**最低成功标准**：

1. E8 在 `test_in` 上 ≤25 km 命中率比 E1 至少高 2 倍。
2. 结果有 bootstrap 置信区间。
3. 报告了 `test_region` 的结果，不管好坏。

只要这三点做到，"负结果"（例如 kNN 赢了）也是可写的发现。

---

## 7. 时间线与 TODO（一个人）

**每天的节奏**：代码和长任务都在 DHLAB 上做。长任务放进 `screen`。

### 第 1 周（11/14–11/20）：数据和基准定死

- [x] S1 环境与 `predict`
- [x] S2 全量元数据 → `smapshot.parquet`（355,281 行）
- [x] S3 EDA notebook（Ticino + Bern 建议）
- [x] S4 划分 `data/splits/main.csv.gz`（Ticino + Bern）和 Fig. 1；尚未 commit
- [x] S5 下载完成（308,672 张，~6.7 GB）
- [x] S7 `lib/metrics.py` 与自测
- [x] S8 B0 在 `val` 上的数字（B1/B2 等图）
- [ ] 读 GeoCLIP 和 PIGEON 的方法部分

### 第 2 周（11/21–11/27）：特征和基线

- [x] S6 特征已缓存（308,672×768）；去重仍待做
- [x] S8 B0–B3 在 `val` 及全 split 预测已落盘
- [x] S9 A1 搜索 + A2（B3）；`s9_val_ablation.json`
- [x] 报告草稿：`report/sections/data_benchmark.tex`、Fig. 1 已在 `report/figures/`

### 第 3 周（11/28–12/04）：微调

- [x] S10 首轮 A3a–A4 训练与 `val` 指标（见上表）
- [x] A4_tau5 三种子 + `test_in` / `test_city` / `test_region` 已评（`A4_tau5_multiseed.json`）
- [ ] 最终配置跑 3 个种子
- [ ] E9、E10
- [ ] 在三个测试集上做最终评估（只跑一次）

### 第 4 周（12/05–12/11）：可选部分和收尾实验

- [x] S11 可靠度：特征、模型、风险–覆盖、校准（`metrics/reliability.json`）
- [x] S12 拍摄时间：T0–T3、`metrics/T3_multiseed.json`、`time_analysis.json`
- [ ] 主表填 `test_in` / `test_city` / `test_region` 三列（S4 已留出，最终评估时跑）
- [ ] bootstrap 置信区间；块大小敏感性检查（附录）
- [ ] **12/09 冻结实验**，之后只写作

### 第 5 周（12/12–12/17）：报告和提交

- [ ] 4 页报告 + 伦理段落 + AI 附录
- [ ] README 复现说明，每张表一条命令
- [ ] 确认提交方式（组仓库还是个人仓库）
- [ ] **12/16 提交**（留一天缓冲）

**时间不够时按这个顺序砍**：E13 → E11 → S11 的交叉拟合（只保留 val 拟合）→ A5 → 整个 S11。`test_region` 列尽量保留。

---

## 8. 报告结构（4 页）

### 8.1 篇幅分配

| 部分 | 篇幅 | 内容 |
| --- | --- | --- |
| Abstract + Introduction | 0.5 页 | 档案照片定位的需求；具体问题；3 条贡献 |
| Related Work | 0.4 页 | 四类方法各一句；历史照片的域差距；可定位性 |
| Data & Benchmark | 0.6 页 | SMAPshot、清洗、三个测试集和缓冲带、指标；Fig. 1 |
| Method | 0.9 页 | GeoCLIP 回顾；A1–A4；（可选）可靠度模型 |
| Experiments | 1.3 页 | Table 1 主结果（三列）、Table 2 消融、CDF、风险–覆盖曲线 |
| Discussion / Limitations | 0.3 页 | 相机位置和景物位置的区别；众包偏差；`view_type` 差异 |
| 额外页 | — | Ethical risks、References、Appendix（超参数、AI 使用、样例图） |

**三条贡献的写法示例**：

1. 一个空间不相交、带留出城市和留出州的 SMAPshot 定位基准。
2. 一套把 GeoCLIP 瑞士化的简单方法（瑞士候选库 + 瑞士负样本 + 距离平滑损失），并做了逐项消融。
3. （可选）一个校准的可靠度估计器，以及它在新地区上的表现分析。

### 8.2 写作要点

- 每张表都从 B0 开始，一行加一个改动，标出增益。
- 主表一定要有 `test_region` 列，这是区别于"随机划分刷分"的关键。
- 超参数放进附录的一张表。

### 8.3 伦理段落（"识别出风险"版本）

**风险**：

- **隐私和滥用**：图像定位方法可以迁移到现代私人照片，用于跟踪个人。
- **空间偏差**：众包标注集中在某些地区和热门景点，模型在少标注地区更差，用于档案编目时会放大不均衡。我们的 `test_region` 实验就在量化这一点。
- **版权**：各档案馆的图像许可不同。

**怎么评估**：

- 按州和 `view_type` 报告误差和样本量。
- 对比 `test_in` 和 `test_region`。

**怎么应对**：

- 报告分地区的误差。
- 用可靠度模型把低置信度的预测交给人工。
- 仓库里不放图像，只放 ID 和下载脚本，并遵守各档案馆的许可。

### 8.4 AI 使用附录

如实写：用 AI 辅助过哪些代码、文献检索、写作润色；哪些内容是人工核对过的。

---

## 9. 风险与备选方案

| 风险 | 备选 |
| --- | --- |
| GPU 被占满 | 特征已缓存；B1/B2 推理与 kNN 可分批跑；单卡微调每次几分钟 |
| 微调在 `test_in` 上不如 kNN | 主表强调 `test_region`：kNN 无训练坐标；可记作发现或做简单混合 |
| 4 页写不下 | 主线 RQ1 + 三列测试表；可靠度/S11 压缩进附录 |

---

## 10. 需要确认的问题

**问助教或组里**

1. 课程按组提交。我独立完成的版本以什么形式提交：独立仓库、组仓库里的独立目录，还是并入组报告？

**问导师（DHLAB）**

1. SMAPshot 能否批量导出？API 可接受的请求频率是多少？
2. 对实验室来说，"可靠度"和"对新地区的泛化"哪个更有价值？
3. 倾斜航拍的坐标是相机位置，评估时是否应该改用画面中心的地面位置？`/attributes` 里有方位角和俯仰角，理论上可以算出来。

---

## 11. 参考文献（已核对出处）

1. Vivanco Cepeda, Nayak, Shah. **GeoCLIP**: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization. NeurIPS 2023. [arXiv:2309.16020](https://arxiv.org/abs/2309.16020) · [code](https://github.com/VicenteVivan/geo-clip) · `papers/neurips-2023-1b57aadd.pdf`
2. Haas, Alberti, Skreta. Learning Generalized Zero-Shot Learners for Open-Domain Image Geolocalization (**StreetCLIP**). arXiv 2023. [arXiv:2302.00275](https://arxiv.org/abs/2302.00275) · `papers/arxiv-2302.00275.pdf`
3. Haas, Skreta, Alberti, Finn. **PIGEON**: Predicting Image Geolocations. CVPR 2024. [arXiv:2307.05845](https://arxiv.org/abs/2307.05845)
4. Astruc et al. **OpenStreetView-5M**: The Many Roads to Global Visual Geolocation. CVPR 2024. [arXiv:2404.18873](https://arxiv.org/abs/2404.18873) · [code](https://github.com/gastruc/osv5m)
5. Dufour, Picard, Kalogeiton, Landrieu. Around the World in 80 Timesteps: A Generative Approach to Global Visual Geolocation (**PLONK**). CVPR 2025. [arXiv:2412.06781](https://arxiv.org/abs/2412.06781) · [code](https://github.com/nicolas-dufour/plonk)
6. Shatwell, Dave, Swetha, Shah. **GT-Loc**: Unifying When and Where in Images Through a Joint Embedding Space. ICCV 2025 (oral). [arXiv:2507.10473](https://arxiv.org/abs/2507.10473) · [code](https://github.com/dshatwell23/gtloc)
7. Ghasemi, Ziashahabi, Avestimehr, Shahabi. **GeoToken**: Hierarchical Geolocalization of Images via Next Token Prediction. ICDM 2025. [arXiv:2511.01082](https://arxiv.org/abs/2511.01082)
8. Zhou et al. **Img2Loc**: Revisiting Image Geolocalization using Multi-modality Foundation Models and Image-based Retrieval-Augmented Generation. SIGIR 2024. [arXiv:2403.19584](https://arxiv.org/abs/2403.19584)
9. Jia, Park, Gao, Zhao, Li. **GeoRanker**: Distance-Aware Ranking for Worldwide Image Geolocalization. NeurIPS 2025. [arXiv:2505.13731](https://arxiv.org/abs/2505.13731)
10. Yildiz et al. **AmsterTime**: A Visual Place Recognition Benchmark Dataset for Severe Domain Shift. ICPR 2022. [PDF](https://jvgemert.github.io/pub/yildizICPR22AmsterTime.pdf)
11. Produit, Blanc, Composto, Ingensand, Fürhoff. Crowdsourcing the Georeferencing of Historical Pictures (**SMAPshot**). 2018. [doi:10.7275/awa4-4w48](https://doi.org/10.7275/awa4-4w48)
12. Produit, Ingensand. 3D Georeferencing of Historical Photos by Volunteers. AGILE 2018. [record](https://arodes.hes-so.ch/record/2380)
13. Maiwald, Bruschke, Schneider, Wacker, Niebling. Giving Historical Photographs a New Perspective (**4D Browser**). Remote Sensing 15(7):1879, 2023. [link](https://www.mdpi.com/2072-4292/15/7/1879)

**数据与平台**

- [SMAPshot 开发文档](https://smapshot.heig-vd.ch/development_documentation)
- [Natural Earth 10m 国界](https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip) · [州界](https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip)
- [4Dbrowser API](https://4dbrowser.urbanhistory4d.org/api)
- [DHLAB 学生指南](https://github.com/dh-epfl-students/EPFL-DHLAB-student-projects/blob/main/README-Student-HowTo.md)
