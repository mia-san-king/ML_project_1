# CS-433 Project 2 ：瑞士历史照片的细粒度图像定位（个人从零实现版）

- **【已知】**：已经测出来或已经核实过的事实。
- **【假设】**：待实验验证的预期，不能当结果写进报告。
- **【需确认】**：还没核实的信息。

队友的结果只作为**背景和参照**（§2）。本教程里的数据、划分、代码和实验全部由你自己从零完成。

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
| 算力 | 本机（Mac）写代码、做 EDA、小样本调试、作图、写报告；DHLAB（Quadro RTX 8000 × 最多 2 块）下载图像、提特征、训练、评估 |
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
| 伦理段落 200–400 词 | 建议写"识别出风险"的版本（§9.3） |
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
 本机 Mac                                    DHLAB GPU 服务器
 ─────────────────────────                  ──────────────────────────────────────
 S1 环境与仓库骨架
 S2 拉元数据 (API, ~360 次请求)  ──rsync──▶  data/meta/smapshot.parquet
 S3 EDA：看分布、选留出地区
 S4 清洗 + 划分 (定死, 进 Git)   ──git──▶    data/splits/main.csv
                                            S5 下载图像 (screen, 可断点续传)
                                            S6 CLIP 特征缓存 (GPU, 一次)
 S7 评估库 (metrics, bootstrap)  ──git──▶
                                            S8  基线 B0–B4
                                            S9  零训练适配 A1–A2 (先验、瑞士候选库)
                                            S10 微调 A3–A4 (瑞士负样本、软目标)
                                            S11 (可选) 可靠度模型
                                            S12 (可选) 留出城市/州的泛化评估
 S13 汇总表格、作图、写报告  ◀──rsync──      results/preds/*.parquet, results/*.json
```

**原则**：

- 代码只在 Mac 上写，`git push` 后在 DHLAB 上 `git pull`。
- 数据、特征、checkpoint 只在 DHLAB 上（元数据例外，两边都有）。
- 小结果文件（预测、指标 JSON）用 `rsync` 拉回 Mac 作图。
- 每一步都有**完成标准**。达到了才进入下一步。

路径约定，代码里统一用环境变量 `P2_DATA`：

| 机器 | `P2_DATA` |
| --- | --- |
| Mac | `~/Documents/ML/p2data`（不在仓库里） |
| DHLAB | `/rcp-scratch/students/yijliu/ml-project2/data` |

```text
$P2_DATA/
├── meta/        # 原始 API 页（json）+ smapshot.parquet
├── geo/         # Natural Earth 国界、州界
├── images/      # 只在 DHLAB；按 id//1000 分子目录
├── features/    # clip_vitl14.npy + ids.npy
├── ckpt/        # 微调后的权重
└── results/     # preds/*.parquet、metrics/*.json
```

---

### S1. 环境与仓库骨架【本机 + DHLAB，约半天】

**1a. 本机建环境**（Apple Silicon 上 PyTorch 自带 MPS，小样本调试够用）：

```bash
conda create -n p2 python=3.11 -y
conda activate p2
pip install torch torchvision geoclip pandas pyarrow pyproj shapely geopandas \
            scikit-learn matplotlib requests tqdm pyyaml
pip freeze > requirements-lock.txt   # 报告复现用，提交进 Git
```

**1b. DHLAB 环境补装**（`ml-p2` 已有 torch 和 geoclip）：

```bash
ssh dhlab
ml-p2                                   # 激活环境并进入项目目录
pip install pyarrow pyproj shapely geopandas scikit-learn requests pyyaml
mkdir -p $DHLAB_SCRATCH/ml-project2/data
echo 'export P2_DATA=/rcp-scratch/students/yijliu/ml-project2/data' >> ~/.bashrc
```

**1c. 在 DHLAB 上确认 GeoCLIP 真的能推理**，不只是能加载：

```bash
python - <<'PY'
from geoclip import GeoCLIP
import torch, urllib.request
m = GeoCLIP().to("cuda").eval()
print(m.image_encoder.mlp, m.gps_gallery.shape, m.logit_scale.exp().item())
urllib.request.urlretrieve("https://smapshot.heig-vd.ch/api/v1/data/collections/2/images/thumbnails/566.jpg", "/tmp/t.jpg")
print(m.predict("/tmp/t.jpg", top_k=3))   # 真值约在 Aproz (VS)
PY
```

如果 `predict` 报 `get_image_features` 相关的错，说明 `transformers` 版本太新、和 geoclip 不兼容。执行 `pip install "transformers<5"` 后重试。

**1d. 仓库骨架**（在 Mac 上建，然后提交）：

```text
ML_project_1/
├── README.md              # 安装、数据、复现每张表的命令
├── tutorial.md
├── requirements-lock.txt
├── configs/               # 每个实验一个 yaml
├── lib/
│   ├── paths.py           # P2_DATA 等路径
│   ├── geo.py             # haversine（numpy / torch）、LV95 转换、瑞士网格
│   ├── smapshot.py        # API 抓取与解析
│   ├── splits.py          # 留出地区 + 空间块划分
│   ├── features.py        # CLIP 特征缓存
│   ├── geoclip_ops.py     # 用缓存特征做 GeoCLIP 推理、先验、训练
│   ├── metrics.py         # 指标、bootstrap
│   └── reliability.py
├── scripts/               # 命令行入口（每步一个）+ 已有的 dhlab_*.sh
├── notebooks/             # 01_eda.ipynb、02_figures.ipynb
├── data/splits/           # 划分文件（小，进 Git）
└── report/
```

`.gitignore` 加上 `data/*` 和 `!data/splits/`，以及 `*.npy`、`*.pt`、`images/`。

```python
# lib/paths.py
import os
from pathlib import Path

DATA = Path(os.environ.get("P2_DATA", Path.home() / "Documents/ML/p2data")).expanduser()
META, GEO, IMAGES = DATA / "meta", DATA / "geo", DATA / "images"
FEATS, CKPT, RESULTS = DATA / "features", DATA / "ckpt", DATA / "results"
for p in (META, GEO, IMAGES, FEATS, CKPT, RESULTS):
    p.mkdir(parents=True, exist_ok=True)
```

**完成标准**：两边都能 `import geoclip`；DHLAB 上 `predict` 输出 3 个坐标；空仓库骨架已推送。

---

### S2. 拉取 SMAPshot 元数据【本机，约 30 分钟】

**已核实的 API 信息**【已知】：

| 项 | 值 |
| --- | --- |
| 列表接口 | `GET https://smapshot.heig-vd.ch/api/v1/images?limit=1000&offset=N&state[]=validated` |
| 数量 | validated 共 355,281 张（总 357,113）。`limit=1000` 可用，所以约 356 次请求 |
| 过滤参数 | 必须用数组写法，例如 `state[]=validated`、`view_type[]=terrestrial`（写成 `state=validated` 会返回 400） |
| 返回字段 | `id, original_id, collection_id, owner_id, title, view_type, date_shot_min, date_shot_max, latitude, longitude, height, width, media.image_url` |
| `view_type` | terrestrial 约 101k；nadir（垂直航拍）约 40.5k；其余约 214k 为空（多半是倾斜航拍） |
| 单张详细 | `GET /images/{id}/attributes`：相机位姿（海拔、方位角、俯仰、焦距）、`country_iso_a2`、许可证、摄影师 |
| 缩略图 | 大多约 500×366 px、约 65 KB；部分来自外部 IIIF，宽 200 px |
| 收藏 | 63 个。大多是航拍（Swissair、Comet、Friedli 等），还有 "Views of Zürich"、"Fonds Perrochet (Tessin)"，以及**瑞士以外**的 imagineRio（里约）、Vorarlberg（奥地利） |

**两个必须记住的数据事实**：

1. 坐标是**相机位置**。倾斜航拍的相机在飞机上，可能离画面主体好几公里，这是一部分无法消除的误差。所以**一定要按 `view_type` 分开报告结果**。
2. `title` 里常有地名（例如 "Aproz"）。它**不能**作为模型输入，否则就是标签泄漏。

```python
# lib/smapshot.py
import json, time
import pandas as pd
import requests
from lib.paths import META

BASE = "https://smapshot.heig-vd.ch/api/v1"
UA = {"User-Agent": "EPFL CS-433 student project (contact: <your-epfl-email>)"}

def fetch_pages(page=1000, delay=1.0, max_pages=None):
    """断点续传：已存在的页直接跳过。"""
    s = requests.Session(); s.headers.update(UA)
    offset, n = 0, 0
    while max_pages is None or n < max_pages:
        f = META / f"page_{offset:07d}.json"
        if not f.exists():
            r = s.get(f"{BASE}/images", params={"limit": page, "offset": offset,
                                                "state[]": "validated"}, timeout=120)
            r.raise_for_status()
            f.write_text(r.text)
            time.sleep(delay)
        rows = json.loads(f.read_text())["rows"]
        n += 1
        if len(rows) < page:
            break
        offset += page

def pages_to_parquet():
    rows = [r for f in sorted(META.glob("page_*.json")) for r in json.loads(f.read_text())["rows"]]
    df = pd.json_normalize(rows).rename(columns={"media.image_url": "image_url"})
    keep = ["id", "original_id", "collection_id", "owner_id", "title", "view_type",
            "date_shot_min", "date_shot_max", "latitude", "longitude", "height", "width", "image_url"]
    df = df[keep].drop_duplicates("id").reset_index(drop=True)
    df.to_parquet(META / "smapshot.parquet")
    return df
```

```bash
# Mac，在仓库根目录
python -c "from lib.smapshot import fetch_pages, pages_to_parquet; fetch_pages(); print(pages_to_parquet().shape)"
rsync -avP ~/Documents/ML/p2data/meta/smapshot.parquet dhlab:/rcp-scratch/students/yijliu/ml-project2/data/meta/
```

**完成标准**：`smapshot.parquet` 约 35.5 万行；`id` 唯一；经纬度非空的比例已记录。

---

### S3. 探索性分析（EDA）与选留出地区【本机，约半天】

在 `notebooks/01_eda.ipynb` 里做，结果直接决定 S4 的划分。

1. **基本统计**：按 `collection_id`、`view_type` 计数；`date_shot_min` 的覆盖率和年代直方图；缺坐标的数量。
2. **地图**：把所有点画在 LV95 坐标上，按 `view_type` 着色。
3. **国界过滤**：下载 Natural Earth 国界和州界（已核实可下载，分别约 5 MB 和 15 MB）。

```python
# lib/geo.py
import numpy as np
import torch
import geopandas as gpd
from pyproj import Transformer
from lib.paths import GEO

R_EARTH = 6371.0088
TO_LV95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
TO_WGS = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
NE = "https://naciscdn.org/naturalearth/10m/cultural/"

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

def haversine_torch(a, b):
    """a: (N,2)、b: (M,2)，[lat, lon] 度 → (N,M) km"""
    a, b = torch.deg2rad(a), torch.deg2rad(b)
    dlat = a[:, None, 0] - b[None, :, 0]
    dlon = a[:, None, 1] - b[None, :, 1]
    h = torch.sin(dlat / 2) ** 2 + torch.cos(a[:, None, 0]) * torch.cos(b[None, :, 0]) * torch.sin(dlon / 2) ** 2
    return 2 * R_EARTH * torch.asin(torch.sqrt(h.clamp(0, 1)))

def load_boundaries():
    """返回 LV95 坐标下的 (瑞士国界, 各州 GeoDataFrame)。"""
    f0, f1 = GEO / "ne_admin0.gpkg", GEO / "ne_admin1_ch.gpkg"
    if not f0.exists():
        a0 = gpd.read_file(NE + "ne_10m_admin_0_countries.zip")
        a0[a0.ADM0_A3 == "CHE"].to_file(f0)
        a1 = gpd.read_file(NE + "ne_10m_admin_1_states_provinces.zip")
        a1[a1.adm0_a3 == "CHE"][["name", "geometry"]].to_file(f1)
    ch = gpd.read_file(f0).to_crs(2056).geometry.union_all()
    cantons = gpd.read_file(f1).to_crs(2056)
    return ch, cantons

def to_lv95(lat, lon):
    e, n = TO_LV95.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(e), np.asarray(n)
```

4. **选留出城市和留出州**。下面的代码统计每个候选城市 10 km 内、每个州内的图片数：

```python
import pandas as pd, numpy as np
from lib.geo import haversine_km, load_boundaries, to_lv95
import geopandas as gpd

df = pd.read_parquet(META / "smapshot.parquet").dropna(subset=["latitude", "longitude"])
CITIES = {"Basel": (47.5596, 7.5886), "Bern": (46.9480, 7.4474), "Lausanne": (46.5197, 6.6323),
          "Luzern": (47.0502, 8.3093), "Zürich": (47.3769, 8.5417), "Genève": (46.2044, 6.1432),
          "St. Gallen": (47.4245, 9.3767)}
for c, (la, lo) in CITIES.items():
    d = haversine_km(df.latitude.values, df.longitude.values, la, lo)
    print(f"{c:10s} ≤10km: {(d <= 10).sum():6d}   其中 terrestrial: {((d <= 10) & (df.view_type == 'terrestrial')).sum()}")

ch, cantons = load_boundaries()
e, n = to_lv95(df.latitude, df.longitude)
pts = gpd.GeoDataFrame(df[["id", "view_type"]], geometry=gpd.points_from_xy(e, n), crs=2056)
print(gpd.sjoin(pts, cantons, predicate="within").groupby("name").size().sort_values())
```

**选择标准**：

- 每个留出集至少 2,000 张，并且 terrestrial 和航拍两类都要有。
- 留出州的语言、建筑、地形最好和训练区不同。**Ticino** 是阿尔卑斯以南、讲意大利语的州，还有单独的收藏 "Fonds Perrochet (Tessin)"，是天然的"没见过的地区"。
- 留出城市不要选数据占比太大的城市，否则训练集损失太多。如果苏黎世占了很大比例（有专门的 "Views of Zürich" 收藏），就选 Basel 或 Bern。

**为什么不用 4Dbrowser 做"没见过的城市"**【已知】：它的覆盖接口返回 342,688 个点，但几乎都在捷克、斯洛伐克、德国和荷兰。瑞士外接框里只有 685 个点，而且看起来是现代全景。可以把它作为**跨国家的压力测试**（可选，见 S12d），但不能作为瑞士城市的留出集。

**完成标准**：notebook 里有地图、各收藏和 `view_type` 的计数表，以及你选定的留出城市、留出州和理由，写进 README。

---

### S4. 清洗与划分（下载前就定死）【本机，约半天】

**4a. 清洗规则**（每一步记录删掉多少张，报告里要用）：

1. 去掉缺坐标的图。
2. 只保留瑞士国界向外 5 km 缓冲区内的点。航拍相机可能在边境湖面上空，所以要留缓冲；这一步也会去掉 imagineRio 和 Vorarlberg。
3. **主基准去掉 `view_type == "nadir"`**（垂直航拍更像正射影像，是另一种任务），但把它们单独存一个附加测试集。
4. 去重放到 S6 之后做：CLIP 余弦相似度 > 0.97 且距离 < 50 m 视为重复，只保留一张。

**4b. 划分方案**：三个测试集 + 训练 + 验证。

| 集合 | 定义 | 用途 |
| --- | --- | --- |
| `test_region` | 留出州（例如 Ticino）内的全部图 | RQ3：完全没见过的地区 |
| `test_city` | 留出城市中心 10 km 内的全部图 | RQ3：没见过的城市 |
| `buffer` | 留出州向外 10 km 的环带、留出城市 10–20 km 的环带 | **直接丢弃**，避免同一座山或同一个湖跨集合泄漏 |
| 其余部分 | 按 10 km × 10 km LV95 块随机分配：train 70% / val 10% / `test_in` 20% | 主实验 |
| `random_*`（对照） | 在"其余部分"里做一次纯随机划分 | 量化空间泄漏 |

```python
# lib/splits.py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from lib.geo import load_boundaries, to_lv95

def make_splits(df, region="Ticino", city_center=(47.5596, 7.5886), city_r_km=10,
                buffer_km=10, block_km=10, frac=(0.7, 0.1, 0.2), seed=0):
    ch, cantons = load_boundaries()
    e, n = to_lv95(df.latitude.values, df.longitude.values)
    pts = gpd.GeoSeries(gpd.points_from_xy(e, n), crs=2056)

    keep = pts.within(ch.buffer(5_000)).values
    reg = cantons.loc[cantons.name == region].geometry.union_all()
    in_reg = pts.within(reg).values
    near_reg = pts.within(reg.buffer(buffer_km * 1000)).values & ~in_reg
    ce, cn = to_lv95(np.array([city_center[0]]), np.array([city_center[1]]))
    d_city = np.hypot(e - ce[0], n - cn[0]) / 1000        # LV95 中近似欧氏距离，km
    in_city = d_city <= city_r_km
    near_city = (d_city > city_r_km) & (d_city <= city_r_km + buffer_km)

    split = np.full(len(df), "", dtype=object)
    split[~keep] = "outside"
    split[keep & (near_reg | near_city)] = "buffer"
    split[keep & in_city] = "test_city"
    split[keep & in_reg] = "test_region"

    rest = keep & (split == "")
    bx = np.floor(e / (block_km * 1000)).astype(np.int64)
    by = np.floor(n / (block_km * 1000)).astype(np.int64)
    block = bx * 100_000 + by
    blocks = np.unique(block[rest])
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    n_tr, n_va = int(frac[0] * len(blocks)), int(frac[1] * len(blocks))
    lookup = {b: s for s, bs in (("train", blocks[:n_tr]), ("val", blocks[n_tr:n_tr + n_va]),
                                 ("test_in", blocks[n_tr + n_va:])) for b in bs}
    split[rest] = [lookup[b] for b in block[rest]]

    rnd = np.full(len(df), "", dtype=object)
    r = rng.random(rest.sum())
    rnd[rest] = np.where(r < frac[0], "train", np.where(r < frac[0] + frac[1], "val", "test_in"))

    out = pd.DataFrame({"id": df.id.values, "split": split, "random_split": rnd, "block": block})
    out.loc[df.view_type.values == "nadir", "split"] = "nadir_" + out.loc[df.view_type.values == "nadir", "split"]
    return out
```

```bash
python -c "
import pandas as pd
from lib.paths import META
from lib.splits import make_splits
df = pd.read_parquet(META/'smapshot.parquet').dropna(subset=['latitude','longitude'])
s = make_splits(df); s.to_csv('data/splits/main.csv.gz', index=False)
print(s.split.value_counts())"
git add data/splits/main.csv.gz && git commit -m "Fix data splits (Ticino + Basel held out, 10 km blocks)" && git push
```

**注意**：

- 按块分配，所以各集合的实际张数不会正好是 70/10/20，在报告里写实际张数。
- **从此以后不准再改划分**。超参数只在 `val` 上选，三个测试集只在最终评估时跑一次。
- 块大小的敏感性检查（5 km、20 km）放到最后，作为附录。

**完成标准**：`main.csv.gz` 已进 Git；一张"划分地图"（每个集合一种颜色），这张图之后直接放进报告。

---

### S5. 下载图像【DHLAB，`screen` 里跑 1–4 天】

**规模估算**：35 万张 × 约 65 KB ≈ 23 GB。scratch 空间够用，瓶颈是请求延时：

| 延时 | 全量耗时 |
| --- | --- |
| 1 s | 约 4 天 |
| 0.3 s | 约 1.5 天 |

**策略**：按优先级下载，可以随时中断、续传，下载期间就能开始后面的步骤。

| 优先级 | 内容 |
| --- | --- |
| 1 | 三个测试集和 val（全量） |
| 2 | train 的随机 50k |
| 3 | train 剩下的部分 |

先用 0.5 s 延时跑。如果对方服务器响应变慢或返回 429，就加大延时。可以在 SMAPshot 开发文档里找联系人，问一下能否批量导出。

```python
# scripts/download_images.py
import io, sys, time
import pandas as pd, requests
from PIL import Image
from tqdm import tqdm
from lib.paths import META, IMAGES

def img_path(i):
    return IMAGES / f"{i // 1000:04d}" / f"{i}.jpg"

def main(delay=0.5, max_side=512):
    meta = pd.read_parquet(META / "smapshot.parquet")[["id", "image_url"]]
    sp = pd.read_csv("data/splits/main.csv.gz")
    df = meta.merge(sp, on="id")
    df = df[~df.split.isin(["outside", "buffer"])]
    pri = {"test_region": 0, "test_city": 0, "test_in": 0, "val": 0, "train": 1}
    df["pri"] = df.split.map(pri).fillna(2)
    df["r"] = pd.Series(range(len(df))).sample(frac=1, random_state=0).values
    df.loc[(df.pri == 1) & (df.r >= 50_000), "pri"] = 2
    df = df.sort_values(["pri", "r"])

    s = requests.Session()
    s.headers["User-Agent"] = "EPFL CS-433 student project (contact: <your-epfl-email>)"
    fail = open(IMAGES / "failed.txt", "a")
    for i, url in tqdm(zip(df.id, df.image_url), total=len(df)):
        p = img_path(i)
        if p.exists():
            continue
        try:
            r = s.get(url, timeout=30)
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            im.thumbnail((max_side, max_side))
            p.parent.mkdir(exist_ok=True)
            im.save(p, quality=90)
        except Exception as e:
            fail.write(f"{i}\t{url}\t{e!r}\n"); fail.flush()
        time.sleep(delay)

if __name__ == "__main__":
    main(delay=float(sys.argv[1]) if len(sys.argv) > 1 else 0.5)
```

```bash
ssh dhlab
ml-p2 && git pull
screen -S dl                                   # 新开一个 screen
python -m scripts.download_images 0.5 2>&1 | tee $P2_DATA/download.log
# 按 Ctrl-A 再按 D 离开 screen；之后用 screen -r dl 回来看
find $P2_DATA/images -name '*.jpg' | wc -l     # 随时查看进度
du -sh $P2_DATA/images
```

**IIIF 小图**：如果某些 `image_url` 里有 `/full/200,/`，可以改成 `/full/!512,512/` 请求更大的图（IIIF 标准语法）。先手动试一张，确认服务器支持再改。

**完成标准**：优先级 1 全部下完；`failed.txt` 失败率 < 1%（失败的图在 split 文件里标记掉）。

---

### S6. 缓存 CLIP 特征【DHLAB GPU，35 万张约 1–3 小时】

CLIP 冻结，所以每张图只算一次 768 维特征（进入 `mlp` 之前），之后所有实验都只用这个 `.npy`。

| 项 | 估算 |
| --- | --- |
| 存储 | 35 万 × 768 × fp16 ≈ 0.55 GB |
| 显存 | batch 256 在 48 GB 卡上没问题 |

```python
# lib/features.py
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from geoclip import GeoCLIP
from lib.paths import FEATS
from scripts.download_images import img_path

class Imgs(Dataset):
    def __init__(self, ids): self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, k): return Image.open(img_path(self.ids[k])).convert("RGB")

@torch.no_grad()
def cache_features(ids, out_name="clip_vitl14", bs=256, workers=8, device="cuda"):
    model = GeoCLIP().to(device).eval()
    enc = model.image_encoder
    proc = lambda ims: enc.image_processor(images=ims, return_tensors="pt")["pixel_values"]
    dl = DataLoader(Imgs(ids), batch_size=bs, num_workers=workers, collate_fn=proc)
    out, first = [], True
    for px in dl:
        px = px.to(device)
        pooled = enc.CLIP.vision_model(pixel_values=px).pooler_output
        f = enc.CLIP.visual_projection(pooled)                      # (B, 768)
        if first:   # 和 GeoCLIP 自己的前向结果对一次，确保特征没取错
            assert torch.allclose(enc.mlp(f), enc(px), atol=1e-3), "特征和 GeoCLIP 前向不一致"
            first = False
        out.append(f.half().cpu())
    np.save(FEATS / f"{out_name}.npy", torch.cat(out).numpy())
    np.save(FEATS / f"{out_name}_ids.npy", np.asarray(ids))
```

```bash
nvidia-smi                                     # 先看哪块卡空着（不要用 GPU 3）
screen -S feat
CUDA_VISIBLE_DEVICES=0 python -c "
import numpy as np, pandas as pd
from scripts.download_images import img_path
from lib.features import cache_features
ids = [i for i in pd.read_csv('data/splits/main.csv.gz').id if img_path(i).exists()]
print(len(ids)); cache_features(ids)"
```

**下载还没完成时**：先对已下载的图缓存一版，后续再对新图增量缓存。每次输出存成带版本号的文件，用 `_ids.npy` 对齐。

**去重**（做完特征后）：对每张图在 50 m 内的邻居里找余弦相似度 > 0.97 的，只保留 id 最小的一张。在 split 文件里加一列 `dup`，并记录删了多少张。

**完成标准**：

- `assert` 通过。
- 随便挑两张同一地点的图，余弦相似度明显高于两张随机图。
- 特征里没有 NaN。

---

### S7. 评估库【本机写，两边共用】

所有方法都输出同一种格式：`results/preds/<method>.parquet`，列为 `id, split, lat_pred, lon_pred`，可靠度方法再加 `conf_*` 列。评估只用这一份代码。

```python
# lib/metrics.py
import numpy as np, pandas as pd
from lib.geo import haversine_km

THRESHOLDS_KM = (1, 5, 10, 25, 50, 200, 750, 2500)

def summarize(err_km):
    e = np.asarray(err_km)
    out = {"n": len(e), "median_km": float(np.median(e)), "mean_km": float(e.mean()),
           "p90_km": float(np.percentile(e, 90))}
    out |= {f"acc@{t}km": float((e <= t).mean() * 100) for t in THRESHOLDS_KM}
    return out

def bootstrap_ci(err_km, blocks, stat=np.median, n=1000, seed=0):
    """按空间块重采样（同一个块的图一起抽），比逐张重采样更诚实。"""
    rng = np.random.default_rng(seed)
    e, b = np.asarray(err_km), np.asarray(blocks)
    ub, inv = np.unique(b, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(ub))]
    vals = [stat(e[np.concatenate([groups[k] for k in rng.integers(0, len(ub), len(ub))])])
            for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])

def evaluate(pred: pd.DataFrame, truth: pd.DataFrame):
    m = pred.merge(truth[["id", "latitude", "longitude", "view_type", "block"]], on="id")
    m["err_km"] = haversine_km(m.latitude, m.longitude, m.lat_pred, m.lon_pred)
    rows = []
    for split, g in m.groupby("split"):
        rows.append({"split": split, "view_type": "all", **summarize(g.err_km),
                     "median_ci": bootstrap_ci(g.err_km, g.block).tolist()})
        for vt, h in g.groupby(g.view_type.fillna("oblique/none")):
            rows.append({"split": split, "view_type": vt, **summarize(h.err_km)})
    return pd.DataFrame(rows), m
```

**自测**：`haversine_km(46.5197, 6.6323, 47.3769, 8.5417)` 应该约等于 173.5 km（洛桑到苏黎世）。

**完成标准**：在一个手写的小例子上，`summarize` 的输出和手算结果一致。

---

### S8. 基线【DHLAB（kNN、GeoCLIP 用 GPU），约 1 天】

| 编号 | 基线 | 做法 | 能回答什么 |
| --- | --- | --- | --- |
| **B0** | 常数预测 | 永远预测 train 坐标的几何中位点（或者 LV95 坐标的中位数） | "什么都不看"能得多少分 |
| **B1** | GeoCLIP 零样本 | 全球 10 万点候选库 | 起点 |
| **B2** | kNN 检索 | CLIP 特征在 train 里找 top-1 近邻，用它的坐标；变体：top-k 的加权中位点 | 同地点照片多的数据上，检索可能非常强 |
| **B3** | GeoCLIP + 瑞士网格候选库，不训练 | 见 S9 的 A2 | 候选库单独的作用 |
| B4（可选） | PLONK-YFCC 零样本 | 官方代码和权重 | 和"最强的现成模型"比较 |

**用缓存特征跑 GeoCLIP**（B1、B3、A1–A4 都用这一套）。同时保存 top-k 和 logsumexp，供 S11 用：

```python
# lib/geoclip_ops.py
import numpy as np, torch, torch.nn.functional as F

@torch.no_grad()
def embed_gallery(model, gallery, bs=20_000):
    g = torch.as_tensor(gallery, dtype=torch.float32)
    return torch.cat([F.normalize(model.location_encoder(g[i:i + bs].to(model.logit_scale.device)), dim=1)
                      for i in range(0, len(g), bs)])

@torch.no_grad()
def predict_feats(model, feats, gallery, penalty=None, lam=0.0, k=20, chunk=1024):
    """feats: (N,768) 缓存的 CLIP 特征；gallery: (M,2)；penalty: (M,) 区域先验 U(g)"""
    dev = model.logit_scale.device
    G = embed_gallery(model, gallery)
    pen = None if penalty is None else lam * torch.as_tensor(penalty, device=dev, dtype=torch.float32)
    scale = model.logit_scale.exp()
    top_i, top_l, lse = [], [], []
    for i in range(0, len(feats), chunk):
        f = torch.as_tensor(feats[i:i + chunk], device=dev).float()
        logits = scale * F.normalize(model.image_encoder.mlp(f), dim=1) @ G.T
        if pen is not None:
            logits = logits - pen
        l, idx = logits.topk(k, dim=1)
        top_i.append(idx.cpu()); top_l.append(l.cpu()); lse.append(torch.logsumexp(logits, 1).cpu())
    top_i, top_l, lse = torch.cat(top_i).numpy(), torch.cat(top_l).numpy(), torch.cat(lse).numpy()
    return np.asarray(gallery)[top_i[:, 0]], top_i, top_l, lse
```

**kNN**（同样分块）：

```python
@torch.no_grad()
def knn_predict(f_query, f_train, gps_train, k=1, chunk=4096, device="cuda"):
    T = F.normalize(torch.as_tensor(f_train, device=device).float(), dim=1)
    sims, idxs = [], []
    for i in range(0, len(f_query), chunk):
        q = F.normalize(torch.as_tensor(f_query[i:i + chunk], device=device).float(), dim=1)
        s, j = (q @ T.T).topk(k, dim=1)
        sims.append(s.cpu()); idxs.append(j.cpu())
    s, j = torch.cat(sims).numpy(), torch.cat(idxs).numpy()
    return gps_train[j[:, 0]], s, j   # s[:,0] 也是 S11 的一个可靠度特征
```

每个基线跑完，就把预测存成 `results/preds/B*.parquet`，再用 `evaluate` 出表。这一步只看 `val`。

**完成标准**：

- B1 在 `val` 上的中位误差在 100–200 km 量级（和队友的数字同一量级）。
- B0 和 B2 的数字已经有了。
- 已经知道"kNN 有多强"。这决定了后面的方法要超过的门槛。

---

### S9. 零训练适配 A1–A2【DHLAB，约 1 天】

**A1：区域先验**（复现 Week 1 的方法，但在你的划分上）：

$$
s(I,g)=s_\text{GeoCLIP}(I,g)-\lambda\,U(g),\qquad U(g)=\Big[\max\big(0,\tfrac{d(g,c)-R}{b}\big)\Big]^2
$$

```python
CH_CENTER = (46.80, 8.23)
def regional_penalty(gallery, R=179.0, b=50.0):
    d = haversine_km(gallery[:, 0], gallery[:, 1], *CH_CENTER)
    return np.clip((d - R) / b, 0, None) ** 2
```

λ 和 b 在 `val` 上做网格搜索，例如 λ ∈ {1, 3, 10, 30}、b ∈ {25, 50, 100}。

**A2：瑞士网格候选库**：在 LV95 坐标下，国界向外 5 km 的范围内铺规则网格。

- 1 km 间距约 5 万个点；0.5 km 间距约 20 万个点。
- 这个网格也覆盖 Ticino 和留出城市，**但网格只是地理坐标，不含任何图像标签**，所以不算泄漏。

```python
import shapely
def swiss_grid(step_m=1000, buffer_m=5000):
    ch, _ = load_boundaries()
    poly = ch.buffer(buffer_m)
    x0, y0, x1, y1 = poly.bounds
    xs, ys = np.meshgrid(np.arange(x0, x1, step_m), np.arange(y0, y1, step_m))
    xs, ys = xs.ravel(), ys.ravel()
    shapely.prepare(poly)
    m = shapely.contains_xy(poly, xs, ys)
    lon, lat = TO_WGS.transform(xs[m], ys[m])
    return np.stack([lat, lon], axis=1).astype(np.float32)
```

**候选库的变体**（做成一个小消融）：

| 变体 | 内容 |
| --- | --- |
| G-global | GeoCLIP 原始的 10 万点 |
| G-grid1km / G-grid500m | 瑞士网格 |
| G-train | train 集里出现过的坐标 |
| G-grid+train | 两者合并 |

**G-train 的问题**：它在 `test_region` 里一个点都没有。这正是 RQ3 要展示的。

**诊断（RQ1′）**：

- oracle 误差 = 候选库里离真值最近的点的距离。用 `sklearn.neighbors.BallTree(np.radians(gallery), metric="haversine")` 计算。
- 如果 oracle 中位数远小于实际中位误差，说明瓶颈在排序，而不是候选点不够。

**完成标准**：在 `val` 上有一张小表：B1 → +A1 → +A2（各个候选库）→ oracle。

---

### S10. 微调 A3–A4【DHLAB GPU，每次训练几分钟到半小时】

所有训练都只读缓存特征：train 特征约 0.4 GB，直接整块放进 GPU 显存，不需要 DataLoader。

**要比较的设定**：

| 编号 | 训练哪些部分 | 负样本 | 损失 |
| --- | --- | --- | --- |
| A3a | 只训图像 MLP | batch 内 + 全球 GPS 队列（原版做法） | InfoNCE |
| A3b | 只训图像 MLP | batch 内 + **瑞士网格随机点** | InfoNCE |
| A3c | MLP + 位置编码器 | 瑞士网格 | InfoNCE |
| **A4** | MLP + 位置编码器 | 瑞士网格 | **haversine 平滑软目标** |

**为什么要软目标**：SMAPshot 里很多照片拍的是同一地点。硬 InfoNCE 会把"同一个 batch 里另一张同地点照片的 GPS"当成负样本，这是错的。软目标按距离给正样本权重（PIGEON 的思路）：

$$
q_{n,j}=\frac{\exp\big(-\text{Hav}(g_n,x_j)/\tau\big)}{\sum_{j'}\exp\big(-\text{Hav}(g_n,x_{j'})/\tau\big)},\qquad
\mathcal L=-\frac1B\sum_n\sum_j q_{n,j}\log\operatorname{softmax}_j(\ell_{n,j})
$$

其中 $x_j$ 是 batch 内的（加噪）GPS 加上负样本队列，$\tau$ 例如取 {1, 5, 25} km。$\tau\to0$ 时退化为硬 InfoNCE。

```python
# lib/geoclip_ops.py（续）
from lib.geo import haversine_torch

def jitter(gps, sigma_km):
    lat = gps[:, 0:1]
    d = torch.randn_like(gps) * sigma_km
    return gps + torch.cat([d[:, :1] / 111.2, d[:, 1:] / (111.2 * torch.cos(torch.deg2rad(lat)))], 1)

def train(model, f_tr, g_tr, neg_pool, *, train_loc=True, soft_tau_km=None, epochs=10, bs=512,
          n_neg=4096, sigma_km=1.0, lr_mlp=1e-4, lr_loc=3e-5, val_fn=None, seed=0):
    """f_tr: (N,768) GPU fp16；g_tr: (N,2) GPU；neg_pool: (M,2) GPU（例如瑞士网格）"""
    torch.manual_seed(seed)
    mlp, loc = model.image_encoder.mlp, model.location_encoder
    groups = [{"params": mlp.parameters(), "lr": lr_mlp}, {"params": [model.logit_scale], "lr": lr_mlp}]
    if train_loc:
        groups.append({"params": loc.parameters(), "lr": lr_loc})
    else:
        loc.requires_grad_(False)
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(f_tr), device=f_tr.device)
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            g = g_tr[idx]
            neg = neg_pool[torch.randint(len(neg_pool), (n_neg,), device=g.device)]
            locs = torch.cat([jitter(g, sigma_km), neg])
            img = F.normalize(mlp(f_tr[idx].float()), dim=1)
            le = F.normalize(loc(locs), dim=1)
            logits = model.logit_scale.exp() * img @ le.T                # (B, B+n_neg)
            if soft_tau_km is None:
                loss = F.cross_entropy(logits, torch.arange(len(idx), device=g.device))
            else:
                q = torch.softmax(-haversine_torch(g, locs) / soft_tau_km, dim=1)
                loss = -(q * F.log_softmax(logits, dim=1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, np.log(100))
        if val_fn is not None:              # 例如：val 子集 5k 张、瑞士网格上的中位误差
            model.eval(); v = val_fn(model)
            print(f"epoch {ep}  loss {loss.item():.3f}  val median {v:.2f} km")
            if v < best:
                best = v
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best
```

**运行方式**：每个设定一个 yaml（`configs/A4_tau5.yaml`），一个入口脚本 `scripts/train.py --config ...`。训练结束保存 `ckpt/<name>.pt` 和 `results/preds/<name>.parquet`（在 `val` 和三个测试集上都用 G-grid500m 预测）。

```bash
screen -S train
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --config configs/A4_tau5.yaml 2>&1 | tee logs/A4_tau5.log
```

**超参数**（只在 `val` 上选，预算控制在 20–30 次运行以内）：

| 超参数 | 搜索范围 |
| --- | --- |
| τ | {1, 5, 25} km |
| σ | {0.5, 2} km |
| 位置编码器学习率 | {1e-5, 3e-5, 1e-4} |
| epochs | 10–20，用早停 |
| 种子 | 最终配置跑 3 个种子，报告均值 ± 标准差 |

**常见坑**：

- **位置编码器训崩**：val 误差突然变大，说明学习率太大。先冻结位置编码器训 2 个 epoch，再一起训。
- **只学会了"预测训练集的热门地点"**：在 `test_region` 上会很明显。这本身就是 RQ3 的一个发现，要记录下来。
- `logit_scale` 一直变大会让概率过度自信，所以代码里 clamp 到 100。这会影响 S11 的可靠度。

**A5（可选，时间多再做）**：先在 0.5 km 网格上取 top-50，再在 top-50 附近用 100 m 的细网格重新打分，"由粗到细"。或者在 top-k 候选里取 softmax 加权的 LV95 坐标平均，看能否降低 ≤1 km 尺度上的误差。

**完成标准**：消融表 B1 → A2 → A3a → A3b → A3c → A4，每一行在 `val` 上都有数字。最终配置在三个测试集上各跑一次。

---

### S11.（可选）可靠度：每个预测给一个可信的置信度【CPU 即可，约 2–3 天】

**目标**：对每张图输出 $\hat p_T=P(e\le T)$，$T\in\{1,5,25\}$ km。要求排序好（AUROC 高）并且校准（ECE 低）。

**特征**（全部来自模型输出，不用标签）：

| 特征 | 计算 |
| --- | --- |
| `p1` | $\exp(\ell_1-\text{logsumexp})$，top-1 在整个候选库上的概率 |
| `margin` | $\ell_1-\ell_2$ |
| `entropy_k` | top-k 归一化后的熵 |
| `spread_k` | top-k 候选点到 top-1 点的距离，按概率加权平均（km）。候选点很分散 = 模型拿不准 |
| `knn_sim` | 和 train 里最近邻的余弦相似度（"类似的照片见过吗"） |
| `knn_agree` | kNN 预测点和 GeoCLIP 预测点的距离（两种方法一致时更可信） |
| `view_type` | one-hot 编码 |
| `img_wh` | 图像宽高比（全景照片） |

**训练数据**：

- 不能用 train 集上的预测，因为它们过拟合，会过度自信。
- 简单做法（推荐）：在 `val` 上训练可靠度模型，用 val 的空间块做 5 折 `GroupKFold` 选超参数；然后在三个测试集上评估。
- 进阶做法：5 折空间交叉拟合。把 train 按块分成 5 折，每次用 4 折微调、对剩下 1 折预测，得到整个 train 的"诚实"预测。缓存特征时训练很快，所以可行，但要花 5 倍训练时间。

**模型**：

- `LogisticRegression`（基线）。
- `HistGradientBoostingClassifier`（主模型）。
- 再用 `CalibratedClassifierCV(method="isotonic")` 校准。

**对照基线**：直接用 `p1` 作为置信度，用 isotonic 映射到概率。

```python
# lib/reliability.py
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

def ece(p, y, n_bins=15):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
               for b in range(n_bins) if (bins == b).any())

def risk_coverage(conf, err_km, t=25):
    order = np.argsort(-conf)
    e = np.asarray(err_km)[order]
    cov = np.arange(1, len(e) + 1) / len(e)
    acc = np.cumsum(e <= t) / np.arange(1, len(e) + 1)
    return cov, acc

def fit_reliability(X_val, err_val, T=25, groups=None):
    y = (err_val <= T).astype(int)
    clf = CalibratedClassifierCV(HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05),
                                 method="isotonic", cv=5)
    clf.fit(X_val, y)
    return clf

def report(clf, X, err, T=25):
    y, p = (err <= T).astype(int), clf.predict_proba(X)[:, 1]
    return {"auroc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "ece": ece(p, y)}
```

**要出的图和表**：

- 风险–覆盖曲线：横轴覆盖率，纵轴 ≤25 km 命中率；每个测试集一条线；对照"p1 基线"。
- 校准图（reliability diagram）：`test_in` 和 `test_region` 各一张。
- 表：AUROC / Brier / ECE × {p1 基线, 可靠度模型} × {T=1, 5, 25 km} × 三个测试集。

**完成标准**：至少在 `test_in` 上，可靠度模型的 AUROC 高于 p1 基线；有一句话说清"在没见过的地区，可靠度是否仍然校准"。

---

### S12.（可选）泛化到没见过的城市和州【DHLAB，约 1 天】

S4 已经把 `test_city` 和 `test_region` 留出来了，所以这一步只是**评估和分析**，不需要额外训练。

**12a. 主对比表**：每个方法三列（`test_in` / `test_city` / `test_region`），指标为中位误差、≤5 km、≤25 km。

**预期现象**【假设】：

- kNN 和 G-train 在 `test_region` 上崩溃，因为它们只能输出训练集里出现过的位置。由于有 10 km 的缓冲带，误差至少是 10 km 左右。
- GeoCLIP + 瑞士网格（A2–A4）理论上能预测到任何位置，所以增益应该部分保留。
- 两者的差距就是"GPS 编码器方法能泛化到新地区"的证据，可以作为报告的一个核心发现。

**12b. 泛化差距**：

$$
\Delta = \text{metric}(T_\text{region}) - \text{metric}(T_\text{in})
$$

- 对比零样本 B1 和微调后的 A4 的 $\Delta$。如果 A4 的 $\Delta$ 很大，说明微调"记住了训练区"。
- 还可以画"误差 vs 到最近训练图的距离"的散点或分箱图，展示误差随距离怎么增长。

**12c. 可靠度的迁移**：S11 的模型在 `test_region` 上的 ECE 是否明显变差？可以给"新地区"的预测加一个 `knn_sim` 很低时降置信的规则，看能否改善。

**12d. 跨国家压力测试**（可选，额外加分）：

- 在 4Dbrowser 的覆盖数据里取一个城市（例如 Dresden），抽 1–2k 张，用 `/images/coverage` 返回的 `imageUrl`。
- 在全球候选库上对比零样本 GeoCLIP 和瑞士微调后的模型。
- 预期微调模型变差（灾难性遗忘）。这能说明"专家化"的代价，一段话加一张小表即可。

**完成标准**：一张三列主表；一张"误差随到训练区距离变化"的图。

---

### S13. 汇总、作图与复现打包【本机】

**拉回结果**：

```bash
rsync -avP dhlab:/rcp-scratch/students/yijliu/ml-project2/data/results/ ~/Documents/ML/p2data/results/
```

**在 `notebooks/02_figures.ipynb` 里出图**：

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

**每天的节奏**：上午在 Mac 上写和测代码，推送；下午在 DHLAB 上 `git pull` 后把长任务丢进 `screen`；晚上把结果拉回本地看。

### 第 1 周（11/14–11/20）：数据和基准定死

- [ ] S1 本机建 `p2` 环境；DHLAB 补装包；`predict` 跑通
- [ ] S1 仓库骨架、`.gitignore`、`lib/paths.py`
- [ ] S2 拉全量元数据 → `smapshot.parquet`，rsync 到 DHLAB
- [ ] S3 EDA notebook：分布、地图、选定留出城市和州
- [ ] S4 写 `splits.py`，生成 `main.csv.gz` 并提交（**本周最重要**）
- [ ] S5 在 DHLAB 上启动下载（先下优先级 1）
- [ ] S7 写 `metrics.py` 并自测
- [ ] 读 GeoCLIP 和 PIGEON 的方法部分

### 第 2 周（11/21–11/27）：特征和基线

- [ ] S6 对已下载的图缓存特征，`assert` 通过；去重
- [ ] S8 B0、B1、B2 在 `val` 上的数字
- [ ] S9 A1（λ、b 搜索）、A2（网格候选库）、oracle 诊断
- [ ] 下载继续跑；下完后增量缓存特征
- [ ] 写报告的 Data & Benchmark 小节和划分地图（趁记忆新鲜）

### 第 3 周（11/28–12/04）：微调

- [ ] S10 `train.py` + yaml 配置；先在 5k 子集上确认 loss 下降
- [ ] A3a → A3b → A3c → A4，在 `val` 上做消融
- [ ] 最终配置跑 3 个种子
- [ ] E9、E10
- [ ] 在三个测试集上做最终评估（只跑一次）

### 第 4 周（12/05–12/11）：可选部分和收尾实验

- [ ] S11 可靠度：特征、模型、风险–覆盖、校准（可选）
- [ ] S12 泛化分析表和"误差 vs 距离"图（可选）
- [ ] bootstrap 置信区间；块大小敏感性检查（附录）
- [ ] **12/09 冻结实验**，之后只写作

### 第 5 周（12/12–12/17）：报告和提交

- [ ] 4 页报告 + 伦理段落 + AI 附录
- [ ] README 复现说明，每张表一条命令
- [ ] 确认提交方式（组仓库还是个人仓库）
- [ ] **12/16 提交**（留一天缓冲）

**时间不够时按这个顺序砍**：E13 → E11 → S11 的交叉拟合（只保留 val 拟合）→ A5 → 整个 S11。S12 的成本很低（S4 已经留出了数据），尽量保留。

---

## 8. 今天到周一的启动清单

**今天（周六）**

1. 读完本教程 §3–§5（1.5 小时）。
2. S1：本机建环境；DHLAB 上跑通 `predict`。
3. S2：开始拉元数据（大约 6 分钟加上请求延时，放在后台跑）。

**周日**

1. S3：EDA notebook，选留出城市和州。
2. 读 GeoCLIP 的 Fig. 2 和 §3，写 5 行笔记：图像怎么变成向量、GPS 怎么变成向量、损失是什么、推理时怎么出坐标、候选库是什么。

**周一**

1. S4：划分定死并提交。
2. S5：DHLAB 上启动下载。
3. S7：`metrics.py`。

从这一刻起，后面的步骤可以和下载并行。

---

## 9. 报告结构（4 页）

### 9.1 篇幅分配

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

### 9.2 写作要点

- 每张表都从 B0 开始，一行加一个改动，标出增益。
- 主表一定要有 `test_region` 列，这是区别于"随机划分刷分"的关键。
- 超参数放进附录的一张表。

### 9.3 伦理段落（"识别出风险"版本）

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

### 9.4 AI 使用附录

如实写：用 AI 辅助过哪些代码、文献检索、写作润色；哪些内容是人工核对过的。

---

## 10. 工作流速查（本机 ↔ DHLAB）

| 动作 | 命令 |
| --- | --- |
| 本机推代码 | `git add -A && git commit -m "..." && git push origin project_2_Yijun_Liu` |
| 服务器拉代码 | `ssh dhlab` → `ml-p2` → `git pull` |
| 本机 → 服务器传小数据 | `rsync -avP ~/Documents/ML/p2data/meta/ dhlab:/rcp-scratch/students/yijliu/ml-project2/data/meta/` |
| 服务器 → 本机拉结果 | `rsync -avP dhlab:/rcp-scratch/students/yijliu/ml-project2/data/results/ ~/Documents/ML/p2data/results/` |
| 长任务 | `screen -S name` → 运行 → `Ctrl-A D` 离开；`screen -r name` 回来；`screen -ls` 列出全部 |
| 选 GPU | 先看 `nvidia-smi`；用 `CUDA_VISIBLE_DEVICES=0`（最多 2 块，不用 GPU 3） |
| 磁盘 | `du -sh $P2_DATA/*`；scratch 是共享的，用完删掉临时文件 |

**规则**：图像、特征、checkpoint 永远不进 Git；划分文件、配置、代码、小的指标 JSON 进 Git。

DHLAB 的登录和环境细节见 [`scripts/dhlab_connect.md`](scripts/dhlab_connect.md)。

---

## 11. 风险与备选方案

| 风险 | 后果 | 备选 |
| --- | --- | --- |
| 下载太慢或被限流 | 第 2 周还没有足够的训练图 | 按优先级下载，测试集先下完；先用 50k 张 train 做完全部实验，之后再用全量重跑最终配置；联系 SMAPshot 问批量导出 |
| geoclip 和新版 transformers 不兼容 | 特征提不出来 | `pip install "transformers<5"`；或者直接用 `open_clip` 的 ViT-L/14（OpenAI 权重）提特征，再加载 geoclip 的 MLP 权重 |
| GPU 被占满 | 训练排队 | 特征缓存后，训练用 1 块卡几分钟就够；kNN 和可靠度可以在 CPU 上跑 |
| 微调打不过 kNN（`test_in` 上） | 主方法的故事变弱 | 强调 `test_region`：kNN 在那里必然崩溃；再加一个 kNN + GeoCLIP 的混合（`knn_sim` 高时用 kNN，否则用 GeoCLIP） |
| 留出州样本太少 | RQ3 的置信区间很宽 | 换更大的州，或者留出两个城市 |
| 4 页写不下 | 被扣写作分 | 主线只讲 RQ1 + `test_region`；可靠度放一张图 + 一段话；其余进附录 |

---

## 12. 需要确认的问题

**问助教或组里**

1. 课程按组提交。我独立完成的版本以什么形式提交：独立仓库、组仓库里的独立目录，还是并入组报告？

**问导师（DHLAB）**

1. SMAPshot 能否批量导出？API 可接受的请求频率是多少？
2. 对实验室来说，"可靠度"和"对新地区的泛化"哪个更有价值？
3. 倾斜航拍的坐标是相机位置，评估时是否应该改用画面中心的地面位置？`/attributes` 里有方位角和俯仰角，理论上可以算出来。

---

## 13. 参考文献（已核对出处）

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
