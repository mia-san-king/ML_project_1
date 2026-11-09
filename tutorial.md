# CS-433 Project 2

本教程根据项目启动会记录、`README.md` 中的资源，以及两篇推荐阅读论文整理而成，帮助你在分支 `project_2_Yijun_Liu` 上从零开始推进 **图像地理定位（image geolocalization）** 相关课题。若课程随后发布正式 Project 2 说明 PDF，以官方文档为准，本教程作为实践路线图补充。

---

## 1. 这个项目在做什么？

**图像地理定位**：给定一张照片，预测它是在地球上的哪个位置（通常是 GPS 经纬度，或落在哪个地理区域）。

两篇起点论文的思路（本地 PDF 在 `papers/`）：


| 论文                         | 本地文件                               | 核心思想（简）                                                                                       |
| -------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------- |
| **StreetCLIP**（arXiv 2023） | `papers/arxiv-2302.00275.pdf`      | 在 CLIP 上做对比学习，用合成 caption 把模型「锚定」到地理领域，强调 **零样本 / 开放域** 定位。                                   |
| **GeoCLIP**（NeurIPS 2023）  | `papers/neurips-2023-1b57aadd.pdf` | **CLIP 式图像–GPS 对齐**：用位置编码（含 Fourier 特征、层次化表示）把经纬度编成向量，与图像特征对齐，做 **image-to-GPS 检索** 而非单纯网格分类。 |


阅读顺序建议：

1. 先读 GeoCLIP 的 **Abstract + Method 图**（结构清晰）。
2. 再读 StreetCLIP 的 **Introduction + 零样本设定**。
3. 在 Google Scholar 上查这两篇的 **「被引用」**，看 2024–2026 是否有更适合你数据（瑞士 / 欧洲历史影像）的 follow-up。

你要做的 Project 2，通常会在「**历史 / 地理标注影像 + 现代深度模型**」这条线上：用 SMAPshot 或 4Dbrowser 拿到带时空信息的图像，再设计或微调一个定位 / 检索模型，并在报告里写清数据、方法、指标与失败案例。

---



## 2. 仓库与分支

```text
ML_project_1/
├── README.md           # 英文资源索引
├── tutorial.md         # 本文件
├── papers/             # 已下载的两篇 PDF
└── （后续）data/ src/ configs/ report/ ...
```

- **Project 1 完整代码**：分支 `main`（不要在其上直接做 Project 2）。
- **Project 2 工作区**：分支 `project_2_Yijun_Liu`（当前为空起点，除 README / papers / 本教程外需自行搭建）。

切换分支：

```bash
cd /path/to/ML_project_1
git checkout project_2_Yijun_Liu
```

---



## 3. 训练数据从哪来？

启动会给了两个 **在线数据源**（需自己写脚本拉取，并遵守礼貌访问）：

### 3.1 SMAPshot（瑞士景观 / 历史地理影像）

- 文档：[SMAPshot development documentation](https://smapshot.heig-vd.ch/development_documentation)
- 特点：瑞士相关、带地理参考的历史 / 景观影像，适合 **瑞士范围内** 的定位或检索实验。

**建议第一步（探索，不写大模型）：**

1. 通读 API / 数据字段说明：有哪些 endpoint、返回的是图像 URL 还是元数据、坐标字段叫什么。
2. 用 Python `requests` 写一个小脚本：请求 **1–5 条** 样本，打印 JSON 结构，存到 `data/raw/smapshot_sample.json`。
3. 确认：经纬度格式、时间戳、图像分辨率、许可与引用要求（写进报告 References）。



### 3.2 4Dbrowser（欧洲城市 / 时空 API）

- API 入口：[4Dbrowser API](https://4dbrowser.urbanhistory4d.org/api)
- 特点：欧洲城市 **历史维度（4D）** 数据，适合「同一地点不同年代」或 **城市尺度** 的定位 / 检索。

**建议第一步：**

1. 打开 API 文档（站点上通常有 OpenAPI 或说明页），列出可用资源（城市、图层、时间片）。
2. 同样先 **少量请求** 摸清 schema，再设计批量下载计划。
3. 与 SMAPshot 对比：你更想做 **瑞士全国** 还是 **某几个欧洲城市**？选题越窄，越容易在期内做完。



### 3.3 API 礼貌访问（必做）

会议明确要求：**两次请求之间加延时**，避免压垮服务器。

示例模式：

```python
import time
import requests

DELAY_SEC = 1.0  # 先用 1 秒；若文档有更严格限制则调大

def get_with_delay(url, **kwargs):
    time.sleep(DELAY_SEC)
    r = requests.get(url, timeout=60, **kwargs)
    r.raise_for_status()
    return r
```

- 开发阶段：缓存响应到 `data/cache/`，重复实验 **不要重复打 API**。
- 批量下载：记录失败 URL，支持断点续传。
- 若需 API key：不要提交到 Git；用环境变量或本地 `config.local.yaml`（加入 `.gitignore`）。

---



## 4. 推荐的项目阶段（按周推进）



### 阶段 A：问题定义（1–2 天）

- 选定 **主数据集**（SMAPshot **或** 4Dbrowser，或两者但角色分明：train / eval）。
- 写清 **任务形式**（三选一或组合，需与导师/说明一致）：
  - **回归**：预测 (lat, lon)；
  - **检索**：在候选 GPS 集合里找最近邻；
  - **分类**：落在预定义网格 cell（GeoCLIP 也讨论 grid，但更强调连续 GPS 编码）。
- 定义 **评估指标**：例如 Great-circle 距离误差（km）、Top-k 检索准确率、或 cell 分类 accuracy。



### 阶段 B：数据管道（约 1 周）

建议目录：

```text
data/
  raw/           # API 原始 JSON / 小样本
  images/        # 下载后的图像（可按 hash 命名）
  processed/     # 清洗后的 CSV：path, lat, lon, time, source
  splits/        # train/val/test（按 **地理或时间** 划分，避免泄漏）
```

注意：

- **划分方式**：同一地点不同时间的照片不要同时出现在 train 和 test（时间泄漏）。
- **坐标系**：统一 WGS84；记录缺失坐标的样本如何处理。
- 本地 `data/` 建议 **gitignore**；只提交下载脚本与统计摘要。



### 阶段 C：基线模型（约 1 周）

由浅入深：

1. **启发式基线**：预测训练集 GPS 均值 / 按国家或城市众数（证明任务非 trivial）。
2. **传统特征**：ResNet 全局池化 + 线性层回归 (lat, lon)（MSE on sin/cos 编码更稳）。
3. **论文复现方向**（需 GPU）：
  - 若有 GeoCLIP / CLIP 预训练权重，做 **冻结 backbone + 只训 location head** 的小实验；
  - 或实现简化版 **图像–GPS 对比损失**（读 GeoCLIP 式 location encoder 一节）。

Project 1 用的是 NumPy；Project 2 通常会用到 **PyTorch + torchvision + 预训练 CLIP**。这与课程 deep learning 部分一致，但具体允许库以 Project 2 说明为准。

### 阶段 D：实验与报告（持续）

- 固定随机种子；记录 hyperparameter 与 commit hash。
- 至少：**主结果表 + 失败样例图**（预测点 vs 真值在 map 上）。
- 讨论：历史影像域 gap、API 采样偏差、零样本是否可行。

---



## 5. 在 DHLAB 集群（RCP）上训练

官方学生指南（必读）：

[EPFL-DHLAB-student-projects — README-Student-HowTo.md](https://github.com/dh-epfl-students/EPFL-DHLAB-student-projects/blob/main/README-Student-HowTo.md)

### 5.1 登录与存储

严格按 [DHLAB README-Student-HowTo](https://github.com/dh-epfl-students/EPFL-DHLAB-student-projects/blob/main/README-Student-HowTo.md)：

1. **导师先开通** IC 集群上 DHLAB 机器的访问权。
2. 登录：`ssh yijliu@dhlabsrv1.epfl.ch`（本机别名 `ssh dhlab`）。详见 [`scripts/dhlab_connect.md`](../scripts/dhlab_connect.md)。
3. **数据与 Conda 环境** 放在 **`scratch/students/<你的目录>`**，不要放在 home（`/` 盘仅约 200GB）。
4. Conda 配置（How-To 原文）：
   ```bash
   conda config --add envs_dirs /scratch/$USER/.conda/envs
   conda config --add pkgs_dirs /scratch/$USER/.conda/pkgs
   ```
5. 若出现 **`Connection refused`**，官方文档未给额外步骤，需联系导师确认账号与 `dhlabsrv1` 状态；**`Permission denied`** 才是 Gaspar 用户名/密码问题。



### 5.2 Conda 环境（不要装在集群根目录）

```bash
conda create --prefix "/rcp_scratch/students/<你的_gaspar>/conda/ml-p2" python=3.11
conda activate "/rcp_scratch/students/<your_gaspar>/conda/ml-p2"
pip install torch torchvision requests tqdm pandas  # 按实际需要增减
```

`<你的_gaspar>` 换成 EPFL 用户名。会议提醒：**切勿在共享根路径安装 Conda/env**，否则会阻塞其他同学。

### 5.3 GPU 使用规范

- **一次只用一块 GPU**（除非 HowTo 明确允许多卡）。
- 提交作业前在节点上执行 `nvidia-smi`，确认空闲显存。
- 长时间训练用 **Slurm / 集群作业脚本**（见 DHLAB README），不要在前台占 GPU 过夜而不排队。



### 5.4 典型工作流

1. **本地 Mac**：写数据探索脚本、小 batch 调试代码。
2. **git push** 到 `project_2_Yijun_Liu`。
3. **RCP 上 git pull**，在 scratch 里跑训练，结果写回 scratch 或 wandb。
4. 只把 **脚本、配置、报告、小样本** 放进仓库；权重文件过大则 gitignore，必要时用课程允许的方式交付。

---



## 6. 与两篇论文对齐的「最小可行目标」

若时间紧，可以设定 MVP：


| 级别    | 目标                                                                 | 对应论文思想                  |
| ----- | ------------------------------------------------------------------ | ----------------------- |
| MVP-1 | 在自建 CSV 上训练 ResNet 回归 (lat,lon)，报告 median 误差 (km)                  | 经典 supervised geo       |
| MVP-2 | 实现 GPS 的 sin/cos 或 Fourier 编码 + MLP，与图像 embedding 做对比损失            | GeoCLIP 简化              |
| MVP-3 | 加载 OpenCLIP/CLIP 冻结图像塔，只训 location tower，在瑞士子集上 zero-shot 检索 Top-1 | StreetCLIP / GeoCLIP 精神 |


每升一级都在 `report/` 里加一节 **Ablation**，说明增益来自数据量、编码方式还是 backbone。

---



## 7. 本地开发检查清单

- [ ] 已读 `README.md` 中全部链接
- [ ] 已读 `papers/` 两篇 PDF 的 Abstract + 方法示意图
- [ ] SMAPshot 或 4Dbrowser 至少成功请求并缓存 **10 条** 样本
- [ ] 实现 API 延时 + 本地 cache
- [ ] 写好 `data/README.md`（字段说明、划分规则）
- [ ] `.gitignore` 包含 `data/`、大权重、`__pycache__/`
- [ ] 在 RCP 上按 HowTo 创建 **prefix Conda**，跑通 `import torch` + 一次 forward
- [ ] 确定评估指标与 baseline 数字，再堆模型复杂度

---



## 8. 常见问题

**Q：两个数据集都要用吗？**  
A：不必须。选一个做主训练，另一个可选做 **跨域泛化** 小实验；两个都深度整合往往工作量过大。

**Q：和 Project 1 的 NumPy 限制一样吗？**  
A：Project 1 强制 NumPy；Project 2 通常是 **深度学习 + 地理数据**，以课程 Project 2 说明为准。本教程默认 PyTorch 生态。

**Q：论文代码要复现吗？**  
A：不必须完整复现。理解 **损失函数 + GPS 编码 + 数据形式**，实现适合你们 API 数据的 scaled-down 版本即可。

**Q：API 下载一半失败怎么办？**  
A：增大 `DELAY_SEC`、减小并发（建议 **串行**）、记录失败 ID 夜间重试；永远不要用多线程狂刷 API。

---



## 9. 下一步你可以立刻做的三件事

1. 打开 [SMAPshot 文档](https://smapshot.heig-vd.ch/development_documentation)，找到 **第一个能返回坐标 + 图像 URL 的 endpoint**，在本仓库新建 `scripts/explore_smapshot.py` 跑通。
2. 打开 [4Dbrowser API](https://4dbrowser.urbanhistory4d.org/api)，同样写 `scripts/explore_4dbrowser.py` 打印 JSON 样例。
3. 精读 GeoCLIP PDF 第 3–4 页（location encoder），用中文笔记总结「GPS 如何变成向量」——这会直接决定你模型 head 怎么写。

完成以上三步后，你就从「空分支」进入「有数据、有问题定义」的状态，可以开始和组员分工（数据 / 模型 / 报告）。

---

## 10. GeoCLIP 方法说明（论文 + 官方代码）

本节总结推荐主线的 **GeoCLIP**，便于你们读论文、装代码、设计 SMAPshot 实验。与 Project 2 的衔接见上文「阶段 C / MVP-2」。

### 10.1 要解决什么问题？

**全球图像地理定位（worldwide image geolocalization）**：输入一张任意地点拍摄的图片，输出其 **GPS（纬度、经度）** 或与之等价的地理位置。

难点在于：全球景观差异极大，且很多旧方法把地球划成 **固定网格做分类**——网格中心离真值远时误差很大，也难以扩展到「开放世界」。

GeoCLIP 把问题改写成 **CLIP 式的跨模态对齐**：让 **图像特征** 与 **GPS 特征** 在同一空间里靠近（同一张图与其真实坐标），远离（与错误坐标）。

### 10.2 核心思想（和 OpenAI CLIP 的类比）

| CLIP（原版） | GeoCLIP |
| --- | --- |
| 图像 ↔ 文本 caption | **图像 ↔ GPS 坐标** |
| 对比学习：匹配的图文拉近，不匹配的推远 | 对比学习：**匹配的 (image, GPS) 对** 拉近 |
| 文本塔编码语言 | **Location Encoder（位置编码器）** 编码经纬度 |

训练数据方面，论文使用 **MP-16** 等大规模 **全球带 GPS 的图像**（约 470 万张量级），模型从数据中学会「什么样的视觉外观对应什么样的地理区域」。

### 10.3 模型结构（两部分）

1. **图像编码器（Image Encoder）**  
   基于 **CLIP 视觉 backbone**（与 CLIP 多模态预训练一致），把图片映射为定长向量（例如 512 维）。

2. **位置编码器（Location Encoder）**  
   把 **连续** 的 `(lat, lon)` 映射为与图像同维的向量。关键设计包括：
   - 把地球视为 **连续函数**，而不是离散类别；
   - 使用 **Random Fourier Features（随机傅里叶特征）** 等 positional encoding，使模型能表达多尺度的地理信息；
   - **层次化表示**，在不同「分辨率」上捕捉从全球到区域的信息。

训练完成后，**图像向量** 与 **GPS 向量** 内积（或相似度）高，表示「这张图像在该位置拍摄」的可能性大。

### 10.4 训练方式

- **损失**：与 CLIP 类似的 **对比损失（contrastive loss）**——在一个 batch 内，正确的 image–GPS 对为正样本，其余组合为负样本。
- **推理（常见两种）**：
  - **检索式**：在大量候选 GPS（或 GPS 网格/库）上算相似度，取 **Top-k** 作为预测（官方 `predict(..., top_k=5)` 即此类思路）；
  - **嵌入用途**：单独使用训练好的 **Location Encoder**，把 GPS 变成 512 维向量，供其他 **geo-aware** 任务拼接使用（论文在 Geo-Tagged NUS-Wide 等任务上有实验）。

### 10.5 和你们项目的关系

- **SMAPshot / 4Dbrowser** 提供 **(历史或地理参考图像, 坐标/可推导 GPS)**，格式上与 GeoCLIP 的 **(image, GPS) 对** 一致，适合：
  1. **零样本**：直接加载预训练权重，在你们 holdout 上算 **定位误差（km）** 或 Top-k 命中率；
  2. **微调**：用瑞士/欧洲子集继续训练或只训 location / 部分层，缓解 **历史影像 vs 现代街景** 的域差距。
- GeoCLIP **不是**为瑞士历史地图专门训练的；**直接零样本可能偏差大**，报告里应诚实写 domain gap，并用微调或 baseline 对比说明改进。

### 10.6 官方资源（必读）

| 资源 | 链接 |
| --- | --- |
| **NeurIPS 2023 论文页** | [proceedings.neurips.cc（GeoCLIP）](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b57aaddf85ab01a2445a79c9edc1f4b-Abstract-Conference.html) |
| **arXiv** | [arxiv.org/abs/2309.16020](https://arxiv.org/abs/2309.16020) |
| **本地 PDF** | `papers/neurips-2023-1b57aadd.pdf` |
| **官方 PyTorch 实现（MIT）** | [github.com/VicenteVivan/geo-clip](https://github.com/VicenteVivan/geo-clip) |

仓库说明：这是论文 **GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization** 的官方实现，支持 `pip install geoclip` 或 clone 后 `python -m pip install .`。

### 10.7 最小推理示例（来自官方 API）

安装：

```bash
pip install geoclip
# 或：git clone https://github.com/VicenteVivan/geo-clip && cd geo-clip && python -m pip install .
```

对单张图做 Top-5 GPS 预测：

```python
from geoclip import GeoCLIP

model = GeoCLIP()
image_path = "path/to/your_image.png"

top_pred_gps, top_pred_prob = model.predict(image_path, top_k=5)

for i, (gps, prob) in enumerate(zip(top_pred_gps, top_pred_prob)):
    lat, lon = gps
    print(f"Top-{i+1}: lat={lat:.6f}, lon={lon:.6f}, prob={prob:.6f}")
```

仅使用 **位置编码器**（例如你们已有 GPS 标签、要做特征拼接或检索库）：

```python
import torch
from geoclip import LocationEncoder

gps_encoder = LocationEncoder()
# 纬度、经度顺序与官方示例一致：NYC, LA
gps_data = torch.tensor([[40.7128, -74.0060], [34.0522, -118.2437]])
gps_embeddings = gps_encoder(gps_data)  # shape: (2, 512)
```

在 DHLAB 上请把上述环境装进 **自己的 scratch Conda prefix**（见第 5 节），推理阶段同样 **单 GPU**，先 `nvidia-smi` 再看显存。

### 10.8 建议阅读顺序（GeoCLIP 专项）

1. 论文 **Abstract + Method 图**（理解双塔与对比学习）。  
2. GitHub [README](https://github.com/VicenteVivan/geo-clip) 的 **Getting Started** 与 **Inference**。  
3. 用 SMAPshot 的 **5–10 张** 样例图跑通 `GeoCLIP().predict`，把预测点画在 map 上与真值对比（即使很差也是有效 baseline）。  
4. 再决定是否在本仓库增加 `scripts/eval_geoclip_smapshot.py` 与微调脚本。

---

*文档与* `README.md` *同步维护；链接失效时以课程 Ed / 导师通知为准。*