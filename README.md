# CS-433 Project 2

瑞士历史照片定位：在相同的数据划分和评估标准下比较不同模型。当前分支为 `Yang_Zhang`。

换电脑后可从[中文项目交接](doc/project2_handoff.md)继续。各模型指标见[实验报告](report/geoclip_time_baselines.md)，原版 T3 的前 50 名统计、运行配置和复算脚本在[实验归档](report/experiments/)。

[Mac 开发与 RTX 5070 训练](infrastructure/rtx5070/README.md)说明如何把本地测试过的提交从 GitHub 交给 Windows 台式机，并查看原版 T1 的训练日志。

## 先按模型找文件

```text
models/
├── clip_regression/    # T4：冻结的 CLIP 特征和日期直接回归瑞士坐标
├── baselines/          # 常数预测 B0、图片近邻检索 B2
│   ├── lib/
│   └── scripts/
├── geoclip/            # 原有 GeoCLIP：冻结模型、区域先验、A3/A4 微调
│   ├── lib/
│   ├── configs/
│   └── scripts/
└── geoclip_time/       # 图片＋年代、位置＋年代双塔
    ├── lib/
    ├── configs/        # T0 不加时间；T1 双塔加时间
    ├── scripts/
    └── tests/
```

| 模型目录 | 入口说明 | 方法说明 |
| --- | --- | --- |
| [`models/clip_regression/`](models/clip_regression/) | T4 单层回归训练与 T1 逐图比较 | [T4 实验说明](doc/time_geoclip.md#t4已知拍摄日期直接回归坐标) |
| [`models/baselines/`](models/baselines/) | 常数预测和 kNN 的运行命令 | [`doc/tutorial.md`](doc/tutorial.md) S8 |
| [`models/geoclip/`](models/geoclip/) | 原有 B1/B3、A1、A3/A4 实验 | [`doc/method_notes.md`](doc/method_notes.md) |
| [`models/geoclip_time/`](models/geoclip_time/) | T0/T1 数据检查、训练、预测和测试 | [`doc/time_geoclip.md`](doc/time_geoclip.md) |

每种模型自己的 `lib/` 放具体算法，`configs/` 放参数，`scripts/` 放运行入口，`tests/` 放该模型的检查。小基线没有配置或测试文件时，不创建空目录。原来的 B1/B2/B3 批量运行入口保留在 GeoCLIP 目录，其中 B2 调用 baselines 的实现。

## 所有模型共用的部分

| 顶层目录／文件 | 用途 |
| --- | --- |
| [`lib/`](lib/) | 元数据、空间划分、地理距离、特征表、评估指标和可靠度工具；`features.py` 是复用的 CLIP 特征接口 |
| [`scripts/`](scripts/) | 抓元数据、下图、划分数据、构建瑞士候选网格、缓存 CLIP 特征、EDA 和评估检查 |
| [`data/splits/`](data/splits/) | 统一的训练／验证／测试划分；各模型使用相同划分进行比较 |
| [`doc/`](doc/) | 统一保存教程、方法说明、论文、小组会议资料；开发计划集中在 `doc/development/` |
| [`notebooks/`](notebooks/) | 数据探索和通用绘图 |
| [`report/`](report/) | 统一汇报、结果比较、图表和 LaTeX；队友已有结果见 [`interim_results.ipynb`](report/interim_results.ipynb) |
| [`infrastructure/dhlab/`](infrastructure/dhlab/) | DHLAB 环境安装和连接辅助脚本；连接说明在 [`doc/dhlab_connect.md`](doc/dhlab_connect.md) |
| [`requirements-lock.txt`](requirements-lock.txt) | Python 依赖版本 |

共用 CLIP 缓存可供当前几个模型复用；未来换图像主干时可以另建缓存，不要求所有模型使用 CLIP。`report/` 目前保存的结果来自已有实验，目录整理不会生成新的训练结果。

## 运行时间模型

以下命令均从仓库根目录运行。整理后旧的 `scripts/train_time.py`、`scripts/train.py` 和顶层 `configs/` 已移入各模型目录，请使用新路径。

```bash
# 先检查元数据、日期和特征缓存，不训练
python models/geoclip_time/scripts/train_time.py \
  --config models/geoclip_time/configs/T1_dual_time.yaml --check-data

# 不加时间的对照
python models/geoclip_time/scripts/train_time.py \
  --config models/geoclip_time/configs/T0_no_time.yaml --device cuda

# 双塔都加时间
python models/geoclip_time/scripts/train_time.py \
  --config models/geoclip_time/configs/T1_dual_time.yaml --device cuda

# 小型代码检查，不运行正式数据集训练
python -m pytest models/geoclip_time/tests -q
```

预测命令、日期区间编码、loss 和数据覆盖检查见 [`doc/time_geoclip.md`](doc/time_geoclip.md)。模型超参数、已有 checkpoint 格式和 `P2_DATA` 下的输出位置保持一致。

将来增加新结构时，新建 `models/<模型名>/`，将它自己的实现、配置、脚本和测试放进去；复用顶层数据划分和评估工具。方法介绍统一写进 `doc/`，结果统一汇总到 `report/`。

## DHLAB 与数据位置

| | 路径 |
| --- | --- |
| 代码 | `/rcp-scratch/students/yijliu/ml-project2/ML_project_1` |
| 数据 `P2_DATA` | `/rcp-scratch/students/yijliu/ml-project2/data` |
| 环境 | `/rcp-scratch/students/yijliu/.conda/envs/ml-p2` |

`ml-p2` 会激活环境并进入项目目录。GPU 先看 `nvidia-smi`，最多 2 块，不用 GPU 3。长任务放进 `screen`。

图像、特征、checkpoint 不进 Git。划分文件在 `data/splits/`。数据是 [SMAPshot](https://smapshot.heig-vd.ch/development_documentation) 的 validated 图像，请求之间要留间隔。
