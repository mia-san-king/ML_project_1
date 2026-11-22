# 复现 11 月 22 日的本地实验

脚本位于 `data/local_runs/tools/`，从仓库根目录运行。原始数据、图像、特征、模型权重和生成记录从有权访问的数据存储中读取。原始数据根 `data/t1_original/` 包含 `features/`、`meta/` 和 `geo/`，统一划分位于 `data/splits/main.csv.gz`。各文件的校验值和实验汇总见[同日实验归档](../report/experiments/2026-11-22-local-models/)。

## 训练和检索

设置环境变量 `CS433_EXPERIMENT_ASSETS` 为私有数据卷的绝对路径，卷下有 `public-assets/`（公开来源的预训练权重和原始训练照片）、`aug8192/`（扩散图、元数据、训练缓存及增强 T1）和 `t4_geo4096/`（地理抽样实验）。历史 Windows 机器上默认使用已有的 `D:/CS433-T1`，其他位置默认使用仓库下的 `data/experiment_assets`。本次生成记录中的软件版本包括 PyTorch 2.9.1+cu128、diffusers 0.35.2、transformers 4.57.6、accelerate 1.12.0、NumPy 2.2.6 和 Pillow 12.0.0，增强训练入口另用 portalocker 管理文件锁。

先核验原始输入、真实训练照片的抽样顺序和公开来源。下面三条检查均可离线执行。

```text
python data/local_runs/tools/run_original_t1.py --verify-only
python data/local_runs/tools/fetch_public_augmentation_assets.py --dry-run
python data/local_runs/tools/generate_more.py --selection-only
```

要补齐缓存，可用 `fetch_public_augmentation_assets.py --only-models` 取得固定版本的 SDXL 和 CLIP，用 `--only-images` 读取所选训练照片的公开来源。若已持有对应档案文件，沿用经校验的已有缓存即可。分发仍遵循各档案馆对原始照片的许可。

8,192 张增强集由旧实验的 512 个 CLIP 向量和新增的 7,680 张扩散图构成。将旧 512 行的 float16 原始字节放到 `data/local_runs/clip_variants/variant_rows.bin`。有权访问 DHLAB 的旧 512 张增强缓存时，可在其 `augmented/features/` 目录对 `clip_vitl14.npy` 执行 `tail -c 786432 clip_vitl14.npy > variant_rows.bin`。下面第一条命令会将旧向量与原缓存拼接计算哈希，对照已归档的增强缓存。

```text
python data/local_runs/tools/compare_clip_variants.py
python data/local_runs/tools/run_original_t1.py
python data/local_runs/tools/generate_more.py --cached-text-only
python data/local_runs/tools/augment_train.py calibrate
python data/local_runs/tools/augment_train.py all
python data/local_runs/tools/run_t3_comparison.py original train
python data/local_runs/tools/run_t3_comparison.py original retrieve
python data/local_runs/tools/run_t3_comparison.py augmented train
python data/local_runs/tools/run_t3_comparison.py augmented retrieve
python data/local_runs/tools/run_paired_t4.py
```

闭式 T4 额外读取 `data/local_runs/t3_t4_8192/t4_source/ne_10m_admin_0_countries.zip`。取原实验使用的 Natural Earth 1:10m 国家边界归档，脚本通过瑞士区域生成的网格复核投影。复算原版与增强版的配对指标时，设 `CS433_ORIGINAL_T1_RUN` 和 `CS433_AUGMENTED_T1_RUN` 为这次运行返回的两个带时间戳目录，再运行 `compare_t3_t4_8192.py`。从完整增强缓存开始可以直接跳过生成，T1 使用 `augment_train.py train`。

## 地理抽样与后续模型

各用 4,096 张的随机组和稀疏地区优先组共用真实训练集。以下命令冻结抽样优先级、生成和编码缺失变体、分别拟合回归器，最后按地理块配对评估。

```text
python data/local_runs/tools/select_t4_geo4096.py
python data/local_runs/tools/geo4096_sources.py
python data/local_runs/tools/geo4096_render.py
python data/local_runs/tools/encode_t4_geo4096.py
python data/local_runs/tools/fit_t4_geo4096.py baseline
python data/local_runs/tools/fit_t4_geo4096.py random
python data/local_runs/tools/fit_t4_geo4096.py sparse
python data/local_runs/tools/fit_t4_geo4096.py compare
python data/local_runs/t4_geo4096/error_distribution.py
```

T5、分模型置信度与 T6 使用同一批照片和固定的 T4、T5 权重，各自提供独立入口。四模型精度分布柱状图可通过 `python data/local_runs/t5_aug8192/plot_model_accuracy.py --t1-predictions <增强T1预测.parquet>` 复算。[T5](../models/t5_reranker/README.md) · [置信度](geolocation_confidence.md) · [T6](../models/t6_router/README.md)
