# 数据来源与本地文件指纹

`assets.tsv.gz` 记录 2026-11-22 在实验工作站上实际读取的研究文件。GitHub 保存清单、实验代码和结果摘要。完整训练数据的存放位置为 DHLAB。原始 T1 实验已有 DHLAB 输入归档，可用校验值核对。新增扩散图、T3–T6 模型、校准器、逐图结果和 T6 分发包的 DHLAB 副本在本次盘点时尚未远程验收。

## 如何校验

每行依次为逻辑根、相对路径、字节数和文件完整字节流的 SHA-256。表采用 UTF-8、制表符分隔、LF 换行，首行是表头，按逻辑根与相对路径排序。重复使用的缓存可以有多个路径和同一哈希，文件大小按每条路径分别计数。`summary.json` 记录分组统计及解压后整个 TSV 的 SHA-256。

- `workspace` 指项目根，可见 `data/splits`、`data/t1_original` 和 `data/local_runs`。
- `store` 指实验资产根，可见 `aug8192`、`t4_geo4096` 和 `public-assets`。
- 原始 T1 运行的部分文件在不同机器上可能带操作者前缀。清单和校验器统一使用从 `T1_dual_time_available_s0_20261118` 开始的逻辑文件名。机器登录信息、传输诊断、运行锁、虚拟环境和 Python 缓存不属于研究数据。

在复制回来的一个或两个数据根上运行下列命令。校验器只需要 Python 标准库，返回非零表示存在缺失或不匹配的文件。

```text
python inventory.py --workspace <项目根> --store <实验资产根>
```

## 原始数据与地理留出

SMAPshot validated 快照含 355,281 条元数据，`workspace/data/t1_original/features` 中的 CLIP ViT-L/14 缓存含 308,672 个图像 ID。通过照片 ID 将 `data/splits/main.csv.gz` 的 `split` 列与特征 ID 取交集，得到下表。地理划分使用 10×10 公里的瑞士 LV95 网格，伯尔尼测试取市中心 10 公里内、提契诺测试取州界内，缓冲区与单列的垂直航拍组不进入主实验。查询的原始坐标用于评估。训练图库仅含 `train` 的 176,308 张真实照片。[SMAPshot API](https://smapshot.heig-vd.ch/api/v1) · [项目划分说明](../../../../doc/project2_handoff.md)

| 划分 | 原始划分照片数 | 图像特征可用数 | 可用照片的地理块数 |
| --- | ---: | ---: | ---: |
| train | 176,538 | 176,308 | 325 |
| val | 29,532 | 29,529 | 47 |
| test_in | 47,609 | 47,593 | 93 |
| test_city（伯尔尼市） | 3,378 | 3,377 | 7 |
| test_region（提契诺州） | 13,040 | 13,039 | 42 |

原照片链接保存在 SMAPshot 元数据的 `image_url`。本清单覆盖本机用于增强的 JPEG 子集，包括 `store/public-assets/images` 和 `store/t4_geo4096/parents`。完整原照片语料与这些子集不同。本机原始训练快照以元数据、特征、特征 ID、地理网格和划分文件核对。瑞士边界从 Natural Earth 1:10m 国家边界筛选 `ADM0_A3=CHE`，本机存档的国家边界压缩包及瑞士 GPKG 均有逐文件指纹。[DHLAB 原始 T1 实验归档](../../2026-11-18-geoclip-time/) · [Natural Earth 边界来源](https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip)

## 增强模型与训练产物

| 文件集 | 对应来源与用途 |
| --- | --- |
| `store/public-assets/sdxl-fp16`, `store/public-assets/clip` | 生成用 SDXL 和特征提取用 CLIP，各自的权重与配置均按文件校验。`store/public-assets/public_assets_manifest.json` 和对应 JSONL 记录获取来源。 |
| `store/aug8192`, `workspace/data/local_runs/clip_variants` | 8,192 张训练增强图由原有 512 张和本地新增 7,680 张组成。最初 512 张在本机保存了 CLIP 特征行及已有实验记录。新增 7,680 张有逐 PNG 哈希、生成记录和父图关联。 |
| `workspace/data/local_runs/t1_full_*`, `store/aug8192/t1_dataset`, `store/aug8192/t1_runs` | 原始／增强条件的 T1 数据、权重、训练记录、评估和逐图输出。 |
| `workspace/data/local_runs/t3_t4_8192`, `t3_refine_analysis` | 原始与增强的 T0/T3、T4，含 T3 检索和 T4 回归所需检查点，以及候选与坐标细化诊断。 |
| `store/t4_geo4096`, `workspace/data/local_runs/t4_geo4096` | 随机和稀疏地区优先两组各 4,096 张。随机组复用已有增强，稀疏组复用 192 张并新增 3,904 张。父图、固定选择方案、增强特征、拟合权重和配对评估均有指纹。 |
| `workspace/data/local_runs/t5_aug8192`, `confidence_models` | T5 的前 50 个候选、检查点和逐图结果，以及四个定位器各自的 25 公里概率校准器与评估文件。 |
| `workspace/data/local_runs/t6_aug8192` | T6 部署包 `T6.zip`、解包组件和 64,009 张留出照片的配对评估。分发包在本机验收的 SHA-256 为 `fb881729f8278f753bf205fb974b5cf180073b20ae5828b94e42eba2b9bbd6be`。 |

SDXL 使用 `stabilityai/stable-diffusion-xl-base-1.0` 的 `462165984030d82259a11f4367a4eed129e94a7b` 版本，CLIP 使用 `openai/clip-vit-large-patch14` 的 `32bd64288804d66eefd0ccbe215aa642df71cc41` 版本。按清单下载相应的模型文件和配置，可逐项校验字节。[SDXL 固定版本](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b) · [CLIP 固定版本](https://huggingface.co/openai/clip-vit-large-patch14/tree/32bd64288804d66eefd0ccbe215aa642df71cc41)
