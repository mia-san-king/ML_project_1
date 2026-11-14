# CS-433 Project 2

把 GeoCLIP 适配到瑞士历史照片的细粒度定位。步骤在 [`doc/tutorial.md`](doc/tutorial.md)。

| | 路径 |
| --- | --- |
| 代码 | `/rcp-scratch/students/yijliu/ml-project2/ML_project_1` |
| 数据 `P2_DATA` | `/rcp-scratch/students/yijliu/ml-project2/data` |
| 环境 | `/rcp-scratch/students/yijliu/.conda/envs/ml-p2` |

`ml-p2` 会激活环境并进入项目目录。GPU 先看 `nvidia-smi`，最多 2 块，不用 GPU 3。长任务放进 `screen`。

图像、特征、checkpoint 不进 Git。划分文件在 `data/splits/`。数据是 [SMAPshot](https://smapshot.heig-vd.ch/development_documentation) 的 validated 图像，请求之间要留间隔。
