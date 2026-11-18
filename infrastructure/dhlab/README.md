# DHLAB 环境辅助脚本

这些是服务器环境工具，所有模型共用。已有环境可直接使用，无须重新安装。

| 文件 | 用途 |
| --- | --- |
| `dhlab_probe_scratch.sh` | 检查服务器可用的 scratch 路径 |
| `dhlab_once_setup.sh` | 一次性准备 scratch 和 Conda 目录 |
| `dhlab_full_setup.sh` | 完整的环境安装、仓库克隆和 shell 配置 |
| `dhlab_setup_remote.sh` | 从本地通过 SSH 调用 probe 或完整安装 |

连接说明见 [`doc/dhlab_connect.md`](../../doc/dhlab_connect.md)。这些脚本会安装软件或配置服务器，不是模型训练入口；目录检查只做 shell 语法验证，不执行安装。

`dhlab_full_setup.sh` 保留原来的默认克隆分支 `project_2_Yijun_Liu`。直接运行它并希望克隆本次整理后的分支时，显式设置 `DHLAB_REPO_BRANCH=Yang_Zhang`。已有服务器 checkout 不会自动切换分支；使用新目录命令前应先确认 checkout 包含 `models/`。日常训练使用各模型目录下的脚本。
