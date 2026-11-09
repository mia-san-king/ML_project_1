# DHLAB 远程服务器（与官方 How-To 一致）

官方文档：[README-Student-HowTo.md](https://github.com/dh-epfl-students/EPFL-DHLAB-student-projects/blob/main/README-Student-HowTo.md)

## 1. 谁能登录？

How-To 写明：

1. **先由导师开通权限**（*ask your supervisor to gain access*）。
2. 再用 Gaspar 登录。

若未开通或机器维护，会出现 **`Connection refused`**（连不上 22 端口），与 Gaspar 用户名对错无关。

## 2. 官方 SSH 命令（只有这一条）

```bash
ssh yijliu@dhlabsrv1.epfl.ch
```

本机快捷方式（`~/.ssh/config` 已配置）：

```bash
ssh dhlab
```

- **Gaspar 用户名：** `yijliu`（不是 SCiper 402257，不是邮箱 `yijun.liu@epfl.ch`）
- **密码：** Gaspar 密码

How-To **没有**写跳板机、非 22 端口或 VPN 步骤；若在校内/校外均 `Connection refused`，需联系 **Ziyi Liu / Rémi Petitpierre** 确认账号与 `dhlabsrv1` 是否在运行。

## 3. 机器资源（How-To）

| 项目 | 说明 |
|------|------|
| RAM | 约 256 GB |
| GPU | 2 块（导师邮件：最多同时使用 GPU 0/1/2 中的 2 块） |
| `/` | 约 200 GB（home 很小） |
| **scratch** | 约 12 TB |

## 4. 数据放哪里？（How-To 强调）

**不要**把数据集、模型、中间结果放在 **home**。

放在 **`scratch/students`** 下自建目录，例如：

```bash
# 登录后
mkdir -p /scratch/students/yijliu/ml-project2
cd /scratch/students/yijliu/ml-project2
```

**`dhlabsrv1` 上学生目录：** `/rcp-scratch/students/yijliu`（挂载点名带 **连字符** `-`，不是 `rcp_scratch`）。`df -h` 里对应 `nas3.../ic_u1703_dhlab_scratch` → `/rcp-scratch`。

## 5. Conda 在 scratch（How-To 原文）

登录后执行一次：

```bash
conda config --add envs_dirs /scratch/$USER/.conda/envs
conda config --add pkgs_dirs /scratch/$USER/.conda/pkgs
```

创建环境示例：

```bash
conda create --prefix /scratch/$USER/.conda/envs/ml-p2 python=3.11 -y
conda activate /scratch/$USER/.conda/envs/ml-p2
```

定期清理：`conda clean --all`

也可运行本仓库：

```bash
# 在 Mac 上（会提示 Gaspar 密码一次；装 Miniconda + ml-p2 + GeoCLIP，约 5–15 分钟）
bash /Users/yijunliu/Documents/ML/ML_project_1/scripts/run_dhlab_setup_from_mac.sh
```

登录后直接使用：

```bash
source ~/.bashrc   # 新会话可省略
ml-p2              # 激活环境并 cd 到项目
python scripts/smoke_geoclip.py
```

## 6. GPU

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0   # 或 0,1，总共不超过 2 块
```

## 7. 常见错误对照

| 终端提示 | 含义 | 处理 |
|----------|------|------|
| **Connection refused** | 22 端口无 SSH 服务 / 防火墙 / 机器关 / **未开通访问** | 问导师；稍后再试；校外可试 EPFL VPN 后再 `ssh dhlab` |
| **Permission denied** | 已连上 SSH，**用户名或密码错** | 确认 `yijliu` + Gaspar 密码 |
| **Operation timed out** | 网络/VPN | 连 EPFL VPN |

## 8. 远程开发（How-To 可选）

- [VS Code Remote SSH](https://code.visualstudio.com/docs/remote/ssh-tutorial)
- Jupyter / `screen` 用法见 How-To 同一文档后半部分

## 9. 克隆本项目

```bash
cd /scratch/students/yijliu/ml-project2   # 按实际 scratch 路径改
git clone git@github.com:mia-san-king/ML_project_1.git
cd ML_project_1 && git checkout project_2_Yijun_Liu
```
