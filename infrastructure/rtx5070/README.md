# 在 Mac 开发，在 Windows RTX 5070 训练

在 Mac 的 `Yang_Zhang` 工作副本里修改代码，用本地 Python 运行相关入口和测试，把通过测试的修改提交并推送到 GitHub。用 `git rev-parse HEAD` 取得 **40 位提交号**。这次训练使用该提交的代码，之后的提交可以独立开发。

例如，`python -m pytest -q models/geoclip_time/tests` 会在 CPU 上执行一个包含反向传播、参数更新和重新载入权重的小训练。推送后可用 `git ls-remote origin refs/heads/Yang_Zhang` 核对 GitHub 上的提交号。

在 Mac 的 VS Code 中使用 **Remote - Tunnels** 打开 `cs433-rtx5070` 的项目，在远端工作区新建 **Git Bash** 终端。从含 `.venv` 的 Windows 项目根目录运行下面的命令，将第一行的值换成 Mac 刚刚推送的提交号。

```bash
revision=替换为完整的40位提交号
set -o pipefail
git fetch --no-tags origin Yang_Zhang &&
  git show "${revision}:infrastructure/rtx5070/launch.py" |
  ./.venv/Scripts/python.exe - "$revision" --data-root data/t1_original
```

启动器在 Windows 用户目录下的 `cs433-gpu-runs` 创建一个新的运行目录，在其中从 GitHub 获取指定提交，然后用项目的 Windows Python 启动独立训练进程。开发目录里的未提交修改留在原处。终端会显示运行目录和 PID。在对应目录的 `console.log` 查看数据核验、每轮进度及错误。`run/status.json` 记录提交号、运行状态、设备和产物路径。训练结束后，验证中位误差最低的权重与逐轮指标分别写入数据根目录的 `ckpt/` 与 `results/metrics/`。

本入口训练原版双塔 T1。它从归档配置加载 15 轮、batch size 32、seed 0，并将特征、ID、元数据、坐标网格和划分与原实验的 SHA-256 对齐。运行中请让 Windows 台式机保持开机并避免休眠。查看进度时，在远端 Git Bash 运行 `tail -f "<运行目录>/console.log"`，也可在 VS Code 中打开该文件。
