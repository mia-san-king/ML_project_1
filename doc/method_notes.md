# GeoCLIP（方法笔记，简版）

- **图像向量**：冻结的 CLIP ViT-L/14 输出 768 维，再经可训练 MLP 映射到 512 维并 L2 归一化。
- **GPS 向量**：经纬度经 Equal Earth 投影与多尺度随机傅里叶特征（RFF），由位置编码器得到 512 维并归一化。
- **训练**：图像–GPS 对比学习（InfoNCE）；batch 内配对为正样本，另用 GPS 队列（及本项目的瑞士网格）作负样本；训练时对 GPS 加高斯噪声（jitter）。
- **打分**：`exp(logit_scale) ×` 两路嵌入的余弦相似度；推理时在候选 GPS 库上做 top-k 检索。
- **本项目**：候选库可换为全球 MP-16、瑞士 1 km 网格或 train 坐标；区域先验减分、硬/软（haversine）目标见 `models/geoclip/lib/geoclip_ops.py` 与教程 S9–S10。