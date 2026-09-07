# DABI: Evaluation of Data Augmentation Methods Using Downsampling in Bilateral Control-Based Imitation Learning with Images
## 基本信息
- paper_id: paper_025
- 年份: 2024（arXiv v1，2024-10-06）
- 作者: Masato Kobayashi†, Thanpimon Buamanee†, Yuki Uranishi（†同等贡献；大阪大学）
- 来源: arXiv:2410.04370（预印本）
## 研究问题
双边控制模仿学习（Bi-ACT）中机器人数据 1000 Hz、相机图像 100 Hz，需降采样对齐到低频；评估"下采样式数据增强"的三种做法（纯降采样、降采样+顺延对齐、DABI 对称对齐）对真实机器人任务成功率的影响。
## 方法
- 下采样式增强的定义：一个图像采集周期内（1000Hz vs 100Hz）有 N 组机器人数据；增强 = 给每张图像配不同的机器人数据切片作为训练样本起点，使 5 条演示扩展为 50 条（10 倍），而非只取与图像同时刻的 1 组。
- 评测的三种数据集（各 5 条演示，Put-in-Drawer 任务，Sec IV/V-D）：
  - Method1：纯降采样到 100Hz，无增强（5 演示）。
  - Method2：降采样+增强，取图像采集时刻起、至下一张图像为止的机器人数据（5→50 演示）。
  - Method3（DABI，提出方法）：以图像为对称中心，取图像前后等距的机器人数据作起点，用 floor/ceiling 调整数量（5→50 演示）。
- 模型：Bi-ACT（ACT 改双边控制版，输入 2 路 RGB 360×640 + 15 维关节数据，action chunking 输出 k×15）；5 训练物体 + 3 未见物体，每物体 5 次试验，按 5 阶段（Open/Pick/Move/Place/Close）计成功率。
## 关键发现
- Method1（无增强）：训练物体整体差，Pink/Black bouncy 全阶段 0%；未见物体多止步 Open 阶段（bottle cap Open 40% 后全 0%，eye cream 全 0%）。
- Method2：Foam ball 全阶段 100%；未见物体部分提升（eye cream 各阶段 80%），但 glue jar 各阶段仅 40%，不稳定。
- Method3（DABI）：所有物体（含 3 个未见物体）全阶段 100%；训练物体 80%–100%（Red/Black bouncy 80%，其余 100%）。
- 结论：下采样后的机器人数据对齐方式（增强方法差异）显著影响成功率；DABI 最优。
## 证据等级
- 等级: direct
- 理由: 论文本身即对下采样式增强方法的三组对照评测（Method1/2/3），成功率数字全部直接来自 Table II。
## 引用
- 下采样式增强定义与三种方法：Sec IV（第 3–4 页）、Sec V-D（第 4–5 页）。
- 实验数字：Table II（第 6 页）；实验设置：Sec V-B/V-E（第 4–5 页）。
- 局限：Sec VI（第 5 页）。
## 不确定项
- 各阶段"成功"的判定标准未定义（人工判定还是自动判定未说明）；Total 成功率汇总公式未给出。
- 未见物体 glue jar（45×63 mm，远大于训练物体 ≤40 mm）在 Method2 仅 40%、Method3 达 100%，Method3 是否过拟合或评判偏松未讨论。
- 样本量极小：仅 1 个任务、5 条演示、每条件 5 次试验；无统计检验、无误差棒。
- 与 RQ3.3 的关系：论文将降采样称为数据增强，但未对增强后的训练数据做任何验证或过滤（无模拟 rollout、无失败处置描述），无接触/事件段概念。
- 数量表述不一致：Sec IV-C 称可扩展"N+1 倍"（N=10 应为 11 倍），摘要与 Sec V-D 称 10 倍（5→50 演示）；文中未解释。
- 作者归属含糊：标题作者序为 Kobayashi/Buamanee/Uranishi，页脚†标注"Equal Contribution"，具体贡献未说明。
