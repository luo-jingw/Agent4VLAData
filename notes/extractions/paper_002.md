# 论文标题: Open X-Embodiment: Robotic Learning Datasets and RT-X Models

## 基本信息
- paper_id: paper_002
- 年份: 2023（arXiv:2310.08864；所读版本 v9, 2025-05-14）
- 作者: Open X-Embodiment Collaboration（21 机构大协作）
- 来源: arXiv:2310.08864 [cs.RO]；项目站 robotics-transformer-x.github.io

## 研究问题
- 目标 1：检验在多机器人、多环境数据（X-embodiment）上训练的策略是否获得正迁移，优于仅用本域数据训练的策略。
- 目标 2：组织大规模机器人数据集（Open X-Embodiment, OXE）并标准化，支撑后续 X-embodiment 研究。

## 方法
- OXE 数据集：1M+ 条真实机器人轨迹，22 种机器人本体，由 60 个既有数据集（来自 34 个实验室）聚合而成，527 项技能（160266 任务）。
- 跨数据集标准化（数据格式整合，Section IV-A）：
  1. 存储格式：全部转换为 RLDS 格式（序列化 tfrecord），兼容不同 action space 与输入模态（不同数量的 RGB/深度相机、点云）；支持主流框架并行加载。
  2. 观测对齐：每数据集选取一个 canonical camera view 作为输入图像，resize 到统一分辨率。
  3. 动作对齐：原始动作统一转换为 7 维末端执行器动作（x, y, z, roll, pitch, yaw, gripper 开合或其速率）；动作先归一化再离散化（每维 256 bins；另加 1 维 episode 终止，共 8 维）。
  4. 明确不对齐的部分：相机视角/位姿仍因数据集而异；不统一各数据集的末端坐标系；允许动作值为绝对/相对位置或速度（按各机器人原始控制方案）。
- 模型与训练：RT-1-X（35M 参数 Transformer + FiLM EfficientNet，15 帧图像历史）；RT-2-X（55B VLA，PaLI-X 骨干，动作转文本 token）。训练数据 mix 来自 9 种机械臂的 12 个数据集；RT-2-X 以约 1:1 比例 co-finetune 原 VLM 数据与机器人数据。
- 数据分析：用 PaLM 从指令中抽取物体与行为。
- 评估：3600 次评测，6 种机器人；对比 Original Method 与单数据集训练的 RT-1 基线。

## 关键发现
- 小数据域：RT-1-X 在 5 个数据集中的 4 个超过 Original Method，平均成功率提升约 50%。
- 大数据域：RT-1-X 欠拟合，55B RT-2-X 同时超过 Original Method 与 RT-1。
- RT-2-X 的 emergent skills 比 RT-2 高约 3×；剔除 Bridge 数据后显著下降，说明跨本体迁移带来新技能。
- Web 预训练对泛化关键（从头训练 emergent skills 0%）；短图像历史（2 帧）显著优于无历史。

## 全文相关性（fulltext_relevance）
- 全文相关性: medium
- 理由: 论文对"跨数据集标准化"有直接且具体的描述（RLDS 格式、canonical view、resize、7 维动作归一化与离散化），但它是聚合型工作：没有描述对 60 个源数据集的质量过滤、失败 episode 剔除、人工审核或质量评分机制，也未讨论次优演示；相关细节指向数据集网站而非正文。

## 引用
- Section III-A（数据集构成与 RLDS 格式）
- Section III-B（PaLM 技能/物体分析）
- Section IV-A（Data format consolidation）
- Section IV-C（训练数据 mixture 与 9 本体说明）
- Section V + Table I/II（实验与消融）

## 不确定项
- 60 个源数据集转 RLDS 时是否剔除失败/损坏轨迹：文中未说明。
- canonical camera view 的选择判据：文中未说明。
- 动作归一化与离散化的具体实现细节：文中未说明。
- 图像统一 resize 的目标分辨率数值：文中未说明。
- 各数据集语言标注是否齐全、缺失标注如何处理：文中未说明。
- 未讨论次优演示或数据质量问题；无数据质量评分/自动过滤机制描述。
- 训练用 9 本体 vs 数据集 22 本体：解释为实验时数据集仍在扩展，其余数据未用于训练。
- 涉及数据清洗/过滤的细节被指向项目网站（robotics-transformer-x.github.io），正文未展开。
