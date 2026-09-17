# Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild

## 基本信息
- 作者：Hao Luo、Ye Wang、Wanpeng Zhang、Haoqi Yuan、Yicheng Feng、Haiweng Xu、Sipeng Zheng、Zongqing Lu（通讯）；北京大学、中国人民大学、BeingBeyond
- arXiv：2602.21736v1（cs.RO），2026-02-25
- 项目页：https://research.beingbeyond.com/jala
- 类型：预印本；手部动作生成评测 + LIBERO/RoboCasa/GR1 仿真 + 真机（Franka 臂 + Inspire 灵巧手）

## 研究问题
- VLA 预训练受机器人数据稀缺限制；人视频可选两类数据存在质量-多样性权衡：实验室数据有精确 3D 手部追踪但限于桌面场景，in-the-wild 视频多样性高但无可靠动作标签
- 既有 latent action 方法缺陷（本文指出的批判）：1) LAPA 等重建式范式（IDM 推断 latent action + FDM 重建未来帧）中，latent action 质量依赖 FDM 对帧的建模能力；2) 对精细人手操控，手部运动细微多变，FDM 建模困难，其噪声"degrade the latent action quality rather than stable grounding"；3) 重建式密集像素监督在 in-the-wild 视频中把权重放在外观、背景、相机伪影上，与动作相关动力学不对齐
- 目标：用联合对齐取代重建——使 latent action 空间既可由 VLA 上下文预测（predictable from context），又保留逆动力学信息（informative of inverse dynamics），无需重建像素；统一标注与无标注视频

## 方法
### 联合对齐
- predictive embedding hi,k = 骨干第 19 层（共 28 层）隐藏状态，对每个 motion token 提取
- LAP（Latent Action Perceiver）：2 层 Perceiver 逆动力学模型，输入 chunk 边界帧 (vt, vt+δ) 的视觉特征（DINOv3 或 V-JEPA2），输出 K 个 latent action zi,k；对齐损失 LAlign = ΣΣ∥hi,k − zi,k∥1（无像素重建）
- LSP（Latent State Perceiver）：与 LAP 参数共享的 Perceiver，输入复制初始帧 (v0, v0)，把 VLM 预测上下文接入同一 latent 空间
- 解耦非对称 EMA（α=0.999）：backbone 用 LSP 梯度更新后向 LAP 传播；query 用 LAP 梯度更新后向 LSP 传播
### MCP（Masked Chunk Prediction）
- chunk 级掩码 token 预测（类似 GR-1）；混合掩码：随机选一个 chunk 作主预测目标，其前 chunk 完整保留，目标 chunk 内每 token 以 {0.05,…,1.0} 均匀采样比例掩码，其后 chunk 固定 5% 掩码；无标注视频整 chunk 掩码、MCP 项失效、仅 LAlign
- 总损失 L = 1labeled·LMCP + λLAlign（λ=0.5）
### 预训练与后训练
- 骨干 InternVL3-2B；motion 以 GRVQ 量化（手腕/手指各 64 token/块，码本 4096，共 128 token/chunk）；15 帧/chunk；in-the-wild 数据时间放慢 0.5×
- UniHand-Mix：7.5M 样本（>2,000 小时）= 5M+ 实验室标注 instruction-video-motion 样本（1,000 小时） + 2.5M in-the-wild instruction-video 对（Ego4D 过滤，约 10% 带 HaWoR 伪手姿态，置信阈值 0.65）；8 个数据源（Ego4D 32.9%、EgoDex 14.7%、Oakink2 12.4%、HOI4D 10.0% 等）
- 训练：1 epoch，68 小时，8×NVIDIA A800(80G)；AdamW lr 3×10⁻⁵；batch 128（8 GPU × 16，梯度累积）
- 后训练：flow-matching DiT 头（16 层 32-head，hidden 2048），predictive embeddings 作条件；推理 N=4 去噪步；仅 LM 参数解冻

## 关键发现
- 手部动作生成（Wild split MPJPE ↓）：JALA-dino 11.02 / JALA-vjepa 11.54 vs Being-H0 16.91 / JALA w/o latent 20.34；Lab split 增益较小
- LIBERO 双视图（无任何机器人数据预训练）：JALA-dino 平均 96.9%；LAPA 79.5%、LAPA† 83.5%（同骨干同数据，仍落后 JALA 超 13 点）；UniVLA 95.5%；JALA w/o dec. 56.6%；JALA-act 94.3% vs Being-H0 90.2%
- LIBERO 单视图（≤3B 参数）：JALA-dino 平均 92.3%（新 SoTA）；GR00T N1.5 92.1%；JALA w/o latent 77.0%；Long 套件 87.2%
- RoboCasa（每任务 50 demos）：JALA Syn 27.58 / Human 35.42 vs LAPA 16.25/22.42 vs Being-H0 23.83/31.33；JALA w/o dec. 14.25/19.33
- GR1 tabletop：JALA 26.33% vs GR00T N1.5 20.41% vs LAPA 11.42% vs Being-H0 12.91%；JALA-act 20.25%（约为 Being-H0 两倍）
- 真机（3 任务 × 10 rollouts，子任务完成率 %）：JALA-dino Put-Three-Obj 60.0 seen/58.0 unseen、Wipe-Board 83.3/80.0、Water-Plant 73.3，全部任务最高；出现未被显式监督的自纠正重抓行为
- 训练效率：JALA 68h vs LAPA† 两阶段 29h+57h=86h，不足 80% 墙钟时间且性能更好
- 消融：in-the-wild 数据比例 0%→100% 下游成功率单调提升；第 19 层隐藏状态迁移最优（14 层次之，24/28 层退化）；t-SNE 显示 h 与 z 聚于相近区域，Wild 扩展 Lab 流形

## 证据等级
- 低-中：arXiv 预印本 v1，未经同行评审；仿真（LIBERO 双/单视图、RoboCasa、GR1）+ 真机（3 任务×10 rollouts）+ 消融，结果均为作者自报，未见第三方复现
- 观察性证据：成功率、完成率、四项手部误差指标、t-SNE、rollout 定性分析

## 引用
- Hao Luo, Ye Wang, Wanpeng Zhang, Haoqi Yuan, Yicheng Feng, Haiweng Xu, Sipeng Zheng, Zongqing Lu. "Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild." arXiv preprint arXiv:2602.21736, 2026.

## 不确定项
- 未报告置信区间/统计显著性检验；真机每任务仅 10 rollouts
- LIBERO 微调演示数量的准确值文中未说明（仅写"fewer than 50 demonstrations per task"）
- JALA 模型总参数量文中未说明
- in-the-wild 伪手姿态标注仅约 10% 样本，"JALA⋆"实验（把标注 wild clips 当无标注用）显示该小比例标注有帮助但非必需（95.7% vs 96.9%）
- RoboCasa 表格与 GR1 数值引自表 4（文中 6.3.3 与 6.3.4 间存在排版断裂，GR1 各方法列读数按表 4 推断）
