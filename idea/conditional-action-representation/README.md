# 条件动作表征框架

> 状态：评估中的想法（讨论稿）。本文件记录当前收敛的框架定义及其来源
> 讨论与参考材料，尚未进入设计。

## 定义（六条要点）

**基本对象**：`(o_t, a_t)` 对——观测 + 动作，最 naive 形式。不含未来输入，不含目标。

1. **表征**：`r = f(o_t, a_t)`，映射到共享表征空间 R，与文本目标同域。
   f 任务无关（g 不进输入），表征序列构成轨迹表征。
2. **分布**：每个目标 g 与上下文 c 对应 R 上的条件分布 `p(R | g, c)`，
   支撑集 = 可达结果流形。多模态合法，不保证唯一最优。
3. **query**：`m(r, g)` 度量表征与目标的契合。策略 = 在 `p(R | g, c)` 上
   按 match 采样。match 是查询算子，不是定义。
4. **解码**：采样表征经 embodiment-specific decoder 映射回 action。
   "Action is decoded, not aligned"。
5. **三个学习目标分离**：编码器对齐（R 与文本同域）、分布学习（p 的估计）、
   对比目标（match）。SFT/RL/world model 在三个目标上统一。

## 三条硬约束

- **g 的位置**：g 只出现在 `p(R | g, c)` 的条件与 `m(r, g)` 中，不进 f 的
  输入（进输入会 posterior collapse；Play-LMP 的 β<1、UniVLA 两阶段拆分
  是文献证据）。
- **训练目标未来感知**：f 与 p 的估计以动作条件的未来表征预测为目标
  （Q-learning 存在性证明：输入不含未来、函数里存未来），预测在表征空间
  而非像素空间（VLA-JEPA 的 leakage 警告：未来只作监督、不作输入）。
- **未来与可观测性分工**：未来期望由训练目标烘焙进 f；Case B（同 (o,a)
  不同后续）的可观测性缺口由 (g, c) 补齐（最小充分上下文）。Q-table 类比
  只在完全可观测时直接成立。

## 两条可证伪前提（零新模型可测）

1. Case B（同 (o,a) 不同后续）占比足够低，或 (g,c) 可消歧——测：同 (o,a)
   对聚类，统计跨任务后续动作一致性；关键区域行为 SNR（ISSUE-003 方法）。
2. match 锐度达标——测：关键区域 prompt 行为 SNR 与条件互信息。

## 来源讨论（本仓库归档）

- [多模态对齐与同胚](../../discussions/多模态对齐与同胚.html)——
  raw action 不是可对齐模态；对齐对象是 transition/outcome 级 lifted
  表征；最小充分上下文。
- [理解 RL 与 SFT](../../discussions/理解RL与SFT.html)——
  outcome manifold 取代标量回报；goal ↔ 可达结果分布的一致性匹配；
  trajectory-level outcome 反推需结构假设才可学。
- [解释数据蒸馏插值](../../discussions/解释数据蒸馏插值.html)——
  representation 相对 query 才成立；语义即程序（条件计算/路由）；
  理解 = 按目标构造表征的能力。

## 参考文献（按主题）

**表征分布与标量场**

- Successor Features for Transfer in Reinforcement Learning（Barreto 2017）:
  https://arxiv.org/abs/1606.05312
- Contrastive Learning as Goal-Conditioned Reinforcement Learning（Eysenbach 2022）:
  https://arxiv.org/abs/2206.07568
- METRA: Scalable Unsupervised RL with Metric-Aware Abstraction:
  https://arxiv.org/abs/2310.08894
- Foundation Policies with Hilbert Representations（HILP, ICML 2024）:
  https://arxiv.org/abs/2402.15567
- Reinforcement Learning with Prototypical Representations（ICML 2021）:
  https://arxiv.org/abs/2102.11271
- Diversity is All You Need（DIAYN, ICLR 2019）:
  https://arxiv.org/abs/1802.06070
- Occupancy Reward Shaping（ICLR 2026）:
  https://arxiv.org/abs/2604.20627

**latent action 线（本仓库已入证据库，paper_052-058）**

- Learning Latent Plans from Play（CoRL 2019）:
  https://arxiv.org/abs/1903.01973
- Learning Visually Guided Latent Actions for Assistive Teleoperation（L4DC 2021）:
  https://arxiv.org/abs/2105.00580
- Latent Action Pretraining from Videos（LAPA, ICLR 2025）:
  https://arxiv.org/abs/2410.11758
- UniVLA: Learning to Act Anywhere with Task-centric Latent Actions:
  https://arxiv.org/abs/2505.06111
- CLAP: Contrastive Latent Action Pretraining:
  https://arxiv.org/abs/2601.04061
- Joint-Aligned Latent Action:
  https://arxiv.org/abs/2602.21736
- VLA-JEPA: Enhancing VLA with Latent World Model:
  https://arxiv.org/abs/2602.10098

**本项目证据与状态**

- 综述定稿（含 RQ4/RQ5）：`../../docs/survey/survey_findings.md`
- 框架核心问题（Q1-Q6）：`../../docs/survey/research_questions.md`
- 未决问题：`../../issues.md`（ISSUE-001/002/003）
- 机会候选：`../../opportunities.md`（OPT-002）
