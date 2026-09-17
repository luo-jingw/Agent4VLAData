# 条件动作表征框架

> 状态：评估中的想法（讨论稿）。本文记录该想法的完整内容：动机、定义、
> 约束、关键决策、可证伪前提、文献关系与开放问题。尚未进入设计。

## 1 一句话概括

不把 action 序列当作对齐对象，而是把观测-动作对 `(o_t, a_t)` 编码为共享表征，
以目标为条件形成表征分布，用 match 查询采样，再经本体特定解码器映射回 action。

## 2 动机与演进

**起点**（合作者目标）：通用 VLA 遥操数据增强——任何机器人演示数据进来，
重构/强化其 action 序列，使模型以更少数据达到齐平性能、执行更快。

**第一次转折**：纯几何方法会破坏模态对齐（跳帧使 action 序列与观测帧错位、
必要停顿被忽略）。action 序列应视为一种模态，增强必须与 vision/文本 prompt
对齐。

**第二次转折**（外部讨论）：raw action 不是可对齐模态——它是条件/关系模态，
语义碰撞使几何近邻与语义近邻脱节（本仓库 ISSUE-003 的 home 歧义即实例）。
对齐对象应上移到 transition/outcome 级的 lifted 表征（"Action should be
decoded, not aligned"）。

**第三次收敛**（RL 视角的独立到达）：任务结果是 outcome manifold 而非标量；
决策 = 目标表征与可达结果分布的一致性匹配；trajectory-level outcome 反推
需要结构假设才可学。这与数据增强视角收敛到同一结构——条件表征分布 +
match 查询。

## 3 定义（六条）

**基本对象**：`(o_t, a_t)` 对——观测 + 动作，最 naive 形式。不含未来输入，不含目标。

1. **表征**：`r = f(o_t, a_t)`，映射到共享表征空间 R，与文本目标同域。
   f 任务无关（g 不进输入），表征序列构成轨迹表征。
2. **分布**：每个目标 g 与上下文 c 对应 R 上的条件分布 `p(R | g, c)`，
   支撑集 = 可达结果流形。多模态合法，不保证唯一最优。
3. **query**：`m(r, g)` 度量表征与目标的契合。策略 = 在 `p(R | g, c)` 上
   按 match 采样。match 是查询算子，不是定义。
4. **解码**：采样表征经 embodiment-specific decoder 映射回 action。
5. **三个学习目标分离**：编码器对齐（R 与文本同域）、分布学习（p 的估计）、
   对比目标（match）。SFT/RL/world model 在三个目标上统一。

**标量场是特例**：Successor Features 结构 `Q(s,a) = ψ(s,a)ᵀw` 中，标量值是
表征在目标方向 w 上的投影。本框架保留完整分布结构，标量投影只是 match
查询的一次运算结果。

## 4 三条硬约束

- **g 的位置**：g 只出现在 `p(R | g, c)` 的条件与 `m(r, g)` 中，不进 f 的
  输入。理由：g 进输入会导致 posterior collapse（编码器复制条件、忽略
  (o,a)），文献证据：Play-LMP 的 β<1 防坍塌设计、UniVLA 因语言吃掉码本
  而做的两阶段拆分。备选位置（g 做条件计算路由，即 MoE 语义路由）记录
  但不纳入当前版本——见讨论总结 `semantic-routing-moe.md`。
- **训练目标未来感知**：f 与 p 的估计以动作条件的未来表征预测为目标，
  预测在表征空间而非像素空间。理由：Q-learning 的存在性证明——输入不含
  未来、函数里存未来；VLA-JEPA 的 leakage 警告——未来只作监督、不作输入。
- **未来与可观测性分工**：未来期望由训练目标烘焙进 f；Case B（同 (o,a)
  不同后续）的可观测性缺口由 (g, c) 补齐（最小充分上下文）。Q-table 类比
  只在完全可观测时直接成立；部分可观测时 Q 值是对信念状态的期望，不是
  区分（ISSUE-003 的机理）。

## 5 关键决策记录

| 决策点 | 选择 | 理由 | 备选（未选原因） |
|---|---|---|---|
| 基本对象 | `(o_t, a_t)` | 最 naive；Case B 风险由 (g,c) 消歧承担，可测量 | `(o_t, a_t, o_{t+1})`（效果进输入，消歧更稳，但失去最 naive；若 Case B 测量不达标再升级） |
| g 的位置 | 条件 + match，不进输入 | 避免 posterior collapse | 进输入（坍塌）；路由（模型侧方案，另线） |
| match 是否需要历史 | 不需要 | c 已在 p 的条件里；若单帧 match 坍缩说明 c 不充分，是测量问题非定义缺陷 | — |
| 未来信息 | 训练目标烘焙（未来表征预测） | Q-learning 类比；2026 批判线否定重建目标 | 未来进输入（leakage 警告） |
| 最优性 | 不保证最优，分布采样 | 多模态合法；match 是查询非定义 | argmax（退回标量场） |

## 6 可证伪前提与测量（零新模型）

1. **Case B（同 (o,a) 不同后续）占比足够低，或 (g,c) 可消歧**。
   测：对 (o,a) 对聚类，统计跨任务后续动作一致性；关键区域行为 SNR
   （ISSUE-003 的方法）。一致率接近 100% 则前提成立；显著不一致则升级
   基本对象为转移元组。
2. **match 锐度达标**。测：关键区域 prompt 行为 SNR 与条件互信息
   （ProGAL 式）。锐度不足的区域即需要更强条件 c 的区域。

## 7 与框架核心问题（Q1-Q6）的关系

| 框架问题 | 在本想法中的角色 |
|---|---|
| Q1 相似语义下 action 分布 | 前提：p(R\|g,c) 的紧致性（组内紧致是单射对齐的地基） |
| Q2 相似 action 下语义分布 | 边界：Case B / 歧义地图，定义 (g,c) 需要消歧的区域 |
| Q3 表征中继优化 action | 机制：本框架的分布采样 + 解码就是中继优化的实例 |
| Q4 几何区分任务的界限 | 边界：几何失效处即 (g,c) 必须接管处 |
| Q5 模态组合 | 结构：本框架选择"表征（而非组合视觉轨迹）"；若 Case B 不达标，组合视觉轨迹是备选 |
| Q6 对齐研究现状 | 定位：动作-文本对齐几乎为零，组合缺口成立 |

## 8 文献锚点与组合缺口

| 要点 | 锚 |
|---|---|
| 表征分布、标量场是特例 | Successor Features（1606.05312）、Contrastive RL（2206.07568） |
| g 只做 query | Contrastive RL 的目标-表征内积结构 |
| 未来烘焙进 f | Q-learning / TD；VLA-JEPA 的 L_WM（2602.10098） |
| 可观测性缺口由 (g,c) 补 | 最小充分上下文；POMDP belief |
| 表征任务无关 | UniVLA task-centric 拆分的反面教训（2505.06111） |
| 一致性（同 latent action 异上下文异动作） | Losey 线（2105.00580） |
| 不保证最优、多模态 | 分布采样而非 argmax |

每条要点都有文献锚，但**六条 + 三约束的组合**无现成工作。此组合缺口即
该想法的候选位置；正式新颖性判断仍需 prior-art 检索。

## 9 开放问题

1. 基本对象是否升级为 `(o_t, a_t, o_{t+1})`——由 Case B 测量裁决。
2. c 的最小充分性——Q2 歧义测量的直接输出。
3. 分布学习目标的具体形式——未来表征预测（VLA-JEPA 式）之外是否有更合适
   的目标，未检索穷尽。
4. 条件计算（MoE 语义路由）是否进入未来版本——见 `semantic-routing-moe.md`。

## 10 来源讨论与参考文献

**来源讨论**（本仓库总结版；原始 html 仅存本地）：

- [多模态对齐与同胚](../../discussions/alignment-homeomorphism.md)
- [理解 RL 与 SFT](../../discussions/outcome-manifold-rl-sft.md)
- [条件计算与语义路由](../../discussions/semantic-routing-moe.md)

**表征分布与标量场**：

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

**latent action 线**（本仓库已入证据库，paper_052-058）：

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

**本项目证据与状态**：

- 综述定稿（含 RQ4/RQ5）：`../../docs/survey/survey_findings.md`
- 框架核心问题（Q1-Q6）：`../../docs/survey/research_questions.md`
- 未决问题：`../../issues.md`（ISSUE-001/002/003）
- 机会候选：`../../opportunities.md`（OPT-002）
