# 理解 RL 与 SFT：outcome manifold（外部讨论总结）

> 来源：ChatGPT 讨论归档（原始 html 仅存本地）。本文为内容总结，非逐字记录。

## 核心框架

1. **任务结果不是 scalar，而是 outcome manifold**。"成功/失败"只是流形上的粗糙 partition，不是评价本身；每个 branch 内部又连续变化。
2. **credit assignment 被重新解释**：最终 outcome 本身就是所有前序决策的监督信号。
3. **SFT、RL、world model 统一**为三个 representation/distribution 学习问题：
   - action-conditioned reachability（动作条件的可达性）；
   - goal-conditioned decision（目标条件的决策）；
   - 对这套结构的某种 scalar projection / computational shortcut。
4. 决策 = 目标表征与可达结果分布之间的一致性匹配。
5. 目标是离散的 outcome mode + mode 内连续质量差异。

## 相关研究线（对话检索结论）

| 工作 | 与框架的关系 |
|---|---|
| Contrastive Learning as Goal-Conditioned RL（NeurIPS 2022） | 表征内积 = goal-conditioned value；RL 可重述为表征/对比学习 |
| Successor Features（Barreto） | 标量场是表征在目标方向投影的祖先形式 |
| METRA（ICML 2023） | latent space metric 对齐环境 temporal distance/reachability |
| HILP（ICML 2024） | Hilbert representation + latent direction = 行为，prompting 式控制 |
| Proto-RL（ICML 2021） | 表征 + prototypes 聚类经验空间 |
| DIAYN（ICLR 2019） | 无奖励函数自监督发现技能 |
| Occupancy Reward Shaping（ICLR 2026） | occupancy measure 内含 temporal geometry，optimal transport 提取 goal-reaching 信息 |

## 关键理论结论

**2026 年 trajectory-level outcome supervision 理论工作**：从整体 outcome 反推序列决策在一般情形下不可学习（指数样本复杂度），需要额外结构假设——"manifold/结构假设可能不是装饰，而是让问题可学习所必需的东西"。

## 01 网格（对话结尾提出的四维切分）

评价对象是否 trajectory-outcome（而非 state occupancy）；outcome 表征是否自学习 latent（而非人定义）；是否为分布（而非点估计）；goal 是否与可达性模型分离（可换目标）。据此逐格检查现有研究覆盖。
