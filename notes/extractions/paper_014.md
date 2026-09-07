# Phase-Localized Curation Does Not Help: A Negative Result on Per-Phase Metric Selection for Demonstration Filtering

## 基本信息
- paper_id: paper_014
- 年份: 2026（arXiv:2606.15064v1，13 Jun 2026）
- 作者: Aarav Bedi（Department of Mechanical Engineering, University of California, Berkeley）
- 来源: arXiv cs.LG

## 研究问题
- 已有结果显示：检测缺陷 AUROC 最高的度量可能是最差的策展度量（检测价值与策展价值脱钩）。
- 本文检验一个自然假说：把轨迹按时间相位（phase）切分，在每个 phase 内用局部最优的度量打分再聚合（phase-gated curation），是否能修复全局度量因混入无信息 phase 而失败的问题。

## 方法
- 任务与策略：3 个 LIBERO 接触丰富的 pick-and-place 任务（robosuite）；策略为 phase-conditioned BC MLP（两层 256 单元，tanh 输出，Adam lr 1e-3、weight decay 1e-4，300 epochs）。轨迹分 4 个 phase：PREGRASP、DESCEND、LIFT、TRANSPORT。
- 缺陷：结构性 early-release 缺陷——LIFT phase 内随机点（phase 的 30%–70% 之间）张开夹爪，物品掉落，之后空载完成剩余动作；污染率 80%（16 干净 + 64 缺陷 = 80 条）。
- 三种策展策略，均保留 top 75%（80 条中留 60 条）：Global（全轨迹算单一度量，取前作最强全局度量）；Uniform（同一度量在每个 phase 内同样应用再聚合）；Phase-gated（每 phase 经 sweep 选出局部最具区分力的度量，phase 分数跨演示 rank-normalize 后平均）。
- 度量均在固定长度截断的轨迹上计算（消除时长混杂）。phase 分割用记录的 phase 标签，缺失时退回夹爪状态转移；该数据上分割精确（DESCEND、LIFT 长度逐 timestep 固定）。
- 每条件 5 个随机种子（42, 0, 7, 1, 2），每种子 30 次 rollout；oracle 天花板（干净数据训练）93.3–97.3%。

## 关键发现
- 负面结论：phase-gated 策展在任一任务上都不是最优；在 3 个任务中的 2 个上是三种策略中最差。
- 具体数字（成功率 %，均值±std，Table I）：Task 1：phase-gated 86.0±6.5 vs global 92.0±1.6、uniform 91.3±1.6；Task 2：78.0±39.2 vs global 58.0±46.1、uniform 83.3±27.2；Task 3：22.7±16.5 vs global 30.7±20.6、uniform 48.0±26.3。未策展污染基线：10.7±14.8 / 64.0±38.1 / 13.3±5.6。Task 2 上 phase-gated 胜过 global 但输给 uniform，故其唯一相对胜利不归因于 phase 分解。
- 机制（信号稀释）：缺陷信号集中在 LIFT；其余三个 phase 无缺陷，其局部最优度量是在噪声上选出。聚合把 LIFT 的干净信号与三个无关 phase 的 rank 平均，信号被稀释到约 1/4 权重，聚合排序比单独用 LIFT 排序更差。
- 不迁移：Table III 显示没有任何 phase 在任意两个任务间共享获胜度量；选择须每任务重新 sweep，且 sweep 本身昂贵、对种子敏感。
- 置信度阈值补丁无效：60% sweep-success 阈值回退到 uniform 的策略下，所有已选 phase 均超过阈值，选择与原 phase-gated 完全相同，结果不变。
- 种子方差大（尤其 Task 2、3），作者发布逐种子结果（Table II）；Task 3 上所有策略都远低于 oracle 天花板 97.3%。

## 全文相关性（fulltext_relevance）
- 全文相关性: high
- 理由: 论文在受控缺陷、固定污染率、多任务多种子的设定下直接实施并检验了 phase-localized 演示策展方法，给出可复现的成功率对比与机制分析，是对演示过滤策展问题的直接（负面）证据。

## 引用
- phase-gated 定义与三种策略：III-C Curation Strategies。
- 缺陷与污染设定：III-B（80% 污染率、30%–70% LIFT 内随机释放）。
- 主要结果数字：IV-A、Table I、Table II。
- 信号稀释机制：IV-B Why Phase-Gating Fails（"稀释到约四分之一的权重"）。
- 不迁移：IV-C、Table III。
- 置信度阈值补丁无效：IV-D（60% sweep-success 阈值）。
- 结论边界：V. Discussion（仅适用于单 phase 集中的缺陷；未测多 phase 分布式缺陷）、Limitations。

## 不确定项
- 只测试了一种缺陷类型（LIFT 内的 early-release）、一种策略类（phase-conditioned BC MLP）、一个仿真器（robosuite/LIBERO）、3 个任务；作者明确不声称对"缺陷信号分布于多个 phase"的情况成立。
- 未报告各候选度量的完整清单（文中出现 smoothness、isolation forest、kNN、gripper timing、entropy，未说明是否有其他候选）。
- Uniform 基线的单一度量具体是什么、Global 基线的最强全局度量具体名称均未明确给出（仅称取自前作 [1]）。
- 未说明"固定长度截断"的具体截断长度。
- 未在更低污染率（如 50%、20%）下测试，无法判断污染率对该负面结论的影响。
- 未与 paper_013 式的"在线 rollout 成功/失败分类器"策展方法直接对比。
