# 机会清单

> 2026-09-08 收束后：主方向 = vision / language / action 三模态对齐。
> 原 OPT-001（agent 语义旋钮）已被收束取代，删除；本文件保留 OPT-002 的
> 对齐语境改写版。

# OPT-002

Status: proposed

Design status: incorporated-as-hypothesis（已进入设计，未经实验验证）

Implementation status: not-approved

Area: sim / 跨本体 action 对齐迁移（原 real2sim 捷径的改写）

## Observation

原线索：利用数据集重建环境做"删帧→跳步连接→碰撞检验"的几何捷径。收束后
重新定位：该线索的真正价值不在碰碰撞，而在"**仿真/陌生本体里存在良好但不可
直接迁移的 action label，经对齐后可泛化应用到微调增强**"。首轮 RQ4 检索确认
这是一条已有证据的路线：UAT 通用动作空间新本体微调仅 0.8% 参数（[paper_042](https://arxiv.org/abs/2501.10105)）；
Behavior-Aligned 以 EE 轨迹对齐行为表征（[paper_040](https://arxiv.org/abs/2607.27549)）；MOTIF 动作母题 VQ 聚类
（[paper_041](https://arxiv.org/abs/2602.13764)）。几何捷径部分保留为"对齐候选的可行性过滤器"。

## Opportunity

1. **对齐增强**：把仿真/他本体 action 序列对齐到本项目动作空间（行为表征或
   通用码本对齐），作为微调数据增强——与"prompt 对应 action 模式簇"结合
   （"pick 指令 ↔ 一类同构 action 序列簇"）。
2. **可行性过滤器**：保留静态碰撞检验作为对齐候选的第一级过滤（有限价值）。

## Expected Mechanism

对齐 = 行为表征/码本空间的对齐损失（[paper_040](https://arxiv.org/abs/2607.27549), [paper_042](https://arxiv.org/abs/2501.10105) 路线）或连续潜空间
（[paper_043](https://arxiv.org/abs/2505.04999) CLAM）；时序保真（驻留/停顿）作为对齐约束进入损失
（RQ4.6 缺口，待第二轮）。

## Required Evidence

1. 对齐度量的定义与可测性（RQ4.4 缺口）——第一可测步骤。
2. 本项目 prompt↔action 互信息/对齐度基线（H(ℓ|v) 的 action 侧对应）。
3. 对齐后增强数据在下游成功率上的对比（护栏：放置精度/首帧跳变）。

## Promotion Condition

证据 1（对齐度量）可测且 baseline 非零；证据 3 通过下游验收后进入 plan.md。

## 与主路线关系

OPT-002 是"对齐增强的数据来源"分支；主路线 = 数据面对齐 + RL 行为后训练面
（互补）。原 real2sim 捷径的几何部分降为候选过滤器。
