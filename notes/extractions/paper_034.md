# ATHENA: Accelerated Multi-Task Heterogeneous Influence Functions for Robot Data Curation

## 基本信息
- paper_id: paper_034
- 年份: 2026（arXiv v1：2026-06-15）
- 作者: Tao Xu, Jiaxin Wang, Runhao Zhang, Jiayi Guan, Xianchao Zeng, Weixi Song, Xinyu Zhou, Zhetao Chen, Guang Chen, Yong-Lu Li（同济大学、上海创新院、西安交通大学、上海交通大学）
- 来源: arXiv:2606.16208v1 [cs.RO]

## 研究问题
影响力函数能原则化地量化每条演示对任务结果的因果效应，但扩展到十亿参数级多任务 VLA 微调存在两个瓶颈：计算瓶颈（O(DP) 投影 + O(NP²+P³) Hessian 逆）与多任务不平衡瓶颈（贪心全局排名偏向强梯度任务）。ATHENA 解决：如何把闭环性能影响力函数高效用于多任务 VLA 的演示级数据策展。

## 方法
- 影响力定义（延续 CUPID 闭环归因，Sec.3.1）：动作级影响力 Ψa-inf = −∇f(ẑ)ᵀH⁻¹∇L(z)（Eq.1）；对 flow-matching VLA（π0）以平方流代理 f_sf(s,a;θ)=E_{t,ϵ}‖v_θ(x_t,s,t)‖² 作测试量（Eq.14、App.A.1），按评估 rollout 回报 R(τ)∈{1,−1} 聚合为演示级性能影响力（Eq.2）。
- 加速 1——Kronecker 压缩梯度特征化（Sec.4.1、App.A.2）：线性层梯度 G=δxᵀ，投影矩阵取 Kronecker 结构 Ω=P_in⊗P_out，双侧在激活空间投影、避免物化 D 维梯度，每层投影成本 O(DP)→O(√(DP))。
- 加速 2——rank-r 随机截断近似（RTA）：G≈U_rΣ_rV_rᵀ，damped Gauss-Newton 逆 (GᵀG+λI)⁻¹ 替换为 (Σ_r²+λI_r)⁻¹，成本 O(NP²+P³)→O(NPr)（Eq.6）。
- 多任务平衡——MII（Sec.4.2）：分别计算任务本地影响力与跨任务影响力（Eq.7），组内排序归一化为效用 u_i（Eq.8–9），最终分 f_MII = u_local · u_cross（Eq.10）；保留本地关键样本同时计入跨任务交互。
- 数据/评估：RoboTwin 2.0 demo_clean 2500 条演示 / 50 任务 / 9.34 小时 @16.67Hz；真机 6 个 ALOHA 任务 720 条高质量演示 6.9 小时 @25Hz；π0（3.3B 参数），50K 步微调；过滤比 ρ∈{0.10,0.25,0.50,0.75,0.90}。

## 关键发现
- 计算加速：50 任务（560.5K timesteps）影响力计算 8054.6 → 25.7 GPU·h（313.4×；K=5/10/25 为 405.6×/369.0×/235.5×）（Table 1；App.C.1：特征化 8004.6→23.2，Hessian 50.0→2.5 GPU·h）。
- 仿真（50 任务，下游验证）：保留 50% 演示持平/超过全量联合微调——clean 43.36% vs 43.42%，randomized 17.30% vs 15.44%；ρ=0.1 时 44.70%/17.72% vs 43.42%/15.44%；50% 数据下 randomized 反超单任务 π0 微调（17.30% vs 16.34%）（Sec.5.2.1）。
- 真机（6 任务，下游验证）：ATHENA 用 66.7% 数据平均成功率 68.0%，超 Joint-100% 60.0%、Single-100% 46.7%、Random-66.7% 50.0%、Oracle-66.7% 47.3%；累计比 Joint-100% 高 48 分、比 Single-100% 高 128 分（Fig.4）。
- 质量分与下游成功不必然一致：Oracle/TSS 子集质量分更高但成功率更低（Fig.2/3）。
- 跨模型迁移：π0 上策曲子集训练 π0.5，ρ=0.5 时 clean 63.28% / rand 34.23% vs 全量 57.00% / 25.68%（Table 2）。
- MII 消融：不用 MII 时全局单排名导致任务塌缩（真机 6 任务中 Stack Bowls 仅保留 13/120 条）；"多任务异构"指任务间时长、动作模式、跨任务耦合不同，原始影响力幅值不可跨任务直接比较（App.C.4）。
- 单任务策展（RoboTwin 4 任务）：轻过滤（删 10%）普遍提升（turn switch 35.5%→47.5%、place cans plasticbox 55%→82%）（App.C.4）。

## 证据等级
- 等级: direct
- 理由: 论文直接把影响力函数用于机器人演示策展并做下游验证（仿真 50 任务 + 真机 6 任务的成功率因果收益），给出异构多任务平衡机制，正对 RQ1.3 的策展机制与下游策略收益问题。

## 引用
- 影响力形式与 flow 代理：Sec.3.1（Eq.1–2）、App.A.1（Eq.11–15）
- 加速机制：Sec.4.1（Eq.5–6）、App.A.2
- MII：Sec.4.2（Eq.7–10、Prop 4.1）、Algorithm 1
- 主结果：Sec.5.2.1（Fig.2/3）、Sec.5.2.2（Fig.4）、Table 1/2
- 多任务异构与塌缩分析：App.C.4（Fig.8–11）
- 限制：Sec.8 Limitations

## 不确定项
- 影响力估计依赖已有评估 rollouts（作者自述限制），真机逐任务采集 rollouts 成本高；低 rollout 预算下的性能退化无数字。
- "billion-parameter"仅验证 π0（3.3B），其他 VLA 架构未实验。
- 只针对微调阶段策展，预训练阶段适用性未验证（作者自述未来工作）。
- 最优过滤比 ρ 因任务而异（H1/H2/H3 组最优 ρ 不同），未给出统一自动选 ρ 方法。
- 真机结论仅 6 任务 × 120 条演示，样本量小；"高质量演示集"上仍榨出提升，是否依赖数据中混有负影响样本未量化。
