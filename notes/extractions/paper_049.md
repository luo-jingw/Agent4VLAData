# Streaming Flow Policy: Simplifying Diffusion/Flow-Matching Policies by Treating Action Trajectories as Flow Trajectories

## 基本信息
- 作者：Sunshine Jiang, Xiaolin Fang, Nicholas Roy, Tomás Lozano-Pérez, Leslie Kaelbling, Siddharth Ancha（MIT）
- 发表：CoRL 2025（9th Conference on Robot Learning, Seoul, Korea）；arXiv:2505.21851v2 [cs.RO]，2025-09-24
- 类型：方法论文（行为克隆/模仿学习）

## 研究问题
- 传统 diffusion/flow 策略在"轨迹空间 ℵ^T"中采样"轨迹的轨迹"：丢弃中间轨迹、必须等采样结束才执行，计算开销大、反馈环路松
- 问题的核心转换：把 action 轨迹本身当作 flow 轨迹——在 action 空间 ℵ 中做流传输（flow transport in A, not trajectory space ℵ^T），使 flow 时间与执行时间对齐，支持边采样边执行
- 对本文调研的意义：这是"action 序列作为被建模对象（模态）"的一种形式化——action chunk 不再是被生成的目标分布，而是生成过程本身的一条轨迹；多模态仍被保留（flow matching 保证逐时刻边缘分布匹配）

## 方法
- 条件速度场构造（式 2）：v_ξ(a,t) = ξ̇(t) − k(a−ξ(t))，初始分布 p₀_ξ = ℵ(a|ξ(0), σ₀²)；路径速度 + 稳定化反馈项，构造围绕演示轨迹的"窄高斯管"；定理 1：边缘分布 ℵ(a|ξ(t), σ₀²e^(−2kt))，标准差随时间指数衰减
- 训练：条件 flow matching 损失（式 4）L2 回归 v_θ(a,t|h) 匹配 v_ξ(a,t)；复用现有 diffusion/flow 策略架构，仅把输入输出从 ℵ^T 改为 ℵ，时序卷积/注意力换成全连接层
- 推理：从 ℵ(a_prev, σ₀²) 采样 a₀（a_prev 为上一个 chunk 最后一个动作；模仿状态时用当前已知机器人状态），前向有限差分积分 a(t+Δt) = a(t) + v_θ(a,t|h_chunk)Δt，闭环外开环执行；每步算出即流式发送
- 测试时 σ₀=0 确定性执行（与 ACT 一致；附录 B 给出潜变量变体用于测试时采多模式，但实验表明 σ₀=0 更优）
- 两条规则：action 空间需连续且轨迹连续可微（需用 ξ̇(t)，binary {0,1} 夹爪改为连续开度 [0,1]）；速度场是神经 ODE + 初始噪声分布（连续归一化流）
- 稳定化增益 k（引 Block et al. 2023 的低层稳定控制器可降低分布偏移）；附录 C：chunk 大小最优点 3/4 环境为 8、1/4 环境为 6
- 与"无需显式聚类、以对齐为约束"的衔接点：多模态性不需要任何模式分割/聚类——最优速度场（式 5）是各演示高斯管的"加权平均"，逐时刻边缘分布直接对齐训练分布（对齐约束代替显式聚类）；代价是不匹配联合分布（式 6），轨迹可跨演示组合（compositionality，文中视为特征而非缺陷）

## 关键发现
- Push-T state 输入（action imitation，avg/max）：SFP 95.1%/96.0%、延迟 3.5 ms vs DP(100 DDPM) 92.9%/94.4%、40.2 ms vs 10-DDIM 87.0%/89.0%、4.4 ms；state imitation：SFP 83.9%/84.8%、8.8 ms vs DP 87.0%/90.1%、127.2 ms
- Push-T image 输入（state imitation）：SFP 91.7%/93.7%、3.5 ms vs DP 90.7%/92.8%、40.2 ms；image+action imitation（附录 D）：82.5%/87.0%、8.8 ms（DP: 83.8%/87.0%、127.2 ms）
- RoboMimic（state 输入）：Lift 100.0%/100.0%、Can 98.4%/100.0%、Square 78.0%/84.0%、延迟 4.5 ms；去稳定化（k=0）：Can 90.0%/92.0%、Square 53.2%/60.0%
- 消融 k=0（Push-T state）：84.0%/86.4% vs 95.1%/96.0%（稳定化显著提升）
- rebuttal：前馈速度场基线更差——Push-T-State 82.3% vs 95.1%、Push-T-action 72.1% vs 91.7%、RoboMimic-Can 12.4% vs 95.6%；假设因训练时不采样动作噪声而不获 Lipman et al. 边缘分布保证
- 论文声称对 k、σ₀、Δt 不敏感（唯一显著下降是 k 恰好为 0）；对 k、σ₀、Δt 的具体取值文中未说明
- 真实实验：Franka Research 3 实现 reach+pick、block reorientation，仅定性（视频）对比 diffusion policy，无定量表格
- 局限：只保证逐时刻边缘分布匹配、不保证联合分布（示例中采样出"3"形而非"S"形轨迹）；可学任意位置约束（关节限位）与凸速度约束（区间 [v_min,v_max]），不能学联合分布才能表达的全局约束

## 证据等级
- 实验性：Push-T（200 演示/50 评估）与 RoboMimic（每任务 300 演示/50 评估）双基准 + 4 基线 + k=0 消融 + 真实 Franka 定性验证；数字为"5 best checkpoints 平均分 + 最佳分"口径，未报告多次运行方差

## 引用
- 前驱/基线：Diffusion Policy（Chi et al. RSS 2023）、Flow Matching（Lipman et al. ICLR 2023）、Affordance Flow Matching（Zhang & Gienger 2024）、Streaming Diffusion Policy（Høeg et al. ICRA 2025）、π0（Black et al. 2024）、ACT（Zhao et al. RSS 2023）
- 理论支撑：生成式行为克隆的低层稳定性保证（Block et al. NeurIPS 2023）、神经 ODE（Chen et al. 2018）、FFJORD（Grathwohl et al. 2019）、Implicit BC（Florence et al. CoRL 2022）

## 不确定项
- 文中未说明：k、σ₀、Δt 的具体取值；实际用到的 T_pred、T_chunk（仅 rebuttal 提及 T_pred≈16、T_a≈8）
- 文中未说明：真实机器人实验的演示数与定量成功指标（仅上传视频）
- 状态与动作的 velocity 可能一致也可能不一致（文中承认 ξ̇(t) 未必是物理速度），含义边界需注意
