# From Inference Efficiency to Embodied Efficiency: Revisiting Efficiency Metrics for Vision-Language-Action Models

## 基本信息
- 作者：Zhuofan Li*, Hongkun Yang*, Zhenyang Chen*, Yangxuan Chen, Yingyan (Celine) Lin, Chaojian Li（HKUST + Georgia Tech），前三位为共同一作
- arXiv:2603.19131v1 [cs.LG]，2026-03-19；预印本，未说明同行评审状态
- 类型：实证对照研究（非综述），目录分类：C7 效率

## 研究问题
- 以参数量、FLOPs、token 解码吞吐为特征的"推理效率"是否能反映 VLA 模型在真实机器人平台上的实际表现
- 若不能，则系统级"embodied efficiency"应如何定义与度量，以及针对高效 VLA 的常见手段（模型压缩、token 稀疏化、动作序列压缩、in-context 提示、SFT）对其影响如何

## 方法
- 度量框架（两组共 5 个指标 + 成功率 SR，均为仿真 Kinematics/Control 代理指标，文中说明模拟器无法准确建模执行器动力学与散热，故不直接测能量）：
  - 任务时长类：任务完成时间 τ = T/f（步数/控制频率）；末端执行器路径长度 L_ee = Σ‖p_{t+1}−p_t‖₂；关节空间路径长度 L_joint = Σ‖q_{t+1}−q_t‖₂
  - 运动平滑类：平均 jerk L2 范数 J = f⁴/(T−2)·Σ‖q̈_{t+1}−2q̈_t+q̈_{t−1}‖₂（离散二阶差分）；平均动作率 R = 1/(T−1)Σ‖a_{t+1}−a_t‖₂
  - 指标只对成功 episode 平均：m_succ = Σm_i·1[succ_i] / Σ1[succ_i]
- 实验设置：Libero-Spatial/Object/Goal/10 + Bridge；每个任务 N=50 个测试 rollout（或先取得 10 次成功即停，同 Libero 原设）；模型 π0、π0.5、MolmoAct
- 干预域：(1) 模型压缩（magnitude pruning 5%/10%/20%；GPTQ 式后训练量化 int8/int4，采用 fake quantization 方案）；(2) token 空间（visual token pruning，比例 12%–78%）；(3) 动作空间（FAST 动作 tokenizer）；(4) 适应方法（in-context：Brief/Detail 两类效率感知提示；SFT：auxiliary jerk/action-rate loss，η=0.01）

## 关键发现
- 功率量级对比：Koch v1.1 伺服额定力矩总功耗约 22 W vs Jetson Orin NX GPU 最大功率 20 W；Franka 机械臂+控制器至多 430 W vs NVIDIA H100 NVL 约 400 W——推理与执行功耗同量级
- Fig.1：π0.5 在 Libero 任务上 5% 与 20% 剪枝后成功率均为 100%，但完成时间为 13 s vs 21 s
- 剪枝（Libero 四套件，5%–20%）：成功率最大降 2.7%；末端路径长度最大 +5.6%（π0 10% 剪枝 +5.6%）；平均 jerk 最大 +11.0%（π0 10% 剪枝）；Bridge 上 5% 剪枝 → 末端路径长度 +46.2%、成功率 −0.2%（摘要同时给出任务完成时间 +13.6%，正文未列出 Bridge 的 τ 具体值）
- 量化 int4：jerk 增加 π0 +19.5%、MolmoAct +14.1%、π0.5 +3.5%；成功率最差降 1.8%；int8 影响小（jerk 最大 +0.8%）——int4 量化在大部分指标上接近或差于 int8
- token 剪枝（π0.5, Libero-Spatial）：剪枝比 12%→56%→78%，jerk 升至未剪枝模型的 204%→337%→375%
- FAST 动作压缩：jerk 增加 +28.0%～+50.6%，完成时间减少 1.5%～5.6%（Libero-Object：π0 与 π0-FAST 成功率相同，FAST τ −1.5% 但 J +34.5%）
- in-context 提示：两个提示设计下平均动作率 R 在所有套件下降；jerk 降幅最大至 −25.8%（Libero-Goal, Detail）；代价是完成时间 +4.3%～+11.2%（Brief 提示多数套件时间增或不变）
- SFT 辅助 loss：jerk 降低 16.6%～24.0%，成功率平均 +0.4%，完成时间最大 +3.8%
- 整体结论：（1）降低推理计算不必然改善实体执行；（2）系统级指标能暴露被传统评价掩盖的策略差异；（3）常用适应方法只带来温和、指标特定、且常伴随权衡的改善

## 证据等级
- 实验性（Observational/实验对照）：多模型、多基准、多名执行轮（每任务 N=50），非单实例演示；但全部评测为仿真 + 运动学/控制代理指标，未在真实机器人上直接测量能量/时间，能量结论为推断

## 引用
- 被研究/对比模型：π0（arXiv:2410.24164）、π0.5（arXiv:2504.16054）、MolmoAct（arXiv:2508.07917）、FAST（arXiv:2501.09747）
- 基准：Libero（NeurIPS 2023）、Bridge v2（arXiv:2308.12952）；实现：OpenPi
- 其他：Magnitude pruning（Han et al. 2015）、GPTQ（arXiv:2210.17323）、EfficientVLA token pruning（arXiv:2506.10100）、SAIL 快速比演示执行（Arachchige et al. 2025）

## 不确定项
- Bridge 上 5% 剪枝导致 +13.6% 完成任务时间仅出现在摘要，正文 Bridge 实验只报告路径长度与成功率
- 能量未被直接测量；指标是能量效率的代理，代理与实际能量的相关性文中未量化
- 各干预的跨模型模式为趋势性总结（如"higher token pruning often leads to jerkier trajectories"），未被统一显著性检验覆盖
