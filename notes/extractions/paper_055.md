# Learning Visually Guided Latent Actions for Assistive Teleoperation

## 基本信息
- 作者：Siddharth Karamcheti (Stanford)、Albert J. Zhai (Caltech)、Dylan P. Losey (Virginia Tech)、Dorsa Sadigh (Stanford)
- 会议：L4DC 2021（PMLR vol 144:1–12, 2021）
- arXiv：2105.00580v1（cs.RO），2021-05-02
- 代码：github.com/Stanford-ILIAD/vla

## 研究问题
- 辅助遥操作：人类用低维接口（1-DoF 摇杆）控制高维（7-DoF）机器人；同一低维输入在不同视觉上下文（物体类别/位置）应产生不同动作，故 latent action 嵌入须以视觉状态为条件
- 如何用视觉状态表示条件 latent action 学习，并实现对新物体、新任务的 few-shot 泛化

## 方法
### Latent action 的定义与学习
- 定义：高维动作 a 的低维嵌入 z（本文 z∈R1，1 维），由条件自编码器（CAE）学得：编码器把 (s, a) 联合编码到 z，解码器 φ(z, s): Z×S→A 重构原始高 DoF 动作；是"瞬时动作的降维表示"，非计划、非多帧状态变化；采用 Losey et al. 2020 的 CAE 路线（"latent actions"为伞形术语，含 Jonschkowski & Brock 2015、Lynch 2019）
- 学习：CAE 为 2 隐层 MLP、单节点瓶颈（1 维），L2 动作重构损失；训练状态-动作对来自演示中相邻关节状态差分（固定窗口内）；增强：状态加噪 ε~N(0, σ²)，σ=0.01
- 视觉状态表示三策略（s 为融合状态）：
  1) 端到端 CNN（max-pooling 稠密特征 f；s = sjoint⊕f；与 CAE 联合训练）
  2) localization-only（oracle 给类别 one-hot c，CNN 学定位 g；s = sjoint⊕g⊕c）
  3) 结构化 YOLO-v5（s = sjoint⊕c⊕[x,y]；YOLO-v5 用 75-100 张标注图（每图 2-4 物体）预训练，CAE 训练时冻结）
- 机器人：7-DoF Franka Emika Panda；仿真（Robosuite，MuJoCo）中动作 a∈R7 为关节速度，s' = s + a·dt（dt=1.0）；每 episode 仅第一帧图像送入视觉编码器（持续跑检测无收益且增延迟）

### 与 raw action 的关系
- 解码器 φ(z,s) 直接重建 7-DoF 原始动作；人类 1-D 摇杆输入即 latent action z；模拟遥操作者离散搜索 R1 内使关节空间投影距离最小的 z*（2 个 waypoint，前 k=15s 用第一 waypoint）

### 性质要求
- 论文未显式定义 controllability / consistency 等性质术语；期望性质表述为"平滑、直观"映射与任务成功率；用户研究以 Likert 主观指标（易用、帮助性、直觉性、是否愿再用）度量
- 泛化协议：SEEN（新物体做基础任务，如推麦片盒向南）、NEAR（方向邻近，如向南偏东）、FAR（绕圈一周，未见于训练集）

## 关键发现
- 采样效率（仿真）：YOLO-v5 仅 2 条演示即优于两种 CNN 模型 10 条演示的 final state error；10 演示后端到端模型未接触到物体、localization-only 推不到位，仅 YOLO-v5 平滑推至目标
- oracle 类别并未给 localization-only 带来优于端到端的优势（出乎预期）
- Few-shot：SEEN/NEAR 设置 transfer 模型从 1 条演示起优于 from-scratch；FAR 设置 1 条演示时 from-scratch 大胜，2 条演示起 transfer 反超并保持领先
- 用户研究（9 名参与者，4 女，年龄 24.22±1.6，8/9 有遥操作经验；每策略 4 任务×2 次试验）：YOLO-v5 在所有客观与主观指标上胜 localization-only；后者部分任务 0% 成功率（定位错误所致）
- 末端执行器控制基线（6×1DoF 模式切换）在 SEEN 任务上接近 YOLO-v5，但主观指标全面落后且轨迹锯齿状；YOLO-v5 轨迹平滑自然
- 真实数据：100 张图（90 训练/10 验证）训练 YOLO-v5 检测器；4 基础物体各 9 条演示，unseen 设置各 3 条；仿真：4 物体各 10 条演示、每物体 10 验证任务、30s 时限、YOLO-v5 用 1000 个随机配置预训练

## 证据等级
- 中：L4DC 2021 同行评审；仿真 + 真机用户研究（物理 Franka Emika Panda）；但图 2/图 4 的具体成功率数值未在纯文本中给出，多数结论依赖图表；用户研究样本小（9 人）

## 引用
- S. Karamcheti, A. J. Zhai, D. P. Losey, D. Sadigh. "Learning Visually Guided Latent Actions for Assistive Teleoperation." L4DC 2021, PMLR 144:1–12. arXiv:2105.00580.

## 不确定项
- 各编码策略在仿真与用户研究中的绝对成功率数值见于图 2/图 4，纯文本未列出（具体百分比文中未说明）
- CAE 隐层维度、学习率等训练超参文中未说明
- 仿真 few-shot 中"full set of 5 demonstrations available"（NEAR 结果段）与基础类 10 条/类、真实 3 条/类的收集协议关系文中未说明
- 用户研究各任务逐条成功率与 Likert 具体分数文中未列出
- latent 空间 1 维的设计依据（可解释性 vs 更高维）文中未讨论
