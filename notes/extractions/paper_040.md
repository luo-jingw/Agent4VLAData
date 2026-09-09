# Cross-Embodiment Transfer via Behavior-Aligned Representations

## 基本信息
- 论文：Cross-Embodiment Transfer via Behavior-Aligned Representations（paper_040）
- 作者：Ajay Sridhar，Jensen Gao（共同一作），Jonathan Yang，Jean Mercat，Suneel Belkhale，Dorsa Sadigh
- 单位：Stanford University、Toyota Research Institute
- 出处：arXiv:2607.27549v1（cs.RO），2026-07-30，预印本（IEEE 会议论文格式，未标注发表状态）
- 项目页：https://ajaysridhar.com/barx/

## 研究问题
- VLA 模型中"行为对齐表征"（对象边界框、语言动作、末端执行器轨迹）能否促进跨本体迁移并如何影响迁移。
- 假设：拥有跨本体不变性且能预测动作的表征，可形成跨本体不变的策略推理空间，通过"隐式对齐"统一大规模异构跨本体数据。
- 对齐什么：不对齐观测空间或动作空间，而是对齐行为表征——语言动作（运动语义描述，如"move left and down"）、2D 末端执行器轨迹（机器人未来运动在图像中的像素位置序列）、对象边界框。形式化要求：表征可由数据 D 计算、包含预测最优行为所需信息、对 R 中本体不变（同一任务不同本体表征应相似）。
- 动机：显式对齐观测（相机位姿、inpainting、掩码）或动作空间（专用 action head）需人工努力或部署期本体知识，难以规模化。

## 方法
- 设定：语言条件跨本体模仿学习，πθ(a|o,l)，o∈Or，a∈Ar；每个观测标注表征元组 z=(z(1),…,z(K))；训练时按分布 prep(z̃) 采样表征子集条件化动作，同时训练预测表征。
- 损失形式：L_total(θ) = E_{(o,z,a,l)~D, z̃~prep(z)}[ ℓ(πθ(·|o,l,z̃), a) + ℓ_aux ]，其中 ℓ_aux = Σ_{k=1..K} λ_k·ℓ_rep(k)(πθ(·|o,l), z(k))；实现中 λ_k=1。无显式"对齐损失"——对齐是隐式的（相同表征预测目标 + 动作条件化）。
- 实现：策略为 MiniVLA 架构（从预训练 VLM 初始化，无机器人预训练），单第三人称相机视点；表征与动作自回归地作为单一文本序列预测，用不同文本提示指示应预测的表征子集。
- 表征标注：边界框=现成管线（VLM 场景描述 + Grounding DINO）；语言动作=本体感受状态变化阈值化（类似 RT-H）；EE 轨迹=预训练 LLARVA 模型检测的未来 2D 位置；仿真实验用特权真值标签。
- 融合变体：No Reps（仅动作）／Single Rep／ECoT（边界框→EE 轨迹→语言动作→动作 顺序链）／Joint Reps（单模型多表征，均训练为"预测某一表征或仅预测动作"）；推理时只预测动作。

## 关键发现
- 基准 RoboCasa-X（基于 RoboCasa）：3 个原任务（PnP Counter to Sink、PnP Sink to Counter、Turn On Sink Faucet）+ 新任务 Flip Mug Upright；源本体 IIWA、Kinova3、UR5e（MimicGen 生成；XP-900=每任务 900 条、XP-3K=每任务 3000 条）；目标本体 Panda、Jaco（Robotiq 2F-85）、Panda-OG（原装 Franka Hand），每任务 50 条人类演示；100 rollouts/评估，取 3 个 checkpoint 最高成功率。仿真中所有本体动作空间与控制器相同（delta 笛卡尔末端执行器位姿控制）。
- (Q1) 每种表征单独使用均优于无表征；EE 轨迹一般最有效，其次语言动作，最后边界框。ECoT/Joint Reps 融合在符合 Robotiq 2F-85 的 Panda、Jaco 上比单独表征更好；但 Panda-OG（末端执行器未见）上仅 EE 轨迹更优——猜测最佳表征选择取决于本体差异。
- (Q2) 表征收益随先验数据规模增长：无先验 +5%、XP-900 +15%、XP-3K +19%（5-shot 对比）；唯一例外 Turn On Sink Faucet（无先验 +22%、XP-900 +4%、XP-3K +1%，猜测因任务变体少、易饱和）。Panda/Jaco 上 XP-3K+表征可达或超过同本体先验 SP-900；Panda-OG 显著落后（本体差距大）。
- (Q3) 推理时先预测表征再动作 vs 直接预测动作：影响总体不显著（印证 concurrent work [40]），表征主要在训练期起隐式作用。
- (Q4) 无动作先验（action-free，仅预测表征）：整体比从零学 +14%，比"带动作但无表征的完整先验"+11%；略低于带动作+表征的先验；对 Panda-OG 无效（Panda 与 Jaco 有效）。
- 真实世界（目标本体 FR3、ViperX 300 S；PnP 任务变体；XP-3K 预训练 + 50 demos 适配；10–30 rollouts/对）：Joint Reps 使带跨本体先验模型的任务进度（task progress）整体提升 28%；仅目标数据时 +8%；无表征时先验数据仅 +7%。真实收益大于仿真（观测与动作域差更大：控制频率不同、阻塞/非阻塞控制不同）。
- 局限：设置中各本体高度对齐（相同相机姿态分布、场景、任务），使表征更易对齐；表征偏向物体中心操作任务。

## 证据等级
- 实证证据等级较高：仿真基准（100 rollouts/评估 × 3 checkpoint）+ 2 个真实机器人平台；但"提升占比"以柱状图/任务进度为主，逐任务精确值经图像提取不可靠。文章为 arXiv 预印本（2026-07-30），无正式发表记录；正文未披露建模的可变性（如相机随机化程度在真实任务中被降低：真实任务只用一个厨房、4 种对象类型/1 种）。

## 引用
- [16] RT-H（语言动作）、[17] RT-Trajectory、[18] LLARVA（EE 轨迹）、[30] ECoT 链式推理、[40] Chen et al. Training Strategies for Efficient Embodied Reasoning（concurrent；表征收益主要来自训练）、[42] Grounding DINO、[43] MimicGen、[19] RoboCasa、[25] MiniVLA、[9] 泛化评估分类 RA-L 2026（同组）、[35] CoT-VLA 等其他中间表征工作。

## 不确定项
- 文中未说明：Joint Reps/ECoT 中 λ_k 是否按表征类型区分（文中 λ_k=1）；文本提示的具体格式；推理时预测表征引入的延迟量化数值。
- 文中未说明：真实世界每个数值的 rollout 数（10–30 范围内，未逐项给出）；action-free 预训练的 epoch/步数。
- 数据边界：仿真表征用特权真值标注，真实用 off-the-shelf 管线标注，两路径的表征质量差异未比较；Q2 中 "+5%/+15%/+19%" 为"Overall"柱状图总体值，逐任务数值未文本化。
