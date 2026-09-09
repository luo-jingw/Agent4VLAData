# FlowWAM: Optical Flow as a Unified Action Representation for World Action Models

## 基本信息
- 作者：Yixiang Chen*, Peiyan Li*, Yuan Xu, Qisen Ma 等（中科院自动化所 NLPR、UCAS、FiveAges、MBZUAI、阿里巴巴）；*共同一作
- 发表：Preprint，arXiv:2607.13017v1 [cs.RO]，2026-07-14（未注明会议/期刊稿件状态）
- 类型：方法论文（World Action Model，WAM；光流动作表征，中文/世界模型）

## 研究问题
- WAM 复用预训练视频生成器做世界建模 + 动作预测，核心开放问题：如何用"与预训练视频生成器对齐且携带足够运动线索"的形式表示动作
- 现有方案三分缺陷：数值动作 token 精确但跨形态不可迁移；隐式潜动作（frame-transition 学习）抽象、丧失稠密空间锚定的运动线索；图像空间动作（ray map/embodiment mask/多视角 action image）是"静态空间线索"——只指示动作发生在哪，而非每个可见部件如何跨帧移动，故只是注入型的 frame-level 条件而非与时序演变共生的动作表征
- 对跨数据/跨本体对齐的意义：光流可直接从无动作标签的原始视频提取，提供"视频原生的统一动作表征"，是跨本体（人类视频→机器人）、跨数据（unlabeled→labeled）对齐的另一条路径

## 方法
- 统一动作表征：相邻帧光流 f_t ∈ R^{H×W×2}（逐像素位移 (u,v)）→ HSV 色轮编码 F_t（式 1：H=atan2(v,u)+π/2π 方向→色相，S=‖f_t‖/m 幅值→饱和度，V=1）；编码可逆 φ⁻¹ 恢复数值流场；与 RGB 帧格式一致，可直接用同一 VAE 编码器与视频生成器处理
- 双流架构：RGB 与 flow 流用同一冻结 VAE 编码为潜在 z 与 z_f，仅 patch embedding 层与输出头为流特定，transformer 块全共享；每层 self-attention 内拼接 RGB/flow token 联合注意力，各流独立 RoPE
- 两模式（同一生成器）：policy 模式——双流均从高斯噪声去噪，jointly 生成未来 RGB+flow，action expert 从逐层隐藏状态解码动作；world-model 模式——flow 潜在固定为期望运动轨迹（clean flow 条件），仅 RGB 从噪声去噪，渲染遵循指定运动的未来视频
- Action Expert：AdaLN DiT，约 780M 参数，30 层（与 Wan2.2 视频 DiT 同深）、hidden 1024、16 头、FFN 4096；cross-attend 融合 RGB+flow 逐层隐藏状态，条件含 T5 指令 + 14D 归一化关节位置 qpos（proprioceptive token），flow-matching 目标预测 N=32 步动作 chunk
- 训练两阶段：Stage-1 action-free（EgoDex 无标签人类操纵视频，仅 L_video，LR 5×10⁻⁵，只更新双流 DiT）；Stage-2 联合（RoboTwin 2.0，附 action expert，L = L_video + λ_a L_action，LR 1×10⁻⁴，λ_f=0.1、λ_a=1.0、α=2.0、ref-aug strength 0.1）
- 关键机制：动作专家训练时 p=0.5 比例对潜在加噪 σ~U[0,1] 并传噪声级嵌入（式 2），对齐 clean/generated latent 分布；流动感知重加权 w_motion（式 4）对抗静态背景主导；条件首帧加小噪声模拟自回归残差
- 数据：RoboTwin 2.0 50 双臂任务（Clean 50 演示/任务、Random 500/任务）；flow 用 RAFT 提取，RoboTwin 用 SAPIEN 回放机器人动作渲染"robot-only"视频再提取（剔除背景/物体运动）；HSV 幅值 cap 25px、<0.5px 置零；多目 T 形拼接 320×384（头 320×256 + 左右腕下采样），腕部流区域用常量占位
- 数值：base 为 Wan2.2-TI2V-5B（UMT5-XXL + Wan2.2 causal VAE 冻结）；推理 video/action 去噪步 25/50，execute-and-replan 窗口 25；32×H100 训练；T_pixel=9 帧 → 32 步动作（时序 stride 4）

## 关键发现
- RoboTwin 2.0（平均值，50 任务×100 rollout）：FlowWAM w/ PT 92.94% / 92.14%（Clean/Random），w/o PT 82.40% / 80.80%；基线：π0.5 42.98/43.84、X-VLA 72.88/72.84、Motus 88.66/87.02、GigaWorld-Policy 86.36/85.04、X-WAM 89.76/90.68、Fast-WAM 91.88/91.78
- WorldArena（EWMScore = 16 指标均值）：FlowWAM 63.71（最佳）；次佳 GigaWorld-1 62.34、ABot-PhysWorld 62.63；Traj. Acc. 64.26 vs 次佳 54.27（相对提升 18.4%，论文摘要口径）；Depth Acc. 98.97、JEPA 97.14 均最佳；SC/BC 89.97/82.46 与最佳基线相近；传统视频基础模型（CogVideoX 58.79、Wan 2.6 59.80）低于 action-conditioned 模型
- 真实机器人（100 条远程演示/任务、10 次随机位姿）：FlowWAM 平均 75.7% > π0.5 61.4% > Motus 57.1%；7 任务全最优；双臂差距更大（bimanual 协调受益于 flow 直接在图像空间表示耦合臂运动）
- 消融（policy，custom validation split，与表 2 口径有差异）：数值动作 69.8%、raw (u,v) 72.3%、w/o flow-loss 重加权 83.9%、w/o 随机 AE 条件 82.1%、FlowWAM 完整 89.8%
- 消融（world mode 条件化，EWMScore）：Text only 49.31、数值动作 54.18、raw Flow actions (u,v) 56.72、图像动作(masks) 57.84、FlowWAM 完整 65.23（注 1：官方 WorldArena 需远程提交，消融用自定义验证 split，故与表 2 数值存在差异）
- Flow 可解码性：50 任务 per-task 预测流误差（vs RAFT 伪真值）与成功率 Pearson r = −0.81（策略增益随流预测质量而非解码捷径）
- 结论性表述（原文）：flow 流本身贡献大部分增益，无标签预训练是放大器；Random 场景增益大于 Clean

## 证据等级
- 实验性：仿真基准（RoboTwin 2.0 全 50 任务、WorldArena 16 指标）+ 真实机器人双平台（Franka、ARX 5）+ 逐组件消融 + 流误差-成功率相关性；但为 preprint、未经同行评审；消融与主表评测口径不同（自定义 split）；单次评估协议（100 rollout/任务）

## 引用
- 同线工作：DreamZero（Ye et al. 2026）、UWM（Zhu et al. RSS 2025）、CoVAR、Motus、Cosmos Policy、Fast-WAM、X-WAM、GigaWorld-Policy；latent action（Latent Action Pretraining, 2025）；像素运动系统 LangToMo、DAWN、ATM；flow 中间量：Flow as Cross-domain Manipulation Interface（Xu et al. CoRL 2025）、EC-Flow（ICCV 2025）、FLIP、Future optical flow prediction（Ranasinghe et al. 2026）
- 依赖：π0、X-VLA（对比）；RAFT（流估计）；Wan 2.2（基础生成器）；EgoDex（无标签预训练数据，arXiv:2505.11709）；RoboTwin 2.0、WorldArena 基准；SAPIEN/MUSIQ/V-JEPA/SAM3（评测工具）

## 不确定项
- 消融与主表口径差异的具体幅度（文中仅以脚注说明，未给出对照验证）；Hyperparameter 中无 780M 之外的总参数量（video generator 5B 不动用 VAE/文本编码器，实际可训练参数未说明）
- 文中未说明：FlowWAM 与其他方法在"无标签数据规模/量"上的对等性（EgoDex 具体用量（片段数/时长）正文未给）；WorldArena 上"18.4% 相对提升"在正文中未显式复述（只给 64.26 与次佳 54.27）
- 文中未说明：真实机器人 7 任务中 wrist flow 常量占位对效果的影响；跨本体对齐（人类视频→机器人）带宽度的上限（仅 EgoDex 一个来源）
