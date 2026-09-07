# A Review of Robot Learning for Manipulation: Challenges, Representations, and Algorithms

## 基本信息
- paper_id: paper_012
- 年份: 2020（arXiv:1907.03146v3，2020-11-06；©2020，CC-BY 4.0）
- 作者: Oliver Kroemer, Scott Niekum, George Konidaris（CMU / UT Austin / Brown）
- 来源: arXiv 综述（cs.RO）；未在文本中标注期刊出处

## 研究问题
- 统一形式化机器人操作学习问题（task family P(M)，Mi=(Si, A, Ri, Ti, γ, τi)，Si=Sr×Sei，环境状态因子化为对象状态），并综述其表征与算法：对象/环境表征（§4）、转移模型（§5）、技能策略学习（§6）、技能刻画（preconditions/effects，§7）、组合与层次结构（§8）。

## 方法
- 综述方法：按"关键概念→统一形式化→五类学习问题"组织，各节均为文献归类，无新实验。
- 轨迹/动作表征相关：
  - 演示在 LfD 中通常表示为状态或状态-动作对的轨迹（§6.4）；演示获取方式含 teleoperation、shadowing、kinesthetic teaching、motion capture（Argall et al., 2009），近年新增 keyframe demonstrations（Akgun et al., 2012）、VR、video（LfO）。
  - 动作空间（§6.1）：joint space vs Cartesian task space；object-relative task frames（Ballard, 1984）；策略还可输出控制器增益。
  - 策略结构谱系（§6.2）：非参数（GP、LWPR）→ 固定参数（NN、lookup、Fourier basis）→ 受限参数（LQR/LQG、DMP、ProMP、GMR）→ goal-based（含"between keyframes"的 spline，Akgun et al., 2012）。
- 分割：§8.2 专述"把轨迹分割成组件技能"，两类方法：(1) 基于技能相似性（policy similarity、value/reward function、HMM 隐技能推断、latent space、pre/post-condition similarity）；(2) 基于特定事件（salient sensory events、mode transitions）。
- skill / motion primitive：技能用 options 框架表示 o=(Io, βo, πo)（§3、§8.1）；motion primitive 层面包括 DMP（spring-mass-damper + 非线性形状项）、ProMP、GMR（§6.2）；§8.3 在线发现技能（含 skill chaining：技能以到达显著事件或另一技能 precondition 为目标构建）。
- 对齐：无专门时间对齐章节；相关概念零散出现——IRL 中避免 correspondence problem（§6.4）、跨域/跨视角对应（point-level correspondences §4.1.2；Time Contrastive Networks、first/third-person 视角对应 §6.5）。
- 演示预处理：无专门"preprocessing"章节；最接近的论述是 §4 的感知与特征学习/选择，以及 §8.2 的分割（属后处理）。

## 关键发现
- 演示轨迹的分割是构建技能库的主导途径之一，但"如何识别技能"被作者称为困难且欠定义的问题（§8.1）。
- 与事后分割相比，"边解题边发现技能"的成功方法明显更少，因问题本质更难（§8.3）。
- 作者将关键帧仅作为演示获取形式之一提及（§6.4），未展开定义、选取或与其他演示形式的对比。
- 综述末章列出开放挑战：安全学习、探索、多模态融合、样本复杂度等（§9）。

## 候选相关性（总体）
- 总体相关性: medium
- 理由: 二次文献（综述），无自身实验数据；但系统覆盖 LfD 中的轨迹表征、演示获取、轨迹分割、技能/运动原语，是调研问题的分类框架来源而非直接证据。

## 引用
- 演示=状态或状态-动作对轨迹；keyframe demonstrations：§6.4（p.27）
- 动作空间与 task frames：§6.1（p.21）
- 策略谱系与 keyframe spline、DMP/ProMP/GMR：§6.2（pp.22–24）
- options 形式化：§3（p.7）与 §8.1（p.40）
- 轨迹分割两类方法：§8.2（pp.40–42）
- skill chaining：§8.3（p.42）
- pre/postconditions 的命题/谓词表征与学习：§7.1–7.2（pp.35–37）
- 对象表征层次（point/part/object）：§4.1.2（pp.11–12）
- 状态空间因子化：§3（pp.6–7）

## 不确定项
- 未讨论时间对齐技术（如 DTW）在演示预处理中的角色——对应部分仅在跨域/跨视角对应中被旁及。
- "演示预处理"（重采样、滤波、去噪、重标定）无系统性论述。
- 关键帧无定义、无选取方法对比，仅作为演示形式被引述（Akgun et al., 2012）。
- 作者自述覆盖"representative subset"，非穷尽；各方法无定量对比。
- 成文于 2020 年，不含后续动作表征形式（如 action chunking、flow matching）与神经关键帧选择；与现代 VLA 时代的衔接需另证。
