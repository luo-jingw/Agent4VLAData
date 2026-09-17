# Learning Latent Plans from Play（Play-LMP）

## 基本信息
- 作者：Corey Lynch, Mohi Khansari, Ted Xiao, Vikash Kumar, Jonathan Tompson, Sergey Levine, Pierre Sermanet（Google Brain）
- 会议：3rd Conference on Robot Learning (CoRL 2019)，Osaka, Japan
- arXiv：1903.01973v2（cs.RO），2019-12-20
- 项目页：learning-from-play.github.io

## 研究问题
- 能否用未标注的人类遥操作 play 数据自监督学习任务无关控制（task-agnostic control）：从任意当前状态 sc 到达任意目标状态 sg，任务由 (sc, sg) 对索引
- 多模态问题：同一 (sc, sg) 对存在多种有效高层行为，单峰 goal-conditioned 行为克隆难以拟合；play 数据比专家演示便宜且覆盖更广

## 方法
### Latent action 的定义与学习
- 定义：latent plan（隐计划）z，连续高斯隐变量（对角协方差），代表连接 sc 与 sg 的一个高层行为计划；不是单步状态变化，而是生成整条长度 κ 的状态-动作序列 τ 的"计划"（"unobserved plans generate observed goal-directed behavior"）；对应生成模型中 z ~ p(z|sc,sg) 生成 τ
- 学习框架：条件 seq2seq VAE（CVAE），三组件端到端：
  1) 计划识别（plan recognition）Venc：双向 RNN 序列编码器，输入整条 τ，输出 qφ(z|τ) 的 μφ、σφ（对角高斯，重参数化采样）
  2) 计划提议（plan proposal）CGenc：前馈网络，输入编码后的 (sc, sg)，输出条件先验 pθ(z|sc, sg) 的 μψ、σψ
  3) 计划与目标条件策略 πLMP(at|st, sg, z)：随机 RNN 解码器
- 目标：L_LMP = Lπ + β·LKL；LKL 为 qφ(z|τ) 与 pθ(z|sc,sg) 的 KL 散度（迫使提议分布覆盖实际发生的计划）；Lπ 为动作重构最大似然；β<1（β-VAE 思想）防 posterior collapse
- 维度：隐计划 256 维（出自架构图，正文未文字说明）；窗口 κ=32；控制 30Hz，每 κ 步重采样计划（约 1Hz 重计划）；测试时 Venc 丢弃，仅用 CGenc 采样一次计划后闭环解码

### 与 raw action 的关系
- πLMP 既是表示学习的解码器又是控制策略；条件 (st, sg, z) → 8-DOF 动作分布（每维 256 bins 的 discretized logistic 混合，MODL）；RNN 2 隐层各 2048
- 隐计划起"多模态分解"作用：把多模态策略问题转为以 z 为条件的单峰执行问题

### 性质要求
- 论文未显式定义 controllability / consistency / semantic 对齐等术语；显式动机仅"可复用、把多模态转单峰"；语义对齐只有实证（见关键发现）；基线与 Play-GCBC 对比即验证"解耦计划推断与解码"的价值

## 关键发现
- 状态输入 18 任务平均成功率：Play-LMP 85.5% > Play-GCBC 77.9% > BC 70.3% > Multitask BC 66.2%；像素输入：Play-LMP 69.4%±10.8、BC 66.5%±12.1、Play-GCBC 58.7%±11.6（3 种子 CI）
- 仅 30 分钟 play 数据 Play-LMP 71.8%，超过 90 分钟专家演示训练的 18 个 BC 策略（70.3%）；单任务提升最高约 50 个百分点
- 数据覆盖：同收集时长下 play 覆盖的交互空间区域数是 18 任务专家演示的 4.2 倍、随机探索的 14.4 倍；play 覆盖随收集时长近似线性增长（测至 7 小时）
- 鲁棒性：play 训练模型对初始位姿扰动（0-0.4m）显著更鲁棒，专家 BC 随扰动快速退化
- 涌现重试：Play-LMP 失败后自发重试直至成功（例：关门任务 3 次尝试），专家演示训练的模型不出现该行为
- 无监督任务发现：512 个随机 play 窗口 + 验证任务演示的 t-SNE 显示隐计划空间按功能任务组织（抽屉区域、按钮区域等），训练中从未使用任务标签
- 环境：8-DOF 仿真机器人 + 桌面（滑门、抽屉、方块、3 按钮灯），18 个视觉操作任务；play 至多约 7 小时，专家演示每任务 100 条共 1800 条（约 1.5 小时）

## 证据等级
- 中：CoRL 2019 同行评审会议论文；但仅单一仿真环境（playground），18 任务，无真机验证；像素实验为 3 种子 CI、数据消融为 20 rollout 95% CI，状态实验主表仅单点无 CI

## 引用
- C. Lynch, M. Khansari, T. Xiao, V. Kumar, J. Tompson, S. Levine, P. Sermanet. "Learning Latent Plans from Play." CoRL 2019. arXiv:1903.01973.

## 不确定项
- 状态实验成功率无置信区间（表 6a 仅单点值）
- 隐计划维度 256 出自架构图（Fig. 9），正文未文字说明
- β 的具体数值文中未说明（仅"β<1"）
- 状态实验模型最多用 180 分钟 play（Fig. 8 说明），与像素实验 7 小时的各曲线点对应数据量细节文中未逐点列出
- 多模态假设（z 条件后变单峰）无直接量化验证，仅以整体成功率间接支持
