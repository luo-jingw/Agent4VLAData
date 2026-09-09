# Universal Actions for Enhanced Embodied Foundation Models

## 基本信息
- 论文：UniAct—Universal Actions for Enhanced Embodied Foundation Models（paper_042）
- 作者：Jinliang Zheng，Jianxiong Li，Dongxiu Liu（共同一作），Yinan Zheng，Zhihao Wang，Zhonghong Ou，Yu Liu，Jingjing Liu，Ya-Qin Zhang，Xianyuan Zhan（通讯）
- 单位：清华大学 AIR、SenseTime Research、北京大学、北京邮电大学、上海 AI Lab、BAAI
- 出处：arXiv:2501.10105v2（cs.RO），2025-03-08；据第三方引注（MOTIF 参考文献）最终发表于 CVPR 2025，pp. 22508–22519
- 项目页：https://2toinf.github.io/UniAct/

## 研究问题
- 问题：跨领域机器人数据的动作异质性——(1) 不同本体（自由度/臂-四足-车）动作空间完全不同；(2) 控制接口不同（EEF 位置 vs 速度控制器、关节位置）使动作命令物理含义不同；(3) 同一平台不同操作者的行为多模态性——使动作数据驻留在互不相交的流形上，阻碍跨域数据共享，难以构建通用具身基础模型。
- 现有做法缺陷：把不同动作空间等同处理（相同离散化/归一化 → 可能冲突的动作空间）；或朴素拼接所有动作空间（需大量人工工程、未利用内在结构联系）。
- 目标：建立"通用动作空间"（Universal Action Space）U ∈ R^{N×D}（N=256 个码、D=128 维），离散码本中每个码编码不同机器人可执行的通用原子行为，不受具体本体机制与控制接口约束。

## 方法
- 通用动作定义（U 的性质）：所有由异构控制信号驱动的运动，都能蒸馏为共享的潜在原子行为（如"向前移动"）；优先探索离散空间（利用 LLM/VLM 与 VQ-VAE 的离散推理能力）。
- 通用动作提取：共享 VLM（LLaVA-OneVision-0.5B）微调为提取器，输出 p(u|o,g)——给定观测 o 与任务目标 g（语言指令）选择动作 u 的概率；推理取 u*=argmax_u p(u|o,g)；训练用 Gumbel-Softmax 类别重参数化 w_i = exp((log p(u_i|o,g)+ε_i)/τ)/Σ_k exp(...)，温度 τ 训练期渐进衰减（STE 因码本严重坍塌被弃用）。关键设计：提取器面向"任务进展"而非"观测变化"——两类监督信号（视觉动态变化、观测间隔）导致噪声潜在动作空间；通用动作与控制信号直接因果关系（新物体出现不改变动作）。
- 异构解码：K 个训练域各配轻量 MLP 解码头 h_k：â(k) = h_k(u*, o)，输入 u*（共享码本）+ 共享视觉主干（ImageNet 预训练 ResNet18，224×224）的视觉特征 o；刻意轻量（防过拟合目标域，把学习主要放在通用动作上）。
- 训练目标：min_{U,θ} Σ_{k=1..K} E_{τ_i~D_k}[ L_k(â(k), a(k)) ]；L_k 按动作标签类型定制（离散：交叉熵；连续：MSE/Huber/扩散损失）；码本 U 与提取器每步联合更新，各域头仅在该域 batch 上更新（类元学习全局共享+域特定组件）。
- 数据：28 个异构本体、约 100 万条演示（OXE、Libero、Droid 及 Bridge 等；统一第三人称视角与语言指令、保留动作异质性；按数据集设定采样率）；64×A100 + DeepSpeed，10 天；batch 1024、lr 2e-5、500K 迭代、BFloat16（VLM 384×384，ResNet18 224×224）。

## 关键发现
- 主结果（真机 WidowX，19 任务、每任务 10 trials、共 190 rollouts，5 轴：视觉/运动/物理/语义泛化 + 语言接地；各轴均分 Octo/OpenVLA-7B/UniAct-0.5B）：视觉 2.4/6.6/6.8、运动 3.0/3.5/6.0、物理 1.0/6.0/7.0——三轴超 14 倍大的 OpenVLA-7B 与 LAPA-7B；语义泛化 1.9/4.8/3.9、语言接地 4.8/8.8/7.3（两轴低于 OpenVLA，解释为 0.5B 骨干语义能力有限）。LIBERO 仿真（130 任务、5 套件、每任务 20 trials）：全部套件超过基线；UniAct 对噪声数据鲁棒（OpenVLA 微调动作精度难收敛）。
- 跨本体/跨接口快速适应（新本体 AIRBOT，训练未出现；4 种控制接口：相对/绝对 EEF 位置、相对/绝对关节位置；100 demos；任务"叠小方块"易/难两版；冻结码本+提取器、从头训练 4 个 MLP 解码头）：所有接口均超基线；微调参数量占比 UniAct 4M/500M=0.8%（vs OpenVLA 97M/7000M=1.4%、Octo 2M/100M=2%），4×A100 约 1 小时。
- 跨任务/机型复用（先进解码头）：双机械臂 AIRBOT + ACT 头（顶视 + 2 腕视 + 本体感受，4 任务 × 250 demos）：Sweep plate 45 vs LLaVA-OV-0.5B 7.5、Fold towel 62.5 vs 20、Cup on plate 50 vs 2.5、Transport pen 65 vs 15；ACT 头 + LIBERO 结果进一步提升。
- 通用动作空间的意义（对齐/复用证据）：人工检查全部 256 个通用动作，≥40% 在不同机器人上解码出"完全一致"的行为；JS 散度——同任务跨本体低（pick bowl：WidowX−Franka 0.34；open drawer：0.58–0.60）、跨任务高，说明提取器正确利用共享原子行为而聚焦任务进展；可用通用动作 ID 直接控制机器人（无需运动学知识），可作为未来具身模型的动作 tokenizer。
- 局限：主要在单臂机器人上评估（控制接口多样）；解码头为统一轻量 MLP（双机械臂需要更复杂头，ACT 头为变通）。

## 证据等级
- 较高（CVPR 2025 正式论文）：百万级规模（约 1M 演示）、28 本体；含定性（256 个动作人工目检）与定量（JS 散度表）验证；但训练数据中含评估本体 WidowX 与 Franka（其头为"预训练"而非"未见"），"未见"仅限 AIRBOT；WidowX 各轴为 10 trials 小样本；基线与 UniAct 数据同源但基线做了数据清洗（相对 EEF 位置化、去除关节位置动作），UniAct 未清洗。

## 引用
- 基线/并排：OpenVLA [31]、Octo [61]、CrossFormer [17]、LAPA [69]、BeT [55]、VQ-BeT [34]、QueST [45]、π0 [8]、RDT [41]、Yang et al. [68]（隐式对齐工作）、Genie [11]、IGOR [4]、LAPO [54]（视觉-状态对齐型潜在动作）。
- 数据：Open-X Embodiment [49]、Libero [38]、Droid [30]、BridgeData V2 [64]、RH20T [20]。

## 不确定项
- 文中未说明：通用动作的时序粒度（每时间步一个码？码与动作序列如何对应）；p(u|o,g) 的输出 token 化形式与训练细节（附录 A 正文未展开）；Gumbel 噪声 ε 的分布细节与 τ 的衰减计划数值。
- 边界：语义/语言两轴 UniAct 低于 OpenVLA 7B 且差距不小（3.9 vs 4.8；7.3 vs 8.8），论文未做更大规模验证；LIBERO 上不同编码（UniAct 用 EEF 位置）与基线微调数据的对齐程度未量化。
- 文中未说明：JS 散度表所用"任务"的具体构成（P=pick up the bowl、O=open the drawer 出自图注）；"50-shot"? 无；AIRBOT 结果的具体成功率数值（图 5 仅曲线，正文未给数字）。
