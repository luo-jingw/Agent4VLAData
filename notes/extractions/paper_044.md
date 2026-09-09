# CLASS: Contrastive Learning via Action Sequence Supervision for Robot Manipulation

## 基本信息
- 论文：CLASS: Contrastive Learning via Action Sequence Supervision for Robot Manipulation（paper_044）
- 作者：Sung-Wook Lee，Xuhui Kang，Brandon Yang，Yen-Ling Kuo
- 单位：University of Virginia
- 出处：arXiv:2508.01600v1（cs.RO），2025-08-03，v1 预印本，未标注发表状态
- 关键词：Robot Manipulation, Action Chunking, Supervised Contrastive Learning, Vision Representation Learning

## 研究问题
- BC 在异构演示数据（相机位姿、物体外观漂移）下过拟合单条演示、无法捕获跨演示共享结构，泛化差。
- 目标：学习行为学意义上的视觉表示，使"未来行为相似"的观察序列在隐空间中聚拢，同时支持检索式（非参数）控制与策略微调。
- 与 RQ4.1 的关系（本文核心）：非跨模态对齐——CLASS 用动作序列相似度作为弱监督信号驱动"视觉-视觉"对齐；动作序列不进编码器，只定义正/负对与软权重。文本不参与对齐（弱项声明："current scope of the work is limited to vision modality"）。

## 方法
- 表示定义：观察序列 Ot={o(t−To−1),…,o_t}，动作序列 At={a_t,…,a_(t+Tp−1)}；编码器 f_θ（ResNet-18 + Spatial-Softmax，批量归一化改为分组归一化）映射 Ot→z，L2 归一化。
- 阶段一（正负对构成）：全数据两两计算动作序列 DTW 距离（aeon 实现；DTW horizon=16），DTW(A,B)=min_Γ Σ_{(p,q)∈Γ}‖a_p−a_q‖₂（单调对齐、欧氏距离）；正对 = DTW 距离 < 固定分位数阈值 K（任务级：Square 2.5%、Three-Stack 1.0%、Aloha-Transfer 2.5%、LIBERO-Object 2.5%、Push-T 1.5%）；其余样本为负。图 2 示例：正对 DTW=0.02、负对 DTW=0.56。
- 阶段二（Soft InfoNCE）：每个观测作 anchor（增强=随机裁剪+高斯噪声）；soft 正权重 w_ij=1−CDF(DTW(i,j))（正对）、否则 0；S_ij=ẑ_iᵀẑ_j/τ，τ=0.05；p_ij=exp(S_ij)/Σ_{k≠i} exp(S_ik)；L_CL=−(1/B)Σ_i Σ_j w_ij log p_ij / Σ_j w_ij。与硬对比（权重恒 1）相比用相似度加权正对；与已有软对比法（加权负对）不同——本文加权正对。
- 训练两种模式：仅表示学习；或两阶段（CLASS 预训练 + BC 微调 encoder+policy head，vision encoder lr=1e−5、policy head lr=1e−4）。
- 检索式控制（非参数）：最新观测编码→余弦相似度 kNN（knn=64；τ_nn 任务给定 0.001/0.002 或 0.01/0.02）→exp(c_i/τ_nn) 加权平均 kNN 动作序列；ˆz 为 proprioception 与图像特征拼接后 L2 归一化，无需 policy head。
- 理论：Proposition 1——L_CL 等价于优化 soft 权重分布 q_ij 与隐空间相似度分布 p_ij 之间的 KL 散度 D_KL(Q‖P)（附录 A 证明）。
- 超参：batch 160、LARS、lr 0.4、weight decay 1e−6、momentum 0.9、cosine schedule、EMA power 0.75、梯度裁剪 0.5；预训练 epoch 10–100（按任务）。

## 关键发现
- 5 仿真（Square 200 演示、Three-Stack 1000、Aloha-Transfer 100、LIBERO-Object 10 任务×50、Push-T 206）+ 3 真机（Two-Stack/Mug-Hang/Toaster-Load 各 200 演示）。
- 平均成功率（静态+动态）：CLASS 85%（MLP）、91%（DP），最优基线 63%/77%；动态相机+随机颜色：76%（MLP）、85%（DP），基线 32%/57%。非参数（Rep-Only）：平均 83%，仅比参数化 DP 低 9%（表 1）。
- 摘要：显著视觉漂移下 Diffusion Policy+CLASS 预训练平均 75% 成功率，其余基线无竞争表现。
- 设计选择消融（Square Rep-Only）：soft>hard（去除软权重性能显著下降）；DTW 滑窗增大至 T=16 时成功率提升（此后饱和）；DTW>L2（静态与动态均如此）；K 权衡——小阈值信息受限、大阈值假正增多。
- 真机（每格非参数/参数）：CLASS-DP 优于 ImageNet-DP 所有任务——非参数 +37% 子任务、+45% 完成；参数化 +41%/+55%（Mug-Hang Loaded 0.00/0.05→0.35/0.55；Two-Stack 0.15/0.30→0.70/0.80）。
- 推理延迟（A40）：Rep-Only 5.5 ms、MLP 7.3 ms、DP 84.4 ms；数据规模 20–1000 演示（Three-Stack）呈 scaling law，CLASS 各规模均超 BC。
- 失败模式：真机最常见为未完成当前子任务即进入下一区域；相机超出采集区域时性能显著下降。

## 证据等级
- 实证水平：单篇自报 arXiv v1 预印本；5 仿真+3 真机任务、8 类基线（Random/ImageNet/R3M/TCN/VINN/DynaMo/EquiVar × MLP/DP）、设计消融（soft/hard、窗长、度量、K）与附录扩展实验（腕相机、颜色增强、ImageNet 初始化、scaling）；每单元 50 个随机场景（LIBERO 20 次×3 seed）；无第三方复现、无发表记录。

## 引用
- 基线/对比：Robomimic [31]、MimicGen [32]、Diffusion Policy [3]、ACT [4]、VQ-BET [5]、VINN [25]、R3M [17]、TCN [16]、DynaMo [37]、EquiVar [38]；DTW 工具 aeon [27]；Soft contrastive [21–24] 与 Supervised Contrastive [28]、Soft Neighbors [29]。

## 不确定项
- 摘要"75%"与表 1 逐任务数字的核算关系文中未说明（表 1 动态列非参数项各有 0.64–1.00，无简单对应说明）。
- CDF 的具体拟合/实现细节给出公式后指向"见 Appendix B"，正文未重复（附录 B 只说明用 aeon 与仅对 anchor-positive 对预计算，未给出 CDF 拟合参数）；文中未说明。
- 文本/语言模态：LIBERO-Object 用 BERT 编码语言指令，但 CLASS 编码器不消费文本，该指令与 CLASS 表示的关系文中未说明。
- 附录 F/G 含数值（腕相机 68%→90%、颜色增强 72%→90%）在正文未引用；滑动窗口 T=16 与 DTW horizon 16 两者的确切关系文中未说明。
