# Non-Markovian Long-Horizon Robot Manipulation via Keyframe Chaining

## 基本信息
- paper_id: paper_011
- 年份: 2026（arXiv:2603.01465v1，2026-03-02）
- 作者: Yipeng Chen, Wentao Tan, Lei Zhu, Fengling Li, Jingjing Li, Guoli Yang, Heng Tao Shen（Tongji University 等）
- 来源: arXiv preprint（cs.RO）；代码 https://github.com/cytoplastm/KC-VLA

## 研究问题
- 现有 VLA 依赖"稠密观测"范式（单帧或短稠密窗口），隐含马尔可夫假设，在存在状态别名（state aliasing）的长时程任务上失效；加长上下文受注意力二次复杂度限制。
- 提出 Keyframe-Chaining VLA：从连续观测中自动提取语义关键帧，构成稀疏语义历史，将 π(At|o0:t) 近似为 π(At|ok1,…,okn,ot)，以可忽略计算开销覆盖全局时间感受野。

## 方法
- Keyframe 定义（§4.2）：长时程任务建模为严格有序的语义阶段序列；keyframes = 标记每个阶段完成的 canonical states；N 阶段任务对应 N 个 GT keyframes；第 k 个 keyframe 的检测是 phase k→k+1 的 transition trigger。GT 标注：语义关键帧伴随本体感觉状态明显变化时用该信号自动提取，否则手工标注。
- 选取方法：两阶段 Task-Modulated KSM。Stage I：ResNet-18 编码器 + Triplet Margin Loss（margin 1.0）统一多任务度量学习；anchor=GT keyframe(i,ψ,e)，positive=同相位不同 episode(i,ψ,e′)，negative 等概率三类：temporal neighbors（δmin≤|t′−t|≤δmax）、intra-task phase negatives、inter-task negatives。Stage II：冻结编码器，滑动窗口 k=3 帧经 Self-Attention 得 Hvis；task-ID embedding 经生成器 gϕ 得 [γ,β]，FiLM 调制共享 phase embedding 得 qlogic；Cross-Attention+MLP 输出匹配分 st（BCEWithLogits，pos weight 5.0）。
- 推理平滑：st>τconf 缓存为 provisional keyframe 并进入 temporal validation window；窗口内更高分帧即取代并重置计时；仅当整窗持续 <τconf 才 commit 至 Okey 并 phase pointer+1（greedy temporal smoothing）。
- 链式注入：backbone GR00T-N1.5-3B（frozen，仅 projector+DiT head 0.5B 训练），输入非均匀序列 {ok1,…,okn,ot} + 结构化 system prompt（明确 multi-view 时序拓扑），Flow Matching 动作头。
- 与 trajectory segmentation 的关系：文中未引用/对比 trajectory segmentation 或 motion primitive 文献；其对立面是固定步长（fixed-stride）时间均匀采样。关键帧选取是事件驱动（语义阶段完成检测），在 §7 定量分析中与 GR00T 固定步长采样（Nh=3, I∈{5,…,100}）对比。

## 关键发现
- 模拟（4 个 ManiSkill 非马尔可夫任务，400 episodes=每任务 100）：平均成功率 92.0%，最强 baseline 57.0%。分任务：Spatial 70.0 / Temporal 98.0 / Identity 100.0 / Counting 100.0。
- 固定步长采样存在 horizon-resolution 权衡：I=40 时 Spatial 60.0% 但 Counting 0.0%；I=20 时 Counting 74.0% 但 Spatial 18.0%；无单一采样率可跨任务泛化。
- KSM 诊断（TP 容差 ±10 帧，5 帧聚类取中位）：两阶段 P/R/F1=97.5/97.5/97.5，FPR/FNR=2.5/2.5；Joint End-to-End FNR=20.4（Counting Recall=40.0）；w/o Metric Pre-training F1=85.1。
- Prompt 消融：去掉结构化 prompt，Spatial 70→56，平均 92→88。
- 真机（AgileX Piper，20 trials/task，SR/CR%）：ours 平均 48.75/75.3；DP 3.8/32.6；GR00T 6.3/43.1。
- 局限（作者自述）：存储原始视觉 keyframe 有像素级冗余且线性扩展；未来方向为 latent 压缩/离散语义 token、动态 memory 更新。

## 全文相关性（fulltext_relevance）
- 全文相关性: high
- 理由: 论文本身提出并系统验证了 keyframe 定义、自动选取（KSM）与链式历史注入机制，直接回答"keyframe 如何支撑长时程任务"这一调研问题，含仿真+真机定量结果。

## 引用
- Keyframe 定义与两阶段选取：§4.2（含 Eq.1–5）
- 稀疏历史输入重构 {ok1,…,okn,ot}≈{o0:t}：§3.3
- 与固定步长采样对比：§5.1 定量分析
- Oracle keyframe 定义（Temporal 4 帧 / Counting 5 帧 / Spatial 1 帧 / Identity 3 帧）：附录 A
- 任务级 KSM 指标（Spatial Recall 90.0 为最低）：附录 B Table 5
- 真机成功率：§5.3 Table 4
- 训练超参（50,000 steps、batch 16、LR 1e-4）：§5.1 与附录 E

## 不确定项
- τconf 与 temporal validation window 长度的具体数值未给出（仅"pre-defined"）。
- 未讨论与 trajectory segmentation / motion primitive 方法的显式关系，也未与基于分割的 LfD 工作对比。
- "canonical state" 的视觉操作性判据未明确定义（Counting 任务的 signal edges 是否仅靠视觉可判，未单独验证）。
- 仿真与真机的 oracle keyframe 定义是否同构未说明。
- 声称"negligible computational overhead"，但未给出 KSM 推理延迟/开销量化。
