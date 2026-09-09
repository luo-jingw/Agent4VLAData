# CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations

## 基本信息
- 论文：CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations（paper_043）
- 作者：Anthony Liang，Pavel Czempin（共同一作），Matthew M. Hong，Yutai Zhou，Jingzhen Wang，Erdem Bıyık，Stephen Tu（共同通讯）
- 单位：University of Southern California（USC）
- 出处：arXiv:2505.04999v2（cs.RO），2026-07-30，arXiv v2 预印本（IEEE 格式，未标注发表状态）
- 项目页：clamrobot.github.io

## 研究问题
- 设定：用户想让机器人学新任务，不收集带动作标签的专家演示——(1) 大量无标签机器人视频 D_unlabeled（便宜可扩展）；(2) 少量带标签 D_labeled，主要是任务无关、随机/次优的 play 数据，且训练时无法与任务特定演示区分；(3) 少量任务特定专家演示 D_unlabeled-expert，只含观测、无动作标签（用户自己遥操作）。
- 问题：标准 IL 需要动作标签；IfO 方法依赖在线 rollout 或现成视觉模型。
- 与潜在对齐的关系（本文核心）：不直接学环境动作，而是自监督推断相邻观测之间的"连续潜在动作"z_t，再用小规模带标签数据把潜在空间 grounding 为可执行电机命令——对齐发生在潜在动作层，无显式对齐损失，是共享潜在空间 + 联合训练实现隐式对齐。论文明确限定单任务、单本体设定，跨本体（embodiment gap）列为未来工作。

## 方法
- Stage 1 潜在动作模型（LAM）：潜在逆动力学 f_ϕ(z_t | o_{t−H},…,o_t,o_{t+1}) + 潜在前向动力学 g_ψ(o_{t+1} | o_{t−H},…,o_t, z_t)；自监督信号 L_recon = MSE(ô_{t+1}, o_{t+1})（未来观测重构）；H 步上下文（context len=2 帧）；编码-解码信息瓶颈；与 VQ 离散潜在动作（LAPO/Genie）不同——CLAM 用连续潜在动作（文中：量化会限制低层操控任务的表达力）。
- 动作解码器（grounding）：p_ω(a_t | z_t) 用 D_labeled 训练；刻意不以观测为条件（保持 z_t 为低层动作码而非高层技能标签）；与 LAM 联合训练（每 K=2 步交替更新），总目标 L_CLAM = L_recon + β·L_action-decoder，β=1（解码器损失权重），L_action-decoder = MSE(â_t, a_t)；不假设 D_labeled 来源（随机/play 数据均可）。
- Stage 2 潜在动作策略：冻结 IDM 标注 D_unlabeled-expert 得 D_relabeled-expert；训练 π_θ(z_t|o_t)（L_π = MSE(ẑ_t, z_t)）；推理 z_t=π_θ(o_t)，a_t=p_ω(z_t)。IDM 图像编码器可作策略预训练特征（自监督表征学习视角）。
- 架构：图像用 ST（Space-Time）Transformer（64×64×3、patch 16 → 16 patches + CLS 标记；3 编码 + 3 解码层、d=256、4 头；解码块带潜在动作交叉注意力）；持续 500,000 次更新、潜动维度 16、嵌入维度 128；策略为 ACT 型 transformer decoder（ResNet18 dv=512 视觉 token，块长 5 动作 ≈1 秒）；另有 MLP-/Transformer-CLAM 变体（图像版为 ST-ViViT-CLAM）。
- 环境与数据：DMControl（Hopper、HalfCheetah；D4RL medium-expert 预训练、次优片段作 D_labeled）；MetaWorld（Assembly、Bin Picking、Peg Insert、Shelf Place；单任务 RL replay buffer）；真实 WidowX（4 任务：Reach Block、Push Button、Close Microwave、Put Object in Pot and Slide Pot，>150 timesteps @5Hz）。数据分解（轨迹数，D_unlabeled/D_labeled/D_unlabeled-expert）：MuJoCo 1000/50/20；MetaWorld 1000/50/20；真实机器人 ~500/~50（~5k 转移 labeled、~50k 转移 play）/30–50。

## 关键发现
- 主要结论：CLAM 超越所有基线、接近特权 BC-Expert。状态版（表 I，归一化回报/成功率均值）：Transformer-CLAM 0.83（BC-Expert 0.87）、MLP-CLAM 0.63、VPT 0.28、BC-AL 0.24、LAPA 0.21、DynaMo 0.13、LAPO 0.13；比最佳基线 VPT 提高 >2×（DMControl 归一化回报）与 ~2–3×（MetaWorld 成功率）。图像版（MetaWorld，50 rollouts × 3 seeds）：ST-ViViT-CLAM 76% vs LAPA 20%、LAPO 9%（>3×）。解释：潜在动作模型能用 |D_unlabeled| 扩展，监督式 IDM（VPT）只随 |D_labeled| 扩展。
- 连续潜在动作 + 联合训练都必要（表 II 消融，MetaWorld 均值成功率）：离散、非联合 16%（0.15/0.12/0.18/0.19）；离散 + 联合 15.5%（0.14/0.14/0.17/0.16）；连续、非联合 23%（0.28/0.18/0.21/0.26）；连续 + 联合 74%（0.69/0.82/0.57/0.88）——连续带来 ~1.5×，联合训练再增 ~3×（74% 原文还称"over 3× improvement"相对 23%）。
- 真实 WidowX（每任务 10 trials、部分得分 0.5/1）：ST-CLAM 7/10、8.5/10、8/10、4/10 vs BC-Expert 7.5/10、8/10、7.5/10、2/10；VPT 2.5/4/5/2；LAPA 2/3/3/0；BC-AL 0/0.5/1/0——未采集任何带标签专家演示即学到任务。
- 可调因素：潜动维度 |z|∈{2,4,8,16} → Assembly 成功率 9%/11%/57%/50%（真值动作维度 4 不够，需轻微过参数化，16 无额外收益）；|D_unlabeled| 50→1000 条：39%/56%/61%/69%/73%（收益递增后饱和）；|D_labeled| 增加提升解码精度，BC 用同量非专家数据很快平台化（即使升到 100 条）；随机策略数据与随机/中档数据效果相近，专家标签数据可恢复最优策略。
- 与迁移的关系：结论明确"untested cross-embodiment"——单本体；正文提到动作无标签 + 视频转换（Phantom/Masquerade）可接跨本体，且 CrossFormer 类方法限训练中见过的本体集。

## 证据等级
- 实证较高：3 类环境（2 个仿真基准 状态+图像）+ 物理 WidowX；表 I/表 II 附标准差；MetaWorld 图像 50 rollouts × 3 seeds；具体消融（维度、数据量、联合训练）；真实机器人仅 10 trials/任务、无种子信息。arXiv v2（2026-07-30）预印本、无正式发表记录；明确说明未解决跨本体对齐（多为未来项）。

## 引用
- 基线：BC-AL、VPT [14]、LAPO [18]（离散 VQ 潜在动作）、LAPA [21]（LAM 预训练+头微调）、DynaMo [20]（IDM/FDM 自监督视觉特征）、LAPA 等 [26] Nikulin et al. LAOM（连续潜在动作+少量标签监督，但监督头训练后弃用，且只做仿真 locomotion）；BC-Expert 为上界。
- 相关：IfO [9]、ACT [38]、Space-Time Transformer [37]、VQ-VAE [33]、D4RL [36]、Phantom [7]、Masquerade [8]、CrossFormer [41]。

## 不确定项
- 文中未说明：真实机器人 D_labeled 的确切轨迹数（正文 ~5k 转移，如对应表 VI 中 50 条则单条 ~100 转移，未直接校核）；真实机器人评估时的 rollout 数（10 trials 固定，无置信区间）；表 II 消融（最大 0.88）与图 2（76%）分属不同设置/批次，未交叉验证。
- 边界：无跨本体（论文自身限定单本体单任务）；潜动维度与 K=2 联合训练频率等超参未扫描。
- 文中未说明：重标注 D_relabeled-expert 时 IDM 的置信度/失败传播；无梯度控制证据（"CLAM performs better than BC with same labeled random data"有定图但未见定量标准差对应表）。
