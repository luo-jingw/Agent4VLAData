# Curating Demonstrations using Online Experience（Demo-SCORE）

## 基本信息
- paper_id: paper_013
- 年份: 2025（arXiv:2503.03707；v2 标注 22 Jul 2025）
- 作者: Annie S. Chen*, Alec M. Lessing*, Yuejiang Liu, Chelsea Finn（Stanford University；*同等贡献）
- 来源: arXiv cs.RO（Stanford IRIS 实验室）

## 研究问题
- 演示数据集内部存在异质性：部分策略不可靠或欠采样，但全部演示都标注为成功，人难以辨别。
- 研究如何在单一任务内自动策展（过滤）演示数据，以提升最终模仿学习策略的可靠性；只使用 rollout 的二值成功/失败标注，不用稠密人工标注。

## 方法
- 输入：N 条成功演示 Ddemo；设置属于自主模仿学习（Auto-IL）问题。
- 步骤（Algorithm 1）：(1) 用全部演示训练初始策略 π0；(2) 在初始训练 run 的 C 个均匀间隔 checkpoint 生成 rollout；(3) 用 rollout 训练"数据质量分类器"（二分类：成功/失败状态），并在最后一个 checkpoint 的 rollout 上交叉验证选最佳分类器 qϕ*；(4) 将分类器应用于原始演示与 rollout：对每条轨迹求各状态成功概率的均值 q̄(τ)，阈值 γ 取分类器训练集（最优分类器对应的 Dπ0,i*）上分类器输出的均值，q̄(τ) > γ 则保留；对过滤后的演示 ∪ 过滤后的成功 rollout 重新训练/微调得到 πfinal。
- 分类器：输入 proprioceptive state，输出二值标签；二元交叉熵损失（轨迹内按步平均）；MLP 两层各 8 单元；dropout 0.3、AdamW weight decay 0.1（附录 B-b）。
- 关键设计：跨 checkpoint 交叉验证防过拟合；可 episode 级或 chunk 级过滤；仿真 C=4、真实 C=5。
- 对比方法：Base Policy（全量演示）、Training Loss（按演示 loss 反比加权）、Auto-IL（原始演示+成功 rollout）、RCP（return 条件策略）。

## 关键发现
- 仿真（Robosuite square peg、ALOHA 双臂 peg insertion，Diffusion Policy/ACT）：所有数据混合平均，Demo-SCORE 比 Base Policy 提高 15–35% 绝对成功率；过滤后 square peg 平均 >94%、peg insertion 平均 >90%。
- Demo-SCORE 平均失败率比最优先前方法 Auto-IL 低 2×以上（square task）、1.6×（peg insertion）；Auto-IL 平均成功率 square 88.2%、peg insertion 79.9%。
- 真实世界 ALOHA 四任务（Spoon/Chocolate/Sharpie/Strawberry）：绝对提升 15–40%，平均近 30%；Jellybean（124 条众包+100 条专家演示）聚合子任务分 27→49，长程全任务成功率 0→20%。
- 过滤比例：square peg 按混合过滤掉 16%–67% 的演示（极端不平衡混合如 200 Human/400 SquareA 仍有效）。
- Rollout 成本低：每 checkpoint 少至 10 条 rollout、总计 <100 条即可；10 rollout/checkpoint 时 square 平均 92%（比次优 Auto-IL 高 >10%），peg insertion 87.7%（Base 58.9%）。
- 消融（Table II/III/V）：无交叉验证明显变差（平均 88.5 vs 94.4）；chunk 级、轨迹级（transformer，只取前 100 步）、plateau checkpoint、小分类器均可行；正则化可移除仍稳健（因交叉验证）。
- OOD：扩张初始化区域 50%/100% 时 Demo-SCORE 不损害甚至改善 OOD 泛化（Table VI、VII）；极端情况下（某区域 100% 次优策略）会略低于其他方法。

## 证据等级
- 等级: direct
- 理由: 论文直接实现并实验验证了"用在线 rollout 经验过滤演示"的方法，在 2 个仿真任务、5 个真实 ALOHA 任务上与纯离线/基线方法量化对比（成功率、失败率、过滤比例），直接回答演示策展问题。

## 引用
- 方法判据与阈值 γ 定义：IV. Demo-Score Data Curation（A. 分类器训练、B. 过滤；Algorithm 1）。
- 15–35% 提升：I. Introduction、V-B b) Results（square peg >94%、peg insertion >90%）。
- 真实世界 15–40%、平均近 30%、Jellybean 0→20%：V-C b) Results、Table I。
- rollout 数量消融（10/25/50/100）：V-D b)、Table III。
- 无交叉验证变差、chunk/轨迹/plateau/无正则变体：V-D、Table II。
- 过滤掉 16–67% 演示：V-B b)。
- OOD 结果：附录 F、Table VI、VII。

## 不确定项
- 未说明初始策略训练步数 K 的确切值（只给出 ALOHA 300,000 步、Robosuite 1,000 epochs；checkpoint 位置见附录 B-a）。
- 真实世界实验的演示数量仅列 Spoon 40、Chocolate/Sharpie 50、Strawberry 120，未明确说明不同策略的具体构成比例。
- 论文假设演示全部成功（失败演示直接丢弃），未讨论演示含失败标签或带噪声的场景。
- 分类器只输入 proprioceptive state，未使用图像观测；未讨论视觉观测下的适用性。
- γ 固定为分类器训练集输出均值，作者称可进一步调参，但未报告 γ 敏感度实验。
- 未给出与"人工专家手动策展"的直接对照数字，只在引言中论证人工策展困难。
