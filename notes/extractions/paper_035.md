# Language-Conditioned Change-point Detection to Identify Sub-Tasks in Robotics Domains

## 基本信息
- paper_id: paper_035
- 年份: 2023（arXiv v1：2023-09-01）
- 作者: Divyanshu Raj, Chitta Baral, Nakul Gopalan（Arizona State University）
- 来源: arXiv:2309.00743v1 [cs.RO]

## 研究问题
给定自然语言指令序列与长轨迹（图像帧 + 离散动作），把每条低层指令映射到轨迹中对应的子片段（切点 = 中心坐标 + 宽度），即语言条件切点检测，用于识别轨迹中的子任务边界，供后续学习子任务/子目标/options 使用。

## 方法
- 问题建模为视频时刻检索（moment retrieval），采用 QvHighlights [14] 的 Moment-DETR 式 transformer：encoder-decoder + 可学习 query 集 + 匈牙利匹配 + L1/gIoU 定位 loss + 显著性 hinge loss + 前景/背景分类 loss（λL1=10, λiou=1, λcls=4, λs=1）；另做加对比 loss 的实验（Sec.IV）。
- 特征：轨迹 = 视频 + 动作。视频特征用 HERO 抽取（CLIP 512 维 + SlowFast 2304 维，每 2 秒 clip）；动作用 CLIP 文本编码器编码（512 维）；拼接为 3328 维/2s clip；语言指令特征用 CLIP（Lq×512）（Sec.III-B）。
- 数据构造（ALFRED 改造）：低层指令映射到帧号 → 视频秒 → 2 秒 clip 的 relevant window / clip id，全部赋予最高显著性 4 分；动作映射到秒并对齐帧数（Lv=La）；最终 6.8K 轨迹、146.1K 低层指令、平均约 6 条指令/轨迹（Sec.III、Table III）。
- 语言信号使用方式：每条低层指令（如 "Turn around and go to the desk"）作为 query，预测该子任务在轨迹中的中心/宽度与每 clip 显著性。
- 消融：轨迹定义（仅视频 t1 vs 视频+动作 t2）、训练数据规模（100%→2%）。

## 关键发现
- 主数字：比基线（仅视频轨迹定义）整体提升 1.78±0.82%（摘要）；加入离散动作后切点检测 R1@0.7 +2.1%、mAP@0.75 +2.1%，显著性检测 mAP +3.2%、HIT@1 +5.2%；再加对比 loss 有微小提升（HIT@1 77.6→78.2）（Table I、Sec.V-B）。
- 全量训练数字（视频+动作+对比 loss，Table I/II）：R1@0.5 85.1±0.48、R1@0.7 65.8±0.74、mAP@0.5 90.9±0.31、mAP@0.75 70.4±0.48、avg mAP 66.7±0.22、HIT@1 78.2±1.30。
- 样本复杂度（Table II）：2% 训练数据（131 条轨迹）→ R1@0.5 40.7±2.5、mAP@0.75 30.7±2.9；50%（3280 条）→ R1@0.5 >80%；作者结论：真机可用规模仍需大量样本，问题未解决。
- 与事件窗泛化的关系：本工作只做切点/子任务边界识别，不把子片段用于规划或策略训练；明确引用 Gopalan et al. [9] 说明子片段可被用于学习子任务/子目标/options，但论文本身无下游策略验证；未来工作含 "behavior cloning from language to sub-task trajectories"（Sec.VI）。
- 离散动作信息对切点定位有显著帮助（语言所描述的动作在视觉特征之外提供判别信号）（Sec.V-B）。

## 全文相关性（fulltext_relevance）
- 全文相关性: high
- 理由: 对"旋钮 1 事件检测"（用语言信号识别轨迹内子任务边界）本身是直接证据；但仅仿真（ALFRED 视觉-语言数据）、无真机，且未验证切点/子片段对下游策略的收益，故对"事件检测→下游数据管线收益"的链条为间接。

## 引用
- 问题定义与方法：Sec.I、Sec.IV（Architecture）
- 数据集构造：Sec.III（A/B）、Table III
- 主结果与消融：Sec.V（Table I/II）
- 样本复杂度结论：Sec.V-B、Sec.VI
- 实现细节：Appendix A

## 不确定项
- 摘要称整体提升 1.78±0.82%，正文 Table I 按指标报 2.1%/3.2%/5.2% 等分解值，两者口径关系未明确说明。
- "语言信号"仅用 CLIP 文本特征 + 静态查询，未分析不同语言粒度（高层目标 vs 低层指令）对切点精度的影响。
- 显著性全为 4 分（"所有时刻都重要"）的简化设定，真实演示中段重要性不均，该假设的偏差未讨论。
- 仅 ALFRED 家居仿真域、每轨迹平均约 6 条指令；真机可行性与跨域泛化完全未验证（作者自述未来工作）。
- 与数据策展的直接关联：论文不涉及演示质量筛选或策略训练收益，只提供子任务切分机制。
