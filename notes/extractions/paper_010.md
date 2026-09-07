# OpenGVL - Benchmarking Visual Temporal Progress for Data Curation

## 基本信息
- paper_id: paper_010
- 年份: 2025（CoRL 2025, Seoul；arXiv:2509.17321v4, 9 Feb 2026）
- 作者: Paweł Budzianowski、Emilia Wiśnios、Gracjan Góral、Michał Tyrolski、Igor Kulakov、Viktor Petrenko、Krzysztof Walas（Lute、IDEAS Research Institute、Simple Automation、Poznań University of Technology、University of Warsaw）
- 来源: 9th Conference on Robot Learning (CoRL 2025)

## 研究问题
机器人数据规模快速增长但缺乏策展工具。能否用 VLM 的通用价值/进度预测能力（GVL 范式）做自动化数据标注与策展？开源 VLM 与闭源模型在"时间任务进度预测"上差距多大？

## 方法
- GVL（Generative Value Learning，源自 Ma et al. [20]，ICLR 2024）：不训练/微调，用 VLM 的 in-context 学习，给定少量轨迹示例及其逐帧完成百分比，让 VLM 对被评估轨迹的每帧输出任务完成百分比（0–100）。输入帧随机打乱（shuffle）——利用 VLM 模仿上下文中行为模式的倾向，避免过度依赖时间顺序线索；观测帧数受限，每 episode 取 15 帧。
- VOC（Value-Order Correlation）：GVL 的自动质量度量，VOC = rank-correlation(argsort(v1,…,vT), (1,…,T))，即预测值与帧时序的秩相关（Spearman 或 Kendall，文中未指明用了哪个），范围 [−1,1]。VOC 高是数据质量的必要非充分信号。
- OpenGVL benchmark：4 个公开数据集（nyu door、berkeley mvp、cmu stretch、nyu franka），每集合同一索引抽 50 episodes，zero-shot 与 two-shot 两种条件；另建两个 withheld 隐藏数据集（末端电子装配，亚毫米精度：一个人类执行、一个双臂 7-DOF 机器人），防止数据污染。
- 模型：开源 Gemma-3（4B/12B/27B）、Qwen2.5-VL-Instruct（3B/7B/32B）、GLM-4.1V-9B-Thinking、MiMo-VL-7B-RL-2508、Cosmos-Reason1-7B、Kimi-VL-A3B（16B 总量/3B 激活）；闭源 gpt-4o、gemini-2.5-flash-lite-preview-06-17、gemini-2.5-pro。统一系统提示（附录 A），temperature=1.0。
- 策展应用：用 VOC/进度曲线在 HF LeRobot hub（截至 2025 年 8 月 >13,000 数据集、>260 万 episodes）上识别三类问题：(1) 任务定义问题；(2) 标注歧义；(3) 失败/分布外示例。

## 关键发现
- 开源模型显著落后于闭源：开源模型 VOC 仅约为闭源上界成绩的 60–70%（Abstract 表述"approximately 70%"，Section 3.2 表述"approximately 60–70%"，两处数字不一致）。
- VOC 随模型规模提升（Gemma 与 Qwen 家族均如此）。最好结果：gemini-2.5-pro（nyu door 2-shot 0.9654、0-shot 0.9158；berkeley mvp 2-shot 0.6806）；gpt-4o（nyu door 2-shot 0.870、0-shot 0.720）。开源最好：gemma-3-27b-it（nyu door 2-shot 0.8219、0-shot 0.6372）；GLM-4.1V-9B-Thinking（nyu door 0.6540/0.6420）；MiMo-VL-7B-RL-2508（nyu door 0.5977/0.5314，berkeley mvp 0.4736/0.4391）。小模型接近随机（如 Qwen2.5-VL-3B 各集接近 0 或为负；Gemma-3-4b 多为负值）。
- 隐藏任务：zero-shot VOC 基本处于或低于随机水平；two-shot 普遍改善，但多数仍弱（约 0.1–0.3），少数中等（≥0.4），极少数强（≥0.7）——这些长时程/精细操作任务仍具挑战性（Fig. 3）。
- 策展演示（均定性，无下游策略数字）：excavator_toy_v3 指令"Dig grass and dump in dump truck"任务定义不清，VOC 无法持续上升，可由 VOC 检出；1500_chess_moves 的 VOC 低 + 摄像头被灯光遮蔽；pickplace_joint 指令歧义（"take out a vial and put it into another pocket"）导致 VOC 很低，此类数据可能损害 VLA 预训练；so101_60_new 中 episode 93（150 条中唯一显著异常）进度曲线涨落异常，可识别执行失败/传感器故障/错误任务解读。

## 证据等级
- 等级: direct
- 理由: 直接测量 VLM 预测时间任务进度的能力（VOC 定量、多模型对比），并用该能力做数据标注/过滤演示——与"用进度预测做数据策展"的调研问题直接对应；但策展部分证据为定性案例，无下游训练收益的定量验证。

## 引用
- GVL 与 VOC 定义：Section 2（Related work）；VOC 公式与范围：Section 2 末尾
- 实验设置（数据集、采样 15 帧、shuffle、隐藏任务）：Section 3.1
- 模型评测与规模效应：Section 3.2、Table 1
- 60–70% 差距结论：Section 3.2 正文（及 Abstract）
- 隐藏任务结果：Section 3.1 / Fig. 3
- 策展三类问题：Section 4.1–4.3；局限性：Section 5.1
- 完整 prompt：Appendix A

## 不确定项
- VOC 具体采用 Spearman 还是 Kendall：正文只写"rank correlation (Spearman or Kendall)"，未指明。
- "约 70%"（Abstract）与"约 60–70%"（Section 3.2）不一致，文中未解释；"闭源上界分数本身只是不可观测真值的代理"（脚注 2）。
- 策展应用全部为定性案例（VOC 曲线观察），未报告过滤/标注后的任何下游训练结果；"开源仅达闭源 70%"的换算口径未给出公式。
- 每个 episode 抽样 15 帧的具体帧选择方式（均匀采样，见 5.1）与每帧 VOC 聚合方式（VOC 平均 over 50 episodes，Table 1 注）之外，帧内预测失配（Mism./Empty 列）的影响未讨论。
- gpt-4o 与 MiMo-VL 的参数规模标注（"–"与 9B）在表中与正文描述（MiMo 7B）不完全一致；gpt-4o 版本/日期未标注。
