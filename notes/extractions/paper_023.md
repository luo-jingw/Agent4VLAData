# TextGrad: Automatic "Differentiation" via Text
## 基本信息
- paper_id: paper_023
- 年份: 2024（arXiv v1，2024-06-11）
- 作者: Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, James Zou（均属斯坦福大学）
- 来源: arXiv:2406.07496（预印本）
## 研究问题
复合 AI 系统（compound AI system）由多个 LLM 与非可微组件（模拟器、数值求解器、代码解释器等）编排而成，目前靠专家手工构建、启发式调参；需要一种系统化的自动优化方法。类比神经网络靠反向传播/自动微分成为"turn-key"，TextGrad 用文本反馈实现"文本微分"来优化复合系统。
## 方法
- 文本梯度机制：把系统表示为计算图，变量为输入/输出（多为自然语言文本）；梯度 = LLM 给出的自然语言批评（描述如何修改变量以改进下游目标）；经 ∇LLM 梯度算子沿图反向传播（Eq. 11：对变量 v 的每个后继 w 收集反馈并聚合），再用 Textual Gradient Descent（TGD）优化器更新变量。梯度算子和 TGD 实现与领域无关，全框架固定（Sec 2）。
- 抽象与 PyTorch 同构：tg.Variable、tg.BlackboxLLM、tg.TextLoss、tg.TGD、loss.backward()/optimizer.step()，便于迁移知识（Fig 1c）。
- 目标函数可为任意非可微函数：自然语言 prompt 的 LLM、跑单元测试的代码解释器、分子模拟引擎（Vina/QED）等。
- 两类优化问题：instance optimization（直接优化单个解：代码、答案、分子、放疗计划，test-time）与 prompt optimization（优化跨查询泛化的 system prompt）。
- 优化技巧（类比数值优化）：minibatch SGD 式 batch 优化（tg.sum 聚合梯度）、自然语言约束的 constrained optimization、momentum（更新时可参考早期迭代）。
- 成本：图有 n 条边时，每轮优化最多额外 n 次 LLM 调用（每条边 1 次梯度算子调用）。
## 关键发现
- 代码（Sec 3.1，LeetCode Hard，gpt-4o，5 seeds，5 迭代）：完成率 zero-shot 0.26、Reflexion（1 演示）0.31±0.012、TextGrad（0 演示）0.36±0.018；相对 Reflexion 约 +16%，摘要称 20% 相对提升。
- 解题（Sec 3.2，gpt-4o，3 次 test-time 更新 + majority voting）：GPQA 51.0%（CoT）→55.0%（TextGrad，当时已知最佳，超过此前 best reported 53.6%）；MMLU-ML 85.7→88.4；MMLU-College Physics 91.2→95.1。
- 提示优化（Sec 3.3，前向模型 gpt-3.5-turbo-0125、gpt-4o 提供反馈，batch size 3 × 12 迭代 = 36 训练例，0 演示）：Object Counting CoT 77.8 / DSPy(BFSR, 8 演示) 84.9 / TextGrad 91.9；Word Sorting 76.7 / 79.8 / 79.8；GSM8k 72.9 / 81.1 / 81.1；DSPy 演示 + TextGrad 指令组合在 GSM8k 达 82.1。
- 分子（Sec 3.4）：SMILES 串 instance 优化，Vina + QED 多目标；DOCKSTRING 58 个靶点、各 3 个初始片段 × 10 迭代；29 个有临床批准药的靶点上生成分子与临床药物相比结合亲和力相当、QED 更高。
- 放疗（Sec 3.5）：用 gpt-4o 优化 matRad 内层数值优化器的权重字符串（外层超参优化）；5 例前列腺癌患者计划；TextGrad 计划在 PTV 平均剂量与 D95 上胜过临床计划，膀胱/直肠剂量低于临床允许最大值。
- 局限性（Sec 5 Discussion）：分子与放疗只有 in silico 验证，未经实验/临床检验；梯度算法稳定性（方差缩减、自适应、自验证）未做；框架尚未覆盖 tool use、RAG 等组件；约束过多时 LLM 遵从性下降（Sec 2 引用文献）。
## 全文相关性（fulltext_relevance）
- 全文相关性: high
- 理由: 论文是"用文本梯度优化 agent/复合管线"这一工程模式的一手原始文献，机制定义与全部实验数字直接来自正文。
## 引用
- 文本梯度机制、∇LLM、TGD、计算图定义：Sec 2（第 2–6 页，Eq. 4–12）。
- 实例优化 vs 提示优化：Sec 2 "Instance vs Prompt Optimization"（第 6 页）。
- 实验数字：Table 1（第 8 页）、Table 2（第 10 页）、Table 3（第 11 页）、Sec 3.1–3.5。
- 与 DSPy/ProTeGi 的关系：Sec 4 Related work（第 16 页）。
- 局限：Sec 5 Discussion（第 17 页）。
## 不确定项
- 摘要"20% relative performance gain"与 Table 1（0.36 vs 0.31 ≈ +16%）不一致，正文未解释该数字口径。
- GPQA 的 51%→55% 基于少数题目的 accuracy 提升（约 4 个百分点，样本量未细述），未报告置信区间/方差（Table 2 无误差条）。
- LeetCode 基线中 Zero-shot 0.26 为作者自己复跑（引 [26] 为 7% GPT-4），跨模型版本比较存在口径差异。
- 优化稳定性未报告：各任务上的迭代次数（3/5/12/10）选择依据未说明；失败案例与成本-收益（额外 LLM 调用开销）未量化。
- 文本梯度质量依赖所用 LLM 的批评能力（正文声明假设 SOTA LLM 能推理各子任务），但未系统验证梯度提供者模型较弱时的表现。
- 放疗与分子部分仅 proof-of-concept，无统计显著性检验（放疗 n=5 患者）。
