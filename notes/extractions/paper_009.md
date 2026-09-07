# AgentInstruct: Toward Generative Teaching with Agentic Flows

## 基本信息
- paper_id: paper_009
- 年份: 2024（arXiv:2407.03502v1，2024-07-03）
- 作者: Arindam Mitra, Luciano Del Corro, Guoqing Zheng, Shweti Mahajan, Dany Rouhana, Andres Codas, Yadong Lu, Wei-ge Chen, Olga Vrousgos, Corby Rosset, Fillipe Silva, Hamed Khanpour, Yash Lara, Ahmed Awadallah（Microsoft Research）
- 来源: arXiv 预印本（cs.AI）

## 研究问题
合成数据可用于模型 post-training，但质量和多样性难保证，通常需要大量人工策展。论文提出"Generative Teaching"设定：仅用原始文本/代码作为种子，由 agentic 流程自动生成大量多样、高质量、不同难度的（prompt, response）训练数据，并验证其能否有效教会模型新技能。

## 方法
- 输入只有 raw seed（无结构化文本、代码文件），不需要种子 prompt。
- 三个 agentic flow（Figure 2/3）：
  1. Content Transformation Flow：把原始种子转换为中间表示（如 argument passage、会议记录、API 列表）。阅读理解技能有 9 个变换 agent（Argument Passage Generator、Debate/Conversation/Meeting Transcript/Poem/Satirical/Instructional/Long Text Generator、Identity Agent）；工具使用技能把代码片段合成为 API 描述，用 API retrieval agent 迭代检索相似代码扩充 API 列表，或用 LLM 假设库中其他 API（library reconstruction）。
  2. Seed Instruction Generation Flow：按预定义分类法（taxonomy）由多个 agent 生成多样指令。阅读理解含 43 种题型（strengthen/weaken/assumption/flaw 等，每个题型一个 agent）；文本修改含 18 种任务类型（paraphrase、simplification、code switching 等）；工具使用按"单 API/多 API/缺 API/缺参数"等情形生成任务。全框架 taxonomy 超过 100 个子类别。
  3. Instruction Refinement Flow：Suggester-Editor agent 对迭代提升复杂度与质量——Suggester 提议让指令更复杂/更刁钻的改法，Editor 据此修改（如改文段使问题不可答、改选项、加干扰项）。
- 共实现 17 种技能（含 coding、RAG、tool use、creative writing 等，Table 1）。
- 数据规模：AgentInstruct 生成约 22M 指令，加上来自 Orca-1/Orca-2/Orca-Math 等约 3.8M 配对指令（Orca-2.5-dataset），合计约 25.8M 对。种子来源：KnowledgePile、AutoMathText、OpenStax 子集、apache-2.0 许可代码（CodeParrot github-code-clean）。
- 验证方式：用 25.8M 数据微调 Mistral-7b-v0.1 得到 Orca-3（对照 Orca-2.5 用 3.8M 数据微调），观察下游基准提升。

## 关键发现
- 数据质量验证：论文未对人类标注数据做直接质量对比，验证是间接的（下游表现）。
- Orca-Bench（held-out，17 技能 × 100 样本，GPT-4=10 分）：Orca-3 9.55 > Orca-2.5 7.13 / ChatGPT 8.13 / Mistral-Instruct 8.31；平均提升 33.94%（对 Orca-2.5）、14.92%（对 Mistral-Instruct）。
- 对 Mistral-7B-Instruct 的相对提升（Table 3）：AGIEval +40%（56.80 vs 40.52）、MMLU +19%（69.95 vs 58.61）、GSM8K +54%（83.09 vs 54.06）、BBH +38%（61.83 vs 44.71）、AlpacaEval +45%（24.80 vs 17.1）、DROP +22%、FOFO +12%；GPQA -4%（28.12 vs 29.46，低于基线）。
- 细分：阅读理解平均 +18%（对 Orca-2.5）/+21%（对 Mistral-Instruct）；数学各基准提升 44%–168%；FoFo 格式遵循 +11.5%（84.01，超过 Gemini Pro 的 80.25）；摘要幻觉率降低 31.34%（21.09% vs 30.72%，GPT-4 作评估器），质量 9.14 vs 8.85；MIRAGE RAG 平均 +38.30%，PubMedQA 相对 +92.71%。
- 训练成本：19 节点 152 张 A100 GPU，per-GPU batch 10，AdamW lr 8e-6，cosine 调度，500 步 warmup，3 epoch，约 200 小时。
- 数据生成的美元成本：文中未给出具体金额；Limitations 一节仅称"生成合成数据资源密集"。生成 agent 所用 LLM 为 GPT-4 等强模型（具体版本与调用量文中未说明）。

## 证据等级
- 等级: indirect
- 理由: 论文展示的是 agentic 合成数据通过下游基准间接体现的数据价值，未直接度量数据质量本身、也未与人类标注数据质量对比；对"agent 用于数据策展/质量评分"只有间接启示。

## 引用
- 三个 flow 的定义：Section 2、Figure 2/3（第 3–4 页）
- 各技能 agent 细节：Section 2.1–2.3（第 6–12 页）、Appendix A
- 数据规模与训练：Section 3.1–3.2（第 14 页）
- 验证数字：Section 4、Table 2–8（第 14–20 页）

## 不确定项
- 无合成数据与人类标注数据质量的直接对比数字（质量好坏完全靠下游基准间接推断）。
- 生成数据的 agent 使用哪些具体 LLM、多少调用量、美元成本：文中未说明。
- 未报告数据的过滤/去重/质量控制的具体标准和通过率。
- 未报告与纯"种子 prompt 集"方法（如 Self-Instruct 类）在数据质量上的直接对照。
- GPQA 出现 -4% 负提升，论文未解释原因。
- 数据多样性只有定性论证（taxonomy 子类别数量），无定量多样性指标。
