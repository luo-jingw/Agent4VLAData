# ReAct: Synergizing Reasoning and Acting in Language Models

## 基本信息
- paper_id: paper_007
- 年份: 2022（arXiv:2210.03629，初版 2022-10；v3 2023-03；ICLR 2023）
- 作者: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao（Princeton University 与 Google Research, Brain team）
- 来源: ICLR 2023 会议论文（arXiv:2210.03629）

## 研究问题
- 此前 LLM 的推理能力（如 CoT）与行动能力（如 action plan generation）被分开研究（Abstract、§1）。
- 动机：CoT 是"static black box"，只用模型内部表示生成 thought，不接地于外部世界，导致事实幻觉与错误传播（§1，图 1(1b)）；纯 acting 方法不利用语言模型做高层目标推理或维持工作记忆（§1）。
- 提出 ReAct：交错生成 verbal reasoning traces 与 task-specific actions，实现"reason to act"与"act to reason"的协同。

## 方法
- 核心机制（§2）：把动作空间扩展为 Â = A ∪ L，L 为语言空间；语言空间中的"thought"（reasoning trace）不作用于外部环境、不产生观察反馈，只推理当前上下文并更新上下文 ct+1 = (ct, ât)，以支持后续推理或行动。
- 知识任务中 thought 与 action 交替出现（多步 thought-action-observation）；决策任务中 thought 稀疏、由模型自定异步出现（§2）。
- 实现：frozen PaLM-540B + few-shot in-context prompts（HotpotQA 用 6 个、Fever 用 3 个人工轨迹示例；"more examples do not improve performance"）（§3.2）。
- 知识任务动作空间（§3.1）：Wikipedia API，search[entity]（返回前 5 句或 top-5 相似实体）、lookup[string]（模拟 Ctrl+F）、finish[answer]；question-only 设置。
- 与 CoT-SC 结合（§3.2）：ReAct→CoT-SC（ReAct 超步数未出答案时回退，HotpotQA/Fever 上限 7/5 步）；CoT-SC→ReAct（n 个样本多数票少于 n/2 时回退）。CoT-SC 采样 21 条、温度 0.7。
- 微调（§3.2）：bootstrap 3000 条正确答案轨迹微调 PaLM-8B/62B（类 STaR/Zelikman）。
- 决策任务：ALFWorld（134 个未见游戏，6 任务类型，每类 3 条标注轨迹构造 6 组 prompt）与 WebShop（500 测试指令，1.18M 商品）；基线 BUTLER（10^5 专家轨迹模仿学习）、IL（1012 条轨迹）、IL+RL（另加 10587 指令）（§4）。

## 关键发现
- HotpotQA（EM）/ Fever（Acc），PaLM-540B prompting（Table 1）：Standard 28.7/57.1；CoT 29.4/56.3；CoT-SC 33.4/60.4；Act 25.7/58.9；ReAct 27.4/60.9；CoT-SC→ReAct 34.2/64.6；ReAct→CoT-SC 35.1/62.0。ReAct 全面优于 Act；Fever 上优于 CoT，HotpotQA 上略低于 CoT。两结合方法只用 3–5 个样本即达到 CoT-SC 21 样本的水平（图 2）。
- 幻觉/错误传播（Table 2，HotpotQA 各 50 条成功/失败轨迹人工标注，共 200 例）：成功轨迹中 false positive（幻觉事实）ReAct 6% vs CoT 14%；失败轨迹中 hallucination ReAct 0% vs CoT 56%，reasoning error ReAct 47% vs CoT 16%（含 ReAct 特有的重复生成先前 thought/action 的循环失败），search result error ReAct 23%（CoT 无此项）。
- ReAct 更 grounded、可信、可解释，但交错结构的约束降低推理灵活性，推理错误率高于 CoT（§3.3 A/B/C）。
- ALFWorld 成功率（Table 3）：ReAct best-of-6 平均 71%，Act best-of-6 45%，BUTLER 37%（BUTLERg 22%）；ReAct 最差试验 48% 也超过基线最佳；相对提升 33%–90%，平均 62%。ReAct 71 vs ReAct-IM 53（IM 式密集外部反馈不如稀疏内部推理）。
- WebShop（Table 4）：ReAct score 66.6 / SR 40.0 vs Act 62.3/30.1 vs IL 59.9/29.1 vs IL+RL 62.4/28.7 vs Human 82.1/59.6；相对先前最好成功率绝对提升 10%。
- 微调（§3.3，图 3）：HotpotQA 上 prompting 时 ReAct 在四种方法中最差（PaLM-8/62B 难以同时学会推理与行动），但用 3000 例微调后 ReAct 最好；PaLM-8B 微调 ReAct 超过所有 PaLM-62B prompting 方法，PaLM-62B 微调 ReAct 超过所有 540B prompting 方法。
- 局限（§6）：大动作空间任务需更多演示，易超出 in-context 输入长度限制；作者建议更多人工标注数据微调与结合 RL。

## 候选相关性（总体）
- 总体相关性: high
- 理由: 原始实验论文，直接给出 ReAct 机制定义与四基准（HotpotQA、Fever、ALFWorld、WebShop）上对比 CoT/Act/IL/RL 的量化实验结果，是调研问题"LLM agent 基础（thought-action-observation 循环）"的一手证据。

## 引用
- 机制定义（动作空间扩展、thought 无环境反馈）：Section 2
- Wikipedia API 动作空间与 ReAct prompt 构成：Section 3.1、3.2
- HotpotQA/Fever 对比数字与失败模式分析：Section 3.3（Table 1、Table 2、图 2、图 3）
- ALFWorld/WebShop 对比数字：Section 4（Table 3、Table 4）
- 与 Inner Monologue 的消融（ReAct-IM）：Section 4（"On the value of internal reasoning vs. external feedback"）

## 不确定项
- 主实验基于 PaLM-540B（当时非公开模型）；GPT-3 结果仅在附录 A.1（正文脚注称 GPT-3 超过 PaLM-540B，具体数字未在正文给出）。
- "34% 和 10% 的绝对成功率提升"出现在 Abstract，为相对 BUTLER/Act 与 IL+RL 的综合表述；正文 Table 3/4 的精确对比按"best trial"口径给出，具体口径选择（avg vs best-of-6）在正文中对不同方法不完全一致，文中未解释为何对外口径用 best-of-6 而对内用 avg。
- 重复 thought/action 循环失败的成因仅列为猜测（"sub-optimal greedy decoding"），未验证。
- HotpotQA 部分问题答案标签可能过时（文中提到，图 4 有示例），对 EM 结果的影响程度未量化。
- 文中未说明 thought 数量的显式上限或预算机制（仅知识任务设 7/5 步回退阈值）。
