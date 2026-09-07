# DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
## 基本信息
- paper_id: paper_022
- 年份: 2023（arXiv v1，2023-10-05）
- 作者: Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, Christopher Potts
- 来源: arXiv:2310.03714（预印本）
## 研究问题
现有 LM 管线用硬编码 prompt template（试错法手工构造长字符串）实现，脆弱且不可扩展（类比手调分类器权重）。DSPy 提出一种编程模型：把 LM 管线抽象为"文本变换图"（text transformation graph），由编译器依据指标自动生成/优化提示词与演示，从而减少乃至移除人工 prompt 的作用。
## 方法
- 三大抽象（Sec 3）：(1) Signature：自然语言类型化声明，输入/输出字段元组 + 可选指令（如 `question -> answer`），描述"做什么"而非"如何 prompt"；(2) Module：参数化的声明式模块，内置 Predict、ChainOfThought、ProgramOfThought、MultiChainComparison、ReAct，可任意组合，define-by-run 计算图（借鉴 PyTorch/Chainer），模块参数包括：所调 LM、指令与字段前缀、演示（论文重点优化演示）；(3) Teleprompter：优化器，输入程序 + 训练集 + 指标，返回优化后的程序。
- 管线与指标分离：训练集可以很小（少量例子）、可无标签；通常只需最终输出的标签，中间步骤标签由 bootstrap 自动生成；指标可为简单 EM/F1，也可本身是 DSPy 程序（如 check grounding），编译时仅凭指标过滤多阶段 trace。
- 编译器三阶段（Sec 4）：Stage 1 候选生成（recursively 找所有 Predict 模块，模拟运行收集多阶段演示，用指标筛选"好 trace"作为演示）；Stage 2 参数优化（对离散候选用随机搜索或 HyperOpt/Optuna 类 TPE 选择；BootstrapFinetune 用演示微调 LM 权重）；Stage 3 高阶程序优化（改控制流，如 ensemble 多份程序并行 + majority voting）。
- Teleprompter 可组合：teacher 程序监督（如 Llama2-13b-chat 编译结果做 teacher，无标签数据微调 Flan-T5-large）；BootstrapFewShot、BootstrapFewShotWithRandomSearch、BootstrapFewShotWithOptuna、BootstrapFinetune、Ensemble。
- 工具模块：dspy.Retrieve（ColBERTv2、Pyserini、Pinecone），实验性 dspy.SQL、dspy.PythonInterpreter。
## 关键发现
- GSM8K（Sec 6，训练 200 例/验证 300 例/测试 1.3k）：全部程序由 2–4 个模块与 teleprompter 组成；编译把不同 LM 从 4–20% 准确率提升到 49–88%。GPT-3.5 CoT：fewshot+human CoT dev 78.6/test 72.4；bootstrap dev 80.3/test 72.9（超越人工 CoT）；reflection（5 条推理链 + MultiChainComparison）bootstrap dev 83.0/test 76.0，ensemble dev 88.3/test 81.6。llama2-13b-chat reflection bootstrap test 40.2，ensemble dev 49.0/test 46.9。
- HotPotQA（Sec 7）：multihop 程序（两跳 query 生成 + 检索 + 回答）最优；GPT-3.5 bootstrap dev Ans 48.7/Psg 47.0，test 39.6/43.8，ensemble dev 54.7/test 45.6（*测试集一半，成本原因）；llama2-13b-chat 编译后与 GPT-3.5 相当。微调 T5-Large（770M）multihop：dev 39.3% EM、46.0% 段落准确率，仅用 200 带标签 + 800 无标签输入，推理成本比专有 LM 低几个数量级。
- 摘要总体数字：编译几分钟到几十分钟内，数行 DSPy 代码使 GPT-3.5 与 llama2-13b-chat 自举管线超过标准 few-shot（总体超 25% 与 65%）、超过专家演示（最高 5–46% 与 16–40%）；引言：GPT-3.5 从 33%→82%（Sec 6）、32%→46%（Sec 7）；llama2-13b-chat 从 9%→47%、22%→41%。
- 编译成本：分钟级到几十分钟；昂贵设置也只需运行程序几千次（如 10–20 次试验 × 150–300 验证例），可并行。
## 证据等级
- 等级: direct
- 理由: 论文是"agent 管线工程模式"（模块化 + 声明式 signature + 指标驱动编译）的一手原始文献，方法细节与实验数字直接来自正文。
## 引用
- 编程模型与三大抽象：Sec 3（第 3–6 页）。
- 编译器三阶段与 teleprompter 机制：Sec 4（第 6–7 页）。
- 指标可任意（含 DSPy 程序作指标）、teacher 组合微调：Sec 3.3（第 6 页）。
- GSM8K/HotPotQA 数字：Table 1（第 8 页）、Table 2（第 11 页）、Sec 6–7。
- 总体提升数字：Abstract（第 1 页）与 Introduction（第 2 页）。
## 不确定项
- 论文未报告与 LangChain 等框架的定量对比（仅在 Appendix B 作定性讨论）。
- 未报告 GSM8K 多数行的 test 结果（如 fewshot、bootstrap 单次无 test 列，"–"），部分结论仅基于 dev 集。
- 指标中含 EM/F1 外的"程序化指标"仅给出示例，未系统评估其效果。
- 未说明 teleprompter 在不同任务上的泛化一致性（只做了两个数据集）。
- 版本为 v1 预印本（5 Oct 2023）；后续 ICLR 2024 版本可能有差异，本文提取以该预印本为准。
