# A Survey on Large Language Model based Autonomous Agents

## 基本信息
- paper_id: paper_017
- 年份: 2023（arXiv:2308.11432 首版；v7 为 2025-03-02，期刊版 2025）
- 作者: Lei Wang, Chen Ma*, Xueyang Feng*, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhi-Yuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen（中国人民大学高瓴人工智能学院）
- 来源: arXiv:2308.11432；Frontiers of Computer Science, 2025, DOI 10.1007/s11704-024-40231-1

## 研究问题
系统综述 LLM 自主 agent 的构建（架构设计 + 能力获取）、应用（社会科学/自然科学/工程）与评估（主观/客观），提出可涵盖多数既有工作的统一框架，并总结挑战与未来方向（Abstract、Section 1）。

## 方法
- 综述组织：按"构建—应用—评估"三轴；Fig. 1 给出 2021-01 至 2023-08 论文增长时间线；Table 1–3 将各工作映射到自建分类法；Section 5 自述收录约 100 篇相关工作。
- 统一框架（Section 2.1, Fig. 2）：四模块——Profile（角色画像）、Memory（记忆）、Planning（规划）、Action（行动）；关系为"profile 影响 memory 与 planning，三者共同影响 action"（Section 2.1 末段）。
- Profile（2.1.1）：内容三类——demographic（年龄/性别/职业）、personality、social（agent 间关系）；生成策略三种——Handcrafting（手工指定，如 Generative Agents、MetaGPT、ChatDev）、LLM-generation（种子 profile + LLM 扩写，如 RecAgent 用 ChatGPT）、Dataset Alignment（取自真实人群数据，如 ANES 参与者背景赋给 GPT-3）；Remark 建议组合使用。
- Memory（2.1.2）：结构——Unified（仅短时记忆，经 in-context learning 直接写入 prompt，如 RLP、SayPlan、CALYPSO、DEPS）与 Hybrid（短时 + 长时向量库，如 Generative Agents、AgentSims、GITM、Reflexion）；Remark：仅有长时记忆的结构文献中罕见。格式——自然语言、Embeddings（如 MemoryBank 双塔检索）、Databases（如 ChatDB 用 SQL 操作）、Structured Lists（如 GITM 分层树、RET-LLM 三元组），可混用（GITM：key 为 embedding、value 为自然语言）。操作——Reading（按 recency/relevance/importance 加权提取，式 (1)）；Writing（去重：GITM 同类序列满 N=5 条后用 LLM 压缩成统一方案；溢出：如 RET-LLM 定长缓冲 FIFO 覆盖）；Reflection（归纳高层见解：Generative Agents 由近期记忆生成 3 个问题、检索后产出 5 条 insight，可层级递归；GITM、ExpeL 类似）。
- Planning（2.1.3）：按有无反馈二分。无反馈——Single-path（CoT、Zero-shot-CoT、Re-Prompting、ReWOO、HuggingGPT、SWIFTSAGE）；Multi-path（CoT-SC 取最高频答案、ToT 树 + BFS/DFS、RecMind、GoT 图结构、AoT、LMZSP 零样本规划、RAP 用 MCTS + 世界模型）；External Planner（LLM+P、LLM-DP 转 PDDL 交外部规划器、CO-LLM 高层 LLM 配底层启发式规划器）。有反馈——Environment（ReAct 的 thought-act-observation、Voyager 三种环境反馈、DEPS 失败原因、Inner Monologue 三类反馈等）、Human（Inner Monologue 主动询问）、Model（Self-Refine、SelfCheck、Reflexion 的言语化反馈等）。
- Action（2.1.4）：四个视角——Goal（task completion / communication / environment exploration）；Production（memory recollection 或 plan following）；Space（外部工具：APIs、Databases & KB、External Models；内部知识：LLM 的 planning / conversation / commonsense 能力）；Impact（改变环境 / 改变自身内部状态 / 触发新动作）。
- 能力获取（2.2）：fine-tuning（人类标注数据、LLM 生成数据、真实世界数据）vs 不 fine-tuning（prompt engineering；mechanism engineering 四种：trial-and-error、crowd-sourcing、experience accumulation、self-driven evolution）；Fig. 4 提出"参数学习 → 提示工程 → 机制工程"的范式迁移图。

## 关键发现
- 提出 profile/memory/planning/action 统一框架，宣称可涵盖此前多数研究（Section 2.1 开头）。
- 多 agent 协作：本文无独立分类章节，相关内容散见——action goal 的 communication（ChatDev、Inner Monologue）；crowd-sourcing 机制即多 agent 辩论迭代至共识 [94]；软件工程多角色协作（ChatDev、MetaGPT、Self-collaboration [24] 的"虚拟团队"）；多 agent 系统（SALLM-MS、NLSOM 动态调整角色、CGMI）；评估侧 ChatEval 用多 agent 辩论做评估、RocoBench 测多机器人协作。未见合作/竞争/辩论等模式的系统分类。
- 应用分类（Section 3, Fig. 5）：社会科学（心理学、政治与经济、社会模拟、法学、研究助理）、自然科学（文献与数据管理、实验助手、自然科学教育）、工程（CS&软件工程、工业自动化、机器人&具身 AI）。
- 评估（Section 4）：主观（人工标注、图灵测试）+ 客观（指标：任务成功率/reward/覆盖率/准确率、类人性、效率；协议：真实世界模拟、社会评估、多任务评估、软件测试；基准：ALFWorld、AgentBench、SocKET、WebArena 等）。
- 六大挑战（Section 6）：role-playing capability、generalized human alignment（模拟需允许"错误价值观"角色）、prompt robustness、hallucination、knowledge boundary（需约束 LLM 不用用户未知的知识）、efficiency（每次动作多次查询 LLM 导致推理慢）。
- 与 Xi 等综述（arXiv:2309.07864）框架的关系：文中未提及该综述；Section 5 Related Surveys 仅列 LLM 通用综述 [175]–[181]，无 Xi 等。其 profile/memory/planning/action 划分与 Xi 等 brain/perception/action 式划分的异同，本文未讨论。

## 候选相关性（总体）
- 总体相关性: high
- 理由: 本综述是 profile/memory/planning/action 统一框架的一手来源，四模块子组件、能力获取策略、多 agent 相关散述均可直接引用作教学素材；"与 Xi 框架异同"文中未讨论，已显式标注为未提及。

## 引用
- 统一框架：Section 2.1 开头、Fig. 2；Profile：2.1.1；Memory：2.1.2（式 (1) 在 Memory Reading）；Planning：2.1.3、Fig. 3；Action：2.1.4；能力获取：2.2、Fig. 4、Table 1。
- 应用：Section 3、Fig. 5、Table 2；评估：Section 4、Table 3；挑战：Section 6；相关综述与 100 篇收录数：Section 5。

## 不确定项
- 与 Xi 等综述的框架异同仅能从两文各自划分推断，本文未提供任何对照说明。
- 多 agent 协作无系统分类，无法直接提取可教学的"协作模式"结构（如 cooperative/competitive/debate 等维度均未提出）。
- Profile 影响 memory/planning/action 仅为定性声明（"influence"），无实验证据或影响程度说明。
- 式 (1) 的权重 α/β/γ 只举例（α=γ=0；或全 1.0），未给出一般设定原则。
- "约 100 篇相关工作"未附完整清单；期刊版卷期为占位符"0(0)"，正式卷期未定。
