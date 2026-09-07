# The Rise and Potential of Large Language Model Based Agents: A Survey

## 基本信息
- paper_id: paper_006
- 年份: 2023（arXiv:2309.07864v3，2023-09-19）
- 作者: Zhiheng Xi 等（共同一作，含 Wenxiang Chen, Xin Guo, Wei He 等；通讯 Qi Zhang, Tao Gui），Fudan NLP Group
- 来源: arXiv（cs.AI）

## 研究问题
- 历史上 agent 研究多聚焦算法/训练策略以提升特定能力，缺乏能适应多样场景的通用基础模型；LLM 被视为构建通用 agent 的起点（Abstract、§1）。
- 论文目标：系统综述 LLM-based agent——给出通用框架（brain/perception/action）、梳理应用（单 agent、多 agent、人机协作）、讨论 agent 社会与开放问题。

## 方法
- 综述论文，无原创实验；通过文献梳理构建概念框架与分类学（typology）。
- 框架：agent = brain（§3.1）+ perception（§3.2）+ action（§3.3），可裁剪适配不同应用；运行流程为 感知→大脑处理→行动 的循环（§3）。
- brain 子组件：自然语言交互（§3.1.1：多轮对话、高质量生成、意图与隐含义理解）；知识（§3.1.2：语言学知识、常识知识、领域专业知识，及知识编辑/幻觉问题）；记忆（§3.1.3：提升长度限制、摘要记忆、向量/数据结构压缩；自动检索用 Recency/Relevance/Importance 加权）；推理与规划（§3.1.4：推理—CoT 等；规划—plan formulation 与 plan reflection 两阶段）；迁移与泛化（§3.1.5：未见任务泛化、in-context learning、持续学习）。
- perception 子组件：文本输入（§3.2.1）；视觉输入（§3.2.2：image captioning、ViT 类编码器+可学习接口层如 Q-Former、投影层）；听觉输入（§3.2.3：级联调用工具模型、频谱图+Transformer）；其他输入（§3.2.4：手势指点、眼动、LiDAR/GPS/IMU 等，文中列为未来方向）。
- action 子组件：文本输出（§3.3.1）；工具使用（§3.3.2：理解工具、学习使用工具、自制工具，工具可扩展行动空间）；具身行动（§3.3.3：observation / manipulation / navigation，成本效率、泛化、规划三个优势，Minecraft 等模拟环境）。
- agent 分类（§2.2 技术脉络）：symbolic agents → reactive agents → RL-based agents → transfer/meta learning agents → LLM-based agents。

## 关键发现
- LLM 适合作为 agent 大脑主件的四个属性（§2.3）：Autonomy、Reactivity、Pro-activeness、Social ability，各属性均给出 LLM 对应证据。
- 应用三分（§4）：单 agent（task-oriented、innovation-oriented、lifecycle-oriented 三类部署）；多 agent（cooperative interaction：disordered/ordered 两类；adversarial interaction）；人机交互（instructor-executor 与 equal partnership 两种范式）。
- agent 社会（§5）：个体行为分 input / internalizing / output 三类；群体行为分 positive / neutral / negative；个性维度为 cognition / emotion / character；社会环境分 text-based、virtual sandbox、physical 三类。
- 评价体系（§6.2）四维度：utility（任务完成成功率为主）、sociability（语言沟通、合作谈判、角色扮演）、values（诚实、无害、情境适应）、ability to evolve continually（持续学习、autotelic learning、新环境适应；文中称评价标准尚难建立）。
- 风险与开放问题（§6.3–6.5）：对抗鲁棒性、可信性、滥用；agent 数量扩展（pre-determined 与 dynamic scaling）；是否通向 AGI 的争论等。

## 全文相关性（fulltext_relevance）
- 全文相关性: high
- 理由: 该论文本身即对"LLM agent 基础"（框架、分类、应用、评价）的系统论述，直接覆盖调研问题；但其论断来自对他人文献的归纳，非本文原创实验。

## 引用
- 框架三组件与子组件：Section 3（图 2、图 3、图 4、图 5）
- LLM 作为大脑的四个属性：Section 2.3
- agent 技术脉络分类：Section 2.2
- 应用分类：Section 4（图 6、图 7）
- agent 社会：Section 5
- 评价四维度：Section 6.2

## 不确定项
- 论文未给出统一、可操作的 agent 评价指标集合，仅按四维度罗列已有做法，并自述"quantifying and objectively evaluating them remains a challenge"（§6.2 开头）。
- 持续演化能力（ability to evolve continually）的具体评价标准：文中明确说尚难建立，仅给初步建议。
- 对各子组件的论述以"有哪些代表性工作"为主，未提供跨方法对比的量化结果。
- 文中未说明三类应用/两种人机范式之间的边界判定标准（如 ordered/disordered 合作的精确定义）。
