# AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents
## 基本信息
- paper_id: paper_019
- 年份: 2024（arXiv:2401.12963v2，2024-07-02）
- 作者: Michael Ahn, Debidatta Dwibedi, Chelsea Finn, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Karol Hausman, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Sean Kirmani, Isabel Leal, Edward Lee, Sergey Levine, Yao Lu, Sharath Maddineni, Kanishka Rao, Dorsa Sadigh, Pannag Sanketi, Pierre Sermanet, Quan Vuong, Stefan Welker, Fei Xia, Ted Xiao, Peng Xu, Steve Xu, Zhuo Xu（Google DeepMind）
- 来源: arXiv preprint 2401.12963
## 研究问题
如何用基础模型（VLM+LLM）充当"机器人编排队列指挥者"，让机器人在未知真实环境中自主提出并执行任务，实现大规模、多样化机器人操作数据采集，且人类监督带宽有限（机器人数 > 监督人数）。
## 方法
- Agent 编排结构（每个 episode 循环一次）：
  1. 探索：VLM 构建自然语言地图（对象检测→视觉-语言嵌入 φi + SLAM 位置），按与历史对象的嵌入距离采样导航目标；
  2. 场景描述：VLM 输入当前图像，输出场景文字描述 + 5 个物体名；
  3. 任务生成：LLM 按 prompt（系统角色 + Robot Constitution + 场景/物体描述 + 当前采集策略专用后缀）生成任务列表；LLM 不微调；
  4. Affordance 过滤：另一 LLM 自我批判（类比 Reflexion/Constitutional AI），把每个任务分类到 k 个采集策略之一或 reject，最终从接受任务中随机采样执行；
  5. 数据采集：三种采集策略（teleop、scripted pick、RT-2），每 episode 采样一个，采样概率按人机监督比调整（1 人监督 3 台时 teleop 采样 p<1/3）；episode 后打多样性分并重置。
- Robot Constitution（写入 LLM prompt 的规则）：Foundational（Asimov 三定律改写：去掉"不作为"、互换二三定律顺序）、Safety（不碰人/动物、尖锐物、带电物）、Embodiment（载重上限、单臂限制）、Guidance（可选高层人类指令）。
- 多样性打分：语言多样性 = USE 归一化 512-d 嵌入的 L2 平均距离；视觉多样性 = CLIP（微调过）嵌入 + k-means（k=1000）到最近质心距离。
- Guardrails（Appendix C）：关节力超阈值急停、E-stop、机器人须在监督者视线内、预先移除危险物体、teleop 时人类对生成任务做 sanity check。
- 失败处理：affordance 过滤时 over-reject 优先；被 LLM 漏掉的不合适任务由 teleoperator 在采集时拒绝（human-in-the-loop 作为安全机制与干预数据来源）；RT-2 成功率低（4.7%）故降低其采样频率。
## 关键发现
- 产量：7 个月、4 栋建筑、共 53 台机器人（峰值 >20 台同时）、77,000 个真实机器人 episode、>6,650 条独特指令；1 人监督 3–5 台移动操作臂（固定式机器人 1 人最多 8 台）；新环境部署 <1 天。
- 按策略分布（Table 1）：scripted 73,293 episode（成功率 21%）、teleop 3,060（82%）、RT-2 936（4.7%）。
- 语言多样性（Table 2，平均 L2 距离）：AutoRT w/PaLI 1.100、w/FlexCap 1.137 > Language Table 0.988、BC-Z 1.070、RT-1 1.073（Optimal 1.414）。
- 任务生成质量（Table 3，75 任务）：可行性 templated 52% vs AutoRT 83%；引导相关性 27%/28% → 引导后 61%。
- 安全性（Table 4，对抗性场景）：同时用 constitutional 生成+过滤时 %Safe 最高（67%/83%）；64 场景 259 任务中基础可接受率 88%（228/259），过滤后 93%（200/214）；对 31 个应拒任务召回 55%（17/31），漏网 14 个全在 teleop 采样中被人类拒绝。
- 下游训练（Table 5）：RT-1 用 AutoRT 数据 co-finetune 后，不同高度抓取 0%→12.5%（0/24→3/24），擦拭 10%→30%（1/10→3/10）；仅用 teleop 子集则高度抓取回到 0%。
## 全文相关性（fulltext_relevance）
- 全文相关性: high
- 理由: 直接研究 VLM/LLM 驱动的机器人数据采集管线；给出 7 个月真实部署的产量、成功率、多样性量化指标与下游训练验证，且有安全消融实验。
## 引用
- Abstract/§1: 77k episode、20+ 机器人；§4.1 探索、§4.2 Robot Constitution、§4.3 任务生成、§4.4 Affordance、§4.5 数据采集（采样概率）；§5.1 Diversity Scoring（Table 1/2）；§5.2 Task Generation（Table 3）；§5.3 Affordance 与安全性（Table 4，55% 召回）；§5.4 Model Training（Table 5）；Appendix C Guardrails、Appendix D Prompts。
## 不确定项
- 所用 LLM 具体模型未具名（只比较了 VLM：PaLI vs FlexCap）；两类 VLM 的最终采集数据占比未给出。
- 77k episode 中成功/失败比例、失败的 episode 是否仍入数据集未说明；自动标注的确切输出格式（语言标签之外的元数据）未详述。
- 每机器人每日吞吐量只在 Appendix I 图 9 以图呈现，无数字。
- 安全评估基于人工标注的小样本（64 场景）；prompt 无法保证规则被遵守（论文自述局限）。
