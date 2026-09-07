# 论文标题
RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data for Robot Manipulation

## 基本信息
- paper_id: paper_030
- 年份: 2025（arXiv v3：2025-05-27）
- 作者: Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju 等（北京人形机器人创新中心、北京大学、北京智源人工智能研究院）
- 来源: arXiv:2412.13877v3 [cs.RO]

## 研究问题
构建大规模、多本体机器人操作数据集，回应现有数据集"缺乏统一数据收集标准、高质量数据不足"的问题；并验证其对单任务模仿学习与 VLA 模型的可用性。

## 方法
- 统一数据平台 + 标准化收集协议，遥操收集 107k 条轨迹、305.5 小时、479 个任务、96 类物体、38 项技能、4 种本体（Franka 26,866 条真实 + 模拟约 26,070 条；Tien Kung 15,187；AgileX 10,269；UR5e 25,170；模拟共 30,035）。
- 数据形态：单条轨迹打包为 H5（多视角 RGB-D + 本体状态 + 末端状态 + 遥操身体状态 + 语言任务描述）。
- 质量保证（Section III-B, Fig.4）：8 条 QA 标准（Touch Excess、Movement not Smooth、Secondary Grabbing、Mechanical Arm Shaking、Collision before Grabbing、Image Distortion、Failed Placement、Gripper out of the Camera）；三步检查：Initial Inspection（帧丢失/冻结）、Detailed Inspection（逐帧/慢放核对 8 条标准）、Data Filtering and Issue Logging（记录时间戳与原因并归类）。
- 预处理过滤标准：任务执行准确性、运动轨迹平滑度、视觉遮挡/运动模糊（Section III 平台功能 3）。
- 分类：task-centric 协议，任务名由 4 要素定义（本体、操作技能、物体、场景描述，Section III-B）。
- 附带数据：5k 条真实失败轨迹（带原因）；10k 条帧级语言标注（Gemini 分段生成 + 人工修订，多评审验证）。
- 实验：单任务 IL（ACT、Diffusion Policy、BAKU）45 个任务；VLA（OpenVLA、RDT-1B、CrossFormer）微调与预训练；每任务 10 次实机试验计成功率。

## 关键发现
- "Normative" 的操作化含义：统一收集平台 + 标准化协议（unified data collection standard），使全部数据在相似条件下采集以"减少变异性与噪声、保证一致性与可靠性、ready-to-use"；文中未给出该术语的抽象定义，也无量化"规范性"指标。
- 训练/验证划分：无按任务/场景/episode 的显式 held-out 划分协议。单任务 IL 按任务训练与评测（45 任务，10 次/任务）；VLA 按机器人类型用 multitask 数据微调后在各任务上评测；预训练用全量 107k，微调用"expert multitask datasets"（约 1% 子集，Section V-D）；评测任务与训练数据的重叠关系未声明。
- 覆盖率定义：以多样性分析呈现（本体分布、任务时长/每轨迹 skill 数分布、6 类任务分类、96 类物体按 5 场景分布），无形式化覆盖度量（如每任务最少轨迹数）；失败案例分析指出数据短板（物体放置非随机、夹爪闭合帧数不足）并建议补采（Section V-F）。
- 实验结果：ACT 平均成功率 Franka 30.7%、UR5e 38.0%、Tien Kung 34.0%、AgileX 55.3%（Section V-B）；RDT-1B 微调整体优于 OpenVLA/CrossFormer（Table III）；全量预训练显著提升成功率（CrossFormer 双任务从 0/10 到近全对，Table IV）；含 humanoid 数据预训练带来 13.3% 相对提升（0.68 vs 0.6，Section V-D）。
- sim-real：100 真实 + 500 模拟共训，仿真内成功率 90%，纯模拟实机仅 10%（FR-UprightBlueCup）；sim/real 结果 Pearson 相关 ACT 0.83、DP 0.91（Section V-G）。

## 全文相关性（fulltext_relevance）
- 全文相关性: high
- 理由: 对 RQ3.1（数据质量标准），论文直接给出完整操作机制（8 条 QA 标准 + 三步检查 + 过滤标准 + 失败数据）；对 RQ3.5（训练/验证划分协议）仅部分覆盖，无显式划分描述，已在不确定项标注。

## 引用
- "Normative"/统一标准表述：Abstract、Section I、Section IV-B "Standardized Settings"
- QA 标准与三步检查：Section III-B、Fig.4
- 失败数据 5k：Abstract、Section IV-B "Failure Case Demonstrations"
- 实验协议（10 次/任务、1% 子集微调）：Section V-A/B/D
- 轨迹分布数字：Section I（26,856/15,187/10,269/25,170/30,035；305.5 小时）

## 不确定项
- 训练/验证是否按任务、场景或 episode 划分未说明；评测任务是否包含在训练数据中未声明。
- "Normative" 无形式化定义，仅以标准化流程间接体现；被 QA 过滤的轨迹数量/比例未报告。
- 覆盖率定义未形式化；每任务轨迹数分布、场景均衡性未量化。
- 多视角相机间的时间同步/对齐机制未涉及。
