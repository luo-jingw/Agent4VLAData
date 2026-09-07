# LLM Trainer: Automated Robotic Data Generation via Demonstration Augmentation using LLMs
## 基本信息
- paper_id: paper_021
- 年份: 2025（arXiv:2509.20070v2，v2 日期 2026-05-31）
- 作者: Abraham George, Amir Barati Farimani（Carnegie Mellon University, Dept. of Mechanical Engineering）
- 来源: arXiv preprint 2509.20070
## 研究问题
如何利用 LLM 的世界知识，将极少量人类示范（少至 1 条）加一句任务描述，自动增强为大规模机器人模仿学习数据集——替代 MimicGen/OneACTPlay 中的人工关键点标注与硬编码目标相关变换。
## 方法
- 分解为两个函数：F: (D,s)→(K,D̂) 离线示范标注；G: (D̂,s,o_new,K)→K′ 在线关键位姿重定向。F 与场景无关故只需运行一次、可跨场景复用。
- 标注流程（LLM=GPT-4o，输入含文本+图像）：先给 LLM 初始/最终观察 + 每约 5 步采样一次的物体位姿（加噪声防时间步依赖），LLM 选出重要时间步；再由新的 LLM 实例结合图像与位姿给出关键位姿列表、每个关键位姿的相关物体、以及物体位姿变化时关键位姿的修改指令；LLM 报错/格式不符则丢弃并重启；自动补上起止位姿、用记录值纠正 LLM 错报的位姿。
- 轨迹扭曲：同 OneACTPlay 的刚体变换（对齐新旧起点与终点、约束 z 轴向上），旋转用起点/终点 delta 旋转线性插值。
- 标注优化：将数据生成建模为多臂老虎机（每 arm = 一个标注，二值奖励），Thompson Sampling（Beta 先验）；新增 arm 的期望价值 E_add 用 k=1000 采样近似的虚拟 TS rollout 估计，仅当 E_add > E_T(P) 时生成新标注；优化目标是最少 rollout 数达到目标成功示范数（不计 LLM 调用成本）。
- 数据保存：rollout 成功才保存为示范，失败丢弃。
- 集成策略：前馈 LLM 轨迹 + IL 反馈纠正——幅度感知余弦相似度衡量两策略动作分歧，分歧超阈值切换，重附着条件式 t* 选择（双阈值 τ + 5 timestep cooldown）。
- 硬件位姿估计：LLM 描述 + SAM + Grounding DINO 提取物体掩码与点云，RANSAC+ICP 配准到参考点云求位姿。
## 关键发现
- 生成成功率（Table II，MimicGen 任务生成 1000 成功示范/任务、OneACTPlay 任务 400/任务）：最佳标注下 Mug Cleanup 44% vs 基线 29.5%、Kitchen 60 vs 42.7、Coffee 89 vs 78.2、Pick&Place 92 vs 81.7、Stack 89 vs 80.0、Stack Flipped 58（基线 N/A）；最佳标注较新标注提高 2–3 倍；总成功率（含探索期）除 Stack 外全部超过人工标注基线。
- LLM 推理泛化：Stack Flipped 任务仅凭视觉线索（目标块颜色互换）即调整堆叠顺序，硬编码基线做不到。
- 训练下游 IL：用 LLM 生成数据训练的 IL agent 与 MimicGen 数据相当，优于 OneACTPlay 数据（作者归因于 LLM 生成随机性增加数据集多样性）。
- 集成策略在低数据 regime 显著提升 IL 性能（如 Stack 50 demos：IL 4.8±4.1 → Ens 29.6±7.8；Pick&Place 25 demos：30.4±6.0 → 48.8±5.6），高数据 regime 偶有下降（不可恢复失败时来不及切换）。
- 前馈 LLM 策略脆弱：walking stack 任务仅 14% 成功率。
- 硬件（Franka Panda mug cleanup，物体位置/朝向随机化）：生成 100 成功示范 + 32 失败，总成功率 75.8%；最佳标注平均 82%；无优化仅 45%；IL 60%、前馈 80%、集成 85%（各 20 trials）。
## 证据等级
- 等级: direct
- 理由: 直接研究 LLM 驱动的示范数据增强；在标准基准（RoboMimic/MimicGen、OneACTPlay）上与人工标注基线逐任务对比生成成功率与下游 IL 性能，并有真实硬件端到端验证。
## 引用
- §III-A 假设、§III-B 关键位姿识别与修改（LLM 两阶段）、§III-C 轨迹扭曲、§III-D 多臂老虎机优化、§III-E 集成、§III-F 硬件位姿估计；§IV Results（Table II 生成成功率、Table III/IV 下游 IL 与集成、walking stack 14%）；§IV-A Hardware Experiments（75.8%/82%/45%/60%/80%/85%）；§V 局限（依赖 rollout 验证、优化目标不计 LLM 成本）。
## 不确定项
- 数据质量问题只处理了"生成成功率 + 失败丢弃"：示范最优性、次优示范的取舍、生成数据的状态分布偏移未讨论。
- "LLM 随机性增加数据集多样性"是下游性能优于 OneACTPlay 的事后解释，未直接测量生成数据的多样性指标。
- 集成策略的相似度阈值 τ、切换与重附着参数未给出具体数值。
- 未说明 GPT-4o 的采样参数（温度等）与 prompt 全文；硬件实验仅单任务、无重复组，硬件数据生成的总次数除 mug cleanup 外未报告。
- 示范增强未涉及语言标签/多模态任务规范的生成（与 MUTEX 式标注的关系未讨论）。
