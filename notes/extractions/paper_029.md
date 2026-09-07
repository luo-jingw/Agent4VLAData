# Resolving State Ambiguity in Robot Manipulation via Adaptive Working Memory Recoding
## 基本信息
- paper_id: paper_029
- 年份: 2025（arXiv v1，2512.24638，2025-12-31）
- 作者: Qingda Hu、Ziheng Qiu、Zijun Xu、Kaizhao Zhang、Xizhou Bu、Zuolei Sun、Bo Zhang、Jieru Zhao、Zhongxue Gan、Wenchao Ding（复旦大学、Westwell、上海交通大学）
- 来源: arXiv:2512.24638（预印本）
## 研究问题
状态歧义（state ambiguity）：相同观测可能对应多条有效行为轨迹，只有结合任务上下文（历史）才能确定当前任务阶段。多数 visuomotor 策略依赖马尔可夫假设、只用短期观测，而朴素延长历史窗口计算昂贵且易严重过拟合。论文提出 PAM（带自适应工作记忆的 visuomotor policy），以可负担成本支持 300 帧历史窗口并维持实时推理。
## 方法
- 状态歧义的界定（Fig. 1 五类场景）：(a) 双向重叠轨迹（bidirectional overlap）；(b) 固定时长静止姿态；(c) 顺序子任务依赖；(d) 重复时间循环；(e) 动态交互响应。歧义不靠数据标注解决，靠模型架构处理历史。
- PAM 架构（Sec III-B）：
  - 因果帧特征提取器：DINOv2 图像编码器+adapter；MiniLM（6 层）编码指令；3 层 MLP 编码 proprioception；Transformer（causal mask）加两个 query token——q_act 提动作原语（motion primitives）e_act，q_ctx 提上下文特征 c_t（作为"重编码工作记忆"，可联合注意低层感知线索与高层决策特征）。
  - Context router：历史各步的上下文特征组成序列，用一组跨越不同历史长度的 range-specific query token 提取分层摘要（同时覆盖短程与长程信息），经自注意力+FFN 得紧凑表示 e_ctx；历史不足时叠加 validity mask。
  - 多条件动作头：flow matching 预测动作块；四选一注入方式（拼接/交错交叉注意力两种顺序/ControlNet 式），默认"先注入动作原语、后注入上下文"的交错交叉注意力。
  - 辅助目标：仅以 e_ctx 为条件，用 decoder-only transformer 重建历史图像的 VAE patch 嵌入（L_aux,img = ‖Ê⁻ᴸ − E⁻ᴸ‖₂²）；λ_aux = 0.4，推理时禁用。
- 两阶段训练（Sec III-C）：Stage 1 只用动作原语训练（e_ctx 相关参数不激活，学稳定多模态帧提取器）；Stage 2 冻结提取器并缓存 KV，无需原始感知输入，训练 context router 与辅助头，保持高效并行采样。总损失 L = L_FM + λ_aux·L_aux。
- 推理：每步只编码当前观测，把上下文特征作为工作记忆跨步传递；300 帧窗口（约 10 秒），推理 ≥20Hz。
## 关键发现
- 真机 7 任务（Agilex Piper 六轴臂，320×240 RGB 全局+腕部相机，每任务 50 条 30Hz 遥操作演示，每任务 20 次独立评估，A100 训练 / RTX4060 部署）：Table I 平均成功率 PAM 0.91 vs LongDP（300 帧窗口）0.61、MTIL（∞ 窗口）0.45；相对提升 49.2%（对 LongDP）与 102.2%（对 MTIL）。PAM 分任务：Buttons 0.80、Hold Pot Lid 0.95、Sponge and Square 0.96、Exchange Objects 0.90、Wipe Once 1.00、Wipe Twice 0.89、Guessing Game 0.90；PAM w/ aux head 平均同为 0.91（Guessing Game 0.97 显著受益）。
- 推理频率：PAM 20Hz、MTIL 20Hz、LongDP 5Hz。
- 任务细节：Wipe Twice 第一次与第三次经过右桌侧相隔约 210 帧；Guessing Game 方块被遮挡到允许行动约 100 帧。
- Libero-Long 10 任务（窗口设为 50）：PAM 0.847 vs π0 0.852、MDT 0.653、DP-T 0.582、OpenVLA 0.544；每任务 5 seeds × 20 次 = 100 次试验。
- 消融（Table III，平均成功率）：视觉编码器 ViT-Base 0.91 / ViT-Small 0.72；注入方式 后注入上下文 0.91、并行 0.89、先注入上下文 0.72、拼接 0.69；context router 平均池化 0.59、1 query 0.75、3 queries 0.91、5 queries 0.80、3 queries+avg 0.71；采样间隔 5→0.81、10→0.81、15→0.91、20→0.86、25→0.83。
- 可解释性（Fig. 4，定性）：context router 注意力图显示 PAM 在 Wipe Twice 中准确引用前一任务阶段的关键帧；extractor 注意力图显示 Guessing Game 中上下文 query 提取方块位置视觉线索、动作原语关注关节状态。
## 证据等级
- 等级: direct
- 理由: 状态歧义（RQ3.4 状态别名）定义与解法的一手方法论文；真机消融与对比数字直接来自正文。
## 引用
- 状态歧义定义与五类场景：Sec I、Fig. 1（第 1–2 页）。
- 时序依赖推理公式 Eq. (1) 与 300 帧窗口理由：Sec III-A（第 3 页）。
- 架构与辅助目标：Sec III-B（第 3–4 页）；两阶段训练：Sec III-C（第 4 页）。
- 真机主结果：Table I（第 6 页）；Libero-Long：Table II（第 7 页）；消融：Table III（第 7 页）。
## 不确定项
- 与"逐帧观测可分性标注"的距离：PAM 完全不做数据标注，也不检测哪些帧有歧义——它改模型输入表征（历史重编码为工作记忆），属模型侧方案而非数据侧方案。文中明确声明自适应重编码"不需要额外标注"（Sec I 贡献 1）。作为 RQ3.4 的证据，它只能说明歧义可通过模型架构缓解，不能提供"哪些观测不可分"的逐帧标签或检测算法。
- 状态歧义没有形式化定义：只给了五类场景示意与两个示例任务，无形式化判据，也无逐帧歧义度量（如观测相同但标签不同的统计）。
- 300 帧窗口长度的选择理由为"覆盖绝大多数场景"（经验性，Sec III-A），更长的场景被推给高层任务规划器，未给出判定边界。
- 歧义任务的场景类别是手工设计的，未说明是否存在同类任务失败案例；五类场景每类至少覆盖一个任务的对应关系（Fig. 3 右上角标注）未在正文文字中逐一解释。
- 辅助目标权重过大导致动作平滑度下降（Sec IV-B）为定性观察，无量化。
- 单任务训练与多任务训练不对齐：PAM 统一多任务训练，LongDP/MTIL 单任务训练，比较口径存在差异（正文已声明）。
