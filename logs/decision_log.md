# Decision Log

## 2026-09-06

### seed 轮验证结论

- schema 有效：10 篇抽取全部 ≤80 行、字段完整、证据等级明确。
- 关键词有效：`cat:cs.RO AND abs:"data curation"` 直接命中 2025-2026 演示策展论文。
- seed 暴露的证据缺口（决定第二轮扩展）：
  1. RQ1.2（LfD 轨迹分割/关键帧/时间对齐）零证据。
  2. RQ1.3（次优演示过滤）只有 1 篇受控仿真研究，缺真实机器人数据上的过滤工作。
  3. RQ1.4/RQ2.3 语义策展只有定性案例，缺下游定量验证与机器人 agent 数据管线。
  4. RQ2.4（离线 agent 管线工程模式）零证据。
  5. RQ2.1 缺多 agent 结构与工具调用的一手文献。
- 决定：执行第二轮（≤20 篇），优先填补上述缺口。

### 第二轮候选（按缺口优先级）

| 缺口 | 候选 |
|---|---|
| RQ1.2 | LfD 综述（待搜）、关键帧提取（Akgun 2012 或 arxiv 替代） |
| RQ1.3 | Data Scaling Laws ([2410.21647](https://arxiv.org/abs/2410.21647))、AgiBot World ([2502.12294](https://arxiv.org/abs/2502.12294))、Curating Demonstrations using Online Experience ([2503.03707](https://arxiv.org/abs/2503.03707))、Phase-Localized negative ([2606.15064](https://arxiv.org/abs/2606.15064)) |
| RQ1.1/RQ2.4 | Robo-DM ([2505.15558](https://arxiv.org/abs/2505.15558)) |
| RQ2.1 | Wang agent survey ([2308.11432](https://arxiv.org/abs/2308.11432))、Toolformer ([2302.04761](https://arxiv.org/abs/2302.04761)) |
| RQ2.3 | AutoRT ([2401.12995](https://arxiv.org/abs/2401.12995))、MUTEX ([2503.12097](https://arxiv.org/abs/2503.12097)) |
| RQ2.4 | DSPy ([2310.03714](https://arxiv.org/abs/2310.03714))、TextGrad ([2406.07496](https://arxiv.org/abs/2406.07496)) |

### 其他决策

- 抽取脚本 check_*.py 输出编码问题（GBK 控制台）已修复（reconfigure utf-8）。
- metadata 一致性检查的路径比较已规范化（normpath），Windows 反斜杠误报消除。
- 抽取代理由 5 个并行 subagent 完成，每 agent 2 篇；证据等级在矩阵中沿用
  extraction 结论（[paper_002](https://arxiv.org/abs/2310.08864) 为 indirect：标准化部分 direct、质量过滤部分缺失）。


## 2026-09-08：收束总路线（与协作方确认）

决策：总目标从"表示误差/几何重采样"改为"vision/language/action 三模态对齐"。
- 触发：与协作方会议；几何重采样（跳帧、必要停顿被忽略）可能破坏模态对齐。
- 改变：总体路线 = 找到与 vision / 文本 prompt 对齐（或更好对齐）的 action
  模态增强方式；曲率重采样降为候选策略。事件窗时间保护的去留未评估。
- 提出：action 序列的模式对应（pick 的 instruction 可能与一类同模式 action
  序列对应，簇是举例、不一定显式）；sim 中良好但不可直接迁移的 action label
  对齐后可进微调增强。
- 调研缺口：对齐度量（文献无"度量即目标"的专门工作，GAP-12）。
- 提议（未定）：第一可测步骤 = H(l|v) + prompt 到 action 的互信息/对齐度。
- 待评估：旧问题（次优/停顿/关键帧/歧义）是否属本框架责任范围。
