# CEAM 结构化结果

八份 JSON 来自已完成实验的本地结果汇总。每份文件的来源和校验和见
[来源清单](../../../../synthesis/ceam/source_manifest.json)。下列实验的采样单位和协议不同，不应拼成同一条性能曲线。

| 文件 | 单位与内容 | 解释边界 |
|---|---|---|
| [学生策略](student_policy_comparison.json) | 3 数据版本×3 学生种子，每格 500 局；配对差值和区间 | 唯一完成的编辑后学生网格；相对原始收益区间包含零 |
| [LIBERO-90 BC](libero90_baseline.json) | 90 任务×50 初态×3 种子；任务和场景汇总 | 普通 BC，不是双表征结果 |
| [残差候选可用性](residual_availability.json) | 20 起点、1,473 候选；各类全库与等预算选择 | 完整候选库大小不同；验后 oracle 不是学习选择 |
| [残差排序](residual_ranking.json) | 同输入直接/双表征对照、训练目标和三种子 | 候选对共享起点；不是新策略 SR |
| [残差后续诊断](residual_continuation.json) | 源后续、冻结策略后续和精确恢复子集 | 测量特定后续的衔接能力 |
| [Teacher 汇总](teacher_editing_summary.json) | 等幅编辑机制及 teacher-only 参照 | 非法 proposal 计入预算；teacher-only 幅度和夹爪不同 |
| [Teacher 任务与种子](teacher_editing_by_task_seed.json) | 任务、候选采样种子及质量统计 | 采样种子不是学生训练种子 |
| [偏好排序](preference_ranking.json) | 原始匹配、固定/适配 r 及监督消融 | 留出源已被其他研究检查过，不是全新任务测试 |

表格和解释见 [完整历史](../research_history.md)。这些文件能核对汇总数值，但不能替代逐步仿真轨迹、
检查点或完整训练代码进行独立复现。
