# CEAM 相关调研

这组调研核对了原始论文的机制、监督信息和适用范围。它们沿用各报告注明的版本与核对日期；
外部论文中的成功率不属于 CEAM 复现实验。五组材料有论文重叠，篇数不应简单相加。

| 方向 | 来源范围与日期 | 核心发现 | 详细报告 |
|---|---|---|---|
| 本仓库既有想法与 CEAM 的对应 | Agent4VLAData 公开快照 | 条件动作表征与任务查询结构相通；分布学习和解码器并未在 CEAM 全部实现 | [参考核对](agent4vla_reference_review.md) |
| 长期匹配、价值与偏好 | 8 篇，10 月 2 日；Successor Features、FB、Contrastive RL、CPL、TD-JEPA 等 | 点积可以参与长期价值分解，但语义取决于训练目标、行为分布与后续策略 | [长期匹配](long_horizon_matching_review.md) |
| 跨 episode 效果与非重建学习 | 9 篇，10 月 3 日；DINO-WM、V-JEPA 2、C-SWM、LeWorldModel、SIGReg 等 | 不必以完整重建为目标；动作接地、共享关系和覆盖仍不可省略 | [跨 episode 表征](cross_episode_representation_research.md) |
| teacher 与数据 calibration | 5 篇，10 月 3 日；DemoDiffusion、FRS/DSBC、FlowDAgger、FAST、OneDP | 反演是编辑接口；压缩、校准和更优监督属于不同问题 | [teacher 校准](teacher_dataset_calibration_review.md) |
| action label 优化 | 8 篇，10 月 4 日；S2I、SkiP、HYDRA、SAIL、DemoDiffusion、OLAF、Beyond Success、MBTS | 标签优化有先例；几何、控制器、纠正语义、质量反馈和接续价值贡献不同 | [标签优化](action_label_optimization_review.md) |

## 与当前问题最相关的区别

- **非坍缩与动作接地。** 表征在全数据上丰富，不代表能区分同一起点的动作效果。
- **重建与后果预测。** 学习相对运动或接触信息不要求恢复完整像素或 3D，但也不能凭潜在预测精度认定物理可行。
- **即时效果与长期效用。** 任务无关 r 的输入边界，不意味着长期后果与后续策略无关。
- **反演与改善。** 精确编码再精确解码得到原动作；编辑需要另外改变潜变量、条件或目标。
- **归一化与监督。** 数据尺度或 codebook 可以改善工程表示，不会自行提供“哪条动作更好”的标签。
- **BC 标签与转移证据。** 原观测搭配新动作可以用于 BC；新动作对应的真实下一状态需要实际执行证据。
- **控制机制与数据增益。** 使用绝对 waypoint、改变在线控制器或速度调度的方法，不能只替换 CEAM 的增量动作数组后照搬结论。

后续 CEAM 测量补充在 [实验总览](../../reports/ceam/README.md)。其中 teacher pilot 和残差编辑发生在部分
调研之后；早期报告中的“未测量”描述其当时范围。最新状态以总览和 [开放问题](../../../synthesis/ceam/open_questions.md) 为准。
