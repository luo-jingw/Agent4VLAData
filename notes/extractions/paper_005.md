# What Demonstration Curation Metrics Do to Your Policy

## 基本信息
- paper_id: paper_005
- 年份: 2026（arXiv:2606.10229v1, 8 Jun 2026；文中未注明会议/期刊，仅标 cs.RO）
- 作者: Aarav Bedi（UC Berkeley 机械工程系）
- 来源: arXiv 预印本

## 研究问题
检测缺陷演示（demonstration）的 curation 指标，其"缺陷检测能力"是否等价于"提升下游 BC（behavior cloning）策略质量的能力"？两个量是否脱钩；同时检验 episode 长度是否构成检测评估中的混杂因素。

## 方法
- 任务：LIBERO pick-and-place（接触丰富仿真，基于 robosuite），四阶段：PREGRASP / DESCEND / LIFT / TRANSPORT。策略为 phase-conditioned BC：观察 26 维、动作 4 维、MLP（2 层 × 256 单元，tanh），Adam lr=1e-3、wd=1e-4。
- 缺陷注入：结构性缺陷——LIFT 阶段随机时刻（该阶段 30%–70% 处）提前张开夹爪；收集 80 条演示（16 干净 + 64 缺陷，污染率 80%）。污染基线成功率 3.3%；oracle（仅 16 条干净演示）93.3%。
- 七种 curation 指标（对 T=324 截断轨迹各输出一个标量质量分）：smoothness（SPARC，动作速度谱）、entropy（动作序列标准差的负值）、gripper timing（夹爪首次张开时刻/T，作者自创的内容感知指标）、isolation forest（动作摘要特征离群）、ensemble（0.5×smoothness+0.5×gripper timing）、kNN（k=5，轨迹级特征空间负平均距离）、trajectory alignment（与干净集平均状态轨迹的余弦相似度）。
- 双轴评估：检测 AUROC（对真实成功标签）；下游（保留 top-75% 即 60/80 条，30 rollouts，3 种子 42/0/7，报告均±std）。
- 长度控制：截断所有演示到 T=324（最小成功长度）。

## 关键发现
- 检测与下游强烈脱钩：七指标的 AUROC 与下游成功率的 Spearman 相关 ρ=−0.14（Section V.B）。AUROC 最高者 gripper timing（0.804）下游最差（13.3%，仅略高于污染基线 3.3%）；AUROC 第二低者 trajectory alignment（0.638）下游 90.0%，距 oracle 93.3% 仅 3.3 pp。
- 截断后各指标 AUROC（原始→截断）：length(cumulative path) 1.000→0.500、ensemble 1.000→0.761、isolation forest 1.000→0.440、kNN 1.000→0.712、trajectory alignment 1.000→0.638、smoothness 0.979→0.447、gripper timing 0.957→0.804、entropy 0.000→0.280（Table I）。
- 下游成功率（均值±std，%）：ensemble 91.1±1.6、trajectory alignment 90.0±0.0、entropy 77.8±12.6、smoothness 63.3±30.7、kNN 58.9±41.7、gripper timing 13.3±16.6、isolation forest 3.3±0.0；oracle 93.3±0.0、污染基线 3.3±0.0（Table II）。
- 长度混杂：成功演示约 325 步终止，缺陷演示跑满 500 步时限；91% 缺陷演示在 324 步前已释放夹爪。使用均值/累积特征的五个指标（ensemble、isolation forest、kNN、trajectory alignment、smoothness）原始 AUROC 近 1.0，截断后大幅下降——它们测的是 episode 长度而非缺陷内容。
- 脱钩机制（Section VI.A）：collateral removal（误删与缺陷共享表面特征的干净演示，造成分布缺口）与 ranking granularity（高 AUROC 只奖励全序精细排序，curation 只取粗阈值 top-k）。
- isolation forest 失效原因：提前释放后手臂以近零夹爪动作继续运动，缺陷演示反而在动作特征空间中比干净演示（含高幅值抓握闭合）更"不异常"，故被优先保留（Section V.D）。
- 高方差指标（kNN ±41.7、smoothness ±30.7、entropy ±12.6 pp）单次 curation 不可靠；ensemble、trajectory alignment 方差近零（±1.6、±0.0 pp）。

## 全文相关性（fulltext_relevance）
- 全文相关性: high
- 理由: 构造了带真实缺陷标签的受控测试床，直接回答"检测 AUROC 与下游 BC 策略提升是否脱钩"这一调研核心问题，并给出逐指标定量结果与混杂因素消融。

## 引用
- 问题与结论：Abstract、Section I（Introduction）
- 实验设置：Section III（A 任务/策略、B 缺陷注入、C 评估协议、D 长度控制）
- 指标定义：Section IV
- 主要结果与 ρ=−0.14：Section V.B、Table II、Fig. 1
- 长度混杂数字：Section III.D、Table I
- 机制解释：Section VI.A；局限性：Section VI.C

## 不确定项
- Table I 中 "length (cumulative path)" 指标（AUROC 1.000→0.500）在 Section IV 的七个指标定义中未出现，文中未说明它是独立指标还是仅作长度代理的诊断项。
- "五个指标利用 episode 长度"的具体清单由抽象表述 + Table I 推断（ensemble、isolation forest、kNN、trajectory alignment、smoothness），正文未逐一列名。
- 80% 污染率高于作者自述的现实典型值 20–40%；作者明确表示结论是否跨缺陷类型成立"是未回答的经验问题"（Section VI.C）。
- 只测单一任务、单缺陷类型、3 个种子；每个种子的具体 AUROC 值（Table II 中 AUROC 无方差）未报告。
- 下游评估 rollouts 数（30/种子）与干净演示训练时的 10 rollouts 口径不一致，文中未解释。
