# SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning

## 基本信息
- paper_id: paper_033
- 年份: 2025（arXiv v2：2025-09-09）
- 作者: Yu Zhang, Yuqi Xie, Huihan Liu, Rutav Shah, Michael Wan, Linxi "Jim" Fan, Yuke Zhu（UT Austin / NVIDIA Research）
- 来源: arXiv:2505.22626v2 [cs.RO]

## 研究问题
大规模模仿学习数据质量不均（次优动作 + 冗余重复），现有策展依赖人工标注或粗粒度（轨迹/数据集级），无法评估单个 state-action 对质量。SCIZOR 提出首个无标注、自监督、transition 级策展框架，过滤次优与冗余 state-action 对并验证下游策略收益。

## 方法
- 演示质量信号（无 LLM、无奖励、无人工标注）：以"两帧间真实流逝时间 T"作为任务进度代理标签，训练自监督任务进度预测器（temporal distance classification，借鉴 [16–18]）：
  - 输入：子轨迹首末两帧图像 → 冻结 DINO-V2 编码 → 差分特征 + CLS token → transformer → 分类头预测进度 bin；B=5 个 bin：[0,0.5),[0.5,1),[1,2),[2,5),[5,+∞)；子轨迹窗口固定 2s；每轨迹均分为 5 个时间 bin 采样训练（Sec.3.2、App.A.2/A.3/A.5）。
  - 次优分数：V_{i:i+T} = T − T_p（实际时长 − 预测进度）；预测进度显著低于流逝时间即判定"落后于进度/次优"（Sec.3.2）。
- 单帧分数聚合：子轨迹分数均摊至各帧 → 未来次优影响按 γ 时间折扣 → 与整条轨迹平均分按 α=0.5 混合（V_final = α·V_i + (1−α)·mean），兼顾局部与全局质量（Sec.3.2）。
- 删除规则：分数超阈值 ϵ_s=0.58 的样本删除（阈值在 RoboMimic/OXEMagic 上调参后直接用于真机 Sirius-Fleet）（App.A.1）。
- 冗余去重：Cosmos 视频编码器抽取 chunk 特征 z_v，拼接 delta 末端位姿动作得联合特征 z_{v+a}（chunk N=8 帧）→ K-means 聚类 → 簇内最大余弦相似度超阈值 ϵ_d=0.99 判重复删除（Sec.3.3、App.A.7）。
- 与 Demo-SCORE 的异同（Sec.2）：Demo-SCORE 为轨迹级策展、依赖在线 rollout 性能；SCIZOR 为 transition 级、完全自监督离线、无需 rollout。实验对比的是 DemInf（轨迹级）与 Re-Mix（数据集级），未直接跑 Demo-SCORE。

## 关键发现
- 相对全量数据：RoboMimic +5.4%、OXEMagic +8.1%、Sirius-Fleet（真机）+32.9% 绝对成功率；摘要口径平均提升 15.4%；比 Uniform 随机删除平均高 16.1%（Sec.4.2 RQ1）。
- 细粒度优势：比 Re-Mix 平均 +3.5%；比 DemInf 在 Sirius-Fleet +19.2%，但在 RoboMimic 未超过 DemInf（该数据集有显式三级轨迹质量标签，轨迹级过滤有效）（RQ2）。
- 消融：仅次优删除或仅去重均不如完整版（Table 1）；去掉"帧-轨迹混合"或"时间折扣"均降性能（Table 2）。
- 删除比例（统一阈值）：RoboMimic 29.6%、Sirius-Fleet 7.9%、OXEMagic 15.8%、OXERT-X 15.8%、OXERT-1 9.7%（Table 3）。
- 次优类型人工分类（每数据集 100 条轨迹）：Lagging / Move Back and Forth、Manipulation Failure、Pause、Stuck at Collision、False Positive（Fig.4）。
- 错误分析：Sirius-Fleet 5% 误判段（其中 3% 为任务完成后补帧、2% 为暂停前短段）；25 个人工标注失败中 13 个被正确预测（RQ5）。
- 质量保持：50 条专家演示无一被标次优；54% 恢复行为完全保留、31% 部分保留、15% 删除（RQ6）。
- 明确讨论次优/缺陷演示：是——次优与冗余是两大目标，并对次优类型、假阳/假阴、恢复行为保留做了专项分析；限制自述依赖"多数演示质量良好"假设。

## 全文相关性（fulltext_relevance）
- 全文相关性: high
- 理由: 论文直接实现并验证了"无标注、自监督的机器人演示数据策展机制"（进度预测 + 联合去重），在 3 个基准（含真机）上给出下游策略成功率提升，正对 RQ1.3 的策展信号与下游收益问题。

## 引用
- 进度预测器与次优分数：Sec.3.2、App.A.2/A.3/A.5
- 去重机制：Sec.3.3、App.A.7
- 主结果与消融：Sec.4.2（RQ1–RQ4）、Table 1/2、Fig.3/4
- 阈值与删除比例：App.A.1（Fig.5/6、Table 3）
- 错误与质量保持：App.B.4（RQ5/RQ6）
- 限制：Sec.6 Limitation

## 不确定项
- "Octo 超过 27M 参数"（Intro）与 Table 4 中 OXE 实验 Octo Params=93M 矛盾，模型规格不一致。
- 未与 Demo-SCORE 直接实验对比，仅文献定位异同；"在线 rollout 信号 vs 自监督进度信号"的优劣无直接数字。
- 阈值 ϵ_s/ϵ_d 仅在两个仿真集上调参后外推真机，跨数据集普适性未充分验证。
- 自监督假设"大多数演示质量良好"，若低质量数据占多数则失效（作者自述），失效边界未实验量化。
- 线性进度假设不适于搅拌、等待类循环任务（作者自述），未见针对性实验。
