# 评审记录

对初稿（review_draft.md，后并入 `docs/survey/survey_findings.md`）的逐项审查。只提问题，不改正文。

## C-01 过度泛化："离线指标→过滤没有收益"（第一章 1.2(4)、1.3）

- 位置：1.2(4) 末段、1.3 末段。
- 问题：[paper_005](https://arxiv.org/abs/2606.10229), [paper_014](https://arxiv.org/abs/2606.15064) 均为单一仿真任务、单一缺陷类型（LIFT 早期释放）、污染率 80%
  （作者自述现实典型 20–40%）。"这条路没有被证明能转化"是对的，但表述接近
  "这条路不通"。且 [paper_005](https://arxiv.org/abs/2606.10229) 内部 **ensemble（91.1±1.6）与 trajectory alignment
  （90.0±0.0）两种离线指标策展已接近 oracle（93.3）**——初稿把"指标全部失效"
  与"检出 AUROC 与下游脱钩"混为一谈。前者不成立。
- 建议修订：区分三个命题——(a) 检出 AUROC 不预测下游（有证据）；
  (b) 部分离线指标策展有效（ensemble/alignment，有证据，单设定）；
  (c) 分阶段局部化不带来增益（有证据，单设定）。删除"唯一持续有效的是在线
  rollout"的排他表述，改为"在线 rollout 策展是证据最稳的一类"。

## C-02 未标注的替换解释（[paper_013](https://arxiv.org/abs/2503.03707)）

- 位置：1.2(4)。
- 问题：Demo-SCORE 的分类器训练于 rollout 数据，其收益可能来自"识别状态分布
  覆盖"，而非"识别演示质量"（初稿未提）。[paper_013](https://arxiv.org/abs/2503.03707) 的 OOD 扩张实验（不损害
  OOD）是对该替换解释的部分反驳，初稿没引用。
- 建议修订：补一句"其 OOD 实验部分反驳了'只识别分布覆盖'的替换解释"。

## C-03 证据来源二级化（GVL）

- 位置：1.2(5)、3.2。
- 问题：GVL 方法出自 Ma et al.（ICLR 2024，[paper_010](https://arxiv.org/abs/2509.17321) 的引用 [20]），本调研
  读的是 benchmark 论文而非原方法论文。初稿把 GVL 表述为本 corpus 一手结论。
- 建议修订：标注"GVL 方法定义转引自 [paper_010](https://arxiv.org/abs/2509.17321) 引用 20，未读原文"。

## C-04 能力数字的口径（"60–70%"）

- 位置：1.2(5)、3.2。
- 问题："开源约为闭源 60–70%"是相对 VOC 上界，而该上界本身是不可观测真值的
  代理（[paper_010](https://arxiv.org/abs/2509.17321) 脚注 2 自述）。初稿把它当绝对能力差陈述。
- 建议修订：改为"开源模型 VOC 约为最强闭源模型的 60–70%，且闭源 VOC 本身只是
  真值代理"。

## C-05 教学章的未标记断言

- 位置：2.5 第 1 条（"agent 就是你写的 while 循环 + LLM 调用 + 工具函数"）。
- 问题：这是调研者的教学化断言，全章其余教学表述均无此标记。
- 建议修订：显式标记为教学化简化。

## C-06 二级引用的边界（Akgun 2012）

- 位置：evidence matrix RQ1.2、gap_list GAP-01。
- 问题：Akgun 2012 经 [paper_012](https://arxiv.org/abs/1907.03146) 转引，未读原文，但在证据表中以事实形态出现。
- 建议修订：在证据表与定稿中标注"（转引自 [paper_012](https://arxiv.org/abs/1907.03146)）"。

## C-07 遗漏：推理侧对应物

- 位置：1.2(3)。
- 问题：[paper_003](https://arxiv.org/abs/2304.13705) 的 temporal ensembling 与本项目部署侧的 PCHIP 插值 + 融合
  是同一问题（重规划抖动的平滑）的两个解，初稿未点出。点出后"本项目方法在
  文献中的定位"更完整。
- 建议修订：1.2(3) 补一句对照。

## C-08 "16k 不训练"与"前 40k 训练"的口径

- 位置：1.1 DROID 段。
- 问题：extraction 记录"约 16k 未成功轨迹不计入 DROID 规模"与"训练取前 40k 条
  有标注成功轨迹"是两件事，初稿"失败剔除（约 16k 条不训练）"合并表述，
  可能失准。
- 建议修订：拆开表述。

## C-09 3.4 建议项的推理标记完整性

- 位置：3.4 第 2 条（语义加权与现有事件窗加权同构）。
- 问题：已标记"（推理，无文献）"，符合规范。但第 1 条"语义过滤对应 GAP-03"
  未在 3.4 段内标记推理，仅靠章首总标记。
- 建议修订：3.4 首行标注"本节全部为推理"。

## C-10 未回答问题的边界声明缺失

- 位置：全文。
- 问题：RQ2.4 的成本/批量推理规模、RQ1.2 的 DTW 现代实践等 gap 在定稿中
  必须显式写出"本综述不回答/未找到证据"，避免读者把 gap 当否定结论。
- 建议修订：每份定稿末尾加"边界与未决"一节，直接引用 gap_list。

## 总评

- 证据使用总体忠实于 extraction；主要风险是 C-01（过度泛化）与 C-04（口径），
  修订范围约 15–20 行，低于 30% 改动上限。


## 修订映射

初稿经反方审查后修订，定稿写入
`docs/survey/`。2026-09-06 文档整理后，三份综述（survey_robot_data_preprocessing、
survey_agentic_data_curation、survey_pipeline_modules）已合并为
`docs/survey/survey_findings.md`；下表"落点"仍为修订当时的文件名，
章节内容对应 survey_findings.md 的 §1–§4。
本文件记录修订映射（改动范围约 10%，低于 30% 上限）。

| 审查项 | 处理 | 落点 |
|---|---|---|
| C-01 过度泛化 | 拆分为 (a)(b)(c)(d) 四个命题，各自标注证据范围；删除"唯一持续有效"的排他表述 | survey_robot_data_preprocessing.md §2.4 |
| C-02 替换解释 | 补充 [paper_013](https://arxiv.org/abs/2503.03707) OOD 实验对"分布覆盖"替换解释的部分反驳 | survey_robot_data_preprocessing.md §2.4(d) |
| C-03 GVL 二级来源 | 标注"方法定义转引自 [paper_010](https://arxiv.org/abs/2509.17321) 引用 [20]，未读原文" | 两份 survey §2.5 / §2 |
| C-04 60–70% 口径 | 补"闭源 VOC 本身只是不可观测真值的代理" | 两份 survey §2.5 / §2 |
| C-05 教学断言未标记 | 补"（教学简化）"标记 | agent_basics_tutorial.md §1、§2、§5 |
| C-06 二级引用 | Akgun 2012 标注"转引自 [paper_012](https://arxiv.org/abs/1907.03146)" | survey_robot_data_preprocessing.md §3 |
| C-07 推理侧对应物 | 补 temporal ensembling 与 PCHIP+smoothstep 的同源对照（标记推理） | survey_robot_data_preprocessing.md §2.3 |
| C-08 16k 口径 | 拆开"不计入规模"与"训练取前 40k"两件事 | survey_robot_data_preprocessing.md §2.2 |
| C-09 推理标记 | 结合点小节首行声明"本节全部为推理" | survey_agentic_data_curation.md §4 |
| C-10 边界声明 | 三份定稿末尾加"边界与未决"节，引用 gap_list | 三份 docs |

## 未处理项

无。审查 10 项全部处理。
