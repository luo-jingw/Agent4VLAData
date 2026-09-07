# Agent 基础教程

> 写给没接触过 agent 的读者（入门教学）。概念与数字均来自本调研文献
> `[paper_NNN]`（出处见 `../../papers/metadata.yaml`）；教学化的比喻与总结是编写者的
> 简化表达，单独标出（教学简化）。进阶调研见 `docs/survey/survey_findings.md`。

## 1 一句话定义

**Agent = 以 LLM（或 VLM）为决策核心、可以调用外部工具、为完成一个任务自主地
多步行动的系统** [paper_006](https://arxiv.org/abs/2309.07864), Sec 2-3。

普通 LLM 调用是"一问一答"；agent 是把 LLM 放进一个循环里：

```
思考 → 行动 → 观测结果 → 再思考 → …直到任务完成
```

文献认为 LLM 适合当 agent 核心有四个属性：自主性（自己定计划）、反应性（响应
环境变化）、主动性（为目标主动行动）、社会性（与人/其他 agent 协作）
[paper_006](https://arxiv.org/abs/2309.07864), Sec 2.3。

（教学简化：工程上 agent 往往就是你写的一个 while 循环 + 若干次 LLM/VLM 调用 +
几个工具函数；论文的贡献在于"循环里放什么、什么时候停、怎么验证"。）

## 2 组成：两套等价框架

文献有两套主流切法，都成立，只是视角不同。

### 2.1 Xi 等的三分 [paper_006](https://arxiv.org/abs/2309.07864), Sec 3

- **Brain（大脑）**：自然语言交互、知识、记忆、推理与规划、迁移泛化；
- **Perception（感知）**：文本、视觉（图像/视频）、听觉等输入；
- **Action（行动）**：文本输出、工具使用、具身行动（机器人在这里接入）。

### 2.2 Wang 等的四分 [paper_017](https://arxiv.org/abs/2308.11432), Sec 2.1

- **Profile（角色）**：agent 是谁（职业、性格、社会关系）；
- **Memory（记忆）**：
  - 短时记忆 = 直接写进 prompt 的上下文；
  - 长时记忆 = 向量库检索，按 recency/relevance/importance 加权取回；
  - 两类维护操作：写（去重、超限时用 LLM 压缩成摘要）与反思（从经历归纳
    高层见解）；
- **Planning（规划）**：
  - 无反馈规划：CoT（一步步想）、ToT（树搜索）、交给外部 PDDL 规划器；
  - 有反馈规划：环境反馈、人类反馈、模型自我批评；
- **Action（行动）**：调 API、查数据库、调用外部模型、或直接改变环境。

（教学简化：对数据处理场景，最省事的理解是——**agent = 一个会循环的 VLM/LLM +
一堆可调用的分析工具 + 一段可积累的记忆**。）

## 3 怎么运行：ReAct 循环

ReAct（Reasoning + Acting）[paper_007](https://arxiv.org/abs/2210.03629) 把模型的动作空间扩展成"语言思考 + 行动"：

- **thought（思考）**：不作用于环境、不产生反馈，只更新上下文，支撑后续推理；
- **action（行动）**：作用于环境，产生 observation（观测）；
- 两者交替：`思考 → 行动 → 观测 → 思考 → …`。

实测（PaLM-540B）：

| 指标 | 结果 |
|---|---|
| 知识任务幻觉率（失败轨迹口径） | CoT 56% → ReAct 0% |
| 知识任务推理错误率 | CoT 16% → ReAct 47%（代价） |
| 决策任务 ALFWorld 成功率 | 纯行动 45% → ReAct 71% |

[paper_007](https://arxiv.org/abs/2210.03629), Sec 3.3-4

要点：**把推理接到外部世界反馈上是 agent 的核心增益来源；但循环结构本身会引入
新的错误模式（重复思考、推理错误），不是免费的。**

## 4 工具调用

源头是 Toolformer [paper_018](https://arxiv.org/abs/2302.04761)：让模型自监督学会"何时调用 API、传什么参数、把
结果接回预测"。三步：采样候选调用位置 → 真实执行 API → 按"加权交叉熵改进"筛选
有价值的调用 → 微调。对工程重要的三个发现：

1. 工具能力是涌现的（约 775M 参数以下几乎无效）[paper_018](https://arxiv.org/abs/2302.04761), Sec 4.4；
2. 一次调用值不值，用"它让后续 token 预测变好了多少"衡量 [paper_018](https://arxiv.org/abs/2302.04761), Sec 2；
3. 局限：不能链式调用、不能交互式使用工具（如改写搜索词）、对措辞敏感
   [paper_018](https://arxiv.org/abs/2302.04761), Sec 7。

今天的工程实践通常用 function calling / MCP 这类显式接口替代自监督训练
（此为工程常识补充，非文献证据）。

## 5 入门常见误解澄清

1. agent 不神秘（教学简化）：它是循环 + 调用 + 工具的工程结构，不是魔法。
2. **agent 不一定省钱省时**：每一步思考都是一次模型调用（成本 × 延迟）；
   Toolformer 连 API 调用成本都没计入自己的局限 [paper_018](https://arxiv.org/abs/2302.04761), Sec 7。
3. **没有免费的判断**：LLM-as-judge 有系统性偏差——position bias（交换选项
   顺序后一致率跌到 23.8–65%）、verbosity bias（冗长改写骗过 judge）、
   self-enhancement（给自己加分）；与人类一致约 85%（非平局口径）
   [paper_008](https://arxiv.org/abs/2306.05685), Sec 3.3, 4.2。
4. 多 agent（生成 + 批评者、分工协作）是常见结构，但两篇综述都只有分类、
   没有系统性工程指导（gap_list GAP-05）[paper_006](https://arxiv.org/abs/2309.07864), Sec 4; [paper_017](https://arxiv.org/abs/2308.11432)。

## 6 进一步阅读（本调研文献内）

- 更系统的 agent 综述：[paper_006](https://arxiv.org/abs/2309.07864)、[paper_017](https://arxiv.org/abs/2308.11432)；
- agentic 数据策展方法与应用：`docs/survey/survey_findings.md`；
- 把 agent 管线当程序优化：DSPy [paper_022](https://arxiv.org/abs/2310.03714)、TextGrad [paper_023](https://arxiv.org/abs/2406.07496)；
- 机器人领域的 agent 数据管线：AutoRT [paper_019](https://arxiv.org/abs/2401.12963)、OpenGVL [paper_010](https://arxiv.org/abs/2509.17321)。

## 7 边界与未决

- 本教程只覆盖本调研文献支持的结论；多 agent 协作模式、agent 安全对齐的
  工程细节不在覆盖范围（gap_list GAP-05）。
- 两套框架的异同（Xi 三分 vs Wang 四分）文献未做对照说明 [paper_017](https://arxiv.org/abs/2308.11432)，
  本文并列呈现而非断言等价。
