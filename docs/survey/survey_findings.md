# 调研发现综述

> 证据链：本文件 → `../../synthesis/evidence_matrix.md` → `../../notes/extractions/*.md`
> → `../../papers/`。引用 `[paper_NNN]` 链接 arXiv；无引用处为调研者推理并标记（推理）。
> 缺口与未决见 `../../synthesis/gap_list.md`。

## 1 数据处理有没有范式（RQ1）

**步骤级有范式，判据级没有。** 各大项目的数据流程在步骤层面一致：

```
记录 → 校验 → 标注 → 过滤 → 格式标准化 → 标签重采样/增广
```

但过滤阈值、质量标准、标注体系全是项目自定，没有跨项目标准
[paper_001](https://arxiv.org/abs/2403.12945), [paper_002](https://arxiv.org/abs/2310.08864), [paper_004](https://arxiv.org/abs/2602.22818), [paper_015](https://arxiv.org/abs/2503.06669)。

**各步骤现状：**

- **记录与格式**：社区收敛到 LeRobot（parquet+mp4+元数据，16K 数据集）
  [paper_004](https://arxiv.org/abs/2602.22818)；Robo-DM 把存储压到 RLDS 的 1/70、加载快 50×，但明确不做质量
  判断 [paper_016](https://arxiv.org/abs/2505.15558)；OXE 的跨数据集标准化（RLDS、7 维动作归一化+离散化）
  是聚合型工作，质量过滤细节被指向项目网站 [paper_002](https://arxiv.org/abs/2310.08864)。
  含义：工具链已收敛，**质量环节没有成熟工具**。
- **标签构造（最活跃分支）**：时间等距是默认 [paper_004](https://arxiv.org/abs/2602.22818)；ACT 的 chunking +
  temporal ensembling 是第一代改良 [paper_003](https://arxiv.org/abs/2304.13705)；语义 keyframe 选取是当前
  前沿（KSM，固定步长采样有 horizon-resolution 权衡）[paper_011](https://arxiv.org/abs/2603.01465)。
  本项目曲率重采样（`|q''|^(1/3)`）在文献中无同构对应物（GAP-06）。
- **LfD 遗产**：演示=状态/状态-动作轨迹；轨迹分割两类（技能相似性/显著事件）；
  关键帧从"人工指定"演变为"自动检测的语义阶段标记"；时间对齐（DTW 类）在
  现代遥操管线文献中几乎不出现（GAP-01）[paper_012](https://arxiv.org/abs/1907.03146)。
- **清洗与过滤**：仅两类判据被复用——人工/半自动标记（DROID 成功标记、
  AgiBot 人工核对）[paper_001](https://arxiv.org/abs/2403.12945), [paper_015](https://arxiv.org/abs/2503.06669) 与自动几何校验（标定质量过滤）
  [paper_001](https://arxiv.org/abs/2403.12945)。没有跨项目的"数据质量分"。

| 层面 | 结论 | 证据 |
|---|---|---|
| 步骤级 | 存在共识序列 | [paper_001](https://arxiv.org/abs/2403.12945)/[paper_004](https://arxiv.org/abs/2602.22818)/[paper_015](https://arxiv.org/abs/2503.06669)/[paper_016](https://arxiv.org/abs/2505.15558) |
| 工具级 | 格式/存储/加载已收敛 | [paper_002](https://arxiv.org/abs/2310.08864)/[paper_004](https://arxiv.org/abs/2602.22818)/[paper_016](https://arxiv.org/abs/2505.15558) |
| 判据级 | 无跨项目标准 | [paper_001](https://arxiv.org/abs/2403.12945)/[paper_005](https://arxiv.org/abs/2606.10229)/[paper_015](https://arxiv.org/abs/2503.06669) |
| 语义级 | 有先例但无下游验证 | [paper_010](https://arxiv.org/abs/2509.17321)/[paper_020](https://arxiv.org/abs/2309.14320)/[paper_021](https://arxiv.org/abs/2509.20070) |

## 2 质量过滤与策展的教训（RQ1.3）

区分四个命题，证据强度不同：

- **(a) 检出 AUROC 不预测下游收益（有证据）**：7 种几何/统计指标的检出
  AUROC 与下游 BC 成功率 Spearman 相关 ρ=−0.14；AUROC 最高者下游最差
  （13.3% vs 污染基线 3.3%）；原始 AUROC 近 1.0 的五个指标主要是**时长混杂**
  [paper_005](https://arxiv.org/abs/2606.10229)。
- **(b) 部分离线指标策展有效（有证据，单一设定）**：ensemble（91.1）与
  trajectory alignment（90.0）接近 oracle（93.3），方差近零
  [paper_005](https://arxiv.org/abs/2606.10229)。"指标全部失效"不成立；"哪个指标有效不可由检出 AUROC
  预测"成立。
- **(c) 分阶段局部化不带来增益（有证据，单一设定）**：phase-gated 策展
  3 任务无最优、2 任务最差；缺陷信号被稀释到约 1/4 权重
  [paper_014](https://arxiv.org/abs/2606.15064)。
- **(d) 在线 rollout 策展证据最稳（有证据）**：Demo-SCORE 用 rollout 训练
  成功/失败分类器过滤演示，仿真 +15–35%、真实 ALOHA +15–40%，只需
  <100 条 rollout；OOD 实验部分反驳"只识别分布覆盖"的替换解释
  [paper_013](https://arxiv.org/abs/2503.03707)。
- **(e) 无 LLM 的经典策展仍是主流**：SCIZOR 自监督时间距离分类器（无 LLM/
  无奖励/无标注）[paper_033](https://arxiv.org/abs/2505.22626)；ATHENA 闭环影响力函数策展 + 下游验证
  （加速 313.4×）[paper_034](https://arxiv.org/abs/2606.16208)。

合成含义（推理）：离线评分→过滤**不一定**转化为下游收益；有收益的策展
大多接了执行反馈。**任何策展管线都必须把验证回路通到下游策略性能。**

## 3 语义级标注与 agentic 策展（RQ1.4/RQ2）

**语义级先例（固定管线组件，非 agent）：**

- GVL：VLM in-context 预测逐帧任务进度（0–100%），VOC（进度与帧序的秩相关）
  作质量度量，可检出任务定义不清/标注歧义/失败样例；口径：开源模型 VOC 约为
  最强闭源模型的 60–70%，且闭源 VOC 只是不可观测真值的代理；小模型接近
  随机 [paper_010](https://arxiv.org/abs/2509.17321)（方法定义转引自其引用 [20]，未读原文）。
- LLM Trainer：GPT-4o 从 1 条示范生成关键位姿标注与增强数据，生成成功率超
  人工基线 2–3 倍 [paper_021](https://arxiv.org/abs/2509.20070)。
- MUTEX：LLM 生成 11 种任务规范变体 + 人工过滤 + TTS 合成语音的标注管线
  [paper_020](https://arxiv.org/abs/2309.14320)。
- DROID 的 GPT-4V 场景分类+人工复核 [paper_001](https://arxiv.org/abs/2403.12945)；语言条件切点检测识别
  子任务边界（+1.78±0.82%，无下游验证）[paper_035](https://arxiv.org/abs/2309.00743)。

**agent 是什么**（30 秒版）：以 LLM/VLM 为决策核心 + 工具调用 + "思考→行动→
观测"循环 [paper_006](https://arxiv.org/abs/2309.07864), [paper_007](https://arxiv.org/abs/2210.03629)。教学版见 `agent_basics_tutorial.md`。

**agentic 数据管线的现有形态：**

- AgentInstruct：三个 agentic flow（内容变换→种子指令→精化）生成 25.8M 条
  合成指令；质量无直接度量，只靠下游基准间接验证（Orca-3 平均 +33.94%）
  [paper_009](https://arxiv.org/abs/2407.03502)。教训：agentic 生成的规模容易做，质量验证难做。
- LLM-as-judge：与人类一致 85%（非平局口径）；position/verbosity/
  self-enhancement 偏差的定量证据；数学判分靠 reference-guided 才从 70% 误判
  降到 15% [paper_008](https://arxiv.org/abs/2306.05685)。工程含义：用 LLM 打分必须做去偏设计。
- DSPy：声明式签名+模块+指标，编译器自动优化；GPT-3.5 GSM8K 33%→82%
  [paper_022](https://arxiv.org/abs/2310.03714)。
- TextGrad：文本梯度反向传播优化复合管线；LeetCode 相对 Reflexion +16%、
  GPQA 51→55 [paper_023](https://arxiv.org/abs/2406.07496)。
- AutoRT（机器人侧最完整）：VLM 语言地图→LLM 生成任务→自我批评过滤
  affordance→三类策略采集→多样性打分；7 个月 77k episode、1 人监督 3–8 台；
  安全靠 prompt 宪法+硬 guardrail；下游验证 RT-1 高度抓取 0%→12.5%
  [paper_019](https://arxiv.org/abs/2401.12963)。

**工程模式五条**：①指标与管线分离 [paper_022](https://arxiv.org/abs/2310.03714)；②下游验证回路不可省
[paper_005](https://arxiv.org/abs/2606.10229), [paper_014](https://arxiv.org/abs/2606.15064), [paper_013](https://arxiv.org/abs/2503.03707), [paper_019](https://arxiv.org/abs/2401.12963)；③批评者模式（生成→自批评→拒绝）
[paper_019](https://arxiv.org/abs/2401.12963), [paper_009](https://arxiv.org/abs/2407.03502)；④成本是一等公民（当前文献普遍不报 LLM 成本，
GAP-04）[paper_019](https://arxiv.org/abs/2401.12963), [paper_021](https://arxiv.org/abs/2509.20070)；⑤人机监督带宽设计 [paper_019](https://arxiv.org/abs/2401.12963)。

**严格口径的"VLA 数据处理 agent"不存在（GAP-11）**：组件级先例齐全，
系统组合为空白。

## 4 管线模块级证据（RQ3）

| 模块 | 关键发现 | 依据 |
|---|---|---|
| 训练/验证划分 | episode 级划分是"实践共识、文献沉默"（RoboMIND 107k 轨迹不声明划分协议）；组合泛化 held-out 有先例（微调 15 原子任务、评测 12 组合任务零演示） | [paper_030](https://arxiv.org/abs/2412.13877), [paper_031](https://arxiv.org/abs/2606.16826) |
| 时间戳对齐 | 遥操数据集同步实践无专文；Robo-DM 用相对时间戳不假设同步；可借鉴 SLAM 域时间偏移估计（滤波状态变量形态，退化条件=恒定角速度+恒定线速度，weak） | [paper_016](https://arxiv.org/abs/2505.15558), [paper_032](https://arxiv.org/abs/2201.09170), [paper_030](https://arxiv.org/abs/2412.13877) |
| 数据增强 | 轨迹级增强用"真实执行+成功过滤"验证（10 演示→1000），但 naive 成功过滤有覆盖偏置（43.5–66.4%）；**速度类增强在接触/力控任务中不安全**（真实反应 vs 时间缩放复制 88% vs 31%）；下采样增强的时序对齐方式影响效果（对称对齐最优）；反事实框架统一不变性/等变性/因果性；语义分割+背景生成视觉增强 | [paper_024](https://arxiv.org/abs/2310.17596), [paper_026](https://arxiv.org/abs/2412.03252), [paper_025](https://arxiv.org/abs/2410.04370), [paper_027](https://arxiv.org/abs/2411.16959), [paper_028](https://arxiv.org/abs/2503.18738) |
| 观测可分性（state aliasing） | 歧义五类分类（双向重叠轨迹/固定时长静止/顺序子任务依赖/重复时间循环/动态交互响应）；主流解法是模型侧（300 帧工作记忆），**不做数据标注、无逐帧歧义度量**；语义 keyframe 检测是邻近概念 | [paper_029](https://arxiv.org/abs/2512.24638), [paper_011](https://arxiv.org/abs/2603.01465) |
| 覆盖与补采 | "失败案例→数据短板→补采建议"是成熟形态（RoboMIND）；多样性打分驱动采集（AutoRT）；覆盖影响泛化（DROID 场景多样性消融） | [paper_030](https://arxiv.org/abs/2412.13877), [paper_019](https://arxiv.org/abs/2401.12963), [paper_001](https://arxiv.org/abs/2403.12945) |

## 5 负面证据清单（设计必须规避的坑）

1. 离线指标与下游脱钩 [paper_005](https://arxiv.org/abs/2606.10229), [paper_014](https://arxiv.org/abs/2606.15064) → 一切过滤/加权以下游成功率验收。
2. 开源小 VLM 语义进度预测接近随机 [paper_010](https://arxiv.org/abs/2509.17321) → 选型前先做能力测试。
3. LLM-as-judge 有系统偏差 [paper_008](https://arxiv.org/abs/2306.05685) → 打分设计要随机化位置、控长度、给参考。
4. naive 成功过滤有覆盖偏置 [paper_024](https://arxiv.org/abs/2310.17596) → 过滤后必须查覆盖。
5. 速度无关性在接触任务不成立 [paper_026](https://arxiv.org/abs/2412.03252) → 任务扩展的边界检查。
6. ReAct 式交错推理提高推理错误率 [paper_007](https://arxiv.org/abs/2210.03629)。
7. 主流基准不声明 train/val 划分协议 [paper_030](https://arxiv.org/abs/2412.13877)。
8. agentic 生成无直接质量度量 [paper_009](https://arxiv.org/abs/2407.03502)。

## 6 缺口（详见 `../../synthesis/gap_list.md`）

- GAP-02 语义过滤的下游定量收益——无任何数字。
- GAP-03 "成功但次优"演示的过滤——无人研究（本项目核心问题）。
- GAP-08 逐帧观测可分性标注——无先例。
- GAP-09 train/val 划分协议——文献沉默，需自验。
- GAP-10 遥操数据多传感器同步——需自建工具。
- GAP-11 agent 运营 VLA 数据处理管线——系统组合空白，组件先例齐全。
- 另有 GAP-01（DTW 现代实践）、GAP-04（成本报告）、GAP-06（曲率重采样对应物）、
  GAP-07（语义切分粒度影响）。
