# Gap List

按调研问题编号的未回答点。每条标注缺口来源与补证建议。

## GAP-01 时间对齐（DTW 类）在现代遥操数据预处理中的实践

- 状态：open
- 证据现状：[paper_012](https://arxiv.org/abs/1907.03146) 仅在跨域/跨视角对应旁及，无 DTW 专节；23 篇中无 direct 证据。
- 补证建议：搜 "time alignment trajectory demonstration robot" / DTW survey；
  或接受"时间对齐是旧 LfD 概念，现代管线用 chunk/重采样替代"这一 hypothesis 并
  显式标记为未验证推理。

## GAP-02 语义过滤的下游定量收益

- 状态：open
- 证据现状：[paper_010](https://arxiv.org/abs/2509.17321) 策展为定性案例（VOC 曲线观察），无"过滤后训练策略成功率
  提升"数字；[paper_001](https://arxiv.org/abs/2403.12945) GPT-4V 分类未做下游消融。
- 补证建议：关注后续 GVL/OpenGVL 工作；或把"语义过滤能否提升下游"列为本项目
  实验计划的第一个假设检验。

## GAP-03 "次优解"（非失败但非最优的演示）的过滤

- 状态：open（本项目核心诉求）
- 证据现状：[paper_005](https://arxiv.org/abs/2606.10229), [paper_013](https://arxiv.org/abs/2503.03707), [paper_014](https://arxiv.org/abs/2606.15064) 针对缺陷/失败演示；[paper_005](https://arxiv.org/abs/2606.10229) 明确限定了结构性缺陷
  单类型；无任何论文直接研究"成功但次优"演示的识别与过滤。
- 含义：agent 方案所针对的问题在文献中**没有直接先例**——这是机会也是风险：
  无现成方法可抄，但若成功则有原创性。

## GAP-04 agent 离线数据管线的成本与精度系统报告

- 状态：open
- 证据现状：[paper_019](https://arxiv.org/abs/2401.12963) 未具名所用 LLM、无成本数字；[paper_021](https://arxiv.org/abs/2509.20070) 优化目标不计 LLM
  调用成本；[paper_009](https://arxiv.org/abs/2407.03502) 无美元成本。
- 补证建议：本项目若做 agent 管线，成本必须作为一等指标记录（AGENTS.md 的
  观测型验证要求）。

## GAP-05 多 agent 协作模式的系统分类

- 状态：open
- 证据现状：[paper_017](https://arxiv.org/abs/2308.11432) 无独立多 agent 章节；[paper_006](https://arxiv.org/abs/2309.07864) 有 cooperative/adversarial
  二分但无工程级指导。
- 补证建议：教学文档中多 agent 部分只写两综述有的内容，明确标注缺失。

## GAP-06 曲率重采样类标签构造的文献对应物

- 状态：open
- 证据现状：最接近的是 [paper_003](https://arxiv.org/abs/2304.13705)（action chunking + temporal ensembling）与
  [paper_011](https://arxiv.org/abs/2603.01465)（语义 keyframe 选取）；无"曲率密度等化插值误差"的同构工作。
- 含义：现有几何方法可能具有方法学新颖性；调研未覆盖"knot placement"数值分析
  文献（那是另一领域，本轮排除范围外），若要做创新性声明需补查。

## GAP-07 语义任务分解粒度对训练数据的影响

- 状态：open
- 证据现状：[paper_011](https://arxiv.org/abs/2603.01465) 证明语义阶段（keyframe）粒度影响 VLA 性能；但"用语义
  切分/标注训练数据本身"对 BC 性能的影响无证据。
- 补证建议：与 GAP-02 合并验证。

## GAP-08 逐帧观测可分性（state aliasing）标注

- 状态：open（第三轮更新）
- 证据现状：[paper_011](https://arxiv.org/abs/2603.01465) 以状态别名为动机改模型输入（keyframe 链式历史）；
  [paper_029](https://arxiv.org/abs/2512.24638) 给出歧义五类分类（双向重叠轨迹/固定时长静止/顺序子任务
  依赖/重复时间循环/动态交互响应）但同样走模型侧方案（300 帧工作记忆），
  **不做数据标注、无逐帧歧义度量**。无任何论文做"逐帧观测是否可分"的数据标注。
- 含义：旋钮 4（观测可分性标注）无文献对应物；[paper_029](https://arxiv.org/abs/2512.24638) 的五类分类
  可直接作为标注判据的设计素材。
- 补证建议：先做离线可分性测量；文献补查方向 "observability annotation
  imitation learning" / "state aliasing VLA dataset"。

## GAP-09 train/val 划分协议在机器人数据集中的文献记录

- 状态：open
- 证据现状：[paper_030](https://arxiv.org/abs/2412.13877)（107k 轨迹基准）不声明划分协议；[paper_031](https://arxiv.org/abs/2606.16826)
  给出 held-out 组合划分（微调 15 原子任务、评测 12 组合任务零演示）但针对
  组合泛化而非 episode 级自相关泄漏。**"按 episode 划分防泄漏"是实践共识、
  文献沉默**。
- 含义：管线设计时 episode 级划分只能作为工程规则采用（无文献支撑），
  需在本项目数据上自己验证。
- 补证建议：查 offline RL（D4RL 类）与视觉重识别领域的泄漏文献作旁证。

## GAP-10 遥操数据集的多传感器时间同步实践

- 状态：open
- 证据现状：[paper_016](https://arxiv.org/abs/2505.15558) 用相对时间戳对齐但不给同步方法；[paper_030](https://arxiv.org/abs/2412.13877)
  完全不涉及；[paper_032](https://arxiv.org/abs/2201.09170)（VINS 自标定，弱证据）给出时间偏移作为
  滤波状态变量的估计形态与退化条件。
- 含义：本管线里"格式转换（时间戳对齐）"模块的判据与工具需要自建，
  文献只能提供方法形态参考。
- 补证建议：查相机-IMU 时间标定（Kalibr）与多相机同步文献作方法学来源。

## GAP-11 agent 运营 VLA 数据处理管线的系统

- 状态：open（2026-09-06 定向检索确认）
- 证据现状：严格口径（有工具循环、自主规划、管理清洗/过滤/标注/重采样的
  LLM/VLM agent）在已声明检索范围内未发现（检索词与范围见
  `logs/search_log.md`）。放宽口径的现有形态：(a) VLM/LLM 作为管线固定
  组件（[paper_010](https://arxiv.org/abs/2509.17321)/[paper_021](https://arxiv.org/abs/2509.20070)/[paper_020](https://arxiv.org/abs/2309.14320)/[paper_001](https://arxiv.org/abs/2403.12945)/[paper_030](https://arxiv.org/abs/2412.13877)）；
  (b) 非机器人数据的 agentic 管线（[paper_009](https://arxiv.org/abs/2407.03502)/[paper_022](https://arxiv.org/abs/2310.03714)/[paper_023](https://arxiv.org/abs/2406.07496)）；
  (c) agent 做数据采集非处理（[paper_019](https://arxiv.org/abs/2401.12963)）；
  (d) 无 LLM 的经典策展仍活跃（[paper_033](https://arxiv.org/abs/2505.22626)/[paper_034](https://arxiv.org/abs/2606.16208)/[paper_013](https://arxiv.org/abs/2503.03707)）。
- 含义：系统级组合原创性成立，但任何单一组件（VLM 判断、成功过滤、自监督
  策展、事件检测）均已有先例；专利/论文的立足点只能是系统组合与协同效应。
- 补证建议：若推进系统级工作，定期复检该 GAP（每季度检索一次）。
