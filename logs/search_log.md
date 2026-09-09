# Search Log

## 2026-09-06

### 查询记录

| 查询 | 来源 | 结果 |
|---|---|---|
| `cat:cs.RO AND abs:"data curation"` | arXiv API | 命中 OpenGVL ([2509.17321](https://arxiv.org/abs/2509.17321))、Phase-Localized Curation ([2606.15064](https://arxiv.org/abs/2606.15064))、What Demonstration Curation Metrics ([2606.10229](https://arxiv.org/abs/2606.10229))、Curating Demonstrations using Online Experience ([2503.03707](https://arxiv.org/abs/2503.03707)) |
| `abs:LeRobot AND abs:dataset AND cat:cs.RO` | arXiv API | 命中 LeRobot ([2602.22818](https://arxiv.org/abs/2602.22818))、Robo-DM ([2505.15558](https://arxiv.org/abs/2505.15558)) |
| 按 ID 定向验证 | arXiv API | seed 10 篇全部验证 |

### 决策

- seed 集 10 篇（RQ1 5 篇 / RQ2 5 篇），写于 `papers/candidates.yaml`。
- 记忆中的 LeRobot ID（[2504.05202](https://arxiv.org/abs/2504.05202)）错误，经搜索修正为 [2602.22818](https://arxiv.org/abs/2602.22818)。
- 第二轮扩展候选已记录（见 `logs/search_log.md` 扩展方向）：
  Robo-DM ([2505.15558](https://arxiv.org/abs/2505.15558))、Phase-Localized ([2606.15064](https://arxiv.org/abs/2606.15064))、Curating Demonstrations
  Online ([2503.03707](https://arxiv.org/abs/2503.03707))、Data Scaling Laws in Imitation Learning ([2410.21647](https://arxiv.org/abs/2410.21647))、
  AgiBot World ([2502.12294](https://arxiv.org/abs/2502.12294))、RoboMIND ([2504.12001](https://arxiv.org/abs/2504.12001))、BridgeData V2 ([2410.24116](https://arxiv.org/abs/2410.24116))、
  Wang agent survey ([2308.11432](https://arxiv.org/abs/2308.11432))、Toolformer ([2302.04761](https://arxiv.org/abs/2302.04761))、DSPy ([2310.03714](https://arxiv.org/abs/2310.03714))、
  TextGrad ([2406.07496](https://arxiv.org/abs/2406.07496))、AutoRT ([2401.12995](https://arxiv.org/abs/2401.12995))、MUTEX ([2503.12097](https://arxiv.org/abs/2503.12097))、IAFM ([2305.16291](https://arxiv.org/abs/2305.16291))、
  LfD 综述（待搜）。
- 待 seed 抽取验证后执行第二轮。

### 第二轮（2026-09-06 晚，13 篇，全部下载并抽取）

| arXiv ID | 论文 | 用途 |
|---|---|---|
| [2603.01465](https://arxiv.org/abs/2603.01465) | Keyframe Chaining | RQ1.2 |
| [1907.03146](https://arxiv.org/abs/1907.03146) | Kroemer 综述（Robot Learning for Manipulation） | RQ1.2 |
| [2503.03707](https://arxiv.org/abs/2503.03707) | Curating Demonstrations using Online Experience | RQ1.3 |
| [2606.15064](https://arxiv.org/abs/2606.15064) | Phase-Localized Curation（负面结果） | RQ1.3 |
| [2503.06669](https://arxiv.org/abs/2503.06669) | AgiBot World Colosseo | RQ1.1 |
| [2505.15558](https://arxiv.org/abs/2505.15558) | Robo-DM | RQ1.1/RQ2.4 |
| [2308.11432](https://arxiv.org/abs/2308.11432) | Wang agent survey | RQ2.1 |
| [2302.04761](https://arxiv.org/abs/2302.04761) | Toolformer | RQ2.1 |
| [2401.12963](https://arxiv.org/abs/2401.12963) | AutoRT | RQ2.3/RQ2.4 |
| [2309.14320](https://arxiv.org/abs/2309.14320) | MUTEX | RQ2.3 |
| [2509.20070](https://arxiv.org/abs/2509.20070) | LLM Trainer | RQ2.3 |
| [2310.03714](https://arxiv.org/abs/2310.03714) | DSPy | RQ2.4 |
| [2406.07496](https://arxiv.org/abs/2406.07496) | TextGrad | RQ2.4 |

修正：记忆中的 AutoRT（[2401.12995](https://arxiv.org/abs/2401.12995)）、MUTEX（[2503.12097](https://arxiv.org/abs/2503.12097)）、Data Scaling Laws
（[2410.21647](https://arxiv.org/abs/2410.21647)）、AgiBot World（[2502.12294](https://arxiv.org/abs/2502.12294)）ID 有误，均经搜索/验证修正或替换。
Data Scaling Laws 未找到，替换为 LLM Trainer（更贴近调研主题）。

### 第三轮（2026-09-06 深夜，9 篇，全部下载并抽取）

背景：用户确认数据处理是完整 pipeline，追加 RQ3.1–RQ3.5（划分/泄漏、时间
同步、增强安全性、观测可分性、覆盖分析）。arXiv API 相关性搜索对细分主题
效果差，改用 Semantic Scholar API（`scripts/s2_search.py`，公共接口限流较重）。

| arXiv ID | 论文 | 用途 |
|---|---|---|
| [2310.17596](https://arxiv.org/abs/2310.17596) | MimicGen | RQ3.3 轨迹级增强+执行验证 |
| [2410.04370](https://arxiv.org/abs/2410.04370) | DABI | RQ3.3 增强评测（下采样对齐） |
| [2412.03252](https://arxiv.org/abs/2412.03252) | Variable-Speed Teaching-Playback | RQ3.3 速度增强的接触边界 |
| [2411.16959](https://arxiv.org/abs/2411.16959) | RoCoDA | RQ3.3 反事实增强 |
| [2503.18738](https://arxiv.org/abs/2503.18738) | RoboEngine | RQ3.3 语义视觉增强 |
| [2512.24638](https://arxiv.org/abs/2512.24638) | Resolving State Ambiguity (PAM) | RQ3.4 状态歧义五类+模型侧方案 |
| [2412.13877](https://arxiv.org/abs/2412.13877) | RoboMIND | RQ3.1/3.5 QA 标准+覆盖分析 |
| [2606.16826](https://arxiv.org/abs/2606.16826) | ATOM-Bench | RQ3.1 held-out 组合划分 |
| [2201.09170](https://arxiv.org/abs/2201.09170) | VINS Online Self-Calibration | RQ3.2 时间偏移估计（weak） |

未命中的检索方向（记录为 gap）：train/val 泄漏专文（GAP-09）、遥操数据集
多相机同步专文（GAP-10）、"data coverage"检索被路径规划语义污染。



### 追加（2026-09-06，3 篇）：'agent 是否存在'定向检索

| arXiv ID | 论文 | 用途 |
|---|---|---|
| [2505.22626](https://arxiv.org/abs/2505.22626) | SCIZOR | RQ1.3 无 LLM 自监督策展 |
| [2606.16208](https://arxiv.org/abs/2606.16208) | ATHENA | RQ1.3 影响力函数策展+下游验证 |
| [2309.00743](https://arxiv.org/abs/2309.00743) | Language-Conditioned Change-point Detection | 旋钮 1 组件先例 |

结论记录：'agent 运营 VLA 数据处理管线'严格口径不存在 → GAP-11。
检索词：data curation+agent / agentic / data pipeline+LLM / VLA+curation（均 cs.RO）。


### 社区轮（2026-09-07，GitHub issues）

- 覆盖 repository：physical-intelligence/openpi、huggingface/lerobot、
  huggingface/smolvla、NVIDIA/gr00t、openvla/openvla、real-stanford/diffusion_policy、
  TonyZhao/act、kscalelabs/smolvla。
- 工具：scripts/gh_issue_search.py（匿名 GitHub issues search API）。
- 关键词：data quality / dataset cleaning / missing frames / camera timestamp /
  episode filtering / resample keyframe / train-test split / augmentation crop /
  gripper suction / validation split。
- 产出：docs/survey/community_issue_survey.md（6 个主题聚类，含精确 issue 链接）。
