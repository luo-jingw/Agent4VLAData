# Agent4VLAData

VLA（Vision-Language-Action）动作模态数据增强框架的调研与设计仓库。

- 现有方法：纯几何曲率重采样管线（`reference/g2-orion05-resample-training/`，
  已训练部署）；
- 目标：通用 VLA 遥操动作模态**数据增强框架**——对齐约束下重构/强化 action
  序列，数据效率与执行效率双目标；
- 当前状态：方向调研中（48 篇文献；RQ4 动作模态对齐为主轴），框架部分定稿，
  实验未启动。

## 目录分工

| 目录 | 职责 |
|---|---|
| `docs/survey/` | 调研定义与综述定稿 |
| `docs/reports/` | 报告（顶层视角、综合报告） |
| `docs/design/` | 设计框架 |
| `design/` | 设计草稿（未验证） |
| `synthesis/` | 证据矩阵、缺口清单、评审记录 |
| `papers/` | 文献库：候选/元数据/bibtex/raw 文本 |
| `notes/extractions/` | 48 篇论文的逐篇抽取笔记 |
| `scripts/` | 文献检索/下载/检查脚本（观测型，无 pass/fail） |
| `logs/` | 搜索、下载、决策记录 |
| `reference/` | 现有方法包（只读参照，勿改） |
| `.claude/skills/` | 项目技能（含调研工作流 research_skill） |

## 文件结构（目录形状）

角色标注：地图 / 现状与约束 / 证据链 / 设计与机会 / 过程资产。

```
agent4data/
├── README.md · AGENTS.md
├── PROJECT.md · plan.md · issues.md · opportunities.md
├── docs/                         ── 已验证的长期文档
│   ├── reports/   地图、论证
│   ├── survey/    调研定义、综述、教学
│   └── design/    设计框架
├── design/                       ── 未验证区（草稿、现状综述，不进 docs/）
├── synthesis/                    ── 证据与缺口
├── papers/ · notes/extractions/  ── 文献库与抽取笔记
├── scripts/ · logs/              ── 工具与记录
└── reference/ · .claude/skills/  ── 现有方法包、项目技能
```

## 可跳转索引（按逻辑链分组）

**地图**

- [顶层视角：五个面](docs/reports/top_level_view.md) — 当前调研期地图：多模态对齐/调研进度/证据/方向空缺/未决

**现状与约束**（先读）

- [问题、方案与需求](design/problems_and_requirements.md) — 现状综述：收束总路线（多模态对齐）→ 问题与解法局限 → RL 可缓解性边界 → 证据分级
- [未决问题登记](issues.md) — ISSUE-001 次优解 / ISSUE-002 观测 gap / ISSUE-003 终态歧义
- [项目约束与事实承诺](PROJECT.md) — 稳态假设及包络、文档内容边界、环境与凭据

**证据链**（查证用）

- [综述定稿](docs/survey/survey_findings.md) — 全部文献结论与引用（含 RQ4 动作模态对齐）
- [证据矩阵](synthesis/evidence_matrix.md) / [缺口清单](synthesis/gap_list.md) — (论文, 研究问题) 证据等级；11 缺口含 blocking
- [调研范围与证据标准](docs/survey/research_questions.md) · [术语表](docs/survey/glossary.md) · [RQ4 动作模态对齐](docs/survey/research_questions.md)（主动研究方向）
- [社区共性问题综述](docs/survey/community_issue_survey.md) — GitHub issues 社区证据
- [文献库元数据](papers/metadata.yaml) · [抽取笔记](notes/extractions/) — 48 篇

**研究网格（对齐方向，2026-09-08/09 两轮）**

| 核心点 | 命中论文（链接） | 可信结论/结果 |
|---|---|---|
| C1 动作模式结构（流形/聚类/函数簇） | [MOTIF](https://arxiv.org/abs/2602.13764)、[DiLA](https://arxiv.org/abs/2605.15725)、[CLAM](https://arxiv.org/abs/2505.04999) | 动作母题 = 进度感知 InfoNCE + 本体对抗 GRL 的 VQ 聚类，去运动学规范化使性能降 10.33%（MOTIF）；连续+联合潜动作 74% vs 离散+非联合 16%（CLAM，连续性关键） |
| C2 多模态对齐空间 | [X-Tokenizer](https://arxiv.org/abs/2606.14752)、[UVAM](https://arxiv.org/abs/2503.00200)、[GAM](https://arxiv.org/abs/2606.17046)、[CLASS](https://arxiv.org/abs/2508.01600) | SRQ 非对称量化 + 三头语义对齐把 action 序列编码进 vision/text 共同空间（X-Tokenizer）；DTW 动作相似度监督 Soft InfoNCE，85%/91% vs 基线 63%/77%（CLASS） |
| C3 对齐度量 | [CLASS](https://arxiv.org/abs/2508.01600)、[ProGAL](https://arxiv.org/abs/2604.09824) | 可借形态：DTW-InfoNCE、符号-实体 GAC InfoNCE；语言漠视条件互信息 0.36-0.57 降至 0.08-0.19（ProGAL）。无"度量即目标"专作（GAP-12 上半） |
| C4 增强操作（对齐约束下改写 action） | [Mitrano 综述](https://arxiv.org/abs/2205.02886)、[Streaming Flow Policy](https://arxiv.org/abs/2505.21851) | 增强合法性形式化 = valid/relevant/diverse，增强操作作用于轨迹层（Mitrano）；action 轨迹视为 flow 轨迹（SFP）。"向效率端推"无直接文献 |
| C5 sim/跨本体数据合法化 | [BAR](https://arxiv.org/abs/2607.27549)、[UAT](https://arxiv.org/abs/2501.10105)、[FlowWAM](https://arxiv.org/abs/2607.13017)、[MOTIF](https://arxiv.org/abs/2602.13764) | EE 轨迹对齐行为表征最有效（BAR）；通用动作 256x128 码本统 28 本体，新本体微调仅 0.8% 参数（UAT） |
| C6 数据效率 | [Learning to Discern](https://arxiv.org/abs/2310.14196) | 偏好 + 表示学习处理异构演示。action 侧信息论论证空缺 |
| C7 执行效率度量 | [Embodied Efficiency](https://arxiv.org/abs/2603.19131) | 指标集：任务时间/EE 路径/关节路径/能耗代理 + 成功率，把"执行更快"变成可测（可直接复用） |
| C8 老问题边界（次优/停顿/关键帧/歧义） | 无直接文献 | 对齐视角下：次优≈低效端、停顿≈对齐证据——待自证 |
| RQ4.6 时序保真（驻留/停顿对齐语义） | 无命中 | 多轮检索无直接工作（GAP-12 下半） |

检索过程与关键词见 `logs/search_log.md`；逐篇抽取见 `notes/extractions/paper_036–048.md`。

**设计与机会**

- [机会候选](opportunities.md) — OPT-002 sim/跨本体 action 对齐迁移（OPT-001 已删）

**过程资产**（溯源）

- [完整论证存档](docs/reports/main_report.md) · [评审与修订记录](synthesis/review_record.md) · [调研计划](plan.md) · [搜索/下载/决策日志](logs/)

逻辑流转：

```
issues.md + PROJECT.md（问题与约束，事实登记）
        ↓
top_level_view.md（五面定位）   survey_findings.md + evidence_matrix.md（证据侧翼）
        ↓
opportunities.md（候选方向）
        ↓ 新发现回流
issues.md（登记新问题，如 ISSUE-003）
```

- 日常主要阅读点以本文档导航为准；`logs/`、`plan.md`、`main_report.md`、
  `review_record.md` 为历史档案。
- 证据分级（[社区一致] / [单篇受控] / [本项目实测] / [引用未核验]）见
  `problems_and_requirements.md` §4。

## 文献库使用

- 元数据：`papers/metadata.yaml`（48 篇，含 arXiv ID、screening_relevance）；
- 抽取笔记：`notes/extractions/paper_NNN.md`（每篇 ≤80 行）；
- PDF 不入库：`python scripts/arxiv_fetch.py <arxiv_id> --pdf` 按需重现；
- 一致性检查（观测型输出，clone 后 PDF 缺失属预期）：

```bash
python scripts/check_metadata.py
python scripts/check_extractions.py
python scripts/check_evidence_matrix.py
```

## 规范

项目遵循 [AGENTS.md](AGENTS.md)（一切显式、状态唯一归属、观测型验证）。
