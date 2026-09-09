# Agent4VLAData

VLA（Vision-Language-Action）数据处理 agent 的调研与设计仓库。

- 现有方法：纯几何曲率重采样管线（`reference/g2-orion05-resample-training/`，
  已训练部署）；
- 目标：agent 做语义级数据处理——识别并去除遥操数据的次优解；
- 当前状态：调研完结（35 篇文献）、设计框架与草稿就绪，实验未启动。

## 目录分工

| 目录 | 职责 |
|---|---|
| `docs/survey/` | 调研定义与综述定稿 |
| `docs/reports/` | 报告（顶层视角、综合报告） |
| `docs/design/` | 设计框架 |
| `design/` | 设计草稿（未验证） |
| `synthesis/` | 证据矩阵、缺口清单、评审记录 |
| `papers/` | 文献库：候选/元数据/bibtex/raw 文本 |
| `notes/extractions/` | 35 篇论文的逐篇抽取笔记 |
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

- [顶层视角：五个面](docs/reports/top_level_view.md) — 概念归属：管线/旋钮/决策/验证/过程

**现状与约束**（先读）

- [问题、方案与需求](design/problems_and_requirements.md) — ★ 现状综述：四类问题 → 解法局限 → 必需功能与接口 → 证据分级
- [未决问题登记](issues.md) — ISSUE-001 次优解 / ISSUE-002 观测 gap / ISSUE-003 终态歧义
- [项目约束与事实承诺](PROJECT.md) — 稳态假设及包络、文档内容边界、环境与凭据

**证据链**（查证用）

- [综述定稿](docs/survey/survey_findings.md) — 全部文献结论与引用
- [证据矩阵](synthesis/evidence_matrix.md) / [缺口清单](synthesis/gap_list.md) — (论文, 研究问题) 证据等级；11 缺口含 blocking
- [调研范围与证据标准](docs/survey/research_questions.md) · [术语表](docs/survey/glossary.md)
- [agent 入门教学](docs/survey/agent_basics_tutorial.md) · [社区共性问题综述](docs/survey/community_issue_survey.md) — 零基础教学；GitHub issues 社区证据
- [文献库元数据](papers/metadata.yaml) · [抽取笔记](notes/extractions/) — 35 篇

**设计与机会**

- [设计框架](docs/design/design_framework.md) — L0 十模块流程 / L1 benchmark / L2 旋钮体系 / L3 agent 设计 / L4 输入输出
- [设计草稿](design/design_draft.md) — 十模块接口、指标工具、agent 工具（TBD 未验证）
- [机会候选](opportunities.md) — OPT-001 agent 语义旋钮 / OPT-002 real2sim 捷径

**过程资产**（溯源）

- [完整论证存档](docs/reports/main_report.md) · [评审与修订记录](synthesis/review_record.md) · [调研计划](plan.md) · [搜索/下载/决策日志](logs/)

逻辑流转（沿 ★ 文件展开）：

```
issues.md + PROJECT.md（问题与约束，事实登记）
        ↓ 引用
★ problems_and_requirements.md（现状综述：问题→局限→需求→证据分级）
        ↓ 骨架                    ↓ 证据侧翼
top_level_view.md（五面定位）   survey_findings.md + evidence_matrix.md
        ↓ 被应对
design_framework.md → design_draft.md（方案，TBD 未验证）
        ↑ 候选方向
opportunities.md（OPT-001 / OPT-002）
        ↓ 新发现回流
issues.md（登记新问题，如 ISSUE-003）
```

- ★ = 日常主要阅读点；其余文件按需查证（`logs/`、`plan.md`、`main_report.md`、
  `review_record.md` 为历史档案）。
- 逻辑链 = 地图定归属 → 现状与约束 → 证据侧翼 → 设计应对 → 未决回流现状；
  过程资产只做溯源。
- 证据分级（[社区一致] / [单篇受控] / [本项目实测] / [引用未核验]）见
  `problems_and_requirements.md` §4。

## 文献库使用

- 元数据：`papers/metadata.yaml`（35 篇，含 arXiv ID、screening_relevance）；
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
