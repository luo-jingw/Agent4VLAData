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

## 文件树（含逻辑流转）

角色标注：地图 / 现状与约束 / 证据链 / 设计与机会 / 过程资产。

```
agent4data/
├── README.md ────────────────── 本文件：导航入口
├── AGENTS.md ────────────────── 模板规范（一切显式、状态唯一归属）
├── PROJECT.md ───────────────── 现状 项目约束与事实承诺（稳态假设、文档边界）
├── plan.md ──────────────────── 过程 调研阶段计划（13 phase 已完成）
├── issues.md ────────────────── 现状 未决问题（001 次优 / 002 观测 gap / 003 终态歧义）
├── opportunities.md ─────────── 设计 机会候选（OPT-001 语义旋钮 / OPT-002 real2sim 捷径）
│
├── docs/                        ── 已接受的世界模型（verified）
│   ├── reports/
│   │   ├── [top_level_view.md](docs/reports/top_level_view.md) ─ 地图 五面思维地图（问题怎么归属）
│   │   └── [main_report.md](docs/reports/main_report.md) ───── 过程 完整论证（背景档案）
│   ├── survey/
│   │   ├── [research_questions.md](docs/survey/research_questions.md) ─ 证据 调研范围与证据标准
│   │   ├── [glossary.md](docs/survey/glossary.md) ───────────────── 证据 术语表
│   │   ├── [survey_findings.md](docs/survey/survey_findings.md) ─── 证据 综述定稿（文献结论）
│   │   └── [agent_basics_tutorial.md](docs/survey/agent_basics_tutorial.md) ─ 证据 agent 教学
│   └── design/
│       └── [design_framework.md](docs/design/design_framework.md) ─ 设计 L0–L4 设计框架
│
├── design/                     ── 未验证区（草稿，不属 world model）
│   ├── [problems_and_requirements.md](design/problems_and_requirements.md) ─ ★现状 现状综述：
│   │     问题 → 方案局限 → 必需功能与接口 → 证据分级（见下）
│   └── [design_draft.md](design/design_draft.md) ─────────────────── 设计 具体设计草稿（TBD）
│
├── synthesis/                  ── 证据链（查证用）
│   ├── [evidence_matrix.md](synthesis/evidence_matrix.md) ── 证据 (论文, 研究问题) → 证据等级
│   ├── [gap_list.md](synthesis/gap_list.md) ──────────────────── 证据 缺口（含 blocking 状态）
│   └── [review_record.md](synthesis/review_record.md) ────────── 过程 评审与修订记录
│
├── papers/                     ── 证据 文献库（candidates/metadata/bibtex/raw 文本）
├── notes/extractions/          ── 证据 35 篇逐篇抽取笔记
├── scripts/                    ── 检索/下载/检查脚本（观测型）
├── logs/                       ── 过程 搜索/下载/决策记录
├── reference/                  ── 现有方法包（只读参照）
└── .claude/skills/             ── 项目技能（research_skill 等）
```

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
