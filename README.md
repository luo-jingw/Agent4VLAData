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

## 快速跳转

**从哪开始读**

- [顶层视角（五面思维地图）](docs/reports/top_level_view.md)——先建立整体框架
- [综合报告](docs/reports/main_report.md)——完整论证与结论
- [agent 入门教程](docs/survey/agent_basics_tutorial.md)——面向零基础读者

**调研成果**

- [综述定稿](docs/survey/survey_findings.md)——范式、质量过滤教训、语义与 agentic、管线模块证据
- [研究问题与证据标准](docs/survey/research_questions.md)
- [术语表](docs/survey/glossary.md)
- [证据矩阵](synthesis/evidence_matrix.md) / [缺口清单](synthesis/gap_list.md)

**设计阶段**

- [设计框架](docs/design/design_framework.md)——L0–L4 分层与旋钮体系
- [设计草稿](design/design_draft.md)——十模块接口、指标工具、agent 工具（draft，未验证）
- [问题、方案与需求](design/problems_and_requirements.md)——四类实际问题、现有解法局限、必需功能与接口、社区讨论链接

**项目状态**

- [PROJECT.md](PROJECT.md)——项目特定约束、环境、onboarding
- [plan.md](plan.md)——当前工作（调研阶段已完成，含 13 个 phase 记录）
- [issues.md](issues.md)——未决问题（ISSUE-001 次优解、ISSUE-002 观测 gap）
- [opportunities.md](opportunities.md)——机会清单（OPT-001 agent 四旋钮、OPT-002 real2sim 捷径）

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
