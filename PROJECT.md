# PROJECT.md

Project-specific context that supplements `AGENTS.md`.

`AGENTS.md` is fixed across projects.
`PROJECT.md` accumulates what is specific to this one.

Update this file when a new project-specific constraint, environment
detail, or convention is discovered during work.

Do not duplicate:
- verified architecture → `docs/`;
- current work → `plan.md`;
- unresolved problems → `issues.md`.

## Template

- Initialized from (source, version, or commit):
- Last synced to (source, version, or commit):
- Last applied migration id (see the template's `migrations.json`):

## Purpose

调研与设计用于机器人数据处理管线（遥操数据 → 训练标签）的 agent 方案，
使数据处理具备语义级理解，去除几何方法（曲率重采样）无法处理的次优解。
当前阶段产出：机器人数据预处理范式综述、agent 基础教学、agentic 数据策展
综述（2026-09 完成，定稿见 `docs/survey/`，报告见 `docs/reports/`，
证据链在 `synthesis/`）。

## Environment

- Platform / hardware: Windows 10/11，Python 3.11.7，pymupdf 可用。
- Shared with (other users or projects on the same account or machine):
- Constraints imposed by the environment:
  - Windows 控制台默认 GBK 编码；所有输出中文的脚本必须
    `sys.stdout.reconfigure(encoding="utf-8")`。
  - 需要外网访问 arXiv API 做文献检索与下载（已验证可达）。
  - 训练机器/数据在别处；本机只做调研与文档。

## Credentials

Services that require a scoped identity in this project (git remotes,
package registries, model hubs, cloud APIs).

| Service | Scope | Source | Notes |
|---|---|---|---|
| arXiv API | 公开，无凭据 | http://export.arxiv.org | 文献检索/下载 |
| GitHub remote | 项目仓库 | https://github.com/luo-jingw/Agent4VLAData.git | 项目自身版本控制 |
| 数据集（pick/place） | 未接入本机 | 在训练机器上 | 后续实验需明确访问方式 |

Do not assume a global or default identity applies.
See `AGENTS.md` → Credentials and Environment Isolation.

Do not record token values or other secrets here.

## Project-Specific Constraints

- 文档与交付语言：中文。
- **场景假设（来自用户，2026-09-06）**：当前任务场景全部静态（料箱/传送带
  均静止）。机械臂速度沿路径任意变化不影响成功；成功判据 = 路径几何一致
  （含搬运段工件刚性附着末端）∧ 吸盘状态切换停留时间一致。含义：
  (1) real2sim 只需静态碰撞检测，无需动力学仿真；
  (2) 事件窗（吸盘触发+真空建立）是唯一的时序约束，捷径/稀疏化不得穿越；
  (3) 部署侧非等时标签的 dt 调度在静态场景下无动态风险。
  **边界条件（第三轮调研补充，paper_026）**：速度无关性只在**无接触动力学**
  的段成立；接触/力控任务中速度变化改变环境反应（真实反应 vs 时间缩放复制
  88% vs 31%）。本项目的吸盘抓取-搬运-放置符合无接触段假设（吸盘吸附后
  工件刚性附着），但任何涉及力接触的任务扩展必须先验证该假设。
- 现有数据处理方法 = 包内 `reference/g2-orion05-resample-training/`
  （自 zip 解压，只读参照；方法说明在其 `docs/` 下）。不要改这个参照包，
  后续实现另起文件。
- 调研产物遵循 research_skill 的目录约定：`papers/`（文献）、`notes/extractions/`
  （抽取）、`synthesis/`（证据矩阵与综述）、`logs/`、`scripts/`。
  `docs/` 按用途分层：`docs/survey/`（调研定义与综述定稿）、
  `docs/reports/`（报告）、`docs/design/`（设计框架与草稿）。
- 文献检查脚本只做观测型输出，不写 pass/fail。
- 版本控制：GitHub remote 见 Credentials。`g2-orion05-resample-training.zip`
  与 `papers/raw/*.pdf` 不入库——zip 是协作分发物（解压参照在 `reference/`）；
  PDF 用 `python scripts/arxiv_fetch.py <id> --pdf` 按 arXiv ID 重现
  （`papers/raw/*.txt` 已入库供离线检索）。clone 后如缺 PDF，检查脚本会
  报告"PDF 缺失"，属预期。
- 项目技能：调研工作流 skill 已从全局复制为项目技能
  `.claude/skills/research_skill/`（含 subagents 与检查脚本；用户明确要求）。

## Onboarding

新贡献者上手顺序：

1. 读 `AGENTS.md`（规范）。
2. 读 `docs/reports/top_level_view.md`（五面思维地图，先建立整体框架）。
3. 读 `docs/reports/main_report.md`（综合报告；含全文论证主线）。
4. 读 `docs/survey/research_questions.md`（调研范围）与定稿：
   `docs/survey/survey_findings.md`（综述）、
   `docs/survey/agent_basics_tutorial.md`（agent 教学）、
   `docs/survey/glossary.md`（术语）。
5. 设计阶段材料：`docs/design/design_framework.md`、`docs/design/design_draft.md`。
6. 看 `synthesis/evidence_matrix.md` 与 `synthesis/gap_list.md`（证据与未决）。
7. 现有方法：`reference/g2-orion05-resample-training/README.md`。
8. 下一步工作候选：`opportunities.md` OPT-001/002、`issues.md` ISSUE-001/002。