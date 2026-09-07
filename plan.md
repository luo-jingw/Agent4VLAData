# Problem

## Current

- 现有数据处理管线 = 纯几何动作重采样（`g2-orion05-resample-training.zip`）：
  曲率进度坐标 `|q''|^(1/3)` 稀疏取点 + 吸盘事件窗按时间保护 + 关键帧 loss 加权。
  观测帧全保留，只有标签内部的动作序列被重采样。
- 已知收益：停顿/犹豫段自动失去标签密度（实测 19% 帧只占 0.23% 关节行程）。
- 已知缺陷：无法识别和去除遥操的**次优解**（绕路、犹豫后折返、可完成但非最优的
  轨迹）。几何量看不到语义，次优轨迹在几何上与最优轨迹无法区分。

## Problem

- 调研需求：如何用 agent 做语义级的数据处理，去除一部分次优解。
- 交付物含 agent 入门教学（面向零基础读者）。
- 不清楚传统机器人控制/机器人学习领域的数据预处理有没有公认范式可循。

## Goal

交付三项调研成果，全部落地为可追踪文件：

1. 机器人数据预处理范式综述：常见方法、标准流程、是否存在公认范式。
2. Agent 基础教学文档：面向入门学生的概念、组件、运行方式。
3. Agentic 数据策展综述：agent 用于数据处理/数据优化的现有方法，
   以及机器人领域语义级数据管线的先例。
4. 上述成果对现有曲率重采样管线的可借鉴点 → `opportunities.md`。

调研结论必须可追踪：结论 → evidence matrix → extraction → 论文 → metadata + PDF。
不确定处显式标记，不把假设写成结论。

# Structure

## Modules

| 模块 | 职责 | 位置 |
|---|---|---|
| M1 调研范围与计划 | 研究问题、术语、纳入排除、关键词 | `plan.md`、`docs/survey/research_questions.md`、`docs/survey/glossary.md`、`docs/survey/research_questions.md`、`logs/search_log.md` |
| M2 文献库 | 候选、下载、元数据、bibtex | `papers/candidates.yaml`、`papers/metadata.yaml`、`papers/bibtex.bib`、`papers/raw/` |
| M3 抽取与证据 | 逐篇抽取、证据矩阵、缺口 | `notes/extractions/`、`synthesis/evidence_matrix.md`、`synthesis/gap_list.md` |
| M4 综述与教学 | 初稿、反方审查、修订稿 | `synthesis/review_record.md`（反方审查+修订映射） |
| M5 定稿交付 | 最终文档 | `docs/`（见 Interface） |
| M6 检查脚本 | 观测型检查 | `scripts/check_metadata.py`、`check_extractions.py`、`check_evidence_matrix.py` |
| M7 日志 | 搜索、下载、决策记录 | `logs/search_log.md`、`download_log.md`、`decision_log.md` |

## Responsibilities

- 搜索只产出候选，不判断结论。
- 下载只归档文件与元数据，不读全文。
- 抽取只记录论文内容，不扩展问题、不下结论。
- 综述只引用 evidence matrix 已有条目，不新增来源。
- 反方审查只提问题，不改正文。
- 修订只处理已记录问题，不引入新方向。

## State Ownership

- 候选文献状态唯一归属 `papers/candidates.yaml`。
- 下载状态唯一归属 `papers/metadata.yaml` + `logs/download_log.md`。
- 证据等级唯一归属 `notes/extractions/*.md` 与 `synthesis/evidence_matrix.md`。
- 结论只出现在 `synthesis/` 与 `docs/` 定稿；`docs/` 只收修订后的定稿与
  已验证的调研定义（研究问题、术语、标准），不收草稿与未决假设。

# Interface

## Interfaces

- `candidates.yaml`：每篇 `id/title/authors/year/source/source_id/url/abstract/keywords/relevance/notes`。
- `metadata.yaml`：每篇 `id/title/authors/year/venue/source_id/pdf_path/bibtex/status/evidence_level`。
- extraction 文件：每篇一个 `notes/extractions/paper_NNN.md`，≤80 行，
  含基本信息/研究问题/方法/关键发现/证据等级/引用/不确定项。
- evidence matrix：每问题一张表 + 负证据汇总 + 缺口列表。
- 定稿交付（`docs/survey/`）：
  - `docs/survey/survey_findings.md`（综述定稿，RQ1–RQ3）
  - `docs/survey/agent_basics_tutorial.md`（agent 教学）
  - `docs/survey/glossary.md`、`docs/survey/research_questions.md`（调研定义）

## Inputs

- 任务需求。
- 现有方法包 `g2-orion05-resample-training.zip`（几何管线上下文）。
- arXiv API / Semantic Scholar（文献来源，需网络）。

## Outputs

- 定稿 + 证据矩阵 + 文献库（PDF/元数据/抽取）。
- `opportunities.md` 新条目、`issues.md` 阻断项（如有）、`PROJECT.md` 更新。

## State Changes

- 调研流程：候选 → downloaded → extracted → mapped → reviewed → revised → final。
- 每步由对应 owner 文件记录，检查脚本验证一致性。

# Flow

## Main Flow

1. 问题定义：4 个框架文档（M1）。
2. 关键词生成：按研究问题分组（`logs/search_log.md`）。
3. 文献搜索：先 seed（每问题 ≤5 篇）验证关键词与 schema，再扩展（每问题 ≤20）。
4. 论文下载：arXiv API 下载 PDF，记录 metadata/bibtex/日志。
5. 论文抽取：逐篇按 schema 抽取，保留不确定性。
6. 证据映射：按问题分批（每批 ≤15 篇）填 evidence matrix + gap list。
7. 综述生成：按章生成初稿（每章 ≤150 行），含 agent 教学章节。
8. 反方审查：检查漏洞、替代解释、证据等级错误、过度结论。
9. 修订定稿：写入 `docs/`，更新持久状态文件。

# Code Mapping

## Modules

| 模块 | 文件 |
|---|---|
| 研究问题 | `docs/survey/research_questions.md` |
| 术语表 | `docs/survey/glossary.md` |
| 纳入排除标准 | `docs/survey/research_questions.md` |
| 搜索关键词 | `logs/search_log.md` |
| 候选文献 | `papers/candidates.yaml` |
| 下载元数据 | `papers/metadata.yaml`、`papers/bibtex.bib`、`papers/raw/*.pdf` |
| 抽取 | `notes/extractions/paper_NNN.md` |
| 证据矩阵 | `synthesis/evidence_matrix.md`、`synthesis/gap_list.md` |
| 评审记录 | `synthesis/review_record.md` |
| 定稿 | `docs/survey/survey_findings.md`、`docs/survey/agent_basics_tutorial.md` |
| 检查脚本 | `scripts/check_metadata.py`、`check_extractions.py`、`check_evidence_matrix.py` |
| 日志 | `logs/search_log.md`、`download_log.md`、`decision_log.md` |
| 机会 | `opportunities.md` |
| 阻断问题 | `issues.md` |
| 项目上下文 | `PROJECT.md` |

## Interfaces

- 检查脚本读 metadata.yaml / extractions / evidence_matrix，输出观测指标
  （文件数、字段缺失、重复、证据等级分布、引用覆盖），不写 pass/fail。

## State

- 见 State Ownership。

# Implementation

## Phase 1

### Goal

初始化调研目录结构与检查脚本。

### Files

- 创建 `papers/raw/`、`notes/extractions/`、`synthesis/`、`logs/`、`scripts/`。
- 复制 `scripts/check_metadata.py`、`check_extractions.py`、`check_evidence_matrix.py`。

### Structures

无。

### Affected Modules

M2、M6。

### Observation

目录存在；脚本可执行。

## Phase 2

### Goal

定义研究问题、术语、纳入排除标准、搜索关键词。

### Files

- `docs/survey/research_questions.md`
- `docs/survey/glossary.md`
- `docs/survey/research_questions.md`
- `logs/search_log.md`

### Structures

研究问题编号 RQ1.1–RQ1.4、RQ2.1–RQ2.4；证据等级 direct/indirect/weak/irrelevant。

### Affected Modules

M1。

### Observation

4 个文件存在且字段完整。

## Phase 3

### Goal

seed 文献搜索：RQ1、RQ2 各 ≤5 篇，写入 `papers/candidates.yaml`。

### Files

- `papers/candidates.yaml`
- `logs/search_log.md`

### Structures

candidates.yaml schema。

### Affected Modules

M2、M7。

### Observation

candidates 数量 ≤10；每篇有 title/source_id/url；relevance 已填。

## Phase 4

### Goal

下载 seed PDF，填 metadata.yaml、bibtex.bib、download_log.md。

### Files

- `papers/raw/*.pdf`
- `papers/metadata.yaml`
- `papers/bibtex.bib`
- `logs/download_log.md`

### Structures

metadata.yaml schema。

### Affected Modules

M2、M7。

### Observation

`python scripts/check_metadata.py`：PDF 数与 metadata 条目一致，字段缺失为 0。

## Phase 5

### Goal

逐篇结构化抽取，每篇 ≤80 行，保留不确定性。

### Files

- `notes/extractions/paper_NNN.md`

### Structures

extraction schema。

### Affected Modules

M3。

### Observation

`python scripts/check_extractions.py`：抽取文件数 = 已下载数，无 metadata 存在而
extraction 缺失。

## Phase 6

### Goal

按研究问题构建证据矩阵与缺口列表。

### Files

- `synthesis/evidence_matrix.md`
- `synthesis/gap_list.md`

### Structures

每问题一张证据表 + 负证据汇总 + 缺口。

### Affected Modules

M3。

### Observation

`python scripts/check_evidence_matrix.py`：matrix 引用的 paper_id 全部存在于
metadata.yaml；负证据与缺口非空。

## Phase 7

### Goal

写综述初稿：RQ1 范式综述、agent 教学章节、agentic curation 综述。

### Files

- `synthesis/review_record.md`

### Structures

按研究问题分章，每章 ≤150 行；引用格式 `[paper_NNN]`。

### Affected Modules

M4。

### Observation

每章至少一条 evidence 引用；无引用的结论段显式标记为推理。

## Phase 8

### Goal

反方审查：漏洞、替代解释、证据等级错误、过度结论。

### Files


### Structures

逐条审查意见，长度 ≤ 初稿 50%。

### Affected Modules

M4。

### Observation

每条意见定位到初稿具体章节。

## Phase 9

### Goal

修订定稿写入 `docs/`，更新持久状态。

### Files

- `docs/survey/survey_findings.md`
- `docs/survey/agent_basics_tutorial.md`
- `opportunities.md`（新条目）
- `issues.md`（如有阻断）
- `PROJECT.md`（Purpose/Environment/Onboarding）

### Structures

定稿结构与初稿一致，改动 ≤30%。

### Affected Modules

M5、持久状态。

### Observation

三份定稿存在；opportunities 条目与证据矩阵互链；检查脚本全部通过。

## Phase 10（第三轮：RQ3 搜索与下载）

### Goal

追加 RQ3.1–RQ3.5（划分/泄漏、时间同步、增强安全性、观测可分性、覆盖分析），
搜索并下载 9 篇（paper_024–032）。

### Files

- `docs/survey/research_questions.md`（RQ3 追加）
- `papers/candidates.yaml`、`papers/metadata.yaml`、`papers/bibtex.bib`
- `papers/raw/*.pdf`（9 篇）
- `scripts/s2_search.py`（Semantic Scholar 检索工具）
- `logs/search_log.md`

### Structures

沿用既有 schema；新论文 evidence_level 由抽取后回填。

### Affected Modules

M2、M7。

### Observation

check_metadata：32 条条目、PDF 32、一致性 0 缺失。

## Phase 11（第三轮：抽取）

### Goal

9 篇结构化抽取。

### Files

- `notes/extractions/paper_024.md` … `paper_032.md`

### Affected Modules

M3。

### Observation

check_extractions：32 抽取文件、0 缺失。

## Phase 12（第三轮：矩阵/定稿/状态更新）

### Goal

证据矩阵追加 RQ3 五表、gap_list 更新（GAP-08 更新、GAP-09/10 新增）、
新增 `docs/survey/survey_findings.md`、PROJECT.md 补记稳态约束边界条件。

### Files

- `synthesis/evidence_matrix.md`
- `synthesis/gap_list.md`
- `docs/survey/survey_findings.md`
- `PROJECT.md`
- `opportunities.md`（旋钮 4 标注口径修正）

### Observation

check_evidence_matrix：32 paper_id 全部被定稿引用；证据分布 direct 48 /
indirect 3 / weak 1；检查脚本全部通过。

## Phase 13（追加：SCIZOR/ATHENA/切点检测）

### Goal

"agent 是否存在"定向检索追加 3 篇（paper_033–035），矩阵补行、GAP-11 新增、
综合报告 §5.2 补 GAP-11。

### Files

- `papers/`（candidates/metadata/bibtex/raw）
- `notes/extractions/paper_033.md` … `paper_035.md`
- `synthesis/evidence_matrix.md`、`synthesis/gap_list.md`
- `docs/survey/survey_findings.md`（§2 质量过滤、§3 语义先例补行）
- `docs/reports/main_report.md`（§5.2 补 GAP-11）

### Observation

check 脚本全过：35 篇、35 抽取、引用覆盖 0 缺失。
