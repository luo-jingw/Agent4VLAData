---
name: research_skill
description: Use when the user asks for systematic literature survey and review writing. Covers problem framing, literature search, paper archiving, content extraction, evidence mapping, review synthesis, critical review, and revision. Use ONLY when the user explicitly requests a structured research workflow.
---
# research-skill

用于文献调研和综述形成的系统化工作流。涵盖问题界定、文献搜索、论文归档、内容抽取、证据映射、综述生成、反方审查和修订全流程。

---

## 核心原则

### 工作流原则

1. **问题→范围→证据→结构→任务。** 先定义问题与边界，再定义证据标准，再定义目录结构与文件接口，最后拆分 subagent 任务。
2. **问题先于搜索。** 先明确研究问题、核心对象、因果关系和排除范围，再生成关键词，再搜索文献。禁止先大量下载论文再反向归纳。
3. **结构先于内容。** 先确定目录结构、文件命名、元数据格式和输出文件，再执行搜索、下载、抽取和综述。
4. **证据先于结论。** 每个结论必须来自可追踪证据。每条证据必须对应来源、论文、页码或段落位置。
5. **状态唯一归属。** 每类关键状态只能有一个拥有文件。candidates 只归属 candidates 文件，metadata 只归属 metadata 文件，extraction 只归属 extraction 文件，evidence matrix 只归属 evidence matrix 文件。
6. **接口稳定优先。** 文件 schema 一旦确定，后续任务只填充 schema，不反复修改字段，不在自由文本中隐藏关键状态。
7. **subagent 按职能划分。** 按调研流程划分 subagent，不按主题划分。
8. **读写权限分离。** 搜索 agent 不写综述，下载 agent 不判断结论，抽取 agent 不扩展问题，综述 agent 不重新搜索，审稿 agent 不修改正文。
9. **任务最小可验证。** 每个任务只改变一种文件或一个状态，必须有明确输出文件，必须能通过文件存在、字段完整性、引用完整性进行检查。
10. **先 seed 后扩展。** 第一轮只收集少量 seed papers，先验证关键词、schema 和证据等级是否有效，再扩大搜索范围。
11. **证据等级固定。** 所有论文标注 direct、indirect、weak、irrelevant。
12. **保留负证据。** 必须记录不支持、反驳、替代解释和不可验证点。
13. **文件映射明确。** 每个 subagent 对应配置文件，每个输出对应具体路径，每个阶段声明修改文件。
14. **不过度设计。** 只实现当前阶段必要结构，不做回退设计、冗余设计和多套并行格式。

### 工程原则

1. **一切显式。** 一个文件做一件事，文件名即语义。所有依赖显式声明，接口与实现分离，所有签名完整标注类型。
2. **模块先于实现。** 先确定模块边界与职责，再讨论实现方式。
3. **测试只做观测型输出。** 不写带预期判断的测试代码，不写 pass/fail，只输出指标（文件数量、字段缺失、引用缺失、证据等级分布等）。

---

## 证据等级

| 等级 | 定义 |
|------|------|
| direct | 直接回答调研问题 |
| indirect | 研究相邻机制 |
| weak | 只提供背景、指标或方法启发 |
| irrelevant | 不能支撑当前调研问题 |

---

## 推荐目录结构

```
project/
  AGENTS.md
  docs/
    research_questions.md
    glossary.md
    inclusion_exclusion_criteria.md
    search_keywords.md
  papers/
    raw/                    # 原始 PDF
    candidates.yaml
    metadata.yaml
    bibtex.bib
  notes/
    extractions/            # 逐篇抽取文件
  synthesis/
    evidence_matrix.md
    gap_list.md
    review_draft.md
    critical_review.md
    revised_review.md
  logs/
    search_log.md
    download_log.md
    decision_log.md
  scripts/
    check_metadata.py
    check_extractions.py
    check_evidence_matrix.py
```

---

## 调研计划书框架

1. **问题定义。** 现状、问题、目标、边界、排除范围、核心对象、需要回答的问题、不回答的问题。
2. **证据标准。** direct/indirect/weak/irrelevant 定义、引用格式、纳入/排除标准。
3. **系统结构。** subagent 列表、职责、输入文件、输出文件、关键状态归属文件。
4. **文件接口。** candidates schema、metadata schema、extraction schema、evidence matrix schema、review draft 结构、critical review 结构。
5. **运行流程。** 问题界定流程、关键词生成流程、文献搜索流程、论文下载流程、论文抽取流程、证据映射流程、综述生成流程、反方审查流程、修订流程。
6. **代码映射。** subagent 配置→路径、论文目录→路径、笔记目录→路径、综述目录→路径、日志目录→路径、检查脚本→路径。
7. **实施阶段。** Phase 1~9，每阶段含目标、输入文件、输出文件、修改文件、负责 subagent、观测型检查项。

---

## 8 个 subagent

### 1. problem-framer
- **职责：** 定义问题、边界、术语、搜索方向和证据等级
- **输入：** 用户研究想法
- **输出：** `research_questions.md`、`glossary.md`、`inclusion_exclusion_criteria.md`、`search_keywords.md`
- **禁止：** 不搜索论文、不下载论文、不写综述
- **原则：** 问题→结构→接口→流程→任务

### 2. literature-scout
- **职责：** 发现候选论文
- **输入：** `search_keywords.md`、`inclusion_exclusion_criteria.md`
- **输出：** `candidates.yaml`
- **禁止：** 不下载论文、不写综述、不判断结论
- **原则：** 先 seed 后扩展，第一轮只收集少量

### 3. paper-librarian
- **职责：** 下载论文、命名文件、维护 metadata 和 bibtex
- **输入：** `candidates.yaml`
- **输出：** `papers/raw/*.pdf`、`metadata.yaml`、`bibtex.bib`、`download_log.md`
- **禁止：** 不判断研究结论、不写综述

### 4. paper-extractor
- **职责：** 按 schema 抽取单篇论文信息
- **输入：** `papers/raw/*.pdf`、`metadata.yaml`
- **输出：** `notes/extractions/*.md`（每篇一个文件）
- **禁止：** 不扩展搜索、不写综述
- **原则：** 保留不确定性，不确定处显式标记

### 5. evidence-mapper
- **职责：** 把 extraction 映射到研究问题
- **输入：** `notes/extractions/*.md`
- **输出：** `evidence_matrix.md`、`gap_list.md`
- **禁止：** 不新增论文、不写最终综述
- **原则：** 每条 evidence 可追踪到 paper，标注证据等级

### 6. synthesis-writer
- **职责：** 基于 evidence_matrix 写综述
- **输入：** `evidence_matrix.md`、`gap_list.md`、`metadata.yaml`
- **输出：** `review_draft.md`
- **禁止：** 不重新搜索、不新增未记录来源、不把不确定性写成确定结论
- **原则：** 区分已有证据、缺失证据和推测关系

### 7. critic-reviewer
- **职责：** 检查漏洞、替代解释、证据等级错误和过度结论
- **输入：** `review_draft.md`、`evidence_matrix.md`、`notes/extractions/*.md`
- **输出：** `critical_review.md`
- **禁止：** 不直接改正文

### 8. revision-agent
- **职责：** 根据 critical_review 修订综述
- **输入：** `review_draft.md`、`critical_review.md`
- **输出：** `revised_review.md`
- **禁止：** 只处理已记录问题，不引入新方向

---

## 实施阶段 (9 Phases)

| Phase | 名称 | 负责 subagent | 主要输出 | 调用方式 |
|-------|------|-------------|---------|---------|
| 1 | 初始化结构 | — | 所有目录和空文件 | 单次 |
| 2 | 定义问题与证据标准 | problem-framer | 4 docs | 单次 |
| 3 | 搜索 seed papers | literature-scout | candidates.yaml (seed ≤5) | 单次；第二轮 ≤20 |
| 4 | 下载与归档 | paper-librarian | PDFs, metadata.yaml | 逐篇，不预读全文 |
| 5 | 结构化抽取 | paper-extractor | extractions/*.md | 每篇一次，≤80行/篇 |
| 6 | 构建 evidence matrix | evidence-mapper | evidence_matrix.md, gap_list.md | 按问题分批，每批≤15篇 |
| 7 | 生成综述初稿 | synthesis-writer | review_draft.md | 按章生成，每章≤150行 |
| 8 | 反方审查 | critic-reviewer | critical_review.md | 单次，≤draft 50% |
| 9 | 修订综述 | revision-agent | revised_review.md | 单次，改动≤30% |

---

## 文件 Schema

### candidates.yaml
```yaml
papers:
  - id: "paper_001"
    title: ""
    authors: []
    year: null
    source: ""           # arxiv/doi/url
    source_id: ""        # arxiv_id/doi
    url: ""
    abstract: ""
    keywords: []
    relevance: ""        # high/medium/low
    notes: ""
```

### metadata.yaml
```yaml
papers:
  - id: "paper_001"
    title: ""
    authors: []
    year: null
    venue: ""
    source_id: ""
    pdf_path: ""         # papers/raw/paper_001.pdf
    bibtex: ""
    status: ""           # downloaded/pending/failed
    evidence_level: ""   # direct/indirect/weak/irrelevant
```

### extraction (per-paper, notes/extractions/paper_001.md)
```markdown
# [Title]

## 基本信息
- paper_id: paper_001
- 年份:
- 作者:
- 来源:

## 研究问题

## 方法

## 关键发现

## 证据等级
- 等级: direct/indirect/weak/irrelevant
- 理由:

## 引用

## 不确定项
```

### evidence_matrix.md
```markdown
# Evidence Matrix

## 研究问题 1: [问题描述]

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|----------|---------|----------|---------|---------|-------|
| paper_001 | direct | 支持 | ... | Section 3.2 | 高 |

## 研究问题 2: [问题描述]
...

## 负证据汇总
...

## 证据缺口 (gap_list)
...
```

---

## 观测型检查项

运行 `scripts/check_metadata.py`、`scripts/check_extractions.py`、`scripts/check_evidence_matrix.py` 进行观测型检查：

1. **文件数量：** 候选论文数量、已下载 PDF 数量、metadata 条目数量、extraction 文件数量
2. **字段缺失：** title/year/url/pdf_path/evidence_level 缺失数量
3. **重复项：** 重复 title、重复 DOI、重复 arXiv id、重复 PDF 文件名
4. **证据分布：** direct/indirect/weak/irrelevant 数量
5. **引用覆盖：** 综述段落引用数量、无引用段落数量、未被引用论文数量
6. **状态不一致：** metadata 存在但 PDF 缺失、PDF 存在但 metadata 缺失、extraction 存在但 metadata 缺失、evidence_matrix 引用不存在的 paper_id

---

## Token 管理

### 调用原则

| 原则 | 说明 |
|------|------|
| 先 seed 后扩展 | 第一轮 ≤5 篇，验证关键词有效后再扩 |
| 按需抽取 | 只抽取与 research_questions 直接相关章节 |
| 分批映射 | 按研究问题分批，单批 ≤15 篇 extraction |
| 分段写作 | 按章分别生成，每章引用 ≤10 篇来源 |
| 不重复加载 | 复用前轮结果引用文件路径 |

### 各 subagent 约束

| agent | 输入上限 | 输出上限 |
|-------|---------|---------|
| problem-framer | 用户原始需求 | 4 文件各 ≤40 行 |
| literature-scout | keywords + criteria | 第一轮 ≤5，第二轮 ≤20 |
| paper-librarian | candidates.yaml | 逐篇下载，不提前读全文 |
| paper-extractor | 1 篇 PDF | extraction ≤80 行 |
| evidence-mapper | 每轮 ≤15 篇 | matrix 每问题 ≤20 行 |
| synthesis-writer | matrix + gap_list | 每章 ≤150 行 |
| critic-reviewer | draft + matrix | review ≤ draft 行数 50% |
| revision-agent | draft + review | 改动 ≤原 draft 30% |

---

## 使用方式

当用户提出文献调研或综述需求时：

1. 先按**调研计划书框架**生成计划书（问题定义、证据标准、系统结构、文件接口、运行流程、代码映射、实施阶段）
2. 按 Phase 1→9 逐步执行，每个 Phase 完成后运行对应检查脚本
3. 严格遵循 subagent 读写权限分离和状态唯一归属
4. 不确定处显式标记，不把可能性写成结论

---

## 计划编写原则

1. **信息最小化。** 只保留事实、结构和结论，删除背景与修辞。
2. **结构显式。** 使用稳定层级组织信息，使逻辑关系直接可见。
3. **语义明确。** 关键对象、模块、接口必须显式出现，避免指代。
4. **句式简短。** 短句表达，一句一个信息单元。
5. **一致性。** 术语、命名和结构在全文保持统一。
6. **可执行。** 每个任务必须能被 opencode 执行，每个输出必须能落到文件。
7. **可追踪。** 结论→evidence matrix→extraction→paper→metadata+PDF。
8. **不过度设计。** 不做回退设计、冗余设计、多套格式。
