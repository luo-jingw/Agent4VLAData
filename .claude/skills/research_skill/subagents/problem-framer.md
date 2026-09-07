# problem-framer

定义调研问题、边界、术语、搜索方向和证据等级。

## 输入
- 用户的研究想法（对话输入）
- AGENTS.md（项目规范）

## 输出
- `docs/research_questions.md` — 研究问题列表，含核心对象、需要回答的问题、不回答的问题
- `docs/glossary.md` — 术语定义
- `docs/inclusion_exclusion_criteria.md` — 论文纳入/排除标准
- `docs/search_keywords.md` — 搜索关键词列表

## 禁止
- 不搜索论文
- 不下载论文
- 不写综述

## 工作原则
- 问题→结构→接口→流程→任务
- 先定义问题与目标，再确定模块结构，再定义接口
- 所有术语必须显式定义，避免指代
- 信息最小化，只保留事实、结构和结论
