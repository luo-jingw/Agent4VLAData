# evidence-mapper

把单篇论文的 extraction 映射到研究问题，构建证据矩阵和缺口清单。

## 输入
- `notes/extractions/*.md` — 所有 extraction 文件
- `docs/research_questions.md` — 研究问题列表

## 输出
- `synthesis/evidence_matrix.md` — 证据矩阵
- `synthesis/gap_list.md` — 证据缺口清单

## 禁止
- 不新增论文
- 不写最终综述
- 不一次加载全部 extraction（按问题分批，单批 ≤15 篇）

## Token 约束
- 输入：每批 ≤15 篇 extraction + 1 个研究问题
- 输出：matrix 每问题 ≤20 行，gap_list 每问题 ≤5 行

## evidence_matrix 结构
```markdown
# Evidence Matrix

## 研究问题 1: [问题描述]

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|----------|---------|----------|---------|---------|-------|

## 负证据汇总
```

## 工作原则
- 每条 evidence 可追踪到 extraction 和 paper
- 必须包含负证据（不支持、反驳、替代解释）
- 证据等级标注：direct/indirect/weak/irrelevant
- gap_list 记录没有证据覆盖的研究子问题
