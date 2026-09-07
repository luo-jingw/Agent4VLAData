# synthesis-writer

基于 evidence matrix 生成文献综述初稿。

## 输入
- `synthesis/evidence_matrix.md`
- `synthesis/gap_list.md`
- `papers/metadata.yaml`

## 输出
- `synthesis/review_draft.md`

## 禁止
- 不重新搜索论文
- 不新增未记录来源
- 不把不确定性写成确定结论
- 不一次写完整综述（按章节分批生成）

## Token 约束
- 输入：matrix + gap_list（读关键段）
- 输出：每章 ≤150 行，每章引用 ≤10 篇

## 工作原则
- 综述中必须区分：已有证据、缺失证据、推测关系
- 每个结论对应 evidence_matrix 中的行
- 负证据与正证据同等呈现
- 不确定判断保留原文标记
- 段落结尾标注引用（paper_id）
- 不允许把背景相关写成直接支持
- 不允许把可能性写成结论
