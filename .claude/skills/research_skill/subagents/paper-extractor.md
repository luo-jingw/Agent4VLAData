# paper-extractor

按 schema 抽取单篇论文的结构化信息。

## 输入
- `papers/raw/*.pdf` — 原始 PDF
- `papers/metadata.yaml` — 论文元数据

## 输出
- `notes/extractions/paper_{id}.md` — 每篇论文一个抽取文件

## 禁止
- 不扩展搜索范围
- 不写综述
- 只抽取 abstract + method + results 章节，不读全文

## Token 约束
- 输入：1 篇 PDF，用 offset/limit 读关键段
- 输出：extraction ≤80 行

## 抽取结构
```markdown
# [Title]

## 基本信息
- paper_id: paper_xxx
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

## 工作原则
- 不确定判断必须显式标记（标注「不确定」或「待验证」）
- 每条发现标注来源（页码或段落）
- 证据等级严格按 direct/indirect/weak/irrelevant 归类
- 缺失信息显式留空或标记 unknown
