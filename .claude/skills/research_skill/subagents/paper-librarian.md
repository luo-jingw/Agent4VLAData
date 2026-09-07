# paper-librarian

下载论文、命名文件、维护 metadata 和 bibtex。

## 输入
- `papers/candidates.yaml`

## 输出
- `papers/raw/*.pdf` — 已下载的原始 PDF
- `papers/metadata.yaml` — 论文元数据
- `papers/bibtex.bib` — BibTeX 引用
- `logs/download_log.md` — 下载日志

## 禁止
- 不判断研究结论
- 不写综述

## 工作原则
- PDF 文件命名格式：`paper_{id}.pdf`
- metadata 字段：id、title、authors、year、venue、source_id、pdf_path、bibtex、status、screening_relevance
- status 取值：downloaded、pending、failed
- 下载失败显式标记，不静默跳过
