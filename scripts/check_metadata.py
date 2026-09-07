"""check_metadata.py — 观测型检查脚本，只输出事实指标，不输出 pass/fail。"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import yaml
import os
from pathlib import Path
from collections import Counter

METADATA_PATH = Path("papers/metadata.yaml")
RAW_DIR = Path("papers/raw")

if not METADATA_PATH.exists():
    print("metadata 文件不存在")
    exit(0)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

papers = data.get("papers", [])
print(f"metadata 条目数量: {len(papers)}")

pdf_count = len(list(RAW_DIR.glob("*.pdf"))) if RAW_DIR.exists() else 0
print(f"已下载 PDF 数量: {pdf_count}")

missing_fields = Counter()
duplicate_ids = Counter()
duplicate_titles = Counter()
status_count = Counter()
evidence_count = Counter()
pdf_path_count = 0

for p in papers:
    paper_id = p.get("id", "")
    if paper_id:
        duplicate_ids[paper_id] += 1
    for field in ["title", "year", "source_id", "pdf_path", "screening_relevance"]:
        if not p.get(field):
            missing_fields[field] += 1
    title = p.get("title", "")
    if title:
        duplicate_titles[title] += 1
    status_count[p.get("status", "unknown")] += 1
    evidence_count[p.get("screening_relevance", "unknown")] += 1
    pdf_path = p.get("pdf_path", "")
    if pdf_path and os.path.exists(pdf_path):
        pdf_path_count += 1

print(f"\n## 字段缺失统计")
for field, count in missing_fields.items():
    print(f"{field} 缺失: {count}")

dup_ids = [k for k, v in duplicate_ids.items() if v > 1]
print(f"\n重复 id: {len(dup_ids)}")
dup_titles = [k for k, v in duplicate_titles.items() if v > 1]
print(f"重复 title: {len(dup_titles)}")

print(f"\n## 状态分布")
for status, count in status_count.items():
    print(f"  {status}: {count}")

print(f"\n## 总体相关性分布（screening_relevance）")
for level, count in evidence_count.items():
    print(f"  {level}: {count}")

print(f"\n## 一致性检查")
print(f"pdf_path 有效: {pdf_path_count}/{len(papers)}")

pdf_in_meta = {os.path.normpath(p.get("pdf_path", "")) for p in papers}
pdf_files_in_raw = {os.path.normpath(str(f)) for f in RAW_DIR.glob("*.pdf")} if RAW_DIR.exists() else set()
meta_has_but_pdf_missing = [p for p in pdf_in_meta if p and p not in pdf_files_in_raw]
pdf_has_but_meta_missing = [f for f in pdf_files_in_raw if f not in pdf_in_meta]
print(f"metadata 有但 PDF 缺失: {len(meta_has_but_pdf_missing)}")
print(f"PDF 有但 metadata 缺失: {len(pdf_has_but_meta_missing)}")
