"""check_extractions.py — 观测型检查脚本，只输出事实指标，不输出 pass/fail。"""
import os
from pathlib import Path
from collections import Counter

EXTRACTIONS_DIR = Path("notes/extractions")
METADATA_PATH = Path("papers/metadata.yaml")

if not EXTRACTIONS_DIR.exists():
    print("extractions 目录不存在")
    exit(0)

extraction_files = sorted(EXTRACTIONS_DIR.glob("*.md"))
print(f"extraction 文件数量: {len(extraction_files)}")

if not METADATA_PATH.exists():
    print("metadata 文件不存在，无法做交叉检查")
    exit(0)

import yaml
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
meta_ids = {p.get("id", "") for p in data.get("papers", [])}

extraction_ids = set()
missing_paper_id = 0
missing_fields = Counter()

for fpath in extraction_files:
    content = fpath.read_text(encoding="utf-8")
    for line in content.splitlines():
        if line.startswith("- paper_id:"):
            eid = line.split(":", 1)[1].strip()
            extraction_ids.add(eid)
            break
    else:
        missing_paper_id += 1

    has_title = "## 基本信息" in content
    has_method = "## 方法" in content
    has_findings = "## 关键发现" in content
    has_evidence = "## 证据等级" in content
    if not has_title:
        missing_fields["基本信息"] += 1
    if not has_method:
        missing_fields["方法"] += 1
    if not has_findings:
        missing_fields["关键发现"] += 1
    if not has_evidence:
        missing_fields["证据等级"] += 1

print(f"\npaper_id 缺失: {missing_paper_id}")
print(f"\n## 字段缺失统计")
for field, count in missing_fields.items():
    print(f"  {field}: {count}")

extraction_without_meta = extraction_ids - meta_ids
meta_without_extraction = meta_ids - extraction_ids
print(f"\n## 交叉一致性")
print(f"extraction 有但 metadata 缺失: {len(extraction_without_meta)}")
if extraction_without_meta:
    for eid in extraction_without_meta:
        print(f"  - {eid}")
print(f"metadata 有但 extraction 缺失: {len(meta_without_extraction)}")
if meta_without_extraction:
    for eid in meta_without_extraction:
        print(f"  - {eid}")
