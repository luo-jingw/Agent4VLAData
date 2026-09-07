"""check_evidence_matrix.py — 观测型检查脚本，只输出事实指标，不输出 pass/fail。"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import re
from pathlib import Path
from collections import Counter

EVIDENCE_MATRIX_PATH = Path("synthesis/evidence_matrix.md")
METADATA_PATH = Path("papers/metadata.yaml")

if not EVIDENCE_MATRIX_PATH.exists():
    print("evidence_matrix 文件不存在")
    exit(0)

content = EVIDENCE_MATRIX_PATH.read_text(encoding="utf-8")

research_questions = re.findall(r"^## RQ\d+.*$", content, re.MULTILINE)
print(f"研究问题数量: {len(research_questions)}")
for q in research_questions:
    print(f"  - {q}")

paper_ids_in_matrix = {
    f"paper_{n}"
    for n in re.findall(r"\| \[?paper_(\d+)\]?(?:\([^)]*\))? \|", content)
}
print(f"\nmatrix 中引用的 paper_id 数量: {len(paper_ids_in_matrix)}")

evidence_levels = Counter()
for level in ["direct", "indirect", "weak", "irrelevant"]:
    count = len(re.findall(rf"\| {level} \|", content))
    evidence_levels[level] = count
print(f"\n## 证据等级分布")
for level, count in evidence_levels.items():
    print(f"  {level}: {count}")

print(f"\n## 综述引用检查")
review_paths = sorted(Path("docs").rglob("*.md"))
refs_in_review = set()
for review_path in review_paths:
    if review_path.exists():
        refs_in_review |= set(re.findall(r"paper_\d+", review_path.read_text(encoding="utf-8")))
print(f"综述/定稿文件数: {len(review_paths)}")
print(f"综述中引用的 paper_id 数量: {len(refs_in_review)}")
unreferenced = paper_ids_in_matrix - refs_in_review
print(f"matrix 中有但综述未引用: {len(unreferenced)}")
unreviewed = refs_in_review - paper_ids_in_matrix
print(f"综述引用但 matrix 中无: {len(unreviewed)}")

print(f"\n## 交叉一致性")
if METADATA_PATH.exists():
    import yaml
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    meta_ids = {p.get("id", "") for p in data.get("papers", [])}
    matrix_without_meta = paper_ids_in_matrix - meta_ids
    print(f"matrix 引用但 metadata 缺失: {len(matrix_without_meta)}")
    if matrix_without_meta:
        for pid in matrix_without_meta:
            print(f"  - {pid}")
