import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

p = Path("papers/metadata.yaml")
text = p.read_text(encoding="utf-8")
levels = {
    "paper_011": "direct",
    "paper_012": "indirect",
    "paper_013": "direct",
    "paper_014": "direct",
    "paper_015": "direct",
    "paper_016": "direct",
    "paper_017": "direct",
    "paper_018": "direct",
    "paper_019": "direct",
    "paper_020": "direct",
    "paper_021": "direct",
    "paper_022": "direct",
    "paper_023": "direct",
    "paper_024": "direct",
    "paper_025": "direct",
    "paper_026": "direct",
    "paper_027": "direct",
    "paper_028": "direct",
    "paper_029": "direct",
    "paper_030": "direct",
    "paper_031": "direct",
    "paper_032": "weak",
}
changed = 0
for pid, lv in levels.items():
    idx = text.find(f"id: {pid}")
    if idx == -1:
        idx = text.find(f'id: "{pid}"')
    assert idx != -1, pid
    end = text.find("- id:", idx + 10)
    if end == -1:
        end = len(text)
    block = text[idx:end]
    if 'evidence_level: pending' in block:
        text = text[:idx] + block.replace("evidence_level: pending", f'evidence_level: {lv}') + text[end:]
        changed += 1
p.write_text(text, encoding="utf-8")
print(f"changed: {changed}")
