"""Add arXiv hyperlinks to all paper_* citations in research files.

Idempotent: re-running never double-links. Maps paper_NNN via papers/metadata.yaml.

Handles these citation forms:
    [paper_001]                          bracketed single
    [paper_001, paper_015, Sec V-A]      bracketed group with section text
    paper_005/013/014                    slash-separated
    paper_005                            bare token
    2403.12945  (2505.15558)             bare arXiv IDs (search logs/keywords)
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "papers" / "metadata.yaml", encoding="utf-8") as f:
    data = yaml.safe_load(f)
URL = {p["id"]: f"https://arxiv.org/abs/{p['source_id']}" for p in data["papers"]}

TARGETS = [
    "docs/survey_robot_data_preprocessing.md",
    "docs/agent_basics_tutorial.md",
    "docs/survey_agentic_data_curation.md",
    "docs/search_keywords.md",
    "synthesis/review_draft.md",
    "synthesis/evidence_matrix.md",
    "synthesis/gap_list.md",
    "synthesis/critical_review.md",
    "synthesis/revised_review.md",
    "opportunities.md",
    "issues.md",
    "logs/search_log.md",
    "logs/decision_log.md",
]


def link_paper(m: re.Match) -> str:
    token = m.group(0)
    ids = re.findall(r"\d{3}", token)
    return ", ".join(f"[paper_{i}]({URL['paper_' + i]})" for i in ids)


def link_group(m: re.Match) -> str:
    items = [x.strip() for x in m.group(1).split(",")]
    out = []
    for x in items:
        if re.fullmatch(r"paper_\d{3}", x):
            out.append(f"[{x}]({URL[x]})")
        else:
            out.append(x)
    return ", ".join(out)


def link_arxiv(m: re.Match) -> str:
    aid = m.group(0)
    return f"[{aid}](https://arxiv.org/abs/{aid})"


def process(text: str) -> str:
    text = re.sub(r"paper_\d{3}(?:/\d{3})+", link_paper, text)
    text = re.sub(r"\[([^\[\]]*paper_\d{3}[^\[\]]*)\](?!\()", link_group, text)
    text = re.sub(r"(?<!\[)paper_\d{3}(?!\]\()", link_paper, text)
    text = re.sub(r"(?<![\w./\[])20\d{2}\.\d{5}(?!\d|\.\w|\]\()", link_arxiv, text)
    text = re.sub(r"(?<![\w./\[])1[5-9]\d{2}\.\d{5}(?!\d|\.\w|\]\()", link_arxiv, text)
    text = re.sub(r"(?<![\w./\[])2[1-9]\d{2}\.\d{5}(?!\d|\.\w|\]\()", link_arxiv, text)
    return text


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for rel in TARGETS:
        path = ROOT / rel
        if not path.exists():
            print(f"missing: {rel}")
            continue
        before = path.read_text(encoding="utf-8")
        after = process(before)
        if after != before:
            path.write_text(after, encoding="utf-8")
            n = len(re.findall(r"paper_\d+|20\d{2}\.\d{5}", after))
            print(f"linked: {rel} ({n} refs)")
        else:
            print(f"unchanged: {rel}")


if __name__ == "__main__":
    main()
