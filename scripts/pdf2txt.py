"""Extract PDF text to <pdf>.txt with pymupdf (dependency declared here)."""
import sys
from pathlib import Path

import fitz  # pymupdf


def pdf_to_txt(pdf_path: Path) -> Path:
    txt_path = pdf_path.with_suffix(".txt")
    doc = fitz.open(pdf_path)
    parts = []
    for i, page in enumerate(doc):
        parts.append(f"\n===== PAGE {i + 1} =====\n")
        parts.append(page.get_text("text"))
    txt_path.write_text("".join(parts), encoding="utf-8")
    return txt_path


def main() -> None:
    raw = Path(__file__).resolve().parent.parent / "papers" / "raw"
    for pdf in sorted(raw.glob("*.pdf")):
        txt = pdf.with_suffix(".txt")
        if not txt.exists():
            pdf_to_txt(pdf)
        print(f"{pdf.name} -> {txt.name} ({txt.stat().st_size} B)")


if __name__ == "__main__":
    main()
