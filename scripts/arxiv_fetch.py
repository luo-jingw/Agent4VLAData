"""arXiv fetch by ID (stdlib only).

Usage:
    python scripts/arxiv_fetch.py <arxiv_id> [--pdf] [--meta]
Downloads the paper PDF to papers/raw/<arxiv_id>.pdf (with --pdf),
prints metadata (with --meta). Appends to logs/download_log.md.
"""
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch_meta(arxiv_id: str) -> dict:
    url = f"{API}?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        root = ET.fromstring(resp.read())
    entry = root.find("a:entry", NS)
    if entry is None:
        raise ValueError(f"no entry for {arxiv_id}")
    title = " ".join(entry.findtext("a:title", "", NS).split())
    summary = " ".join(entry.findtext("a:summary", "", NS).split())
    published = entry.findtext("a:published", "", NS)[:10]
    authors = [e.findtext("a:name", "", NS) for e in entry.findall("a:author", NS)]
    link = entry.find("a:id", NS).text
    pdf_url = link.replace("/abs/", "/pdf/")
    return {
        "title": title,
        "authors": authors,
        "year": published[:4],
        "published": published,
        "arxiv_id": arxiv_id,
        "url": link,
        "pdf_url": pdf_url,
        "abstract": summary,
    }


def download_pdf(arxiv_id: str, pdf_url: str) -> Path:
    dest = ROOT / "papers" / "raw" / f"{arxiv_id}.pdf"
    if dest.exists():
        print(f"exists: {dest.name}")
        return dest
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"saved: {dest.name}")
    return dest


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    arxiv_id = sys.argv[1]
    meta = fetch_meta(arxiv_id)
    print(f"{meta['title']}")
    print(f"{meta['published']} | {', '.join(meta['authors'][:5])}")
    print(f"{meta['url']}")
    print(f"pdf: {meta['pdf_url']}")

    if "--pdf" in sys.argv:
        dest = download_pdf(arxiv_id, meta["pdf_url"])
        log = ROOT / "logs" / "download_log.md"
        log.parent.mkdir(exist_ok=True)
        line = (
            f"| {time.strftime('%Y-%m-%d %H:%M')} | {arxiv_id} | "
            f"{meta['title'][:80]} | {dest.stat().st_size} B |\n"
        )
        if not log.exists():
            log.write_text("# Download Log\n\n| Time | arXiv ID | Title | Size |\n|---|---|---|---|\n", encoding="utf-8")
        with log.open("a", encoding="utf-8") as f:
            f.write(line)


if __name__ == "__main__":
    main()
