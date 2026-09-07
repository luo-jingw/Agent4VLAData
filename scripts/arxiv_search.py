"""arXiv API search helper (stdlib only).

Usage:
    python scripts/arxiv_search.py "<query>" [--max N]
"""
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API = "http://export.arxiv.org/api/query"


def search(query: str, max_results: int = 5) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
    )
    with urllib.request.urlopen(f"{API}?{params}", timeout=60) as resp:
        root = ET.fromstring(resp.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in root.findall("a:entry", ns):
        title = " ".join(entry.findtext("a:title", "", ns).split())
        summary = " ".join(entry.findtext("a:summary", "", ns).split())
        link = entry.find("a:id", ns).text
        arxiv_id = link.rsplit("/abs/", 1)[-1]
        published = entry.findtext("a:published", "", ns)[:10]
        authors = [e.findtext("a:name", "", ns) for e in entry.findall("a:author", ns)]
        out.append(
            {
                "title": title,
                "authors": authors,
                "year": published,
                "arxiv_id": arxiv_id,
                "url": link,
                "abstract": summary,
            }
        )
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    query = sys.argv[1]
    max_results = 5
    if "--max" in sys.argv:
        max_results = int(sys.argv[sys.argv.index("--max") + 1])
    for i, r in enumerate(search(query, max_results), 1):
        print(f"[{i}] {r['title']}")
        print(f"    {r['year']} | {', '.join(r['authors'][:4])}")
        print(f"    {r['url']}")
        print()


if __name__ == "__main__":
    main()
