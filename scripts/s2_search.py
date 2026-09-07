"""Semantic Scholar search helper (stdlib only, no API key).

Usage:
    python scripts/s2_search.py "<query>" [--max N]
"""
import json
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.semanticscholar.org/graph/v1/paper/search"


def search(query: str, max_results: int = 8) -> list[dict]:
    fields = "title,abstract,year,authors,externalIds,venue"
    params = urllib.parse.urlencode({"query": query, "limit": max_results, "fields": fields})
    req = urllib.request.Request(f"{API}?{params}", headers={"User-Agent": "Mozilla/5.0"})
    data = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(4 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(3)
    if data is None:
        return []
    out = []
    for p in data.get("data", []):
        ext = p.get("externalIds") or {}
        out.append(
            {
                "title": p.get("title", ""),
                "year": p.get("year"),
                "venue": p.get("venue", ""),
                "arxiv_id": ext.get("ArXiv", ""),
                "abstract": (p.get("abstract") or "")[:300],
                "authors": [a["name"] for a in p.get("authors", [])[:4]],
            }
        )
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    query = sys.argv[1]
    max_results = 8
    if "--max" in sys.argv:
        max_results = int(sys.argv[sys.argv.index("--max") + 1])
    for i, r in enumerate(search(query, max_results), 1):
        print(f"[{i}] {r['title']}  ({r['year']}, {r['venue']})")
        if r["arxiv_id"]:
            print(f"    arXiv:{r['arxiv_id']} | {', '.join(r['authors'])}")
        print(f"    {r['abstract']}")
        print()


if __name__ == "__main__":
    main()
