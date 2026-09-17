"""OpenAlex search helper (no auth needed). Reconstructs abstracts from inverted index.

Usage:
    python scripts/openalex_search.py "<query>" [--max N] [--filter cs]
"""
import json
import sys
import urllib.parse
import urllib.request

API = "https://api.openalex.org/works"


def reconstruct(inv: dict) -> str:
    if not inv:
        return ""
    pos = {}
    for word, positions in inv.items():
        for p in positions:
            pos[p] = word
    return " ".join(pos[i] for i in sorted(pos))


def search(query: str, max_results: int = 8, filter_cs: bool = False) -> list[dict]:
    params = {
        "search": query,
        "per-page": max_results,
    }
    if filter_cs:
        params["filter"] = "concepts.id:C41008148"  # Computer Science
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "research-survey/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    out = []
    for w in data.get("results", []):
        abstract = reconstruct(w.get("abstract_inverted_index"))
        out.append(
            {
                "title": (w.get("title") or "")[:150],
                "year": w.get("publication_year"),
                "doi": w.get("doi", ""),
                "authors": [a["author"]["display_name"] for a in w.get("authorships", [])[:4]],
                "cited_by": w.get("cited_by_count", 0),
                "abstract": abstract[:300],
            }
        )
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    max_results = 8
    filter_cs = False
    if "--max" in args:
        i = args.index("--max")
        max_results = int(args[i + 1])
        del args[i : i + 2]
    if "--filter" in args:
        args.remove("--filter")
        filter_cs = True
    query = " ".join(args)
    for i, r in enumerate(search(query, max_results, filter_cs), 1):
        print(f"[{i}] {r['title']} ({r['year']}, cited {r['cited_by']})")
        print(f"    DOI: {r['doi']}")
        if r["authors"]:
            print(f"    {', '.join(r['authors'])}")
        if r["abstract"]:
            print(f"    {r['abstract']}")
        print()


if __name__ == "__main__":
    main()
