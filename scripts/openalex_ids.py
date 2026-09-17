"""OpenAlex title search + print arXiv ID / pdf url (for download planning)."""
import json
import sys
import urllib.parse
import urllib.request

API = "https://api.openalex.org/works"


def title_search(query: str, max_results: int = 5) -> list[dict]:
    params = {
        "filter": f"title.search:{query}",
        "per-page": max_results,
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "research-survey/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    out = []
    for w in data.get("results", []):
        arxiv_id = ""
        pdf_url = ""
        pl = w.get("primary_location") or {}
        if pl.get("pdf_url"):
            pdf_url = pl["pdf_url"]
        for loc in w.get("locations", []) + [pl]:
            url_ = (loc or {}).get("landing_page_url") or ""
            if "arxiv.org/abs" in url_:
                arxiv_id = url_.rsplit("/", 1)[-1].split("v")[0]
                break
        out.append(
            {
                "title": (w.get("title") or "")[:120],
                "year": w.get("publication_year"),
                "arxiv_id": arxiv_id,
                "pdf_url": pdf_url,
            }
        )
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    query = " ".join(sys.argv[1:])
    for i, r in enumerate(title_search(query), 1):
        print(f"[{i}] {r['title']} ({r['year']}) arxiv={r['arxiv_id'] or '-'} pdf={r['pdf_url'] or '-'}")


if __name__ == "__main__":
    main()
