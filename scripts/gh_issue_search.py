"""GitHub issues search across selected robot-learning repos (unauthenticated).

Usage:
    python scripts/gh_issue_search.py "<query>"            # cross-repo search
    python scripts/gh_issue_search.py "<query>" --repo a,b  # restricted repos
"""
import json
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.github.com/search/issues"
REPOS = [
    "physical-intelligence/openpi",
    "huggingface/lerobot",
    "huggingface/smolvla",
    "NVIDIA/gr00t",
    "openvla/openvla",
    "real-stanford/diffusion_policy",
    "TonyZhao/act",
    "kscalelabs/smolvla",
]


def search(query, repos=None, per_page=20):
    q = query
    if repos:
        q += " repo:" + " repo:".join(repos)
    q += " in:title,body -repo:None"
    params = urllib.parse.urlencode({"q": q, "per_page": per_page})
    req = urllib.request.Request(
        f"{API}?{params}", headers={"Accept": "application/vnd.github+json"}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                time.sleep(20 * (attempt + 1))
                continue
            raise
    return {}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    repos = None
    if "--repo" in args:
        i = args.index("--repo")
        repos = args[i + 1].split(",")
        del args[i : i + 2]
    query = " ".join(args)
    data = search(query, repos)
    total = data.get("total_count", 0)
    print(f"query: {query}\ntotal: {total}")
    for item in data.get("items", [])[:15]:
        repo = item["repository_url"].split("/repos/")[-1]
        title = item.get("title", "")
        state = item.get("state", "")
        comments = item.get("comments", 0)
        url = item.get("html_url", "")
        body = (item.get("body") or "")[:220].replace("\n", " ")
        print(f"- [{repo}] ({state}, {comments}c) {title}")
        print(f"  {url}")
        print(f"  {body}")
        print()
    print("----")


if __name__ == "__main__":
    main()
