"""Download a list of arXiv PDFs via curl (API-independent).

Usage:
    python scripts/arxiv_curl_dl.py <id1> <id2> ...
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "papers" / "raw"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for aid in sys.argv[1:]:
        dest = RAW / f"{aid}.pdf"
        if dest.exists():
            print(f"exists: {aid}")
            continue
        r = subprocess.run(
            ["curl.exe", "-s", "-L", "-m", "120", "-A", UA,
             f"https://arxiv.org/pdf/{aid}", "-o", str(dest)],
            capture_output=True, timeout=150,
        )
        if dest.exists() and dest.stat().st_size > 10000:
            print(f"saved: {aid} ({dest.stat().st_size} B)")
        else:
            print(f"FAILED: {aid}")
            if dest.exists():
                dest.unlink()


if __name__ == "__main__":
    main()
