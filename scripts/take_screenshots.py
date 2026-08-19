"""Render each executed notebook to HTML, then full-page screenshot it.

Produces submission/screenshots/<nb>.png — one per notebook, which is what the
rubric asks for ("at least one screenshot per notebook").

Run AFTER the notebooks have been executed (`make notebooks`), otherwise you
screenshot empty `In [ ]:` cells and the grader sees no evidence.

    .venv/Scripts/python scripts/take_screenshots.py     # Windows
    .venv/bin/python scripts/take_screenshots.py         # macOS / Linux
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "submission" / "screenshots"

NOTEBOOKS = [
    "01_embeddings_index",
    "02_hybrid_search_rrf",
    "03_search_api_benchmark",
    "04_feast_feature_store",
    "05_filtered_search",
    "06_agent_retrieval",
    "07_semantic_cache",
    "08_feature_engineering",
]


def executed_cell_count(ipynb: Path) -> tuple[int, int]:
    """Return (executed, total) code-cell counts, so we can warn on empties."""
    nb = json.loads(ipynb.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    run = [c for c in code if c.get("execution_count") is not None]
    return len(run), len(code)


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        # Wide viewport so the latency/precision tables don't wrap mid-column.
        page = await browser.new_page(viewport={"width": 1400, "height": 1000})

        for nb in NOTEBOOKS:
            ipynb = NB_DIR / f"{nb}.ipynb"
            if not ipynb.exists():
                print(f"  SKIP {nb} — no .ipynb (run `make notebooks` first)")
                missing.append(nb)
                continue

            run, total = executed_cell_count(ipynb)
            if run < total:
                print(f"  WARN {nb} — only {run}/{total} code cells executed")

            # `python -m jupyter` keeps us inside this venv regardless of PATH.
            subprocess.run(
                [sys.executable, "-m", "jupyter", "nbconvert", "--to", "html",
                 str(ipynb)],
                cwd=str(ROOT), check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

            html = NB_DIR / f"{nb}.html"
            await page.goto(html.as_uri())
            await page.wait_for_timeout(1500)
            await page.screenshot(path=str(OUT_DIR / f"{nb}.png"), full_page=True)
            print(f"  OK   {nb}.png  ({run}/{total} cells executed)")

        await browser.close()

    if missing:
        print(f"\nMissing notebooks: {', '.join(missing)}")
        return 1
    print(f"\nWrote {len(NOTEBOOKS)} screenshots to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
