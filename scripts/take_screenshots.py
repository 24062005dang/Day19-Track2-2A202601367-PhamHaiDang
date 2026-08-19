import asyncio
from playwright.async_api import async_playwright
import os
from pathlib import Path

async def main():
    os.makedirs("submission/screenshots", exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        base = Path.cwd() / "notebooks"
        
        # Convert ipynb to HTML using nbconvert
        print("Converting to HTML...")
        os.system(".\\.venv\\Scripts\\jupyter.exe nbconvert --to html notebooks/01_embeddings_index.ipynb")
        os.system(".\\.venv\\Scripts\\jupyter.exe nbconvert --to html notebooks/02_hybrid_search_rrf.ipynb")
        os.system(".\\.venv\\Scripts\\jupyter.exe nbconvert --to html notebooks/03_search_api_benchmark.ipynb")
        os.system(".\\.venv\\Scripts\\jupyter.exe nbconvert --to html notebooks/04_feast_feature_store.ipynb")
        
        for nb in ["01_embeddings_index", "02_hybrid_search_rrf", "03_search_api_benchmark", "04_feast_feature_store"]:
            html_file = base / f"{nb}.html"
            if html_file.exists():
                print(f"Screenshotting {nb}...")
                await page.goto(html_file.as_uri())
                await page.wait_for_timeout(2000)
                await page.screenshot(path=f"submission/screenshots/{nb}.png", full_page=True)
            else:
                print(f"File not found: {html_file}")
                
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
