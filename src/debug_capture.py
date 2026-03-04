"""Debug script to capture full HTML and screenshots from gallery pages."""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from utils import GRADUAL_SCROLL_JS


async def capture_page(url: str, output_dir: str, name: str):
    """Capture full HTML and markdown of a page using Crawl4AI."""
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=True,
    )
    
    crawl_config = CrawlerRunConfig(
        wait_until="domcontentloaded",
        page_timeout=60000,
        word_count_threshold=10,
        js_code=GRADUAL_SCROLL_JS,
    )
    
    print(f"Navigating to {url}...")
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)
        
        if not result.success:
            print(f"ERROR: Failed to crawl {url}")
            return None
        
        # Save full HTML
        html_path = os.path.join(output_dir, f"{name}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(result.html)
        print(f"Saved HTML: {html_path} ({len(result.html)} characters)")
        
        # Save markdown (what the LLM sees)
        markdown_path = os.path.join(output_dir, f"{name}.md")
        markdown_content = result.markdown.fit_markdown if result.markdown else ""
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Saved markdown: {markdown_path} ({len(markdown_content)} characters)")
        
        # Also save raw markdown if available
        if hasattr(result.markdown, 'raw_markdown') and result.markdown.raw_markdown:
            raw_md_path = os.path.join(output_dir, f"{name}_raw.md")
            with open(raw_md_path, "w", encoding="utf-8") as f:
                f.write(result.markdown.raw_markdown)
            print(f"Saved raw markdown: {raw_md_path} ({len(result.markdown.raw_markdown)} characters)")
        
        print(f"\nPage loaded successfully!")
        print(f"  URL: {result.url}")
        print(f"  Status: {result.status_code}")
        
        return result


async def main():
    # Create output directory
    today = datetime.now().strftime("%Y%m%d")
    output_dir = f"/Users/jonas/Documents/Development/experiments/20260303-gallery-artists-scraper/output/{today}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Capture Rodolphe Janssen
    url = "https://rodolphejanssen.com/artists"
    print(f"\n{'='*60}")
    print(f"Capturing: {url}")
    print(f"{'='*60}")
    result = await capture_page(url, output_dir, "rodolphejanssen_debug")
    
    if result:
        # Show a sample of the markdown
        md_content = result.markdown.fit_markdown if result.markdown else ""
        print(f"\n--- First 3000 characters of markdown ---")
        print(md_content[:3000])
        print(f"\n--- ... ---")
        print(f"\n--- Last 3000 characters of markdown ---")
        print(md_content[-3000:] if len(md_content) > 3000 else md_content)


if __name__ == "__main__":
    asyncio.run(main())
