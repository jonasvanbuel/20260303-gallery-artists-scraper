"""Web scraping logic using Crawl4AI and Groq LLM."""

import json
import os
from datetime import datetime
from typing import List, Optional

import uuid
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from groq import Groq

from config import Config
from models import ArtistExtraction, Gallery
from utils import ensure_https, get_base_url, resolve_url


# JavaScript code for gradual scrolling to load lazy-loaded content
GRADUAL_SCROLL_JS = """
async () => {
    const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
    let scrollHeight = document.body.scrollHeight;
    const viewportHeight = window.innerHeight;
    let currentPosition = 0;
    let iterations = 0;
    const maxIterations = 50;
    const initialHeight = scrollHeight;

    console.log(`[Scroll] Starting. Initial height: ${initialHeight}px, viewport: ${viewportHeight}px`);

    // Scroll down gradually to trigger lazy loading
    while (currentPosition < scrollHeight && iterations < maxIterations) {
        currentPosition += Math.floor(viewportHeight * 0.7);  // Scroll 70% of viewport
        window.scrollTo(0, currentPosition);
        await delay(800);  // Wait for lazy loading

        // Update scroll height in case content was added
        const newScrollHeight = document.body.scrollHeight;
        if (newScrollHeight > scrollHeight) {
            console.log(`[Scroll] Height expanded: ${scrollHeight} -> ${newScrollHeight}`);
            scrollHeight = newScrollHeight;
        }

        iterations++;
    }

    console.log(`[Scroll] Completed. Final height: ${scrollHeight}px, iterations: ${iterations}`);

    // Scroll back to top then down once more to ensure everything loaded
    window.scrollTo(0, 0);
    await delay(500);
    window.scrollTo(0, document.body.scrollHeight);
    await delay(1000);

    return {
        initialHeight: initialHeight,
        finalHeight: scrollHeight,
        iterations: iterations,
        scrollComplete: true
    };
}
"""


class GalleryScraper:
    """Scraper for extracting artists from gallery websites."""

    def __init__(self):
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        self.groq_model = Config.GROQ_MODEL

    async def scrape_gallery(self, gallery: Gallery) -> dict:
        """
        Scrape a single gallery and return results.

        Returns dict with:
        - gallery_id
        - gallery_name
        - gallery_url
        - artists_page_url
        - artists: list of extracted artists
        - error: error message if failed
        """
        # Ensure URL has protocol
        gallery_url = ensure_https(gallery.url)

        result = {
            "gallery_id": str(gallery.id),
            "gallery_name": gallery.name,
            "gallery_url": gallery_url,
            "artists_page_url": None,
            "artists": [],
            "error": None,
        }

        try:
            # Phase 1: Discover artists page URL
            artists_page_url = await self._discover_artists_page(gallery, gallery_url)

            if not artists_page_url:
                result["error"] = "Could not find artists page"
                return result

            result["artists_page_url"] = artists_page_url

            # Phase 2: Extract artist list
            artists = await self._extract_artists(gallery, gallery_url, artists_page_url)
            result["artists"] = artists

        except Exception as e:
            result["error"] = str(e)
            print(f"   ⚠️  Error scraping {gallery.name}: {e}")

        return result

    async def _discover_artists_page(self, gallery: Gallery, gallery_url: str) -> Optional[str]:
        """
        Discover the artists page URL.

        Strategy:
        1. Try /artists first
        2. If that fails, fetch homepage and use LLM
        """
        base_url = get_base_url(gallery_url)

        # Try /artists first
        artists_url = f"{base_url}/artists"

        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=False,
        )

        crawl_config = CrawlerRunConfig(
            wait_until="networkidle",
            page_timeout=Config.PAGE_TIMEOUT,
            word_count_threshold=10,
            js_code=GRADUAL_SCROLL_JS,
        )

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=artists_url, config=crawl_config)

                # Check if page loaded successfully and has content
                if result.success and result.markdown and len(result.markdown) > 100:
                    # Quick check: does it look like an artists page?
                    # We look for multiple artist-like patterns
                    markdown_lower = result.markdown.lower()
                    artist_indicators = [
                        "artist",
                        "represented",
                        "roster",
                        "view profile",
                        "biography",
                    ]
                    indicator_count = sum(
                        1 for indicator in artist_indicators if indicator in markdown_lower
                    )

                    if indicator_count >= 2:
                        return artists_url

        except Exception:
            pass  # Fall back to LLM discovery

        # Fallback: Fetch homepage and use LLM
        return await self._llm_discover_artists_page(gallery, gallery_url)

    async def _llm_discover_artists_page(self, gallery: Gallery, gallery_url: str) -> Optional[str]:
        """Use LLM to discover artists page from homepage."""
        print("   Using LLM to discover artists page...")

        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=False,
        )

        crawl_config = CrawlerRunConfig(
            wait_until="networkidle",
            page_timeout=Config.PAGE_TIMEOUT,
            word_count_threshold=10,
            js_code=GRADUAL_SCROLL_JS,
        )

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=gallery_url, config=crawl_config)

                if not result.success or not result.markdown:
                    print(f"   ⚠️  Failed to fetch homepage for {gallery.name}")
                    return None

                # Use raw_markdown as fit_markdown can be empty or incomplete
                markdown = result.markdown.raw_markdown or result.markdown.fit_markdown

                # Ask LLM to find the artists page URL
                prompt = f"""System: You are a web scraping assistant. Find the artists page URL.
Respond with JSON only, no explanation.

User: Find the URL on {gallery.name}'s website ({gallery_url}) that lists
their represented artists. Look for navigation links or menu items like
"Artists", "Our Artists", "Represented Artists", "Roster", or similar.

Return: {{"artists_page_url": "https://..."}} or {{"artists_page_url": null}}

Only return a URL that appears in the content. Do not invent URLs.

---
{markdown}"""

                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )

                content = response.choices[0].message.content
                if not content:
                    print("   ⚠️  LLM returned empty content")
                    return None
                parsed = json.loads(content)
                artists_page_url = parsed.get("artists_page_url")

                if artists_page_url:
                    # Resolve relative URLs
                    base_url = get_base_url(gallery_url)
                    artists_page_url = resolve_url(base_url, artists_page_url)
                    print(f"   ✓ LLM found artists page: {artists_page_url}")
                    return artists_page_url

                print(f"   ⚠️  LLM could not find artists page for {gallery.name}")
                return None

        except Exception as e:
            print(f"   ⚠️  Error in LLM discovery for {gallery.name}: {e}")
            return None

    async def _extract_artists(
        self, gallery: Gallery, gallery_url: str, artists_page_url: str
    ) -> List[ArtistExtraction]:
        """Extract artists from the artists page."""
        print(f"   Extracting artists from {artists_page_url}")

        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=False,
        )

        crawl_config = CrawlerRunConfig(
            wait_until="networkidle",
            page_timeout=Config.PAGE_TIMEOUT,
            word_count_threshold=10,
            js_code=GRADUAL_SCROLL_JS,
        )

        all_markdown = ""
        pages_crawled = 0
        max_pages = 10
        current_url = artists_page_url

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                while current_url and pages_crawled < max_pages:
                    result = await crawler.arun(url=current_url, config=crawl_config)

                    if not result.success:
                        break

                    # Use raw_markdown as it contains the complete content
                    page_markdown = result.markdown.raw_markdown or result.markdown.fit_markdown
                    all_markdown += page_markdown + "\n\n"
                    pages_crawled += 1

                    # Check for pagination
                    next_url = self._find_next_page(page_markdown, current_url)
                    if next_url and next_url != current_url:
                        current_url = next_url
                    else:
                        break

        except Exception as e:
            print(f"   ⚠️  Error crawling artists page: {e}")
            if not all_markdown:
                return []

        # Save debug output for troubleshooting
        await self._save_debug_output(gallery, all_markdown)

        # Use LLM to extract artists
        return await self._llm_extract_artists(gallery, gallery_url, all_markdown)

    async def _save_debug_output(self, gallery: Gallery, markdown: str):
        """Save markdown content to debug output directory."""
        try:
            today = datetime.now().strftime("%Y%m%d")
            debug_dir = f"output/{today}"
            os.makedirs(debug_dir, exist_ok=True)

            # Create safe filename from gallery name
            safe_name = "".join(c if c.isalnum() else "_" for c in gallery.name.lower())

            # Save markdown
            md_path = os.path.join(debug_dir, f"{safe_name}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Gallery: {gallery.name}\n\n")
                f.write(f"URL: {gallery.url}\n\n")
                f.write(f"Total characters: {len(markdown)}\n\n")
                f.write("---\n\n")
                f.write(markdown)
            print(f"   💾 Saved debug markdown: {md_path}")

        except Exception as e:
            print(f"   ⚠️  Failed to save debug output: {e}")

    def _find_next_page(self, markdown: str, current_url: str) -> Optional[str]:
        """Find next page URL from pagination indicators."""
        import re

        # Common pagination patterns
        patterns = [
            r'href=["\']([^"\']*page[=\/]\d+[^"\']*)["\']',
            r'href=["\']([^"\']*[?&]paged?=\d+[^"\']*)["\']',
            r'href=["\']([^"\']*\/\d+\/?)["\'][^>]*next',
            r'href=["\']([^"\']*)["\'][^>]*next page',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, markdown.lower())
            for match in matches:
                resolved = resolve_url(current_url, match)
                if resolved and resolved != current_url:
                    return resolved

        return None

    async def _llm_extract_artists(
        self, gallery: Gallery, gallery_url: str, markdown: str
    ) -> List[ArtistExtraction]:
        """Use LLM to extract artists from markdown content."""
        print(f"   Using LLM to extract artists ({len(markdown)} chars)...")

        # Use larger context limit to capture all artists
        # Groq models typically have 8K context, ~6000 tokens available for input
        # 1 token ~ 4 characters, so 20000 chars ~ 5000 tokens is safe
        MAX_CHARS = 20000
        truncated_markdown = markdown[:MAX_CHARS]
        if len(markdown) > MAX_CHARS:
            print(f"   ⚠️  Truncated markdown from {len(markdown)} to {MAX_CHARS} characters")

        prompt = f"""System: You are extracting artist data from a gallery website.
Return valid JSON only, no explanation.

User: Extract ALL artists from this gallery page. I need every single artist listed, not just a sample. For each artist:
- artist_display_name: Full name as shown
- artist_gallery_url: Link to their profile page (if available)
- is_represented: true if they appear to be formally represented (e.g., under "Represented" section),
  false if they're only in "Projects" or secondary offerings

Important:
- Extract EVERY artist you can find in the content below
- Look for patterns like "[Artist Name](url)" or headings with artist names
- Ignore artists mentioned only in passing (press quotes, etc.)
- If the page distinguishes "Represented" vs "Projects" artists, mark accordingly
- Resolve relative URLs to absolute URLs

Return format: {{"artists": [{{"artist_display_name": "...", "artist_gallery_url": "...", "is_represented": true/null}}]}}

---
{truncated_markdown}"""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            content = response.choices[0].message.content
            if not content:
                print("   ⚠️  LLM returned empty content")
                return []
            parsed = json.loads(content)
            artists_data = parsed.get("artists", [])

            # Convert to ArtistExtraction objects
            artists = []
            base_url = get_base_url(gallery_url)

            for artist_data in artists_data:
                artist = ArtistExtraction(
                    artist_display_name=artist_data.get("artist_display_name", ""),
                    artist_gallery_url=resolve_url(
                        base_url, artist_data.get("artist_gallery_url")
                    ),
                    is_represented=artist_data.get("is_represented", True),
                )
                artists.append(artist)

            print(f"   ✓ LLM extracted {len(artists)} artists")
            return artists

        except Exception as e:
            print(f"   ⚠️  Error in LLM extraction: {e}")
            return []
