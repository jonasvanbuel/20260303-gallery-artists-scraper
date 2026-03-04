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
from utils import ensure_https, get_base_url, resolve_url, GRADUAL_SCROLL_JS


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
            wait_until="domcontentloaded",
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
            wait_until="domcontentloaded",
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
            wait_until="domcontentloaded",
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
        """Use LLM to extract artists from markdown content using multi-pass consensus."""
        print(f"   Using LLM to extract artists ({len(markdown)} chars)...")

        # Use larger context limit to capture all artists
        MAX_CHARS = 20000
        truncated_markdown = markdown[:MAX_CHARS]
        if len(markdown) > MAX_CHARS:
            print(f"   ⚠️  Truncated markdown from {len(markdown)} to {MAX_CHARS} characters")

        base_url = get_base_url(gallery_url)

        # Multi-pass extraction for better completeness
        all_extractions = []
        num_passes = 3

        for pass_num in range(num_passes):
            prompt = self._build_extraction_prompt(truncated_markdown, pass_num)

            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": "You are a precise data extraction assistant. Your task is to extract ALL artist names from gallery websites with 100% completeness."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,  # Slight variation between passes
                )

                content = response.choices[0].message.content
                if content:
                    parsed = json.loads(content)
                    artists_data = parsed.get("artists", [])
                    artists = []
                    for artist_data in artists_data:
                        display_name = artist_data.get("artist_display_name", "").strip()
                        if display_name:
                            artists.append(ArtistExtraction(
                                artist_display_name=display_name,
                                artist_gallery_url=resolve_url(
                                    base_url, artist_data.get("artist_gallery_url")
                                ),
                                is_represented=artist_data.get("is_represented", True),
                                normalized_name=display_name.lower().strip(),
                            ))
                    all_extractions.append(artists)

            except Exception as e:
                print(f"   ⚠️  Pass {pass_num + 1} failed: {e}")
                all_extractions.append([])

        # Merge using union - include all unique artists from all passes
        merged = self._merge_extractions(all_extractions)
        print(f"   ✓ Multi-pass extraction complete")

        return merged

    def _build_extraction_prompt(self, markdown: str, pass_num: int) -> str:
        """Build extraction prompt with pass-specific strategies but full context."""
        extraction_strategies = [
            "Strategy: Carefully scan through the content once, extracting every artist name you see in order.",
            "Strategy: Look specifically for markdown links in the format [Artist Name](url) - these are the primary artist listings.",
            "Strategy: Verify by counting - scan for numbered lists, bullet points, or alphabet sections to ensure no artist was missed."
        ]

        return f"""Extract ALL artists from this gallery page. You must find every single artist with 100% completeness.

{extraction_strategies[pass_num % len(extraction_strategies)]}

Requirements:
- Extract EVERY artist name from the complete content below
- Look for patterns like "[Artist Name](https://.../artists/name)"
- Check for bullet lists, numbered lists, or alphabetically organized sections
- Count the total number of artists you found

For each artist, return in JSON:
- artist_display_name: The exact name as shown
- artist_gallery_url: Full URL to their artist page
- is_represented: true

CRITICAL: Your response must include ALL artists from the content. Return valid JSON format: {{"artists": [{{"artist_display_name": "...", "artist_gallery_url": "...", "is_represented": true}}]}}

---
{markdown}"""

    def _merge_extractions(
        self, extractions: List[List[ArtistExtraction]]
    ) -> List[ArtistExtraction]:
        """Merge multiple extraction passes using union with best-metadata selection."""
        if not extractions:
            return []

        if len(extractions) == 1:
            return extractions[0]

        # Build union of all artists, keeping the best metadata for duplicates
        best_artist: dict[str, ArtistExtraction] = {}

        for extraction in extractions:
            for artist in extraction:
                normalized = artist.normalized_name
                if normalized not in best_artist:
                    best_artist[normalized] = artist
                else:
                    # Prefer the version with a gallery URL
                    existing = best_artist[normalized]
                    if artist.artist_gallery_url and not existing.artist_gallery_url:
                        best_artist[normalized] = artist

        merged = list(best_artist.values())

        # Sort by display name for consistent output
        merged.sort(key=lambda a: a.artist_display_name.lower())

        # Show extraction variance for debugging
        for i, extraction in enumerate(extractions, 1):
            print(f"   Pass {i}: {len(extraction)} artists")
        print(f"   Union result: {len(merged)} unique artists")

        return merged
