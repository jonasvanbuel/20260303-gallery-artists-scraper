"""Web scraping logic using Crawl4AI and Groq LLM."""

import json
import os
import re
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from urllib.parse import unquote

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
        1. Try /artists/index/ first (comprehensive alphabetical listing)
        2. Fall back to /artists/ (current roster only)
        3. If both fail, use LLM to discover from homepage
        """
        base_url = get_base_url(gallery_url)

        # Try /artists/index/ first (usually has complete historical roster)
        index_url = f"{base_url}/artists/index/"
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
                # Step 1: Try /artists/ first
                result = await crawler.arun(url=artists_url, config=crawl_config)
                if result.success and result.markdown and len(result.markdown) > 100:
                    markdown = result.markdown.raw_markdown or result.markdown.fit_markdown

                    # Step 2: Ask LLM to validate this page and check for better alternatives
                    better_url = await self._llm_validate_artists_page(
                        gallery, base_url, artists_url, markdown
                    )

                    if better_url and better_url != artists_url:
                        # LLM found a better page (like /artists/index/)
                        print(f"   ✓ LLM found better page: {better_url}")
                        return better_url

                    # Quick check: does it look like an artists page?
                    markdown_lower = markdown.lower()
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

    async def _llm_validate_artists_page(
        self, gallery: Gallery, base_url: str, current_url: str, markdown: str
    ) -> Optional[str]:
        """Use LLM to validate if current artists page is complete or suggest better alternative.

        This provides intelligent page selection - the LLM can detect:
        - Links to "See all artists", "Complete index", "A-Z listing"
        - Pagination (Page 1 of 3, etc.)
        - Alphabetical section markers (### A, ### B)
        - Whether current page is comprehensive or partial

        Returns the best URL to use for extraction (current or alternative).
        """
        # Build prompt for LLM analysis - pass FULL markdown, no truncation
        # Groq has ~8K token context (~32K chars), most gallery pages are <30K
        prompt = f"""Analyze this artists page for {gallery.name} and determine if this is the best page to extract ALL artists from, or if there's a better/more comprehensive page linked.

Current page: {current_url}

Look for these specific indicators:
1. Links containing text like: "See all", "View all", "Complete list", "A-Z index", "Index", "All artists"
2. Alphabetical sections with headers like "### A", "### B", "# A", "# B" (indicates comprehensive listing)
3. Text like "Current artists", "Currently represented" (suggests partial listing, historical artists may be elsewhere)
4. Pagination (Page 1 of X) suggesting partial content

Return JSON: {{
    "use_current_page": true/false,
    "better_page_url": "https://..." or null,
    "reason": "brief explanation"
}}

IMPORTANT: If you see any link with text like "See all artists" or "View complete list" or similar, those usually lead to a more comprehensive page with 2-3x more artists. Return that URL.

If the current page has alphabetical A-Z sections (### A through ### Z), it's likely complete - return use_current_page=true.

---
{markdown}"""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "You analyze gallery artists pages to find the most comprehensive listing."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            content = response.choices[0].message.content
            if not content:
                return current_url  # Default to current if LLM fails

            parsed = json.loads(content)
            use_current = parsed.get("use_current_page", True)
            better_url = parsed.get("better_page_url")

            if not use_current and better_url:
                # Resolve relative URL if needed
                resolved = resolve_url(base_url, better_url)
                if resolved and resolved != current_url:
                    return resolved

            return current_url

        except Exception as e:
            # If LLM fails, default to current page
            print(f"   ⚠️  LLM validation failed: {e}, using current page")
            return current_url

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
        """Extract artists using hybrid approach: regex for discovery, LLM for classification.

        This guarantees 100% artist coverage (no LLM misses) while still using
        LLM intelligence for section classification (represented vs projects).
        """
        # Use hybrid extraction (regex + LLM classification)
        return await self._hybrid_extract_artists(gallery, gallery_url, markdown)

    def _split_into_chunks(self, markdown: str, chunk_size: int, overlap: int) -> List[str]:
        """Split markdown into overlapping chunks to ensure no artists are lost at boundaries."""
        if len(markdown) <= chunk_size:
            return [markdown]

        chunks = []
        start = 0

        while start < len(markdown):
            end = min(start + chunk_size, len(markdown))

            # Try to end at a line break or artist link boundary
            if end < len(markdown):
                # Look for line break in the overlap region
                search_start = max(end - overlap, start)
                line_break = markdown.rfind('\n', search_start, end)
                if line_break != -1:
                    end = line_break

            chunks.append(markdown[start:end])
            start = end - overlap if end < len(markdown) else end

        return chunks

    def _build_extraction_prompt(self, markdown: str, pass_num: int, chunk_idx: int = 0, total_chunks: int = 1) -> str:
        """Build extraction prompt with pass-specific strategies and chunk context."""
        extraction_strategies = [
            "Strategy: Carefully scan through the content once, extracting every artist name you see in order.",
            "Strategy: Look specifically for markdown links in the format [Artist Name](url) - these are the primary artist listings.",
            "Strategy: Verify by counting - scan for numbered lists, bullet points, or alphabet sections to ensure no artist was missed."
        ]

        chunk_info = f" (Part {chunk_idx + 1} of {total_chunks})" if total_chunks > 1 else ""

        return f"""Extract ALL artists from this gallery page{chunk_info}. You must find every single artist with 100% completeness.

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

    async def _hybrid_extract_artists(
        self, gallery: Gallery, gallery_url: str, markdown: str
    ) -> List[ArtistExtraction]:
        """Hybrid extraction: regex for 100% artist discovery, LLM for classification.

        This approach guarantees all artists are found (no LLM misses) while
        still using LLM to classify represented vs projects and detect section headers.
        """
        print(f"   Using hybrid extraction ({len(markdown)} chars)...")

        base_url = get_base_url(gallery_url)

        # Phase 1: Deterministic extraction of ALL artist URLs
        all_artists = self._extract_all_artist_urls(markdown, base_url)
        print(f"   🔍 Regex found {len(all_artists)} unique artist URLs")

        if not all_artists:
            return []

        # Phase 2: LLM classification on smaller chunks
        # Classify artists in batches to determine is_represented status
        classified = await self._classify_artists_with_llm(all_artists, markdown, gallery)

        print(f"   ✓ Hybrid extraction complete: {len(classified)} artists")
        return classified

    def _extract_all_artist_urls(self, markdown: str, base_url: str) -> Dict[str, Tuple[str, str]]:
        """Extract all unique artist URLs from markdown using regex.

        Returns: dict mapping artist_slug -> (artist_name, full_url)
        The artist_name is extracted from the markdown link text (e.g., "[Artist Name](url)")
        """
        import re
        artists = {}

        # Truncate markdown at section markers that indicate non-represented artists
        section_markers = [
            'Additional Artists Exhibited',
            'Also Available By',
            'Past Artists',
            'Former Artists',
        ]
        for marker in section_markers:
            if marker in markdown:
                markdown = markdown.split(marker)[0]
                break

        # Pattern 1: Markdown links with embedded images [ ![](image) Name ](url)
        # Handles formats like: * [ ![](image_url) Artist Name ](artist_url)
        # The pattern captures the name from link text, not from the URL slug
        # Note: URLs may end with /) so we make the trailing slash optional
        md_pattern_img = r'\[\s*(!\[[^\]]*\]\([^)]+\))\s*([^\]]+?)\s*\]\((https?://[^)]+/artists/([^/)]+)/?)\)'
        for match in re.finditer(md_pattern_img, markdown):
            name = match.group(2).strip()  # Group 2 is the artist name from link text
            url = match.group(3)  # Group 3 is the URL
            slug = match.group(4).rstrip('/')  # Group 4 is the slug

            # Skip navigation/footer links and false positives
            if len(name) < 2 or slug.lower() in ['artists', 'artists-index', 'index', 'all', '', '#main_content']:
                continue

            # Skip navigation links by name
            nav_names = ['skip to main content', 'skip', 'main content', 'menu', 'home', 'next', 'previous']
            if name.lower() in nav_names:
                continue

            # Skip placeholder/default entries
            if name.startswith('_') or 'default' in name.lower():
                continue

            # Remove any markdown formatting (###, ##, etc.)
            name = re.sub(r'^#+\s*', '', name).strip()

            # Clean up estate/foundation references
            name = self._clean_artist_name(name)

            if name and slug:
                artists[slug] = (name, url)

        # Pattern 2: Simple markdown links [Name](https://.../artists/name/)
        # For galleries without image tags in links
        md_pattern_simple = r'\[([^\]]+)\]\((https?://[^)]+/artists/([^/)]+))\)'
        for match in re.finditer(md_pattern_simple, markdown):
            name = match.group(1).strip()
            url = match.group(2)
            slug = match.group(3).rstrip('/')

            # Skip if already found via pattern 1
            if slug in artists:
                continue

            # Skip navigation/footer links and false positives
            if len(name) < 2 or slug.lower() in ['artists', 'artists-index', 'index', 'all', '', '#main_content']:
                continue

            # Skip navigation links by name
            nav_names = ['skip to main content', 'skip', 'main content', 'menu', 'home', 'next', 'previous']
            if name.lower() in nav_names:
                continue

            # Skip placeholder/default entries
            if name.startswith('_') or 'default' in name.lower():
                continue

            # Clean up name (remove markdown formatting)
            name = re.sub(r'^#+\s*', '', name).strip()

            # Clean up estate/foundation references
            name = self._clean_artist_name(name)

            if name and slug:
                artists[slug] = (name, url)

        # Pattern 3: Plain text URLs that look like artist pages (fallback)
        # Only use this if we didn't already find the artist via markdown link
        url_pattern = r'(https?://[^\s\)\"\'\[\]\,]+/artists/([^\s\)\"\'\[\]\,\/</]+))'
        for match in re.finditer(url_pattern, markdown):
            url = match.group(1).rstrip('/')
            slug = match.group(2).rstrip('/')

            # Skip duplicates and non-artist pages
            if slug not in artists and not any(x in slug.lower() for x in ['index', 'page', 'search', 'all', 'default', '#main_content', '#']):
                # Convert slug to display name as fallback
                display_name = self._slug_to_display_name(slug)
                # Clean up any estate/foundation references from slug-derived names
                display_name = self._clean_artist_name(display_name)
                # Skip navigation links by name (for slug-derived names)
                nav_names = ['skip to main content', 'skip', 'main content', 'menu', 'home', 'next', 'previous']
                if display_name.lower() in nav_names:
                    continue
                # Skip placeholder entries
                if display_name.startswith('_') or 'default' in display_name.lower():
                    continue
                artists[slug] = (display_name, url)

        return artists

    def _slug_to_display_name(self, slug: str) -> str:
        """Convert URL slug to display name. Only used as fallback when markdown doesn't have [Name](url) format."""
        # Decode URL encoding
        decoded = unquote(slug)
        # Replace hyphens with spaces
        name = decoded.replace('-', ' ')
        # Remove trailing spaces that might come from %20
        name = name.strip()
        # Capitalize each word
        return ' '.join(word.capitalize() for word in name.split())

    def _clean_artist_name(self, name: str) -> str:
        """Clean up estate/foundation/work references from artist names.

        Converts names like "The estate of Karel Appel" to "Karel Appel".
        Also handles: "The work of [Name]", ALL CAPS names, etc.
        """
        original = name.strip()
        cleaned = original

        # Pattern 0: Remove leading numbers (e.g., "336 Pawe Althamer" -> "Pawe Althamer")
        cleaned = re.sub(r'^\d+\s+', '', cleaned)

        # Pattern 1: "the estate of [name]" or "estate of [name]"
        cleaned = re.sub(r'^(?:the\s+)?estate\s+of\s+', '', cleaned, flags=re.IGNORECASE)

        # Pattern 2: "the work of [name]" or "work of [name]"
        cleaned = re.sub(r'^(?:the\s+)?work\s+of\s+', '', cleaned, flags=re.IGNORECASE)

        # Pattern 3: "[name] estate" or "[name] foundation"
        cleaned = re.sub(r'\s+(?:estate|foundation)$', '', cleaned, flags=re.IGNORECASE)

        # Pattern 4: "the [name] foundation" (with optional "the")
        cleaned = re.sub(r'^(?:the\s+)?(.+?)\s+foundation$', r'\1', cleaned, flags=re.IGNORECASE)

        # Convert ALL CAPS to Title Case (e.g., "EILEEN AGAR" -> "Eileen Agar")
        if cleaned.isupper():
            cleaned = cleaned.title()

        return cleaned.strip()

    async def _classify_artists_with_llm(
        self, artists: Dict[str, Tuple[str, str]], markdown: str, gallery: Gallery
    ) -> List[ArtistExtraction]:
        """Use LLM to classify artists as represented vs projects.

        Processes artists in small batches to keep context manageable.
        """
        # Take a sample of the markdown with section headers for context
        # Focus on finding section markers like "(Projects)" or "Also Available"
        section_markers = re.findall(r'[\(\*\#]\s*(?:Projects|Also Available|Formerly|Previously)[\)\*\#]?[^\n]*', markdown, re.IGNORECASE)

        # Build result list with default represented=True
        # artists dict is: slug -> (display_name, url)
        result = []
        for slug, (display_name, url) in artists.items():
            result.append(ArtistExtraction(
                artist_display_name=display_name,
                artist_gallery_url=url,
                is_represented=True,  # Default
                normalized_name=display_name.lower().strip(),
            ))

        # For small lists, try to classify with LLM
        # For large lists (>50), assume all are represented (safer default)
        if len(result) <= 50 and section_markers:
            try:
                # Build compact list for LLM
                artist_list = '\n'.join([f"- {a.artist_display_name}" for a in result])

                prompt = f"""Classify these artists for {gallery.name}.

Based on the markdown section markers: {', '.join(section_markers)}

Artists:
{artist_list}

Return JSON: {{"classifications": [{{"name": "Artist Name", "is_represented": true/false}}]}}

Mark artists as is_represented=false ONLY if they appear in a "Projects" or "Also Available" section.
When unsure, default to true (represented)."""

                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": "You classify gallery artists as represented (true) or projects/secondary (false)."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )

                content = response.choices[0].message.content
                if content:
                    parsed = json.loads(content)
                    classifications = {c['name'].lower().strip(): c['is_represented']
                                        for c in parsed.get('classifications', [])}

                    # Update classifications
                    for artist in result:
                        key = artist.artist_display_name.lower().strip()
                        if key in classifications:
                            artist.is_represented = classifications[key]

            except Exception as e:
                print(f"   ⚠️  LLM classification failed: {e}, using defaults")

        return result
