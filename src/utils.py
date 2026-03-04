"""Utility functions for URL resolution, name normalization, and browser automation."""

import re
import unicodedata
from urllib.parse import urljoin, urlparse


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


def normalize_artist_name(display_name: str) -> str:
    """
    Normalize artist name for matching:
    - Lowercase
    - Remove accents (é → e)
    - Remove punctuation except hyphens
    - Remove extra whitespace
    - Handle common suffixes (Jr., Sr., III)
    """
    # Lowercase
    name = display_name.lower().strip()

    # Remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Remove parentheses and their contents (common for birth years)
    name = re.sub(r"\s*\([^)]*\)", "", name)

    # Remove punctuation except hyphens and spaces
    name = re.sub(r"[^\w\s\-]", "", name)

    # Normalize whitespace
    name = re.sub(r"\s+", " ", name)

    # Handle suffixes
    name = re.sub(r"\s+(jr|sr|iii|ii|iv)\b", r" \1", name)

    return name.strip()


def resolve_url(base_url: str, url: str | None) -> str | None:
    """Resolve relative URLs to absolute URLs."""
    if not url:
        return None
    if url.startswith("http"):
        return url
    return urljoin(base_url, url)


def ensure_https(url: str) -> str:
    """Ensure URL has https:// prefix if no protocol specified."""
    if not url.startswith("http://") and not url.startswith("https://"):
        return f"https://{url}"
    return url


def get_base_url(url: str) -> str:
    """Extract base URL (scheme + netloc) from a full URL."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"
