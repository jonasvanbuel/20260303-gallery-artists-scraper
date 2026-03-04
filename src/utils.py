"""Utility functions for URL resolution and name normalization."""

import re
import unicodedata
from urllib.parse import urljoin, urlparse


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
