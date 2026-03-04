"""Pydantic data models for the scraper."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ArtistExtraction(BaseModel):
    """Raw artist extracted from gallery website."""

    artist_display_name: str
    artist_gallery_url: Optional[str] = None
    is_represented: bool = True
    normalized_name: Optional[str] = None  # computed post-extraction


class GalleryScrapeResult(BaseModel):
    """Complete result for one gallery scrape."""

    gallery_id: uuid.UUID
    gallery_name: str
    gallery_url: str
    artists_page_url: Optional[str] = None
    artists: list[ArtistExtraction]
    error: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class ArtistMatch(BaseModel):
    """Result of fuzzy matching an extracted artist to database."""

    extracted_name: str
    normalized_name: str
    matched_artist_id: Optional[int] = None  # NOTE: artists.id is integer
    matched_display_name: Optional[str] = None
    confidence_score: float  # 0.0 - 1.0
    match_type: str  # 'exact', 'fuzzy', 'fuzzy_display', 'new', 'uncertain'
    gallery_url: Optional[str] = None
    is_represented: bool = True
    # For uncertain matches - store the potential duplicate info for later review
    potential_duplicate_id: Optional[int] = None
    potential_duplicate_name: Optional[str] = None


class Gallery(BaseModel):
    """Gallery record from Supabase."""

    id: uuid.UUID
    name: str
    url: str
    priority: Optional[bool] = None


class Artist(BaseModel):
    """Artist record from Supabase."""

    id: int  # NOTE: integer, not uuid
    artist_name: str
    artist_display_name: str
