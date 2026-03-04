"""Fuzzy matching logic for artist reconciliation."""

import logging
from typing import List

from thefuzz import fuzz, process

from models import Artist, ArtistExtraction, ArtistMatch
from utils import normalize_artist_name

logger = logging.getLogger(__name__)


def match_artists(
    extracted_artists: List[ArtistExtraction],
    existing_artists: List[Artist],
    threshold_high: float = 0.9,
    threshold_medium: float = 0.7,
) -> List[ArtistMatch]:
    """
    Match extracted artists against existing database.

    Returns list of ArtistMatch objects with match results.
    """
    matches = []

    for extracted in extracted_artists:
        # Normalize the extracted name
        normalized_name = normalize_artist_name(extracted.artist_display_name)
        extracted.normalized_name = normalized_name

        # Find best match
        match = fuzzy_match_artist(
            extracted.artist_display_name,
            normalized_name,
            existing_artists,
            threshold_high,
            threshold_medium,
        )

        # Add gallery URL from extraction
        match.gallery_url = extracted.artist_gallery_url
        match.is_represented = extracted.is_represented

        matches.append(match)

    return matches


def fuzzy_match_artist(
    extracted_name: str,
    normalized_name: str,
    existing_artists: List[Artist],
    threshold_high: float = 0.9,
    threshold_medium: float = 0.7,
) -> ArtistMatch:
    """
    Match a single extracted artist against existing database.

    Tiers:
    1. Exact match on normalized_name (100%)
    2. Fuzzy match on normalized_name using token_sort_ratio
    3. Fuzzy match on artist_display_name
    4. No match - likely new artist
    """

    # Tier 1: Exact normalized name match
    exact_matches = [a for a in existing_artists if a.artist_name == normalized_name]
    if exact_matches:
        return ArtistMatch(
            extracted_name=extracted_name,
            normalized_name=normalized_name,
            matched_artist_id=exact_matches[0].id,
            matched_display_name=exact_matches[0].artist_display_name,
            confidence_score=1.0,
            match_type="exact",
        )

    # Build lookup lists for fuzzy matching - just use display names as strings
    # thefuzz will return the matched string and we look up the artist
    normalized_names_lookup = {a.artist_name: a for a in existing_artists}
    display_names_lookup = {a.artist_display_name: a for a in existing_artists}

    normalized_name_list = list(normalized_names_lookup.keys())
    display_name_list = list(display_names_lookup.keys())

    # Tier 2: Fuzzy match on normalized names
    best_normalized = process.extractOne(
        normalized_name, normalized_name_list, scorer=fuzz.token_sort_ratio
    )

    if best_normalized and best_normalized[1] >= threshold_high * 100:
        matched_name = best_normalized[0]
        artist = normalized_names_lookup[matched_name]
        return ArtistMatch(
            extracted_name=extracted_name,
            normalized_name=normalized_name,
            matched_artist_id=artist.id,
            matched_display_name=artist.artist_display_name,
            confidence_score=best_normalized[1] / 100.0,
            match_type="fuzzy",
        )

    # Tier 3: Fuzzy match on display names
    best_display = process.extractOne(
        extracted_name, display_name_list, scorer=fuzz.token_sort_ratio
    )

    if best_display and best_display[1] >= threshold_high * 100:
        matched_name = best_display[0]
        artist = display_names_lookup[matched_name]
        return ArtistMatch(
            extracted_name=extracted_name,
            normalized_name=normalized_name,
            matched_artist_id=artist.id,
            matched_display_name=artist.artist_display_name,
            confidence_score=best_display[1] / 100.0,
            match_type="fuzzy_display",
        )

    # Tier 4: Medium confidence (potential duplicate - flag for review)
    best_score = max(
        best_normalized[1] if best_normalized else 0,
        best_display[1] if best_display else 0,
    )

    if best_score >= threshold_medium * 100:
        # Determine which match had the higher score and capture that artist
        if best_normalized and best_normalized[1] >= (best_display[1] if best_display else 0):
            matched_name = best_normalized[0]
            matched_artist = normalized_names_lookup[matched_name]
        else:
            matched_name = best_display[0]
            matched_artist = display_names_lookup[matched_name]

        return ArtistMatch(
            extracted_name=extracted_name,
            normalized_name=normalized_name,
            confidence_score=best_score / 100.0,
            match_type="uncertain",
            potential_duplicate_id=matched_artist.id,
            potential_duplicate_name=matched_artist.artist_display_name,
        )

    # No match - new artist
    return ArtistMatch(
        extracted_name=extracted_name,
        normalized_name=normalized_name,
        confidence_score=0.0,
        match_type="new",
    )
