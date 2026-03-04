"""Supabase database operations."""

from typing import List, Optional, Dict
from datetime import datetime

import uuid
from supabase import create_client, Client

from config import Config
from models import Artist, Gallery


class SupabaseClient:
    """Client for Supabase database operations."""

    def __init__(self):
        self.client: Client = create_client(
            Config.SUPABASE_URL, Config.SUPABASE_SECRET_KEY
        )

    def get_priority_galleries(self) -> List[Gallery]:
        """Get galleries marked as priority first."""
        print("   Fetching priority galleries from database...")
        response = self.client.table("galleries").select("*").eq("priority", True).execute()
        galleries = [Gallery(**g) for g in response.data]
        print(f"   ✓ Found {len(galleries)} priority galleries")
        return galleries

    def get_all_galleries(self) -> List[Gallery]:
        """Get all galleries."""
        print("   Fetching all galleries from database...")
        response = self.client.table("galleries").select("*").execute()
        galleries = [Gallery(**g) for g in response.data]
        print(f"   ✓ Found {len(galleries)} total galleries")
        return galleries

    def get_all_artists(self) -> List[Artist]:
        """Load all artists for fuzzy matching."""
        print("   Fetching all artists from database...")
        # Consider pagination for large datasets (>1000 artists)
        response = (
            self.client.table("artists").select("id, artist_name, artist_display_name").execute()
        )
        artists = [Artist(**a) for a in response.data]
        print(f"   ✓ Found {len(artists)} artists for matching")
        return artists

    def get_gallery_artists(self, gallery_id: uuid.UUID) -> List[Dict]:
        """Get current artist relationships for a gallery."""
        response = (
            self.client.table("gallery_artists")
            .select("artist_id, is_represented, artists!inner(artist_name, artist_display_name)")
            .eq("gallery_id", str(gallery_id))
            .eq("is_represented", True)
            .execute()
        )
        return response.data

    def create_artist(self, artist_name: str, artist_display_name: str) -> int:
        """Create new artist and return ID (integer).

        If artist already exists (duplicate key), fetch and return existing ID.
        """
        try:
            response = (
                self.client.table("artists")
                .insert(
                    {
                        "artist_name": artist_name,
                        "artist_display_name": artist_display_name,
                    }
                )
                .execute()
            )
            new_id = response.data[0]["id"]
            return new_id
        except Exception as e:
            # Check if it's a duplicate key error
            error_msg = str(e)
            if "duplicate key" in error_msg or "23505" in error_msg:
                # Fetch the existing artist by artist_name
                response = (
                    self.client.table("artists")
                    .select("id")
                    .eq("artist_name", artist_name)
                    .execute()
                )
                if response.data:
                    existing_id = response.data[0]["id"]
                    return existing_id
                else:
                    raise
            else:
                # Re-raise other errors
                raise

    def create_gallery_artist_link(
        self,
        gallery_id: uuid.UUID,
        artist_id: int,
        artist_gallery_url: Optional[str],
        is_represented: bool = True,
        confidence: float = 1.0,
    ) -> None:
        """Upsert gallery-artist relationship with verification timestamp."""
        self.client.table("gallery_artists").upsert(
            {
                "gallery_id": str(gallery_id),
                "artist_id": artist_id,
                "artist_gallery_url": artist_gallery_url,
                "is_represented": is_represented,
                "last_scraped_at": "now()",
                "last_verified_at": "now()",
                "scrape_confidence": confidence,
                "removed_at": None,  # Clear removed_at if re-adding
                "removal_reason": None,
            },
            on_conflict="gallery_id,artist_id",
        ).execute()

    def mark_artist_removed(
        self,
        gallery_id: uuid.UUID,
        artist_id: int,
        reason: str = "rescrape_not_found",
    ) -> None:
        """Soft delete - mark artist as no longer represented by gallery."""
        self.client.table("gallery_artists").update(
            {
                "is_represented": False,
                "removed_at": "now()",
                "removal_reason": reason,
            }
        ).eq("gallery_id", str(gallery_id)).eq("artist_id", artist_id).execute()

    def update_gallery_indexed_at(self, gallery_id: uuid.UUID) -> None:
        """Update the gallery's last_indexed_at timestamp."""
        self.client.table("galleries").update(
            {"last_indexed_at": "now()"}
        ).eq("id", str(gallery_id)).execute()
