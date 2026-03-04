"""Entry point for the gallery artists scraper."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from database import SupabaseClient
from matcher import match_artists
from models import Artist, Gallery
from scraper import GalleryScraper


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(text)
    print(f"{'=' * 60}")


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n{'─' * 60}")
    print(text)
    print(f"{'─' * 60}")


def print_artist_preview(matches: list, gallery_name: str):
    """Print a preview of artists to be saved."""
    print_subheader(f"ARTISTS TO SAVE FOR: {gallery_name}")
    print(f"{'Match Type':<15} {'Artist Name':<25} {'Display Name':<30} {'Gallery URL':<50}")
    print(f"{'─' * 15} {'─' * 25} {'─' * 30} {'─' * 50}")

    for match in matches:
        match_type = match.match_type.upper()
        if match.match_type == "uncertain":
            match_type = "NEW (review)"
        artist_name = match.normalized_name[:24]
        display_name = match.extracted_name[:29]
        url = (match.gallery_url or "N/A")[:49]

        print(f"{match_type:<15} {artist_name:<25} {display_name:<30} {url:<50}")

        # For uncertain matches, show the potential duplicate inline
        if match.match_type == "uncertain" and match.potential_duplicate_name:
            note = f"  ↳ Possible duplicate of: '{match.potential_duplicate_name}' ({match.confidence_score:.0%} match)"
            print(f"{'':<15} {note}")

    # Print stats
    exact = sum(1 for m in matches if m.match_type == "exact")
    fuzzy = sum(1 for m in matches if m.match_type in ["fuzzy", "fuzzy_display"])
    uncertain = sum(1 for m in matches if m.match_type == "uncertain")
    new_artists = sum(1 for m in matches if m.match_type == "new")
    total_new = new_artists + uncertain

    print()
    print(f"  📊 Summary:")
    print(f"     • Exact matches (link existing): {exact}")
    print(f"     • Fuzzy matches (link existing): {fuzzy}")
    print(f"     • New artists to create: {new_artists}")
    print(f"     • Uncertain (create new, flag for review): {uncertain}")
    print(f"     • Total new artists: {total_new}")


def print_database_summary(matches: list):
    """Print what data will be saved to the database."""
    print_subheader("DATABASE OPERATIONS PREVIEW")

    new_count = sum(1 for m in matches if m.match_type == "new")
    uncertain_count = sum(1 for m in matches if m.match_type == "uncertain")
    total_new = new_count + uncertain_count
    link_count = sum(1 for m in matches if m.match_type in ["exact", "fuzzy", "fuzzy_display"])

    print("Tables affected:")
    print(f"  1. artists table:")
    print(f"     • {new_count} confirmed new artists will be created")
    if uncertain_count > 0:
        print(f"     • {uncertain_count} uncertain artists will be created (flagged for review)")
    print(f"     • Total new artists: {total_new}")
    print(f"     • Fields: artist_name (normalized), artist_display_name (as shown)")
    print()
    print(f"  2. gallery_artists table (junction):")
    print(f"     • {link_count} existing artists will be linked")
    print(f"     • {total_new} new artists will be linked after creation")
    print(f"     • Fields: gallery_id, artist_id, artist_gallery_url, is_represented, scrape_confidence")
    if uncertain_count > 0:
        print()
        print("  ⚠️  Note: Uncertain matches may be duplicates. A review script will identify")
        print("     potential duplicates for manual consolidation in the future.")


async def process_gallery(
    gallery: Gallery,
    scraper: GalleryScraper,
    db: SupabaseClient,
    existing_artists: list[Artist],
    dry_run: bool = False,
) -> dict:
    """Process a single gallery: scrape, match, and store."""
    print_header(f"Processing: {gallery.name}")
    print(f"Gallery URL: {gallery.url}")
    print(f"Gallery ID: {gallery.id}")

    # Get current gallery artists before scraping (for comparison)
    print("\n📋 Checking existing gallery artists...")
    current_links = db.get_gallery_artists(gallery.id)
    current_artist_ids = {link["artist_id"] for link in current_links}
    print(f"   Found {len(current_links)} currently represented artists")

    # Scrape gallery
    print("\n🌐 Scraping gallery website...")
    result = await scraper.scrape_gallery(gallery)

    if result.get("error"):
        print(f"❌ Failed to scrape {gallery.name}: {result['error']}")
        return result

    # Extract artists data
    artists_data = result.get("artists", [])
    if not artists_data:
        print(f"⚠️  No artists found for {gallery.name}")
        return result

    print(f"✅ Extracted {len(artists_data)} artists from {gallery.name}")

    # Match artists
    print("\n🔍 Matching artists against database...")
    matches = match_artists(
        artists_data,
        existing_artists,
        threshold_high=Config.FUZZY_MATCH_HIGH_THRESHOLD,
        threshold_medium=Config.FUZZY_MATCH_MEDIUM_THRESHOLD,
    )

    # Print preview of what will be saved
    print_artist_preview(matches, gallery.name)
    print_database_summary(matches)

    if dry_run:
        print("\n🏃 [DRY RUN] Not writing to database")
        return result

    # Store results automatically (no prompt)
    print("\n💾 Saving to database...")
    artists_created = 0
    artists_matched = 0
    found_artist_ids = set()  # Track which artists were found in this scrape

    for match in matches:
        artist_id = None

        if match.match_type == "new":
            # Create new artist
            new_id = db.create_artist(match.normalized_name, match.extracted_name)
            artists_created += 1
            artist_id = new_id

            # Add to existing_artists cache so subsequent galleries can match against it
            from models import Artist
            new_artist = Artist(
                id=new_id,
                artist_name=match.normalized_name,
                artist_display_name=match.extracted_name
            )
            existing_artists.append(new_artist)

            # Link to gallery
            db.create_gallery_artist_link(
                gallery_id=gallery.id,
                artist_id=new_id,
                artist_gallery_url=match.gallery_url,
                is_represented=match.is_represented,
                confidence=match.confidence_score,
            )
            print(f"  ✓ Created & linked: {match.extracted_name}")

        elif match.match_type in ["exact", "fuzzy", "fuzzy_display"]:
            # Link existing artist
            if match.matched_artist_id is None:
                print(f"  ⚠️  Error: Matched artist ID is None for {match.extracted_name}")
                continue
            artist_id = match.matched_artist_id
            db.create_gallery_artist_link(
                gallery_id=gallery.id,
                artist_id=match.matched_artist_id,
                artist_gallery_url=match.gallery_url,
                is_represented=match.is_represented,
                confidence=match.confidence_score,
            )
            artists_matched += 1
            print(f"  ✓ Linked: {match.extracted_name}")

        else:
            # Uncertain - create as NEW artist but log the potential duplicate for review
            new_id = db.create_artist(match.normalized_name, match.extracted_name)
            artists_created += 1
            artist_id = new_id

            # Add to existing_artists cache
            from models import Artist
            new_artist = Artist(
                id=new_id,
                artist_name=match.normalized_name,
                artist_display_name=match.extracted_name
            )
            existing_artists.append(new_artist)

            # Link to gallery
            db.create_gallery_artist_link(
                gallery_id=gallery.id,
                artist_id=new_id,
                artist_gallery_url=match.gallery_url,
                is_represented=match.is_represented,
                confidence=match.confidence_score,
            )

            # Log the potential duplicate for future review
            dup_name = match.potential_duplicate_name or "Unknown"
            dup_id = match.potential_duplicate_id or "Unknown"
            print(f"  ⚠️  Created (possible duplicate): {match.extracted_name}")
            print(f"      ↳ Similar to: '{dup_name}' (ID: {dup_id}, confidence: {match.confidence_score:.2f})")

        # Track that we found this artist
        if artist_id:
            found_artist_ids.add(artist_id)

    # Check for artists that were previously linked but not found in current scrape
    removed_count = 0
    missing_artists = current_artist_ids - found_artist_ids

    if missing_artists:
        print(f"\n🗑️  Artists no longer represented ({len(missing_artists)} found):")
        for artist_id in missing_artists:
            # Find artist name from current_links
            link_info = next((l for l in current_links if l["artist_id"] == artist_id), None)
            if link_info:
                artist_name = link_info.get("artists", {}).get("artist_display_name", f"ID:{artist_id}")
                print(f"   • {artist_name} - marking as removed")
            else:
                print(f"   • Artist ID {artist_id} - marking as removed")

            db.mark_artist_removed(gallery.id, artist_id, reason="rescrape_not_found")
            removed_count += 1

    # Update gallery's last_indexed_at timestamp
    db.update_gallery_indexed_at(gallery.id)

    print(f"\n✅ Stored: {artists_matched} matched, {artists_created} new, {removed_count} removed")

    return result


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape gallery websites and extract artist relationships"
    )
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="Only scrape priority galleries",
    )
    parser.add_argument(
        "--gallery-id",
        type=str,
        help="Scrape a single gallery by UUID",
    )
    parser.add_argument(
        "--gallery-url",
        type=str,
        help="Scrape a single gallery by URL (e.g., 'rodolphejanssen.com' or 'https://example.com')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract but don't write to database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of galleries to process (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/results.jsonl",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)

    Config.ensure_output_dir()

    print_header("GALLERY ARTISTS SCRAPER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize database client
    print("\n📡 Connecting to database...")
    db = SupabaseClient()
    scraper = GalleryScraper()

    # Load existing artists for matching
    print("📊 Loading existing artists...")
    existing_artists = db.get_all_artists()
    print(f"   Found {len(existing_artists)} artists in database")

    # Get galleries to process
    if args.gallery_id:
        # Single gallery by ID
        all_galleries = db.get_all_galleries()
        galleries = [g for g in all_galleries if str(g.id) == args.gallery_id]
        if not galleries:
            print(f"❌ Gallery not found: {args.gallery_id}")
            sys.exit(1)
    elif args.gallery_url:
        # Single gallery by URL
        url = args.gallery_url.lower().strip()
        url = url.replace("https://", "").replace("http://", "")
        url = url.rstrip("/")

        all_galleries = db.get_all_galleries()
        galleries = [g for g in all_galleries if g.url.lower().strip().rstrip("/") == url]

        if not galleries:
            # Try partial match
            galleries = [g for g in all_galleries if url in g.url.lower()]

        if not galleries:
            print(f"❌ Gallery not found with URL: {args.gallery_url}")
            sys.exit(1)

        print(f"✅ Found gallery: {galleries[0].name} ({galleries[0].url})")
    elif args.priority_only:
        galleries = db.get_priority_galleries()
    else:
        # Priority first, then all others
        priority = db.get_priority_galleries()
        all_galleries = db.get_all_galleries()
        # Combine without duplicates
        seen_ids = {g.id for g in priority}
        galleries = priority + [g for g in all_galleries if g.id not in seen_ids]

    if args.limit:
        galleries = galleries[: args.limit]

    total = len(galleries)
    print(f"\n📋 Processing {total} galleries")

    # Process galleries
    results = []
    for i, gallery in enumerate(galleries, 1):
        print(f"\n\n[{i}/{total}] {gallery.name}")

        try:
            result = await process_gallery(
                gallery=gallery,
                scraper=scraper,
                db=db,
                existing_artists=existing_artists,
                dry_run=args.dry_run,
            )
            results.append(result)

            # Save to JSONL immediately (for resume capability)
            with open(args.output, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")

            # Delay between galleries (unless last one)
            if i < total:
                await asyncio.sleep(Config.DELAY_BETWEEN_GALLERIES)

        except Exception as e:
            print(f"❌ Error processing {gallery.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print_header("SUMMARY")
    print(f"Total galleries processed: {len(results)}")
    successful = sum(1 for r in results if not r.get("error"))
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    total_artists = sum(len(r.get("artists", [])) for r in results)
    print(f"Total artists extracted: {total_artists}")

    print(f"\n📝 Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
