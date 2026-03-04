"""Review script to identify potential duplicate artists in the database.

This script queries the database and identifies artists with similar names
that might need manual review and consolidation.

Usage:
    python src/review_duplicates.py
    python src/review_duplicates.py --threshold 0.85
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from thefuzz import fuzz
from database import SupabaseClient


def find_similar_artists(artists: list, threshold: float = 0.8):
    """Find pairs of artists with similar names."""
    similar_pairs = []

    for i, artist1 in enumerate(artists):
        for artist2 in artists[i + 1:]:
            # Compare normalized names
            score_normalized = fuzz.token_sort_ratio(
                artist1.artist_name,
                artist2.artist_name
            ) / 100.0

            # Compare display names
            score_display = fuzz.token_sort_ratio(
                artist1.artist_display_name,
                artist2.artist_display_name
            ) / 100.0

            best_score = max(score_normalized, score_display)

            if best_score >= threshold:
                similar_pairs.append({
                    "artist1": artist1,
                    "artist2": artist2,
                    "score": best_score,
                    "score_normalized": score_normalized,
                    "score_display": score_display,
                })

    # Sort by similarity score (highest first)
    similar_pairs.sort(key=lambda x: x["score"], reverse=True)
    return similar_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Review potential duplicate artists in the database"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold (0.0-1.0, default: 0.8)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=4,
        help="Minimum name length to consider (default: 4, to avoid 'A', 'B' etc.)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ARTIST DUPLICATE REVIEW TOOL")
    print("=" * 80)
    print(f"Similarity threshold: {args.threshold}")
    print(f"Minimum name length: {args.min_length}")
    print()

    # Connect to database
    print("📡 Connecting to database...")
    db = SupabaseClient()

    # Get all artists
    print("📊 Loading artists...")
    artists = db.get_all_artists()
    print(f"   Found {len(artists)} artists")

    # Filter out very short names (to avoid matching "A" with "A. Smith")
    artists = [a for a in artists if len(a.artist_name) >= args.min_length]

    # Find similar pairs
    print(f"\n🔍 Analyzing for potential duplicates (threshold: {args.threshold})...")
    similar_pairs = find_similar_artists(artists, args.threshold)

    if not similar_pairs:
        print("\n✅ No potential duplicates found!")
        return

    # Group by confidence level
    high_confidence = [p for p in similar_pairs if p["score"] >= 0.9]
    medium_confidence = [p for p in similar_pairs if 0.8 <= p["score"] < 0.9]
    low_confidence = [p for p in similar_pairs if p["score"] < 0.8]

    print(f"\n⚠️  Found {len(similar_pairs)} potential duplicate pairs:\n")

    if high_confidence:
        print("HIGH CONFIDENCE (90%+ - Likely duplicates):")
        print("─" * 80)
        for pair in high_confidence:
            a1, a2 = pair["artist1"], pair["artist2"]
            print(f"\n  Match: {pair['score']:.0%}")
            print(f"    Artist 1: '{a1.artist_display_name}' (ID: {a1.id}, normalized: '{a1.artist_name}')")
            print(f"    Artist 2: '{a2.artist_display_name}' (ID: {a2.id}, normalized: '{a2.artist_name}')")
            print(f"    → Action: Consider merging these artists")
        print()

    if medium_confidence:
        print("MEDIUM CONFIDENCE (80-89% - Review recommended):")
        print("─" * 80)
        for pair in medium_confidence:
            a1, a2 = pair["artist1"], pair["artist2"]
            print(f"\n  Match: {pair['score']:.0%} (normalized: {pair['score_normalized']:.0%}, display: {pair['score_display']:.0%})")
            print(f"    Artist 1: '{a1.artist_display_name}' (ID: {a1.id})")
            print(f"    Artist 2: '{a2.artist_display_name}' (ID: {a2.id})")
            print(f"    → Action: Manual review needed")
        print()

    if low_confidence:
        print(f"LOW CONFIDENCE ({int(args.threshold * 100)}-79% - Possible duplicates):")
        print("─" * 80)
        for pair in low_confidence:
            a1, a2 = pair["artist1"], pair["artist2"]
            print(f"\n  Match: {pair['score']:.0%} (normalized: {pair['score_normalized']:.0%}, display: {pair['score_display']:.0%})")
            print(f"    Artist 1: '{a1.artist_display_name}' (ID: {a1.id})")
            print(f"    Artist 2: '{a2.artist_display_name}' (ID: {a2.id})")
            print(f"    → Action: Review if time permits (may be different artists)")
        print()

    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review each pair above and determine if they're the same artist")
    print("2. If they ARE the same artist:")
    print("   - Choose which artist record to keep (usually the one with more gallery links)")
    print("   - Update gallery_artists table to point to the kept artist")
    print("   - Delete the duplicate artist record")
    print("3. If they are NOT the same artist:")
    print("   - No action needed - they can coexist in the database")
    print()
    print("SQL to check gallery links for an artist:")
    print("  SELECT g.name, ga.* FROM gallery_artists ga")
    print("  JOIN galleries g ON g.id = ga.gallery_id")
    print("  WHERE ga.artist_id = <artist_id>;")
    print()


if __name__ == "__main__":
    main()
