# Gallery Artists Scraper

AI-powered scraper that extracts artist relationships from gallery websites and stores them in Supabase.

## Features

- **Smart Discovery**: Tries `/artists` first, then uses LLM to find the correct page
- **Lazy Loading Support**: Scrolls pages to trigger JavaScript lazy loading
- **Fuzzy Matching**: Matches extracted artists against existing database with confidence scores
- **Many-to-Many Relationships**: Links galleries to artists with metadata
- **Priority Queue**: Process priority galleries first

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   crawl4ai-setup  # Installs Playwright browsers
   ```

## Usage

### Scrape priority galleries only
```bash
python3 src/main.py --priority-only
```

### Scrape a specific gallery by ID
```bash
python3 src/main.py --gallery-id "uuid-here"
```

### Scrape a specific gallery by URL
```bash
python3 src/main.py --gallery-url "rodolphejanssen.com"
```

### Scrape all galleries (priority first, then others)
```bash
python3 src/main.py
```

### Dry run (extract but don't save)
```bash
python src/main.py --priority-only --dry-run
```

### Limit for testing
```bash
python src/main.py --priority-only --limit 3
```

## How It Works

1. **Phase 1 - Discovery**:
   - Tries `gallery-url.com/artists` first
   - If that fails, fetches homepage and asks LLM to find artists page

2. **Phase 2 - Extraction**:
   - Fetches artists page (handles pagination)
   - Scrolls to trigger lazy loading
   - LLM extracts structured artist data

3. **Phase 3 - Matching**:
   - Fuzzy matches against existing artists
   - Creates new artists for unknown names
   - Links artists to galleries with confidence scores

## Output

Results are saved to `output/results.jsonl` for debugging/audit purposes. The main data goes into your Supabase `gallery_artists` table.

## Architecture

- **config.py**: Environment variables and settings
- **models.py**: Pydantic data models
- **database.py**: Supabase client operations
- **matcher.py**: Fuzzy matching logic (thefuzz)
- **scraper.py**: Crawl4AI + Groq integration
- **main.py**: CLI entry point and orchestration
- **review_duplicates.py**: Review tool for finding potential duplicate artists
- **debug_capture.py**: Standalone script to capture page HTML for debugging

## Database Schema

### `galleries` table
- `id` (UUID, PK)
- `name`, `url`, `priority`
- `last_indexed_at` (TIMESTAMP) - when gallery was last scraped

### `artists` table
- `id` (INTEGER, PK, auto-increment)
- `artist_name` (normalized, lowercase) - for fuzzy matching
- `artist_display_name` (as shown on gallery websites)

### `gallery_artists` junction table
- `gallery_id` (UUID, FK), `artist_id` (INTEGER, FK)
- `artist_gallery_url` - link to artist's page on gallery website
- `is_represented` (BOOLEAN) - true = currently represented, false = formerly represented
- `scrape_confidence` (DECIMAL) - match confidence 0.0-1.0
- `last_scraped_at` (TIMESTAMP) - when first added
- `last_verified_at` (TIMESTAMP) - when last confirmed in rescrape
- `removed_at` (TIMESTAMP) - when artist was no longer found
- `removal_reason` (VARCHAR) - e.g., "rescrape_not_found"

## Fuzzy Matching & Duplicate Handling

The scraper uses fuzzy matching to reconcile extracted artists with your existing database:

| Match Score | Action | Example Output |
|------------|--------|----------------|
| 100% (Exact) | Auto-link existing artist | `EXACT: David Adamo` |
| 90-99% (Fuzzy) | Auto-link with confidence | `FUZZY: David Adamo (95%)` |
| **70-89% (Uncertain)** | **Create NEW artist + flag for review** | `NEW (review): Matthew Hansel (79%)` |
| <70% (New) | Create new artist | `NEW: John Smith` |

### Why Create New Artists for Uncertain Matches?

Rather than risk incorrectly merging different artists, uncertain matches are **created as new artist records** and flagged for future review. This ensures:
- No data loss or incorrect associations
- Each gallery's representation is accurately captured
- Potential duplicates can be reviewed and merged manually later

### Reviewing Potential Duplicates

Run the review script periodically (e.g., monthly) to find and consolidate duplicates:

```bash
# Check for potential duplicates (80%+ similarity)
python src/review_duplicates.py

# Use lower threshold to catch more possibles
python src/review_duplicates.py --threshold 0.7

# Very strict (only likely duplicates)
python src/review_duplicates.py --threshold 0.9
```

The script will output:
- HIGH confidence (90%+) - Likely duplicates, consider merging
- MEDIUM confidence (80-89%) - Review recommended  
- LOW confidence (threshold-79%) - Possible duplicates

## Rescraping & Artist Removal Tracking

When you rescrape a gallery, the system:
1. **Compares existing artists** with newly scraped artists
2. **Updates `last_verified_at`** for all found artists
3. **Soft-deletes missing artists** by setting `is_represented = false` and `removed_at` timestamp
4. **Updates gallery's `last_indexed_at`** timestamp

### Soft Delete (No Data Loss)

Instead of hard-deleting relationships, the system marks them as removed:
- `is_represented` → `false`
- `removed_at` → current timestamp
- `removal_reason` → `"rescrape_not_found"`

This preserves historical data: "Gallery X represented Artist Y from 2023-2025"

### SQL to Query Removed Artists

```sql
-- Find all removed artist relationships
SELECT g.name as gallery, a.artist_display_name, ga.removed_at, ga.removal_reason
FROM gallery_artists ga
JOIN galleries g ON g.id = ga.gallery_id
JOIN artists a ON a.id = ga.artist_id
WHERE ga.is_represented = false
ORDER BY ga.removed_at DESC;

-- Find stale representations (not verified in 30 days)
SELECT g.name, COUNT(*) as stale_count
FROM gallery_artists ga
JOIN galleries g ON g.id = ga.gallery_id
WHERE ga.is_represented = true
  AND ga.last_verified_at < NOW() - INTERVAL '30 days'
GROUP BY g.name;
```

## Troubleshooting

**Import errors**: Ensure your virtual environment is activated and run:
```bash
pip install -e .
```

**Playwright errors**: Run `crawl4ai-setup` or `playwright install chromium`

**RLS errors**: Ensure you're using the service_role key, not anon key

**Rate limiting**: Increase `DELAY_BETWEEN_GALLERIES` in .env
