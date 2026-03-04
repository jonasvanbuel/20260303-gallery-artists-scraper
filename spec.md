# Gallery Artists Scraper — Implementation Spec v2

## Goal

A Python script that:
1. Reads galleries from Supabase (prioritizing `priority=true` galleries first)
2. Navigates each gallery website to find its artists page
3. Extracts artist data and fuzzy-matches against existing Supabase `artists` table
4. Creates/updates many-to-many `gallery_artists` relationships
5. Stores raw extraction results for audit/debugging

## Tech Stack

- **Python 3.12+** with `venv` + `pip` for dependency management
- **[Crawl4AI](https://github.com/unclecode/crawl4ai)** — async web crawler with JS rendering
- **[Groq](https://console.groq.com)** — fast LLM inference (llama-3.3-70b-versatile)
- **[Supabase Python Client](https://github.com/supabase/supabase-py)** — database operations
- **[thefuzz](https://github.com/seatgeek/thefuzz)** — fuzzy string matching for artist names
- **Pydantic v2** — data models and validation
- **python-dotenv** — environment variable management

---

## Database Schema

### Your Current Tables

**galleries table:**
```sql
- id: uuid PRIMARY KEY (gen_random_uuid())
- name: text NOT NULL
- url: text NOT NULL
- priority: boolean (nullable) -- you've marked some as true
```

**artists table:**
```sql
- id: integer PRIMARY KEY (auto-increment)
- artist_name: text NOT NULL UNIQUE
- artist_display_name: text NOT NULL
- birth_year: integer (nullable)
- death_year: integer (nullable)
- gender: text (nullable)
- nationality: text (nullable)
- created_at: timestamp with time zone DEFAULT now()
```

### New Tables to Create

```sql
-- Many-to-many relationship: galleries ↔ artists
-- IMPORTANT: artist_id references artists.id which is INTEGER (not uuid)
CREATE TABLE IF NOT EXISTS gallery_artists (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  gallery_id uuid REFERENCES galleries(id) ON DELETE CASCADE,
  artist_id integer REFERENCES artists(id) ON DELETE CASCADE,
  artist_gallery_url text,          -- URL to artist page on gallery site
  is_represented boolean DEFAULT true,
  last_scraped_at timestamp with time zone,        -- when we last confirmed this relationship
  first_seen_at timestamp with time zone DEFAULT now(),
  scrape_confidence float,            -- fuzzy match confidence score (0-1)
  created_at timestamp with time zone DEFAULT now(),
  
  UNIQUE(gallery_id, artist_id)
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_gallery_artists_gallery_id ON gallery_artists(gallery_id);
CREATE INDEX IF NOT EXISTS idx_gallery_artists_artist_id ON gallery_artists(artist_id);

-- Enable RLS
ALTER TABLE gallery_artists ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Allow authenticated users to read gallery_artists"
ON gallery_artists FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow service role full access"
ON gallery_artists FOR ALL TO service_role USING (true) WITH CHECK (true);
```

---

## Project Structure

```
gallery-artists-scraper/
├── .env                            # Environment variables
├── pyproject.toml
├── output/                         # Local backup of results (optional)
│   └── results.jsonl
└── src/
    ├── main.py                     # Entry point
    ├── config.py                   # Settings & env vars
    ├── database.py                 # Supabase client & queries
    ├── scraper.py                  # Crawl4AI + Groq logic
    ├── matcher.py                  # Fuzzy artist matching
    ├── models.py                   # Pydantic data models
    └── utils.py                    # URL resolution, name normalization
```

---

## Data Models (`models.py`)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import uuid

class ArtistExtraction(BaseModel):
    """Raw artist extracted from gallery website"""
    artist_display_name: str
    artist_gallery_url: Optional[str] = None
    is_represented: bool = True
    normalized_name: Optional[str] = None  # computed post-extraction

class GalleryScrapeResult(BaseModel):
    """Complete result for one gallery scrape"""
    gallery_id: uuid.UUID
    gallery_name: str
    gallery_url: str
    artists_page_url: Optional[str] = None
    artists: list[ArtistExtraction]
    error: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)

class ArtistMatch(BaseModel):
    """Result of fuzzy matching an extracted artist to database"""
    extracted_name: str
    normalized_name: str
    matched_artist_id: Optional[int] = None  # NOTE: artists.id is integer
    matched_display_name: Optional[str] = None
    confidence_score: float  # 0.0 - 1.0
    match_type: str  # 'exact', 'fuzzy', 'new', 'uncertain'
    gallery_url: Optional[str] = None
    is_represented: bool = True
```

---

## Architecture: Three-Phase Per Gallery

### Phase 1 — Discover Artists Page URL

**Strategy: Try /artists FIRST, then use LLM on homepage as fallback**

1. **Try `{gallery_url}/artists`**
   - Fetch with Crawl4AI (with scrolling for lazy loading)
   - If returns 200 and has artist-like content (multiple names, artist list pattern), use it
   - If 404 or empty, proceed to step 2

2. **Fetch homepage and use LLM discovery**:
   - Crawl4AI fetches homepage with scrolling, converts to markdown
   - Send to Groq LLM: "Find the URL that lists this gallery's represented artists"
   - LLM identifies the correct URL path from the navigation/menu

3. **Result**: `artists_page_url` or null (with error logged)

### Phase 2 — Extract Artist List

1. Fetch the artists page URL from Phase 1 with Crawl4AI (with scrolling)
2. Handle pagination (follow "Next" links, max 10 pages)
3. Send combined markdown to Groq for extraction
4. LLM returns list of artists with display names and URLs
5. Normalize names: `display_name` → `normalized_name` for matching

### Phase 3 — Match & Store

**This is where the magic happens:**

```python
# 1. Load existing artists from Supabase into memory
existing_artists = db.get_all_artists()  # List of {id, artist_name, artist_display_name}

# 2. For each extracted artist, find best match
for extracted in scraped_artists:
    match_result = fuzzy_match(extracted, existing_artists)
    
    if match_result.confidence_score > 0.9:
        # High confidence: link to existing artist
        db.create_gallery_artist_link(
            gallery_id=gallery_id,
            artist_id=match_result.matched_artist_id,  # integer
            artist_gallery_url=extracted.artist_gallery_url,
            confidence=match_result.confidence_score
        )
    elif match_result.confidence_score > 0.7:
        # Medium confidence: flag for manual review
        db.flag_for_review(match_result)
    else:
        # Low confidence: likely new artist
        new_artist_id = db.create_artist(
            artist_name=extracted.normalized_name,
            artist_display_name=extracted.artist_display_name
        )
        db.create_gallery_artist_link(
            gallery_id=gallery_id,
            artist_id=new_artist_id,  # integer
            artist_gallery_url=extracted.artist_gallery_url
        )
```

---

## Fuzzy Matching Strategy (`matcher.py`)

**Algorithm: Multi-tier matching with thefuzz**

```python
from thefuzz import fuzz, process
from typing import List, Optional

def fuzzy_match_artist(
    extracted_name: str,
    normalized_name: str,
    existing_artists: List[dict],
    threshold_high: float = 90.0,
    threshold_medium: float = 70.0
) -> ArtistMatch:
    """
    Match extracted artist against existing database.
    
    Tiers:
    1. Exact match on normalized_name (100%)
    2. Fuzzy match on normalized_name using token_sort_ratio
    3. Fuzzy match on artist_display_name
    4. No match - likely new artist
    """
    
    # Build lookup lists
    normalized_names = [(a['id'], a['artist_name']) for a in existing_artists]
    display_names = [(a['id'], a['artist_display_name']) for a in existing_artists]
    
    # Tier 1: Exact normalized name match
    exact_matches = [a for a in existing_artists if a['artist_name'] == normalized_name]
    if exact_matches:
        return ArtistMatch(
            extracted_name=extracted_name,
            normalized_name=normalized_name,
            matched_artist_id=exact_matches[0]['id'],  # integer
            matched_display_name=exact_matches[0]['artist_display_name'],
            confidence_score=1.0,
            match_type='exact'
        )
    
    # Tier 2: Fuzzy match on normalized names
    best_normalized = process.extractOne(
        normalized_name, 
        normalized_names, 
        scorer=fuzz.token_sort_ratio
    )
    
    if best_normalized and best_normalized[1] >= threshold_high:
        artist_id = best_normalized[2][0]  # integer
        artist = next(a for a in existing_artists if a['id'] == artist_id)
        return ArtistMatch(
            extracted_name=extracted_name,
            normalized_name=normalized_name,
            matched_artist_id=artist_id,  # integer
            matched_display_name=artist['artist_display_name'],
            confidence_score=best_normalized[1] / 100.0,
            match_type='fuzzy'
        )
    
    # Tier 3: Fuzzy match on display names
    best_display = process.extractOne(
        extracted_name,
        display_names,
        scorer=fuzz.token_sort_ratio
    )
    
    if best_display and best_display[1] >= threshold_high:
        artist_id = best_display[2][0]  # integer
        artist = next(a for a in existing_artists if a['id'] == artist_id)
        return ArtistMatch(
            extracted_name=extracted_name,
            normalized_name=normalized_name,
            matched_artist_id=artist_id,  # integer
            matched_display_name=artist['artist_display_name'],
            confidence_score=best_display[1] / 100.0,
            match_type='fuzzy_display'
        )
    
    # Tier 4: Medium confidence (flag for review)
    best_score = max(
        best_normalized[1] if best_normalized else 0,
        best_display[1] if best_display else 0
    )
    
    if best_score >= threshold_medium:
        return ArtistMatch(
            extracted_name=extracted_name,
            normalized_name=normalized_name,
            confidence_score=best_score / 100.0,
            match_type='uncertain'
        )
    
    # No match - new artist
    return ArtistMatch(
        extracted_name=extracted_name,
        normalized_name=normalized_name,
        confidence_score=0.0,
        match_type='new'
    )
```

**Name Normalization Rules:**
```python
import re
import unicodedata

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
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    
    # Remove parentheses and their contents (common for birth years)
    name = re.sub(r'\s*\([^)]*\)', '', name)
    
    # Remove punctuation except hyphens and spaces
    name = re.sub(r'[^\w\s\-]', '', name)
    
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name)
    
    # Handle suffixes
    name = re.sub(r'\s+(jr|sr|iii|ii|iv)\b', r' \1', name)
    
    return name.strip()
```

---

## Database Operations (`database.py`)

```python
from supabase import create_client, Client
from typing import List, Optional
import uuid

class SupabaseClient:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)
    
    def get_priority_galleries(self) -> List[dict]:
        """Get galleries marked as priority first"""
        response = self.client.table('galleries') \
            .select('*') \
            .eq('priority', True) \
            .execute()
        return response.data
    
    def get_all_galleries(self) -> List[dict]:
        """Get all galleries (after priority ones are done)"""
        response = self.client.table('galleries') \
            .select('*') \
            .execute()
        return response.data
    
    def get_all_artists(self) -> List[dict]:
        """Load all artists for fuzzy matching"""
        # Consider pagination for large datasets (>1000 artists)
        response = self.client.table('artists') \
            .select('id, artist_name, artist_display_name') \
            .execute()
        return response.data
    
    def create_artist(self, artist_name: str, artist_display_name: str) -> int:
        """Create new artist and return ID (integer)"""
        response = self.client.table('artists').insert({
            'artist_name': artist_name,
            'artist_display_name': artist_display_name
        }).execute()
        return response.data[0]['id']  # integer
    
    def create_gallery_artist_link(
        self,
        gallery_id: uuid.UUID,
        artist_id: int,  # NOTE: integer, not uuid
        artist_gallery_url: Optional[str],
        is_represented: bool = True,
        confidence: float = 1.0
    ):
        """Upsert gallery-artist relationship"""
        self.client.table('gallery_artists').upsert({
            'gallery_id': gallery_id,
            'artist_id': artist_id,  # integer
            'artist_gallery_url': artist_gallery_url,
            'is_represented': is_represented,
            'last_scraped_at': 'now()',
            'scrape_confidence': confidence
        }, on_conflict='gallery_id,artist_id').execute()
```

---

## Crawl4AI Configuration with Lazy Loading Support (`scraper.py`)

**CRITICAL: Must scroll to bottom for lazy-loaded content**

```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
import os

browser_cfg = BrowserConfig(
    browser_type="chromium",
    headless=True,
    verbose=False,
)

crawl_cfg = CrawlerRunConfig(
    wait_until="networkidle",
    page_timeout=30000,
    word_count_threshold=10,
    # Magic: Scroll to bottom to trigger lazy loading
    js_code="""
        async () => {
            await new Promise((resolve) => {
                let totalHeight = 0;
                const distance = 100;
                const timer = setInterval(() => {
                    const scrollHeight = document.body.scrollHeight;
                    window.scrollBy(0, distance);
                    totalHeight += distance;
                    
                    if (totalHeight >= scrollHeight) {
                        clearInterval(timer);
                        resolve();
                    }
                }, 100);
            });
            // Wait a bit more for any final lazy loads
            await new Promise(r => setTimeout(r, 2000));
        }
    """
)

# Groq configuration
GROQ_MODEL = "llama-3.3-70b-versatile"
```

Alternative scrolling approach using Crawl4AI's built-in support:

```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

crawl_cfg = CrawlerRunConfig(
    wait_until="networkidle",
    page_timeout=30000,
    word_count_threshold=10,
    # Simulate scrolling by waiting and checking for content changes
    js_code="""
        async () => {
            const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
            let lastHeight = document.body.scrollHeight;
            let attempts = 0;
            const maxAttempts = 20;
            
            while (attempts < maxAttempts) {
                window.scrollTo(0, document.body.scrollHeight);
                await delay(500);
                const newHeight = document.body.scrollHeight;
                if (newHeight === lastHeight) {
                    attempts++;
                } else {
                    lastHeight = newHeight;
                    attempts = 0;
                }
            }
            // Scroll back to top then back down to ensure everything is loaded
            window.scrollTo(0, 0);
            await delay(500);
            window.scrollTo(0, document.body.scrollHeight);
            await delay(1000);
        }
    """
)
```

### Phase 1 Prompt — Find Artists Page (LLM Fallback)

Used only when `/artists` doesn't work. The LLM examines the homepage markdown to find the correct artists page URL.

```
System: You are a web scraping assistant. Find the artists page URL. 
Respond with JSON only.

User: Find the URL on {gallery_name}'s website ({gallery_url}) that lists 
their represented artists. Look for navigation links or menu items like 
"Artists", "Our Artists", "Represented Artists", "Roster", or similar.

Return: {"artists_page_url": "https://..."} or {"artists_page_url": null}

Page content:
{markdown}
```

### Phase 2 Prompt — Extract Artists

```
System: You are extracting artist data from a gallery website. 
Return valid JSON only.

User: Extract all artists from this gallery page. For each artist:
- artist_display_name: Full name as shown
- artist_gallery_url: Link to their profile page (if available)
- is_represented: true if they appear to be formally represented, 
  false if they're only available as secondary/resale

Ignore artists mentioned only in passing (press quotes, etc.).

Return: {"artists": [...]}

Page content:
{markdown}
```

---

## Environment Variables (`.env`)

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SECRET_KEY=eyJ...
GROQ_API_KEY=gsk_...
DELAY_BETWEEN_GALLERIES=2
FUZZY_MATCH_HIGH_THRESHOLD=0.9
FUZZY_MATCH_MEDIUM_THRESHOLD=0.7
LOG_LEVEL=INFO
```

---

## Entry Point (`main.py`)

```python
Usage:
  python src/main.py                          # scrape priority galleries first, then all
  python src/main.py --priority-only          # only scrape priority galleries
  python src/main.py --gallery-id <uuid>      # scrape single gallery by ID
  python src/main.py --dry-run                # extract but don't write to database
  python src/main.py --limit 5                # limit for testing
```

**Execution Flow:**
1. Connect to Supabase
2. Load all artists into memory for fuzzy matching
3. Get priority galleries first (if `--priority-only` or no args)
4. For each gallery:
   - Try `{gallery_url}/artists` first (fast, no LLM tokens) with scrolling
   - If 404 or empty, fetch homepage (with scrolling) and use LLM to find correct URL
   - Extract artist list (handle pagination with scrolling)
   - Fuzzy match each artist to database
   - Create/update gallery_artists links
   - Delay before next gallery
5. Generate summary report

---

## Dependencies (`pyproject.toml`)

```toml
[project]
name = "gallery-artists-scraper"
version = "0.2.0"
requires-python = ">=3.12"

dependencies = [
  "crawl4ai>=0.5.0",
  "groq>=0.13.0",
  "supabase>=2.0",
  "thefuzz>=0.22.0",
  "python-levenshtein>=0.25.0",  # Speeds up thefuzz
  "pydantic>=2.0",
  "python-dotenv>=1.0",
]
```

---

## Future Enhancements (Post-MVP)

- **Incremental updates**: Only check galleries that haven't been scraped in N days
- **Scheduled runs**: Weekly cron job
- **Review queue UI**: Dashboard for reviewing medium-confidence matches
- **Duplicate detection**: Find potential duplicate artists in database
- **Gallery health monitoring**: Alert if gallery site structure changes significantly
- **Retry logic**: Failed galleries auto-retry with exponential backoff
