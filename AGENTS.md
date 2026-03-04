# AI Agent Instructions for Gallery Artists Scraper

> This file provides instructions for AI assistants (Cursor, Claude, OpenCode, etc.) working on this repository.

## Critical Rules

### Git & GitHub Workflow ⭐ HIGHEST PRIORITY

**NEVER push to GitHub unless explicitly instructed.**

- ✅ **ALLOWED without asking**: `git status`, `git add`, `git commit`, `git diff`, `git log`
- ❌ **REQUIRES explicit permission**: `git push`, `git push origin`, pushing to remote

**Always ask before pushing:**
```
I've made the changes. Would you like me to commit and push to GitHub?
```

**The user must explicitly say things like:**
- "push to GitHub"
- "commit and push"
- "deploy"
- "push to remote"

Only then should you run `git push`.

## Project Context

### Technology Stack
- Python 3.12 with asyncio
- Crawl4AI for web scraping
- Groq LLM for artist extraction
- Supabase for database
- Pydantic for data models

### Architecture
```
src/
├── main.py              # Entry point
├── scraper.py           # Web scraping + LLM extraction
├── matcher.py           # Fuzzy matching
├── database.py          # Supabase operations
├── models.py            # Pydantic models
├── config.py            # Environment config
└── utils.py             # Shared utilities
```

### Key Design Decisions
- **Hybrid extraction**: Regex finds 100% of artists, LLM classifies
- **No truncation**: All markdown content is processed
- **LLM validation**: Intelligent page discovery vs hardcoded routes
- **Multi-pass consensus**: Multiple LLM calls for completeness

## Code Style

### Python
- Use type hints
- Prefer `pathlib.Path` over `os.path`
- Use f-strings for formatting
- Keep functions focused and under 50 lines when possible

### Comments
- No obvious comments ("# Increment counter")
- Explain WHY, not WHAT
- Document non-obvious behavior and trade-offs

### Error Handling
- Use specific exceptions, not bare `except:`
- Log errors with context
- Fail gracefully when possible

## Testing
- Create temporary test scripts in repo root (clean up after)
- Use `--gallery-url` flag for testing specific galleries
- Check `output/` directory for debug markdown
