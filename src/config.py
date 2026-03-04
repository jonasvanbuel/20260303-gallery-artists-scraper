"""Configuration and environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""

    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SECRET_KEY: str = os.getenv("SUPABASE_SECRET_KEY", "")

    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    # Available models on Groq (as of 2026-03-04):
    # - llama-3.3-70b-versatile (default, good balance)
    # - qwen-2.5-32b (excellent for structured data extraction)
    # - deepseek-r1-distill-llama-70b (good for complex reasoning)
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Scraping
    DELAY_BETWEEN_GALLERIES: int = int(os.getenv("DELAY_BETWEEN_GALLERIES", "2"))
    PAGE_TIMEOUT: int = int(os.getenv("PAGE_TIMEOUT", "30000"))

    # Fuzzy matching thresholds
    FUZZY_MATCH_HIGH_THRESHOLD: float = float(
        os.getenv("FUZZY_MATCH_HIGH_THRESHOLD", "0.9")
    )
    FUZZY_MATCH_MEDIUM_THRESHOLD: float = float(
        os.getenv("FUZZY_MATCH_MEDIUM_THRESHOLD", "0.7")
    )

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Output
    OUTPUT_DIR: Path = Path("output")

    @classmethod
    def validate(cls) -> None:
        """Validate that required environment variables are set."""
        required = [
            "SUPABASE_URL",
            "SUPABASE_SECRET_KEY",
            "GROQ_API_KEY",
        ]
        missing = [var for var in required if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    @classmethod
    def ensure_output_dir(cls) -> None:
        """Ensure output directory exists."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
