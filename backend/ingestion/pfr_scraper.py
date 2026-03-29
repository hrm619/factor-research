"""PFR HTML fetcher with rate limiting and filesystem caching."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from backend.ingestion.config import PFR_BASE_URL, REQUEST_DELAY, MAX_RETRIES, CACHE_DIR

logger = logging.getLogger(__name__)


class PFRScraper:
    """Fetches HTML from Pro Football Reference with caching and rate limiting."""

    def __init__(self, cache_dir: str = CACHE_DIR, delay: float = REQUEST_DELAY) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "factor-research/0.1 (academic research)"
        })
        self._last_request_time: float = 0.0

    def fetch_season_schedule(self, season: int) -> str:
        """Fetch the season schedule/games page from PFR."""
        cache_key = f"schedule_{season}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.info("Cache hit for season %d schedule", season)
            return cached

        url = f"{PFR_BASE_URL}/years/{season}/games.htm"
        html = self._rate_limited_get(url)
        self._write_cache(cache_key, html)
        return html

    def fetch_game_page(self, game_id: str) -> str:
        """Fetch an individual game's box score page from PFR."""
        cached = self._read_cache(game_id)
        if cached is not None:
            logger.debug("Cache hit for game %s", game_id)
            return cached

        url = f"{PFR_BASE_URL}/boxscores/{game_id}.htm"
        html = self._rate_limited_get(url)
        self._write_cache(game_id, html)
        return html

    def _rate_limited_get(self, url: str) -> str:
        """GET with rate limiting and retries."""
        # Enforce delay between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._last_request_time = time.time()
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                logger.debug("Fetched %s (attempt %d)", url, attempt)
                return response.text
            except requests.RequestException as e:
                logger.warning("Request failed (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(self.delay * attempt)  # Exponential-ish backoff

    def _read_cache(self, key: str) -> str | None:
        path = self.cache_dir / f"{key}.html"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def _write_cache(self, key: str, content: str) -> None:
        path = self.cache_dir / f"{key}.html"
        path.write_text(content, encoding="utf-8")
