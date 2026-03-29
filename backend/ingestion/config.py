"""Ingestion configuration constants."""

import os

PFR_BASE_URL = "https://www.pro-football-reference.com"

# Rate limiting
REQUEST_DELAY = float(os.environ.get("PFR_REQUEST_DELAY", "3.0"))
MAX_RETRIES = int(os.environ.get("PFR_MAX_RETRIES", "3"))

# Target seasons
SEASONS = list(range(2014, 2025))  # 2014 through 2024

# Stat validation ranges (spec Section 2.7)
STAT_RANGES = {
    "total_yards": (0, 800),
    "pass_yards": (0, 700),
    "rush_yards": (0, 500),
    "points_scored": (0, 100),
    "points_allowed": (0, 100),
    "pass_attempts": (0, 120),
    "rush_attempts": (0, 80),
    "penalties": (0, 30),
    "penalty_yards": (0, 250),
    "first_downs": (0, 50),
    "third_down_attempts": (0, 30),
    "third_down_conversions": (0, 30),
    "fourth_down_attempts": (0, 15),
    "fourth_down_conversions": (0, 15),
    "sacks": (0, 15),
    "sacks_allowed": (0, 15),
    "turnovers": (0, 12),
    "interceptions_thrown": (0, 10),
    "fumbles_lost": (0, 8),
    "time_of_possession": (0, 3900),  # in seconds, max ~65 min (OT)
}

# Cache directory for raw HTML
CACHE_DIR = os.environ.get("PFR_CACHE_DIR", ".pfr_cache")
