"""Parse PFR HTML pages into structured data."""

from __future__ import annotations

import logging
import re
from datetime import date

from bs4 import BeautifulSoup, Comment

from backend.models.research_models import TEAM_ABBREVIATION_MAP

logger = logging.getLogger(__name__)


def parse_schedule_page(html: str, season: int) -> list[dict]:
    """Extract game metadata from a PFR season schedule page.

    Returns a list of dicts with keys: game_id, week, game_date, game_type,
    home_team, away_team, home_score, away_score.
    """
    soup = BeautifulSoup(html, "lxml")
    games = []

    table = soup.find("table", id="games")
    if table is None:
        logger.warning("No games table found for season %d", season)
        return games

    tbody = table.find("tbody")
    if tbody is None:
        return games

    for row in tbody.find_all("tr"):
        # Skip header rows within tbody
        if row.get("class") and "thead" in row.get("class", []):
            continue

        cells = row.find_all(["td", "th"])
        if len(cells) < 8:
            continue

        try:
            game_data = _parse_schedule_row(cells, season)
            if game_data:
                games.append(game_data)
        except (ValueError, IndexError, AttributeError) as e:
            logger.warning("Failed to parse schedule row: %s", e)
            continue

    logger.info("Parsed %d games from %d schedule", len(games), season)
    return games


def _parse_schedule_row(cells: list, season: int) -> dict | None:
    """Parse a single row from the schedule table."""
    # Find the boxscore link to extract game_id
    boxscore_cell = None
    for cell in cells:
        link = cell.find("a", href=re.compile(r"/boxscores/\d+"))
        if link:
            boxscore_cell = link
            break

    if boxscore_cell is None:
        return None

    href = boxscore_cell["href"]
    game_id_match = re.search(r"/boxscores/(\w+)\.htm", href)
    if not game_id_match:
        return None
    game_id = game_id_match.group(1)

    return {"game_id": game_id, "season": season}


def parse_game_page(html: str, game_id: str) -> dict | None:
    """Extract game metadata and team stats from a PFR game box score page.

    Returns a dict with keys: game (metadata dict), home_stats (dict), away_stats (dict).
    """
    soup = BeautifulSoup(html, "lxml")

    # Extract scorebox for game metadata
    scorebox = soup.find("div", class_="scorebox")
    if scorebox is None:
        logger.warning("No scorebox found for game %s", game_id)
        return None

    game_meta = _extract_game_metadata(soup, scorebox, game_id)
    if game_meta is None:
        return None

    # Extract team stats from the "team_stats" table
    # PFR sometimes hides tables in comments
    team_stats_table = _find_table(soup, "team_stats")
    if team_stats_table is None:
        logger.warning("No team_stats table found for game %s", game_id)
        return None

    home_stats, away_stats = _parse_team_stats_table(team_stats_table)

    # Extract spread
    spread = _extract_spread(soup)

    return {
        "game": {**game_meta, "home_spread_close": spread},
        "home_stats": home_stats,
        "away_stats": away_stats,
    }


def _find_table(soup: BeautifulSoup, table_id: str) -> BeautifulSoup | None:
    """Find a table by ID, including tables hidden in HTML comments."""
    table = soup.find("table", id=table_id)
    if table:
        return table

    # PFR hides some tables in comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if table_id in str(comment):
            comment_soup = BeautifulSoup(str(comment), "lxml")
            table = comment_soup.find("table", id=table_id)
            if table:
                return table
    return None


def _extract_game_metadata(soup: BeautifulSoup, scorebox, game_id: str) -> dict | None:
    """Extract game metadata from the scorebox div."""
    teams = scorebox.find_all("div", recursive=False)
    if len(teams) < 2:
        return None

    # PFR scorebox: first team div is away, second is home
    away_div, home_div = teams[0], teams[1]

    def _extract_team_from_div(div) -> tuple[str, int] | None:
        links = div.find_all("a", href=re.compile(r"/teams/"))
        if not links:
            return None
        team_link = links[0]
        abbr_match = re.search(r"/teams/(\w+)/", team_link["href"])
        if not abbr_match:
            return None
        abbr = abbr_match.group(1).upper()
        abbr = _normalize_abbr(abbr)

        # Score
        score_div = div.find("div", class_="scores")
        if score_div is None:
            score_div = div.find("div", class_="score")
        if score_div:
            score_text = score_div.get_text(strip=True)
            # Sometimes scores are in individual divs
            score_divs = score_div.find_all("div")
            if score_divs:
                # Last score div is the final score
                score_text = score_divs[-1].get_text(strip=True)
            try:
                score = int(re.search(r"\d+", score_text).group())
            except (AttributeError, ValueError):
                score = None
        else:
            score = None

        return abbr, score

    away_info = _extract_team_from_div(away_div)
    home_info = _extract_team_from_div(home_div)
    if away_info is None or home_info is None:
        return None

    away_abbr, away_score = away_info
    home_abbr, home_score = home_info

    # Game date
    game_date = _extract_game_date(soup)

    # Determine game type from meta or URL patterns
    game_type = "regular"  # Default; playoff detection happens at schedule level

    return {
        "game_id": game_id,
        "home_team": home_abbr,
        "away_team": away_abbr,
        "home_score": home_score,
        "away_score": away_score,
        "game_date": game_date,
        "game_type": game_type,
    }


def _extract_game_date(soup: BeautifulSoup) -> date | None:
    """Extract game date from the page."""
    # Look for the scorebox_meta div
    meta = soup.find("div", class_="scorebox_meta")
    if meta:
        date_div = meta.find("div")
        if date_div:
            date_text = date_div.get_text(strip=True)
            # Try parsing common PFR date formats
            date_match = re.search(r"(\w+ \d+, \d{4})", date_text)
            if date_match:
                from datetime import datetime
                try:
                    return datetime.strptime(date_match.group(1), "%B %d, %Y").date()
                except ValueError:
                    pass
    return None


def _parse_team_stats_table(table) -> tuple[dict, dict]:
    """Parse the team_stats table into (home_stats, away_stats) dicts.

    PFR team_stats table has rows like:
    Stat Name | Away Value | Home Value
    """
    home_stats = {}
    away_stats = {}

    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue

        stat_name = cells[0].get_text(strip=True).lower()
        away_val = cells[1].get_text(strip=True)
        home_val = cells[2].get_text(strip=True)

        _map_stat(stat_name, away_val, away_stats)
        _map_stat(stat_name, home_val, home_stats)

    return home_stats, away_stats


def _map_stat(stat_name: str, value_str: str, stats: dict) -> None:
    """Map a PFR stat name/value pair to our schema fields."""
    try:
        if "first down" in stat_name and "penalt" not in stat_name:
            stats["first_downs"] = _parse_int(value_str)
        elif "rush" in stat_name and "att" in stat_name.lower():
            # "Rushing-Yards-TDs" format: "25-120-1" or "Rush-Yds-TDs"
            parts = value_str.split("-")
            if len(parts) >= 2:
                stats["rush_attempts"] = _parse_int(parts[0])
                stats["rush_yards"] = _parse_int(parts[1])
                if len(parts) >= 3:
                    stats["rush_touchdowns"] = _parse_int(parts[2])
        elif "pass" in stat_name and ("cmp" in stat_name or "att" in stat_name or "yd" in stat_name):
            # "Cmp-Att-Yd-TD-INT" format
            parts = value_str.split("-")
            if len(parts) >= 3:
                stats["pass_completions"] = _parse_int(parts[0])
                stats["pass_attempts"] = _parse_int(parts[1])
                stats["pass_yards"] = _parse_int(parts[2])
                if len(parts) >= 4:
                    stats["pass_touchdowns"] = _parse_int(parts[3])
                if len(parts) >= 5:
                    stats["interceptions_thrown"] = _parse_int(parts[4])
        elif "sack" in stat_name and "allow" not in stat_name:
            # Sacks made by this team's defense
            parts = value_str.split("-")
            stats["sacks"] = _parse_int(parts[0])
        elif "net pass" in stat_name:
            pass  # Skip net passing yards (we use raw)
        elif "total yard" in stat_name or stat_name == "total yards":
            stats["total_yards"] = _parse_int(value_str)
        elif "fumble" in stat_name and "lost" in stat_name:
            parts = value_str.split("-")
            if len(parts) >= 2:
                stats["fumbles_lost"] = _parse_int(parts[1])
            else:
                stats["fumbles_lost"] = _parse_int(value_str)
        elif "turnover" in stat_name:
            stats["turnovers"] = _parse_int(value_str)
        elif "penalt" in stat_name:
            parts = value_str.split("-")
            if len(parts) >= 2:
                stats["penalties"] = _parse_int(parts[0])
                stats["penalty_yards"] = _parse_int(parts[1])
        elif "third down" in stat_name:
            parts = value_str.split("-")
            if len(parts) >= 2:
                stats["third_down_conversions"] = _parse_int(parts[0])
                stats["third_down_attempts"] = _parse_int(parts[1])
        elif "fourth down" in stat_name:
            parts = value_str.split("-")
            if len(parts) >= 2:
                stats["fourth_down_conversions"] = _parse_int(parts[0])
                stats["fourth_down_attempts"] = _parse_int(parts[1])
        elif "time of poss" in stat_name or "possession" in stat_name:
            stats["time_of_possession"] = _parse_time_of_possession(value_str)
    except (ValueError, IndexError) as e:
        logger.debug("Failed to parse stat '%s' = '%s': %s", stat_name, value_str, e)


def _extract_spread(soup: BeautifulSoup) -> float | None:
    """Extract closing spread from a PFR game page.

    PFR shows the Vegas line in the game info section.
    """
    # Look in game_info table/comment
    game_info = _find_table(soup, "game_info")
    if game_info is None:
        return None

    for row in game_info.find_all("tr"):
        header = row.find("th")
        if header and "vegas" in header.get_text(strip=True).lower():
            td = row.find("td")
            if td:
                text = td.get_text(strip=True)
                # Format: "Team -3.5" or "Pick" (for pick'em)
                if "pick" in text.lower():
                    return 0.0
                # Extract the number
                match = re.search(r"(-?\d+\.?\d*)", text)
                if match:
                    spread_val = float(match.group(1))
                    # PFR shows "Favorite -X" — we need to determine if it's home or away
                    # and convert to home_spread_close convention
                    # This requires knowing which team is the favorite
                    # For now return the raw number; the ingest pipeline normalizes
                    return spread_val
    return None


def _normalize_abbr(abbr: str) -> str:
    """Normalize a PFR team abbreviation to canonical form."""
    return TEAM_ABBREVIATION_MAP.get(abbr, abbr)


def _parse_int(s: str) -> int | None:
    """Parse a string to int, returning None on failure."""
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return None


def _parse_time_of_possession(s: str) -> int | None:
    """Parse MM:SS time of possession to total seconds."""
    match = re.match(r"(\d+):(\d+)", s.strip())
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    return None
