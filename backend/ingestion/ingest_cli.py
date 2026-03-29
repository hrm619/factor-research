"""CLI entry point for data ingestion."""

from __future__ import annotations

import json
import logging
from datetime import datetime, UTC

import click
from sqlalchemy import text

from backend.research.db import get_engine, get_session, init_db
from backend.models.research_models import (
    Game, Team, TeamGameStats, IngestionLog,
    CANONICAL_TEAMS, get_expected_game_count,
)
from backend.ingestion.pfr_scraper import PFRScraper
from backend.ingestion.pfr_parser import parse_schedule_page, parse_game_page
from backend.ingestion.cleaning import clean_game_data, normalize_team_abbr
from backend.ingestion.derived_metrics import compute_derived_metrics
from backend.ingestion.config import SEASONS

logger = logging.getLogger(__name__)


@click.group()
@click.option("--db-url", default=None, help="Database URL (default: sqlite:///factor_research.db)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx, db_url, verbose):
    """Factor Research data ingestion CLI."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["db_url"] = db_url


@main.command()
@click.option("--season", type=int, default=None, help="Season to ingest (default: all)")
@click.option("--cache-dir", default=".pfr_cache", help="Directory for cached HTML")
@click.option("--skip-derived", is_flag=True, help="Skip derived metric computation")
@click.pass_context
def ingest(ctx, season, cache_dir, skip_derived):
    """Ingest game data from Pro Football Reference."""
    engine = get_engine(ctx.obj["db_url"])
    init_db(engine)

    # Seed teams
    _seed_teams(engine)

    scraper = PFRScraper(cache_dir=cache_dir)
    seasons = [season] if season else SEASONS

    for s in seasons:
        _ingest_season(engine, scraper, s, skip_derived)


@main.command("validate-db")
@click.pass_context
def validate_db(ctx):
    """Run validation checks on the database."""
    engine = get_engine(ctx.obj["db_url"])
    session = get_session(engine)

    try:
        # Check team count
        team_count = session.query(Team).count()
        click.echo(f"Teams: {team_count} (expected 32)")

        # Check game counts per season
        result = session.execute(
            text("SELECT season, COUNT(*) as cnt FROM games GROUP BY season ORDER BY season")
        )
        for row in result:
            season, count = row
            expected = get_expected_game_count(season)["total"]
            status = "OK" if abs(count - expected) <= 2 else "WARN"
            click.echo(f"Season {season}: {count} games (expected {expected}) [{status}]")

        # Check for NULL scores
        null_scores = session.execute(
            text("SELECT COUNT(*) FROM games WHERE home_score IS NULL OR away_score IS NULL")
        ).scalar()
        click.echo(f"Games with NULL scores: {null_scores}")

    finally:
        session.close()


@main.command("recompute-derived")
@click.option("--season", type=int, default=None, help="Season to recompute (default: all)")
@click.pass_context
def recompute_derived(ctx, season):
    """Recompute derived metrics without re-scraping."""
    engine = get_engine(ctx.obj["db_url"])
    seasons = [season] if season else SEASONS

    for s in seasons:
        rows = compute_derived_metrics(engine, s)
        click.echo(f"Season {s}: {rows} derived metric rows computed")


def _ingest_season(engine, scraper: PFRScraper, season: int, skip_derived: bool) -> None:
    """Ingest a single season."""
    session = get_session(engine)
    log = IngestionLog(source="pfr", season=season, status="running")
    session.add(log)
    session.commit()

    errors = []
    rows_ingested = 0
    rows_skipped = 0

    try:
        # Fetch schedule
        schedule_html = scraper.fetch_season_schedule(season)
        schedule = parse_schedule_page(schedule_html, season)
        click.echo(f"Season {season}: found {len(schedule)} games in schedule")

        for game_info in schedule:
            game_id = game_info["game_id"]
            try:
                game_html = scraper.fetch_game_page(game_id)
                parsed = parse_game_page(game_html, game_id)

                if parsed is None:
                    errors.append(f"Failed to parse game {game_id}")
                    rows_skipped += 1
                    continue

                cleaned = clean_game_data(parsed)
                _upsert_game(session, cleaned, season)
                rows_ingested += 1

            except Exception as e:
                errors.append(f"Game {game_id}: {str(e)}")
                rows_skipped += 1
                logger.error("Failed to ingest game %s: %s", game_id, e)

        session.commit()

        # Compute derived metrics
        if not skip_derived:
            compute_derived_metrics(engine, season)

        log.status = "success" if not errors else "partial"
        log.rows_ingested = rows_ingested
        log.rows_skipped = rows_skipped
        log.errors = json.dumps(errors) if errors else None
        log.completed_at = datetime.now(UTC)
        session.commit()

        click.echo(f"Season {season}: ingested {rows_ingested}, skipped {rows_skipped}")

    except Exception as e:
        log.status = "failed"
        log.errors = json.dumps([str(e)])
        log.completed_at = datetime.now(UTC)
        session.commit()
        raise
    finally:
        session.close()


def _upsert_game(session, cleaned: dict, season: int) -> None:
    """Insert or update a game and its team stats."""
    game_data = cleaned["game"]
    game_id = game_data["game_id"]

    # Upsert game
    existing = session.query(Game).filter_by(game_id=game_id).first()
    if existing:
        for key, value in game_data.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
        existing.season = season
    else:
        game = Game(season=season, **game_data)
        session.add(game)

    session.flush()

    # Upsert team stats for both teams
    for is_home, stats_key in [(True, "home_stats"), (False, "away_stats")]:
        team_abbr = game_data["home_team"] if is_home else game_data["away_team"]
        stats = cleaned[stats_key]

        # Add points scored/allowed from game scores
        if is_home:
            stats["points_scored"] = game_data.get("home_score")
            stats["points_allowed"] = game_data.get("away_score")
        else:
            stats["points_scored"] = game_data.get("away_score")
            stats["points_allowed"] = game_data.get("home_score")

        existing_stats = session.query(TeamGameStats).filter_by(
            game_id=game_id, team_abbr=team_abbr
        ).first()

        if existing_stats:
            for key, value in stats.items():
                if hasattr(existing_stats, key):
                    setattr(existing_stats, key, value)
        else:
            tgs = TeamGameStats(
                game_id=game_id, team_abbr=team_abbr, is_home=is_home, **stats
            )
            session.add(tgs)


def _seed_teams(engine) -> None:
    """Seed the teams table with all 32 NFL teams."""
    session = get_session(engine)
    existing = {t.team_abbr for t in session.query(Team).all()}

    # Minimal team data — full names and divisions can be enriched later
    team_data = {
        "ARI": ("Arizona Cardinals", "NFC", "West"),
        "ATL": ("Atlanta Falcons", "NFC", "South"),
        "BAL": ("Baltimore Ravens", "AFC", "North"),
        "BUF": ("Buffalo Bills", "AFC", "East"),
        "CAR": ("Carolina Panthers", "NFC", "South"),
        "CHI": ("Chicago Bears", "NFC", "North"),
        "CIN": ("Cincinnati Bengals", "AFC", "North"),
        "CLE": ("Cleveland Browns", "AFC", "North"),
        "DAL": ("Dallas Cowboys", "NFC", "East"),
        "DEN": ("Denver Broncos", "AFC", "West"),
        "DET": ("Detroit Lions", "NFC", "North"),
        "GB": ("Green Bay Packers", "NFC", "North"),
        "HOU": ("Houston Texans", "AFC", "South"),
        "IND": ("Indianapolis Colts", "AFC", "South"),
        "JAX": ("Jacksonville Jaguars", "AFC", "South"),
        "KC": ("Kansas City Chiefs", "AFC", "West"),
        "LAC": ("Los Angeles Chargers", "AFC", "West"),
        "LAR": ("Los Angeles Rams", "NFC", "West"),
        "LVR": ("Las Vegas Raiders", "AFC", "West"),
        "MIA": ("Miami Dolphins", "AFC", "East"),
        "MIN": ("Minnesota Vikings", "NFC", "North"),
        "NE": ("New England Patriots", "AFC", "East"),
        "NO": ("New Orleans Saints", "NFC", "South"),
        "NYG": ("New York Giants", "NFC", "East"),
        "NYJ": ("New York Jets", "AFC", "East"),
        "PHI": ("Philadelphia Eagles", "NFC", "East"),
        "PIT": ("Pittsburgh Steelers", "AFC", "North"),
        "SEA": ("Seattle Seahawks", "NFC", "West"),
        "SF": ("San Francisco 49ers", "NFC", "West"),
        "TB": ("Tampa Bay Buccaneers", "NFC", "South"),
        "TEN": ("Tennessee Titans", "AFC", "South"),
        "WAS": ("Washington Commanders", "NFC", "East"),
    }

    for abbr, (name, conf, div) in team_data.items():
        if abbr not in existing:
            session.add(Team(team_abbr=abbr, team_name=name, conference=conf, division=div))

    session.commit()
    session.close()
