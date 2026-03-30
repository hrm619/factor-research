# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NFL factor research framework for backtesting game-outcome hypotheses against historical data (2014-2024). Tests whether measurable team attributes predict against-the-spread (ATS) performance. Part of the broader fin-arb project.

The Phase 1 spec lives in `factor_research_phase1_spec.docx`. Amendments in `SPEC_AMENDMENTS.md` (game count corrections, spread source strategy, FDR correction).

## Build & Run

This project uses **uv** for Python dependency management (Python 3.13).

```bash
uv sync                                     # Install dependencies
uv run pytest                                # Run all tests (154 tests)
uv run pytest tests/path/to/test.py::test_name  # Run a single test

# CLI — research
uv run factor-research validate -h path.yaml # Validate a hypothesis YAML
uv run factor-research run -h path.yaml      # Run one hypothesis pipeline
uv run factor-research run-all               # Run all hypotheses in hypotheses/
uv run factor-research list-metrics          # Print available metrics

# CLI — ingestion
uv run factor-ingest ingest --season 2024    # Ingest one season from nflverse
uv run factor-ingest ingest                  # Ingest all seasons (2014-2024)
uv run factor-ingest validate-db             # Validate DB row counts
uv run factor-ingest recompute-derived       # Recompute derived metrics
```

## Architecture

### Module Layout

```
backend/
  models/research_models.py    # SQLAlchemy ORM (Game, Team, TeamGameStats, DerivedMetrics, IngestionLog)
  research/
    models.py                  # Dataclasses (HypothesisDefinition, BucketStats, MeasurementResult, etc.)
    db.py                      # Engine/session helpers, init_db()
    define.py                  # Stage 1: YAML → HypothesisDefinition
    extract.py                 # Stage 2: DB query → DataFrame with outcomes
    classify.py                # Stage 3: Bucket observations (quartile/percentile/binary/custom)
    measure.py                 # Stage 4: Stats, FDR correction, quality scoring
    report.py                  # Stage 5: Terminal + JSON output
    harness.py                 # Pipeline orchestrator (run_hypothesis, run_all_hypotheses)
    statistical.py             # Pure stat functions (binomial, z-test, chi-sq, Wilson CI, Cohen's h, FDR)
    metrics_catalog.py         # Metric registry (16 metrics with definitions)
    cli.py                     # Click CLI entry point
    hypotheses/                # 10 hypothesis YAML files (H1-H10)
    results/                   # Output (gitignored)
  ingestion/
    nflverse_source.py         # nflreadpy data fetching (schedule, team stats, PBP aggregation)
    cleaning.py                # Team abbr normalization, stat range validation
    derived_metrics.py         # STD/L4 metric computation (no look-ahead)
    ingest_cli.py              # Click CLI entry point
    config.py                  # Constants (seasons, game type mapping, stat ranges)
```

### Data Source

Game data is sourced from [nflverse](https://github.com/nflverse) via the `nflreadpy` Python package. Three data endpoints are used per season:

- **`load_schedules()`** — game scores, spreads, schedule metadata
- **`load_team_stats()`** — per-team-per-game stats (passing, rushing, penalties, sacks, turnovers, etc.)
- **`load_pbp()`** — play-by-play data, aggregated for: 3rd/4th down conversions, red zone stats, time of possession

Data is fetched as Polars DataFrames, converted to pandas, then mapped to our SQLAlchemy ORM. nflreadpy caches data in `~/.nflverse/` automatically.

### Pipeline Flow

YAML → **Define** → HypothesisDefinition → **Extract** → DataFrame → **Classify** → DataFrame (with buckets) → **Measure** → MeasurementResult → **Report** → Terminal + JSON

Each stage receives only the previous stage's output plus the HypothesisDefinition. Only Extract touches the database.

### Key Data Design Decisions

- **No look-ahead bias**: `_std` and `_l4` derived metrics represent entering-game profile. Week 1 = NULL.
- **Within-season ranking**: All classification ranks within season via `groupby('season')`.
- **Aggregate rates**: SUM(numerator)/SUM(denominator), not AVG(rate).
- **Push = non-cover**: margin_vs_spread == 0 → covered_spread = False.
- **Spread convention**: `home_spread_close` is negative when home is favored. For away teams: `margin_vs_spread = (away_score - home_score) - home_spread_close`.
- **FDR correction**: Benjamini-Hochberg applied within-hypothesis (measure.py) and cross-hypothesis (report.py).
- **Quality scoring**: Composite of sample_size (30%) + significance (30%) + effect_size (20%) + consistency (20%). Max 3.0. HIGH >= 2.5, MEDIUM >= 1.8, LOW >= 1.0.
- **Game ID format**: nflverse format (e.g., `2024_01_KC_BAL`), used as PK in `games` table and FK in `team_game_stats` and `derived_metrics`.

### Team Abbreviation Normalization

The map in `research_models.py` normalizes historical and variant abbreviations to canonical form. Key mappings include: STL/LA→LAR, OAK/LV→LVR, SD→LAC, GNB→GB, KAN→KC, NWE→NE, NOR→NO, SFO→SF, TAM→TB, HTX→HOU, CLT→IND, RAV→BAL, OTI→TEN, CRD→ARI.

## Code Conventions

- Pure functions throughout — consistent with fin-arb architecture
- Dataclasses for value types, SQLAlchemy ORM for persistence
- Python standard logging (INFO normal, WARNING data quality)
- Tests in `tests/` mirroring module structure; in-memory SQLite for DB tests
