# Factor Research Framework — Quickstart Tutorial

**Goal:** Go from zero to running your first hypothesis test, end to end.

This tutorial walks you through every step: installing dependencies, verifying the tooling, ingesting real NFL data from Pro Football Reference, and running a live hypothesis to see if third-down conversion efficiency is underpriced by the betting market.

---

## Prerequisites

- Python 3.13+ installed
- [uv](https://docs.astral.sh/uv/) package manager installed
- Terminal access
- Internet connection (for PFR scraping)

---

## Step 0: Clone and Install

```bash
git clone https://github.com/hrm619/factor-research.git
cd factor-research
uv sync
```

This installs all dependencies: pandas, scipy, statsmodels, beautifulsoup4, SQLAlchemy, click, etc.

**Verify it worked:**

```bash
uv run factor-research --help
```

You should see:

```
Usage: factor-research [OPTIONS] COMMAND [ARGS]...

  Factor Research hypothesis testing CLI.

Options:
  --db-url TEXT  Database URL (default: sqlite:///factor_research.db)
  -v, --verbose  Enable verbose logging
  --help         Show this message and exit.

Commands:
  list-metrics  List all available metrics in the catalog.
  run           Run a single hypothesis through the full pipeline.
  run-all       Run all hypotheses in a directory.
  validate      Validate a hypothesis YAML file without running it.
```

Also verify the ingestion CLI:

```bash
uv run factor-ingest --help
```

---

## Step 1: Run the Tests (Smoke Check)

Before touching live data, make sure everything compiles and the logic is sound. The test suite includes an integration test that runs the full pipeline against synthetic data in an in-memory database — no scraping required.

```bash
uv run pytest -v
```

You should see 154 tests pass. If anything fails, stop here and debug before proceeding.

**Key tests to look for:**

- `test_harness.py::TestRunHypothesis::test_end_to_end` — this runs the full 5-stage pipeline (Define → Extract → Classify → Measure → Report) on synthetic data. If this passes, the pipeline is wired correctly.
- `test_statistical.py` — validates p-values, confidence intervals, and FDR correction against known answers.
- `test_define.py` — validates that hypothesis YAML files parse correctly.

---

## Step 2: Validate a Hypothesis Definition (No Data Needed)

This tests Stage 1 (Define) in isolation. It parses the YAML, checks the schema, and resolves metric references against the catalog. No database required.

```bash
uv run factor-research validate \
  -h backend/research/hypotheses/conversion_efficiency_ats_mispricing.yaml
```

**Expected output:**

```
Valid: conversion_efficiency_ats_mispricing
  Metrics: ['third_down_rate_std', 'yards_per_game_std']
  Classification: quartile on third_down_rate_std
  Outcome: ats
  Lookback: season_to_date
  Time windows: ['2014-2018', '2019-2021', '2022-2024']
```

If you see this, the research package is correctly installed and the hypothesis file is well-formed.

**Try validating another one:**

```bash
uv run factor-research validate \
  -h backend/research/hypotheses/defensive_underpricing.yaml
```

---

## Step 3: Explore the Metrics Catalog

See what metrics are available for hypothesis testing:

```bash
uv run factor-research list-metrics
```

This prints all 16 metrics with descriptions. These are the building blocks you can reference in hypothesis YAML files.

---

## Step 4: Ingest Your First Season of Live Data

Now the real work begins. We'll scrape one season from Pro Football Reference to test the full pipeline with real data before committing to the multi-hour full backfill.

**Start with the 2024 season** (most recent, 284 games):

```bash
uv run factor-ingest ingest --season 2024 -v
```

### What happens under the hood

1. The scraper fetches the 2024 schedule page from PFR
2. It parses out every game ID from the schedule
3. For each game, it fetches the box score page
4. The parser extracts: scores, team stats, third/fourth down data, penalties, turnovers, time of possession, and the Vegas closing spread
5. Data is cleaned (team abbreviations normalized, stat ranges validated)
6. Everything is upserted into a local SQLite database (`factor_research.db`)
7. Derived metrics (season-to-date and last-4-games rolling averages) are computed

### What to watch in the terminal output

```
Season 2024: found 284 games in schedule     ← Good: correct game count
...
Season 2024: ingested 280, skipped 4         ← Small skip count is normal
```

### How long it takes

At the default 3-second rate limit between requests: approximately **15-20 minutes** for one season. Each game page is cached in `.pfr_cache/` after download, so if the process crashes or you re-run, it picks up where it left off without re-downloading.

### If something goes wrong

- **Network error:** The scraper retries up to 3 times with backoff. If PFR is down or rate-limiting you, wait and retry.
- **Parsing errors for specific games:** The scraper logs errors but continues. A few skipped games is normal — some special games (e.g., international games, Pro Bowl) may have non-standard page formats.
- **`403 Forbidden`:** PFR may block aggressive scraping. The 3-second delay should prevent this, but if it happens, increase the delay: `PFR_REQUEST_DELAY=5 uv run factor-ingest ingest --season 2024`

---

## Step 5: Validate the Database

Confirm the data landed correctly:

```bash
uv run factor-ingest validate-db
```

**Expected output (for one season):**

```
Teams: 32 (expected 32)
Season 2024: 280 games (expected 284) [OK]
Games with NULL scores: 0
```

The game count might be slightly under 284 if a few games failed to parse. The `[OK]` status means it's within the ±2 tolerance.

### Quick manual sanity check

You can also poke at the database directly:

```bash
uv run python -c "
from backend.research.db import get_engine
from sqlalchemy import text
engine = get_engine()
with engine.connect() as conn:
    # How many games?
    r = conn.execute(text('SELECT COUNT(*) FROM games'))
    print(f'Total games: {r.scalar()}')

    # Sample a game
    r = conn.execute(text('SELECT game_id, home_team, away_team, home_score, away_score, home_spread_close FROM games LIMIT 3'))
    for row in r:
        print(row)

    # Check derived metrics exist
    r = conn.execute(text('SELECT COUNT(*) FROM derived_metrics'))
    print(f'Derived metric rows: {r.scalar()}')
"
```

You should see real game data — teams, scores, and spreads. The derived metrics row count should be approximately 2x the game count (one row per team per game).

---

## Step 6: Run Your First Hypothesis — FOR REAL

This is the moment. Run the conversion efficiency hypothesis against your live data:

```bash
uv run factor-research run \
  -h backend/research/hypotheses/conversion_efficiency_ats_mispricing.yaml \
  -v
```

### What the pipeline does

1. **Define:** Parses the YAML file
2. **Extract:** Queries the database for all 2024 regular season games (excluding Week 1), joins team stats with derived metrics, attaches ATS outcomes
3. **Classify:** Ranks each team-game by third_down_rate_std within the season, assigns to quartiles (Q1 = best converters, Q4 = worst)
4. **Measure:** Computes cover rates per quartile, runs binomial tests, calculates confidence intervals, compares Q1 vs Q4, scores quality
5. **Report:** Prints color-coded terminal output and writes JSON

### How to read the output

```
======================================================================
  conversion_efficiency_ats_mispricing
======================================================================
  Rows: 496  |  Seasons: [2024]

  Bucket      N   Cover%    p-raw    p-adj       CI 95%      Sig
  ------------------------------------------------------------
  Q1        124    52.4%   0.3921   0.5228   [0.433, 0.614]
  Q2        124    48.4%   0.6312   0.6312   [0.394, 0.574]
  Q3        124    50.8%   0.8714   0.8714   [0.418, 0.598]
  Q4        124    48.4%   0.6312   0.6312   [0.394, 0.574]

  Comparison: Q1 vs Q4
  Rate difference: +0.040  |  p-value: 0.5123  |  Cohen's h: 0.081

  Quality: LOW (1.45/3.00)
  [size=2.00 sig=1.00 eff=1.00 cons=1.00]
======================================================================
```

**What each column means:**

| Column | Meaning |
|--------|---------|
| Bucket | Q1 = top quartile (best converters), Q4 = bottom quartile |
| N | Number of team-game observations in this bucket |
| Cover% | What percentage of the time teams in this bucket covered the spread |
| p-raw | Raw p-value from binomial test (is this cover rate different from 50%?) |
| p-adj | FDR-adjusted p-value (corrected for testing multiple buckets) |
| CI 95% | 95% Wilson confidence interval for the true cover rate |
| Sig | `*` = significant (p < 0.05), `~` = suggestive (p < 0.10), blank = not significant |

**Quality grade breakdown:**

| Dimension | What it measures |
|-----------|-----------------|
| size | Are the sample sizes large enough? (need 200+ per bucket for HIGH) |
| sig | Is the result statistically significant? |
| eff | Is the effect size (Cohen's h) meaningful? |
| cons | Is the edge consistent across time windows? |

### Important: one season = small sample

With only one season of data (~500 team-game observations split across 4 quartiles), you will almost certainly see:

- **Quality: LOW or INSUFFICIENT_DATA** — this is correct and expected
- Wide confidence intervals that cross 50%
- Non-significant p-values

This does NOT mean the hypothesis is wrong. It means you don't have enough data yet to tell. This is the statistical rigor layer doing its job.

---

## Step 7: Add More Seasons

The hypothesis was designed to test across 2014-2024 (11 seasons, ~5,000+ observations). One season is just a smoke test. Now backfill:

```bash
# One at a time (recommended for your first backfill — lets you monitor)
uv run factor-ingest ingest --season 2023 -v
uv run factor-ingest ingest --season 2022 -v
uv run factor-ingest ingest --season 2021 -v
# ... continue through 2014

# Or all at once (will take 2-3 hours)
uv run factor-ingest ingest
```

**After each season**, you can re-run the hypothesis and watch the results change:

```bash
uv run factor-research run \
  -h backend/research/hypotheses/conversion_efficiency_ats_mispricing.yaml
```

As sample sizes grow, confidence intervals tighten, p-values become more informative, and the quality score improves. This is where you start to see whether the edge is real.

---

## Step 8: Run All 10 Hypotheses

Once you have multiple seasons loaded:

```bash
uv run factor-research run-all
```

This runs all 10 hypothesis YAML files in `backend/research/hypotheses/`, applies cross-hypothesis FDR correction (to account for the fact that you're testing multiple hypotheses simultaneously), and writes results for each.

To see only the high-confidence findings:

```bash
uv run factor-research run-all --high-confidence
```

This filters to only HIGH and MEDIUM quality grades.

---

## Step 9: Check Your Results

All outputs are saved in `backend/research/results/`:

```bash
ls backend/research/results/
```

You'll see:

- **JSON files** — one per hypothesis run, with full structured data (bucket stats, comparisons, quality scores, time window breakdowns). These are machine-readable for downstream analysis.
- **summary.csv** — one row per hypothesis run, appended after each execution. Open this in a spreadsheet to compare hypotheses side by side.

To inspect a JSON result:

```bash
cat backend/research/results/conversion_efficiency_ats_mispricing_*.json | python -m json.tool | head -50
```

---

## Step 10: Write Your Own Hypothesis

Once you're comfortable with the output, try creating your own hypothesis. Copy an existing YAML as a starting point:

```bash
cp backend/research/hypotheses/defensive_underpricing.yaml \
   backend/research/hypotheses/my_custom_hypothesis.yaml
```

Edit the new file. The key fields to change:

- `hypothesis_name` — unique snake_case name
- `classification.metric` — which metric to bucket on (run `list-metrics` to see options)
- `classification.type` — `quartile`, `percentile`, `binary`, or `custom`
- `outcome` — `ats` (against the spread), `su` (straight up), or `ou` (over/under)
- `comparison_buckets` — which buckets to compare head-to-head

Validate it before running:

```bash
uv run factor-research validate -h backend/research/hypotheses/my_custom_hypothesis.yaml
uv run factor-research run -h backend/research/hypotheses/my_custom_hypothesis.yaml -v
```

---

## Cheat Sheet

| What you want to do | Command |
|---|---|
| Install dependencies | `uv sync` |
| Run all tests | `uv run pytest -v` |
| Validate a hypothesis YAML | `uv run factor-research validate -h path/to/hypothesis.yaml` |
| List available metrics | `uv run factor-research list-metrics` |
| Ingest one season | `uv run factor-ingest ingest --season 2024 -v` |
| Ingest all seasons (2014-2024) | `uv run factor-ingest ingest` |
| Validate database integrity | `uv run factor-ingest validate-db` |
| Recompute derived metrics | `uv run factor-ingest recompute-derived` |
| Run one hypothesis | `uv run factor-research run -h path/to/hypothesis.yaml` |
| Run all hypotheses | `uv run factor-research run-all` |
| Run with high-confidence filter | `uv run factor-research run-all --high-confidence` |
| Verbose logging | Add `-v` to any command |

---

## What's Next

After you've ingested all 11 seasons and run the full backlog:

1. **Read the results.** Which hypotheses show signal? Which are noise? Look at the quality grades and the time window trends.
2. **Create new hypotheses.** Use what you learn to formulate new questions. The framework is designed to make testing cheap and fast once data is loaded.
3. **Look for compound factors.** The most interesting edges often come from combining two factors (e.g., "efficient offense meets elite defense"). The existing hypotheses include some compound tests (H4, H9).
4. **Watch the time windows.** If a factor's edge is declining in the most recent window (2022-2024), it may be getting priced in by the market. If it's increasing, the market may be slow to adjust.
