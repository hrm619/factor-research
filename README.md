# NFL Factor Research

A framework for backtesting game-outcome hypotheses against historical NFL data (2014-2024). Tests whether measurable team attributes predict against-the-spread (ATS) performance.

Part of the broader fin-arb project.

---

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
uv sync
```

### Verify

```bash
uv run pytest                    # 170 tests
uv run factor-research --help
uv run factor-ingest --help
```

---

## Data Ingestion

Game data is sourced from [nflverse](https://github.com/nflverse) via the `nflreadpy` package. No API keys or web scraping required — data is fetched directly as structured DataFrames.

### Ingest all seasons (2014-2024)

```bash
uv run factor-ingest ingest
```

### Ingest a single season

```bash
uv run factor-ingest ingest --season 2024
```

### Validate ingested data

```bash
uv run factor-ingest validate-db
```

### Recompute derived metrics without re-ingesting

```bash
uv run factor-ingest recompute-derived
```

Data sources per season:
- **Schedule + scores + spreads**: `nflreadpy.load_schedules()`
- **Team game stats** (passing, rushing, penalties, sacks, turnovers): `nflreadpy.load_team_stats()`
- **Play-by-play aggregation** (3rd/4th down conversions, red zone, time of possession): `nflreadpy.load_pbp()`

Ingesting a single season takes ~5 seconds. All 11 seasons takes ~1 minute.

---

## Research Pipeline

### Run all hypotheses

```bash
uv run factor-research run-all
```

### Run a single hypothesis

```bash
uv run factor-research run -h backend/research/hypotheses/third_down_rate_meta_shift.yaml
```

### Validate a hypothesis YAML

```bash
uv run factor-research validate -h path/to/hypothesis.yaml
```

### List available metrics

```bash
uv run factor-research list-metrics
```

### Pipeline stages

```
YAML -> Define -> Extract -> Classify -> Measure -> Report
```

1. **Define** — parse hypothesis YAML into a HypothesisDefinition
2. **Extract** — query DB, attach outcomes (ATS cover, win, over/under)
3. **Classify** — bucket observations by metric (quartile/percentile/binary/custom)
4. **Measure** — compute stats, FDR correction, quality scoring
5. **Report** — terminal output + JSON results

Results are written to `backend/research/results/` (gitignored).

### Cross-repo integration (data contracts)

```bash
# Import a Contract 1 hypothesis from research-assistant
uv run factor-research import --contract ~/.fin-arb/contracts/hypotheses/name.json

# Export validated edges as Contract 2 for fin-arb
uv run factor-research export-edges --output ~/.fin-arb/contracts/edges/nfl_edges.json
```

---

## Hypotheses

10+ hypotheses in `backend/research/hypotheses/` (new ones can be imported via Contract 1):

| File | Factor | Outcome |
|---|---|---|
| `third_down_rate_meta_shift.yaml` | 3rd down conversion rate | ATS |
| `conversion_efficiency_ats_mispricing.yaml` | 3rd down rate (STD) | ATS |
| `conversion_efficiency_lookback_comparison.yaml` | 3rd down rate (L4 vs STD) | ATS |
| `defensive_efficiency_compound.yaml` | Points allowed + yards allowed | ATS |
| `defensive_underpricing.yaml` | Points allowed per game | ATS |
| `fourth_down_aggressiveness.yaml` | 4th down attempt rate | ATS |
| `penalty_rate_coaching_quality.yaml` | Penalty rate | ATS |
| `red_zone_efficiency_divergence.yaml` | Red zone TD rate | ATS |
| `sack_rate_differential.yaml` | Sack rate | ATS |
| `turnover_margin_mean_reversion.yaml` | Turnover margin | ATS |

---

## Architecture

```
backend/
  models/research_models.py    # SQLAlchemy ORM (Game, Team, TeamGameStats, DerivedMetrics, IngestionLog)
  research/
    models.py                  # Dataclasses (HypothesisDefinition, BucketStats, MeasurementResult, etc.)
    db.py                      # Engine/session helpers, init_db()
    define.py                  # Stage 1: YAML -> HypothesisDefinition
    extract.py                 # Stage 2: DB query -> DataFrame with outcomes
    classify.py                # Stage 3: Bucket observations
    measure.py                 # Stage 4: Stats, FDR correction, quality scoring
    report.py                  # Stage 5: Terminal + JSON output
    harness.py                 # Pipeline orchestrator
    statistical.py             # Pure stat functions (binomial, z-test, chi-sq, Wilson CI, Cohen's h, FDR)
    metrics_catalog.py         # Metric registry (16 metrics)
    cli.py                     # Click CLI entry point
    contract_import.py         # Contract 1 JSON -> hypothesis YAML
    contract_export.py         # Results -> Contract 2 edge registry
    hypotheses/                # 10+ hypothesis YAML files
    results/                   # Output (gitignored)
  ingestion/
    nflverse_source.py         # nflreadpy data fetching (schedule, team stats, PBP aggregation)
    cleaning.py                # Team abbr normalization, stat range validation
    derived_metrics.py         # STD/L4 metric computation (no look-ahead)
    ingest_cli.py              # Click CLI entry point
    config.py                  # Constants (seasons, game type mapping, stat ranges)
tests/                         # 170 tests mirroring module structure
```
