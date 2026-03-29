# Phase 1 Spec Amendments

Amendments to `factor_research_phase1_spec.docx` agreed before implementation.

## A1: Game Count Validation (Section 10.1)

The spec states "exactly 267 games" per season. Correct per-era counts:

| Seasons    | Regular | Playoff | Total |
|------------|---------|---------|-------|
| 2014–2019  | 256     | 10      | 266   |
| 2020       | 256     | 12      | 268*  |
| 2021–2024  | 272     | 12      | 284   |

\* Minus COVID cancellations. The wildcard round expanded in 2020 (6→7 playoff teams per conference, +2 games). The 17-game regular season started in 2021.

Validation should use per-era expected counts with a tolerance of ±2.

## A2: Spread Data Source Strategy (Sections 2.6, 2.8)

- **Primary source:** PFR game pages (already scraped for stats — single source, single parse).
- **Fallback:** Games with NULL spreads are flagged in `ingestion_log`. Manual backfill from covers.com only if the gap exceeds the 2% missing-data threshold.
- **No automated covers.com scraping in Phase 1.** Different site structure and rate-limiting concerns make it a separate effort.
- **Early validation step:** Confirm PFR spread coverage for 2014–2024 at the start of ingestion implementation. If coverage is patchy, revisit this decision.

## A3: FDR Correction for Multiple Comparisons (Section 6)

Add Benjamini-Hochberg FDR correction to the statistical pipeline:

- **Within-hypothesis:** After all per-bucket binomial tests, apply FDR correction across the bucket-level p-values. Add `p_value_adjusted` field to `BucketStats` alongside raw `p_value`.
- **Cross-hypothesis:** When running the full hypothesis backlog, apply a second FDR pass across hypothesis-level results in the Report stage.
- **Quality score** uses the adjusted p-value, not the raw p-value.
- **Both raw and adjusted p-values** are reported — raw for exploration, adjusted for conclusions.
- **Implementation:** `statsmodels.stats.multitest.multipletests(method='fdr_bh')`.
