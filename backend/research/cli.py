"""CLI entry point for factor research."""

from __future__ import annotations

import logging

import click

from backend.research.metrics_catalog import MetricsCatalog


@click.group()
@click.option("--db-url", default=None, help="Database URL (default: sqlite:///factor_research.db)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx, db_url, verbose):
    """Factor Research hypothesis testing CLI."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["db_url"] = db_url


@main.command()
@click.option("--hypothesis", "-h", required=True, help="Path to hypothesis YAML file")
@click.option("--output-dir", "-o", default="backend/research/results", help="Output directory")
@click.option("--high-confidence", is_flag=True, help="Show only HIGH/MEDIUM quality results")
@click.pass_context
def run(ctx, hypothesis, output_dir, high_confidence):
    """Run a single hypothesis through the full pipeline."""
    from backend.research.harness import run_hypothesis
    run_hypothesis(
        hypothesis, db_url=ctx.obj["db_url"],
        output_dir=output_dir, high_confidence=high_confidence,
    )


@main.command("run-all")
@click.option("--hypothesis-dir", "-d", default="backend/research/hypotheses", help="Hypothesis directory")
@click.option("--output-dir", "-o", default="backend/research/results", help="Output directory")
@click.option("--high-confidence", is_flag=True, help="Show only HIGH/MEDIUM quality results")
@click.pass_context
def run_all(ctx, hypothesis_dir, output_dir, high_confidence):
    """Run all hypotheses in a directory."""
    from backend.research.harness import run_all_hypotheses
    results = run_all_hypotheses(
        hypothesis_dir, db_url=ctx.obj["db_url"],
        output_dir=output_dir, high_confidence=high_confidence,
    )
    click.echo(f"\nCompleted {len(results)} hypotheses.")


@main.command()
@click.option("--hypothesis", "-h", required=True, help="Path to hypothesis YAML file")
def validate(hypothesis):
    """Validate a hypothesis YAML file without running it."""
    from backend.research.define import load_hypothesis, validate_hypothesis, resolve_metrics
    from backend.research.models import HypothesisValidationError

    try:
        raw = load_hypothesis(hypothesis)
        definition = validate_hypothesis(raw)
        catalog = MetricsCatalog()
        resolve_metrics(definition, catalog)
        click.echo(f"Valid: {definition.hypothesis_name}")
        click.echo(f"  Metrics: {definition.metrics}")
        click.echo(f"  Classification: {definition.classification.type} on {definition.classification.metric}")
        click.echo(f"  Outcome: {definition.outcome}")
        click.echo(f"  Lookback: {definition.lookback}")
        if definition.time_windows:
            click.echo(f"  Time windows: {[tw.label for tw in definition.time_windows]}")
    except (HypothesisValidationError, KeyError) as e:
        click.echo(f"Invalid: {e}", err=True)
        raise SystemExit(1)


@main.command("list-metrics")
def list_metrics():
    """List all available metrics in the catalog."""
    catalog = MetricsCatalog()
    for name in catalog.list_metrics():
        metric = catalog.get_metric(name)
        click.echo(f"  {name:<40} {metric.description[:60]}")
