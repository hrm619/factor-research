"""CLI entry point for factor research."""

from __future__ import annotations

import logging

import click

from backend.research.metrics_catalog import MetricsCatalog


@click.group()
@click.option("--db-url", default=None, help="Database URL (default: sqlite:///factor_research.db)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--contracts-dir",
    default=None,
    help="Contracts directory (default: ~/.fin-arb/contracts)",
)
@click.pass_context
def main(ctx, db_url, verbose, contracts_dir):
    """Factor Research hypothesis testing CLI."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["db_url"] = db_url
    if contracts_dir is None:
        from pathlib import Path
        contracts_dir = str(Path.home() / ".fin-arb" / "contracts")
    ctx.obj["contracts_dir"] = contracts_dir


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


@main.command("export-edges")
@click.option(
    "--output", "-o", default=None,
    help="Output path (default: $CONTRACTS_DIR/edges/<domain>_edges.json)",
)
@click.option("--hypothesis", "-h", "hyp_name", default=None, help="Export only this hypothesis")
@click.option("--results-dir", default="backend/research/results", help="Results directory")
@click.option("--hypotheses-dir", default="backend/research/hypotheses", help="Hypotheses directory")
@click.option("--domain", default="nfl", help="Domain name")
@click.pass_context
def export_edges(ctx, output, hyp_name, results_dir, hypotheses_dir, domain):
    """Export validated edges to the edge registry JSON."""
    from backend.research.contract_export import export_edges as _export_edges

    if output is None:
        from pathlib import Path
        contracts_dir = ctx.obj.get("contracts_dir", str(Path.home() / ".fin-arb" / "contracts"))
        output = str(Path(contracts_dir) / "edges" / f"{domain}_edges.json")

    registry = _export_edges(
        results_dir=results_dir,
        hypotheses_dir=hypotheses_dir,
        output_path=output,
        hypothesis_filter=hyp_name,
        domain=domain,
    )
    n_edges = len(registry.get("edges", []))
    click.echo(click.style(f"Exported {n_edges} edges to {output}", fg="green"))


@main.command("import")
@click.option("--contract", "-c", required=True, help="Path to Contract 1 JSON file")
@click.option(
    "--output-dir", "-o", default="backend/research/hypotheses",
    help="Output directory for generated YAML",
)
@click.option("--dry-run", is_flag=True, help="Validate only, do not write YAML")
def import_contract_cmd(contract, output_dir, dry_run):
    """Import a hypothesis from a Contract 1 JSON file."""
    from backend.research.contract_import import import_contract
    from backend.research.models import HypothesisValidationError

    try:
        yaml_path = import_contract(contract, output_dir, dry_run=dry_run)
    except (ValueError, HypothesisValidationError) as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1)

    if dry_run:
        click.echo(click.style("Validation passed (dry run)", fg="green"))
        click.echo(f"Would write to: {yaml_path}")
    else:
        click.echo(click.style(f"Hypothesis imported: {yaml_path}", fg="green"))


@main.command("list-metrics")
def list_metrics():
    """List all available metrics in the catalog."""
    catalog = MetricsCatalog()
    for name in catalog.list_metrics():
        metric = catalog.get_metric(name)
        click.echo(f"  {name:<40} {metric.description[:60]}")
