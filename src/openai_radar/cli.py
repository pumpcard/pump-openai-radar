"""pump-openai-radar CLI."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from openai_radar import __version__
from openai_radar.client import RadarClient, RadarError
from openai_radar.findings import Finding
from openai_radar.models.base import ModelUsage
from openai_radar.pump_login import LoginError, clear_credentials, load_credentials
from openai_radar.pump_login import login as pump_login
from openai_radar.runner import RunConfig, Runner, RunResult

console = Console()

app = typer.Typer(
    add_completion=False,
    help="OpenAI Radar — infrastructure FinOps scanner. Part of the Hyperscaler Radar suite.",
)

SEVERITY_STYLE = {
    "critical": "bold white on red",
    "high": "bold red",
    "medium": "yellow",
    "low": "cyan",
    "info": "dim",
}

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _build_client(api_key: str | None, admin_key: str | None, project: str | None) -> RadarClient:
    try:
        return RadarClient(api_key=api_key, admin_key=admin_key, project_id=project)
    except RadarError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=2)


def _render_inventory(result: RunResult) -> None:
    table = Table(
        title="📡  Inventory",
        title_justify="left",
        header_style="bold magenta",
    )
    table.add_column("Resource")
    table.add_column("Count", justify="right")
    table.add_column("Notes")

    total_gb = sum(vs.usage_gb for vs in result.vector_stores)
    failed_ft = sum(1 for ft in result.fine_tunes if ft.status == "failed")
    failed_batch = sum(bj.request_counts_failed for bj in result.batch_jobs)
    total_tokens = sum(u.total_tokens for u in result.usage)

    table.add_row("Assistants", str(len(result.assistants)), "")
    table.add_row("Vector stores", str(len(result.vector_stores)), f"{total_gb:.2f} GB total")
    table.add_row("Fine-tunes", str(len(result.fine_tunes)), f"{failed_ft} failed")
    table.add_row("Batch jobs", str(len(result.batch_jobs)), f"{failed_batch:,} failed requests")
    table.add_row("Usage records", str(len(result.usage)), f"{total_tokens:,} tokens")
    table.add_row("Relationships", str(len(result.relationships)), "")

    console.print(table)


def _render_relationships(result: RunResult) -> None:
    if not result.relationships:
        return
    table = Table(
        title="🔗  External service relationships",
        title_justify="left",
        header_style="bold magenta",
    )
    table.add_column("Assistant")
    table.add_column("Service", width=12)
    table.add_column("Signal", width=18)
    table.add_column("Evidence", overflow="fold")

    for rel in result.relationships:
        table.add_row(
            rel.source_name or rel.source_id,
            rel.kind.value,
            rel.signal,
            rel.evidence,
        )
    console.print(table)


def _render_findings(findings: list[Finding]) -> None:
    if not findings:
        console.print("[green]No findings — nothing flagged against the current rule set.[/green]")
        return

    ranked = sorted(findings, key=lambda f: SEVERITY_ORDER.get(f.severity.value, 9))

    table = Table(
        title="🔎  Findings",
        title_justify="left",
        header_style="bold magenta",
    )
    table.add_column("Sev", width=8)
    table.add_column("Rule", width=10)
    table.add_column("Resource", width=24, overflow="ellipsis")
    table.add_column("Finding", overflow="fold")

    for f in ranked:
        style = SEVERITY_STYLE.get(f.severity.value, "")
        table.add_row(
            f"[{style}]{f.severity.value.upper()}[/{style}]",
            f.rule_id,
            f.resource_id,
            f"{f.title}\n[dim]{f.recommendation}[/dim]",
        )

    console.print(table)

    counts: dict[str, int] = {}
    for f in ranked:
        counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
    summary = "  ".join(
        f"[{SEVERITY_STYLE.get(s, '')}]{c} {s}[/{SEVERITY_STYLE.get(s, '')}]"
        for s, c in sorted(counts.items(), key=lambda kv: SEVERITY_ORDER.get(kv[0], 9))
    )
    console.print(f"\n{summary}\n")


def _resolve_pump_upload(
    *,
    upload: bool,
    upload_token: str | None,
    api_base: str | None,
) -> tuple[str | None, str]:
    """Return the token and API base for this run.

    ``--upload-token`` wins. ``--upload`` uses the token stored by
    ``pump-openai-radar login``. An explicit ``--api-base`` (or ``PUMP_API_BASE``)
    overrides the base saved at login.
    """
    from openai_radar.pump_login import (
        DEFAULT_API_BASE,
        LoginError,
        load_credentials,
        token_is_expired,
    )

    if upload_token:
        return upload_token, api_base or DEFAULT_API_BASE
    if not upload:
        return None, api_base or DEFAULT_API_BASE

    try:
        creds = load_credentials()
    except LoginError as exc:
        console.print(f"[bold red]Upload failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    if creds is None:
        console.print(
            "[bold red]Upload failed:[/bold red] Not logged in. "
            "Run `pump-openai-radar login`, or pass --upload-token."
        )
        raise typer.Exit(code=1)
    if token_is_expired(creds):
        console.print(
            "[bold red]Upload failed:[/bold red] Pump login expired. "
            "Run `pump-openai-radar login` again."
        )
        raise typer.Exit(code=1)
    return creds.access_token, api_base or creds.api_base


def _report_destination(report_file: str | None, csv_dir: str | None) -> Path:
    if report_file:
        return Path(report_file)
    if csv_dir:
        return Path(csv_dir) / "report.csv"
    return Path("report.csv")


def _write_and_maybe_upload_report(
    client: RadarClient,
    *,
    lookback_days: int,
    project_id: str | None,
    upload_token: str | None,
    api_base: str,
    report_file: str | None,
    csv_dir: str | None,
    usage: list[ModelUsage],
) -> None:
    """Write the cost report and usage CSV, and PUT them to Pump when a token is set.

    Costs upload as role ``billing``. Usage uploads as role ``inventory``.
    An empty usage list still writes and uploads a header-only inventory file,
    so Pump can tell the scan finished with no usage rows.
    """
    from openai_radar.scanners.report import ReportError, fetch_cost_report, write_report_csv
    from openai_radar.scanners.usage import write_usage_csv
    from openai_radar.upload import UploadError, upload_csvs

    destination = _report_destination(report_file, csv_dir)
    try:
        report = asyncio.run(
            fetch_cost_report(client, lookback_days=lookback_days, project_id=project_id)
        )
    except (ReportError, RadarError) as exc:
        console.print(f"[bold red]Report failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if report.truncated:
        console.print("[yellow]Cost report may be incomplete.[/yellow]")

    written = write_report_csv(destination, report.rows)
    console.print(f"[green]Wrote[/green] {written}")

    # Always upload inventory, including a header-only file when the scan
    # found no usage. Pump starts analysis only after both objects exist.
    usage_path = write_usage_csv(destination.with_name("usage.csv"), usage)
    console.print(f"[green]Wrote[/green] {usage_path}")
    files = {"billing": str(written), "inventory": str(usage_path)}

    if not upload_token:
        return

    if not usage:
        console.print("[yellow]No usage rows; uploading an empty inventory file.[/yellow]")

    console.print(f"Uploading to Pump ({api_base})")
    try:
        upload_csvs(api_base=api_base, token=upload_token, files=files)
    except UploadError as exc:
        console.print(f"[bold red]Upload failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print("[green]Your OpenAI cost and usage data is on its way to Pump.[/green]")


def _payload(result: RunResult) -> dict:
    return {
        "assistants": [a.model_dump(mode="json") for a in result.assistants],
        "vector_stores": [v.model_dump(mode="json") for v in result.vector_stores],
        "fine_tunes": [f.model_dump(mode="json") for f in result.fine_tunes],
        "batch_jobs": [b.model_dump(mode="json") for b in result.batch_jobs],
        "usage": [u.model_dump(mode="json") for u in result.usage],
        "relationships": [r.model_dump(mode="json") for r in result.relationships],
        "findings": [f.as_dict() for f in result.findings],
    }


@app.command()
def run(
    project: str | None = typer.Option(
        None, "--project", help="Scope the scan to this project ID."
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="Overrides OPENAI_API_KEY."),
    admin_key: str | None = typer.Option(
        None, "--admin-key", help="Overrides OPENAI_ADMIN_KEY. Unlocks org-wide usage data."
    ),
    lookback: int = typer.Option(30, "--lookback", help="Days of usage history."),
    output: str = typer.Option("table", "--output", "-o", help="table | json"),
    out_file: str | None = typer.Option(None, "--out-file", help="Write the JSON payload here."),
    csv_dir: str | None = typer.Option(None, "--csv-dir", help="Write per-resource CSVs here."),
    drawio_file: str | None = typer.Option(
        None, "--drawio-file", help="Write a draw.io architecture diagram here."
    ),
    upload: bool = typer.Option(
        False,
        "--upload",
        help="Upload costs as billing and usage as inventory, using `pump-openai-radar login`.",
    ),
    upload_token: str | None = typer.Option(
        None,
        "--upload-token",
        help="Pump upload token. Overrides the token stored by `pump-openai-radar login`.",
    ),
    api_base: str | None = typer.Option(
        None,
        "--api-base",
        envvar="PUMP_API_BASE",
        help="Pump API origin. Overrides the base stored by login.",
    ),
    report_file: str | None = typer.Option(
        None,
        "--report-file",
        help="Write the cost report CSV here. With --upload, defaults to report.csv.",
    ),
) -> None:
    """Scan an OpenAI organization."""
    if output not in ("table", "json"):
        console.print(f"[bold red]--output must be 'table' or 'json', got '{output}'.[/bold red]")
        raise typer.Exit(code=2)

    client = _build_client(api_key, admin_key, project)
    token, pump_base = _resolve_pump_upload(
        upload=upload, upload_token=upload_token, api_base=api_base
    )
    config = RunConfig(project_id=project, usage_lookback_days=lookback)

    started = time.time()
    try:
        result = asyncio.run(Runner.run(client, config))
    except RadarError as exc:
        console.print(f"[bold red]Scan failed:[/bold red] {exc}")
        raise typer.Exit(code=1)
    elapsed = time.time() - started

    if output == "json":
        text = json.dumps(_payload(result), indent=2, default=str)
        if out_file:
            with open(out_file, "w", encoding="utf-8") as fh:
                fh.write(text)
            console.print(f"[green]Wrote[/green] {out_file}")
        else:
            sys.stdout.write(text + "\n")
    else:
        _render_inventory(result)
        _render_relationships(result)
        _render_findings(result.findings)
        if out_file:
            with open(out_file, "w", encoding="utf-8") as fh:
                json.dump(_payload(result), fh, indent=2, default=str)
            console.print(f"[green]Wrote[/green] {out_file}")

    if csv_dir:
        for path in result.export_csv(csv_dir):
            console.print(f"[green]Wrote[/green] {path}")

    if drawio_file:
        console.print(f"[green]Wrote[/green] {result.export_drawio(drawio_file)}")

    if token or report_file:
        _write_and_maybe_upload_report(
            client,
            lookback_days=lookback,
            project_id=project,
            upload_token=token,
            api_base=pump_base,
            report_file=report_file,
            csv_dir=csv_dir,
            usage=result.usage,
        )

    if output != "json":
        console.print(f"[dim]Scan complete in {elapsed:.2f}s[/dim]")


@app.command()
def findings(
    project: str | None = typer.Option(
        None, "--project", help="Scope the scan to this project ID."
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="Overrides OPENAI_API_KEY."),
    admin_key: str | None = typer.Option(None, "--admin-key", help="Overrides OPENAI_ADMIN_KEY."),
    lookback: int = typer.Option(30, "--lookback", help="Days of usage history."),
    output: str = typer.Option("table", "--output", "-o", help="table | json"),
) -> None:
    """Scan, then print only the findings table."""
    client = _build_client(api_key, admin_key, project)
    config = RunConfig(project_id=project, usage_lookback_days=lookback)

    try:
        result = asyncio.run(Runner.run(client, config))
    except RadarError as exc:
        console.print(f"[bold red]Scan failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if output == "json":
        sys.stdout.write(
            json.dumps([f.as_dict() for f in result.findings], indent=2, default=str) + "\n"
        )
    else:
        _render_findings(result.findings)


@app.command()
def login(
    api_base: str | None = typer.Option(
        None,
        "--api-base",
        help="Pump API origin. Defaults to $PUMP_API_BASE or https://api.pump.co.",
    ),
    app_base: str | None = typer.Option(
        None,
        "--app-base",
        help="Pump app origin. Defaults to $PUMP_APP_BASE or https://app.pump.co.",
    ),
) -> None:
    """Log in with Pump (OAuth 2.0 authorization code + PKCE) and store the token."""
    try:
        pump_login(api_base=api_base, app_base=app_base)
    except LoginError as exc:
        typer.echo(f"Login failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def logout() -> None:
    """Forget the Pump token stored by `login`."""
    typer.echo("Logged out." if clear_credentials() else "Not logged in.")


@app.command()
def status() -> None:
    """Show whether a Pump token is stored, without printing the token."""
    try:
        creds = load_credentials()
    except LoginError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if creds is None:
        typer.echo("Not logged in. Run `pump-openai-radar login`.")
        raise typer.Exit(code=1)
    upload = f" upload {creds.upload_id}" if creds.upload_id else ""
    typer.echo(f"Logged in to {creds.api_base}.{upload} Token expires {creds.expires_at}.")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"pump-openai-radar {__version__}")


def main() -> None:
    app(prog_name="pump-openai-radar")


if __name__ == "__main__":
    main()
