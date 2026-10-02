"""openai-radar CLI, plus Pump login.

``cli.py`` is the scanner. This module loads that app and adds ``login``,
``logout``, and ``status`` so the authorization-code + PKCE flow is a
subcommand:

    python cli_login_app.py login
    python cli_login_app.py logout
    python cli_login_app.py status

``python pump_login.py login`` does the same login without loading the scanner.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import typer

from pump_login import LoginError, clear_credentials, load_credentials
from pump_login import login as pump_login


def _load_app() -> typer.Typer:
    path = Path(__file__).resolve().with_name("cli.py")
    spec = importlib.util.spec_from_file_location("openai_radar_cli", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app = module.app
    if not isinstance(app, typer.Typer):
        raise SystemExit(f"{path} did not define a Typer app")
    return app


app = _load_app()


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
        raise typer.Exit(code=1)


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
        raise typer.Exit(code=1)
    if creds is None:
        typer.echo("Not logged in. Run `openai-radar login`.")
        raise typer.Exit(code=1)
    upload = f" upload {creds.upload_id}" if creds.upload_id else ""
    typer.echo(f"Logged in to {creds.api_base}.{upload} Token expires {creds.expires_at}.")


if __name__ == "__main__":
    app(prog_name="openai-radar")
