"""Log in to Pump from the openai-radar CLI.

OAuth 2.0 authorization code with PKCE (S256), the same shape `gh` and the
Stripe CLI use for a public client that cannot keep a secret:

1. Bind ``127.0.0.1`` on an ephemeral port.
2. Open the Pump app so the user can approve the login in the browser.
3. Receive a one-time code on the loopback callback.
4. Exchange the code and the code verifier at the Pump API for an upload token.

The verifier never leaves this process. The browser only sees the code.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlsplit

CLIENT_ID = "openai-radar"
SCOPE = "radar"
DEFAULT_API_BASE = "https://api.pump.co"
DEFAULT_APP_BASE = "https://app.pump.co"
USER_AGENT = "openai-radar/2.1.0"
LOGIN_TIMEOUT_SECONDS = 180


class LoginError(Exception):
    """The user can act on this: retry login, or fix the URL they were shown."""


@dataclass(frozen=True)
class PumpCredentials:
    access_token: str
    token_type: str
    expires_at: str
    upload_id: str | None
    scope: str
    api_base: str


def credentials_path() -> Path:
    override = os.environ.get("OPENAI_RADAR_CONFIG_DIR")
    if override:
        return Path(override) / "credentials.json"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    root = Path(xdg) if xdg else Path.home() / ".config"
    return root / "openai-radar" / "credentials.json"


def generate_pkce() -> tuple[str, str]:
    """Return ``(code_verifier, S256 code_challenge)``."""
    # token_urlsafe(64) is 86 characters, inside the RFC 7636 43–128 range,
    # and its alphabet is a subset of the allowed verifier characters.
    verifier = secrets.token_urlsafe(64)
    return verifier, code_challenge_s256(verifier)


def code_challenge_s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def build_authorize_url(
    *,
    app_base: str,
    redirect_uri: str,
    code_challenge: str,
    state: str,
) -> str:
    query = urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "scope": SCOPE,
        }
    )
    return app_base.rstrip("/") + "/api/oauth/cli?" + query


def _success_page() -> bytes:
    return (
        b"<!DOCTYPE html><html><head><meta charset='utf-8'><title>openai-radar</title></head>"
        b'<body style="font-family: system-ui, sans-serif; max-width: 32rem; margin: 4rem auto;">'
        b"<h1>You are logged in</h1><p>Return to the terminal. You can close this window.</p>"
        b"</body></html>"
    )


class _CallbackRequestHandler(BaseHTTPRequestHandler):
    server: "_CallbackServer"

    def do_GET(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        if parts.path != "/callback":
            self.send_error(404)
            return
        # Do not log the query string: it contains the authorization code.
        self.server.query = parse_qs(parts.query)
        body = _success_page()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        self.server.done.set()

    def log_message(self, fmt: str, *args: object) -> None:
        return


class _CallbackServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _CallbackRequestHandler)
        self.query: dict[str, list[str]] = {}
        self.done = threading.Event()


def _start_callback_server() -> tuple[_CallbackServer, threading.Thread]:
    server = _CallbackServer()
    thread = threading.Thread(target=server.serve_forever, name="openai-radar-login", daemon=True)
    thread.start()
    return server, thread


def exchange_code(
    *,
    api_base: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> PumpCredentials:
    """POST the code and verifier to the Pump token endpoint.

    Uses urllib with an explicit User-Agent. Cloudflare in front of api.pump.co
    rejects the default Python agent, same as the aws-radar uploader.
    """
    body = urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": CLIENT_ID,
            "code_verifier": code_verifier,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        api_base.rstrip("/") + "/api/v1/oauth/token",
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        description = _error_description(exc.read())
        raise LoginError(f"Pump rejected the login ({exc.code}): {description}") from exc
    except urllib.error.URLError as exc:
        raise LoginError(f"Could not reach Pump at {api_base}: {exc.reason}") from exc

    if status != 200:
        raise LoginError(f"Pump rejected the login ({status}): {_error_description(raw)}")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise LoginError("Pump returned a token response that was not JSON.") from exc

    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise LoginError("Pump token response did not include an access_token.")

    expires_in = int(payload.get("expires_in") or 0)
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
    upload_id = payload.get("upload_id")
    return PumpCredentials(
        access_token=token,
        token_type=str(payload.get("token_type") or "Bearer"),
        expires_at=expires_at,
        upload_id=upload_id if isinstance(upload_id, str) else None,
        scope=str(payload.get("scope") or SCOPE),
        api_base=api_base.rstrip("/"),
    )


def _error_description(raw: bytes) -> str:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "the token endpoint did not return JSON"
    if isinstance(payload, dict):
        description = payload.get("error_description") or payload.get("detail")
        if isinstance(description, str) and description:
            return description
    return "the token endpoint rejected the authorization code"


def save_credentials(creds: PumpCredentials, path: Path | None = None) -> Path:
    destination = path or credentials_path()
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(destination.parent, 0o700)
    payload = {
        "access_token": creds.access_token,
        "token_type": creds.token_type,
        "expires_at": creds.expires_at,
        "upload_id": creds.upload_id,
        "scope": creds.scope,
        "api_base": creds.api_base,
    }
    # Write as the owner-only file from the start, then chmod again in case
    # the umask created it more loosely.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(destination, flags, 0o600)
    try:
        os.write(fd, json.dumps(payload, indent=2).encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(destination, 0o600)
    return destination


def load_credentials(path: Path | None = None) -> PumpCredentials | None:
    destination = path or credentials_path()
    if not destination.is_file():
        return None
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoginError(f"Could not read {destination}: {exc}") from exc
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise LoginError(f"{destination} does not contain a token. Run `openai-radar login` again.")
    upload_id = payload.get("upload_id")
    return PumpCredentials(
        access_token=token,
        token_type=str(payload.get("token_type") or "Bearer"),
        expires_at=str(payload.get("expires_at") or ""),
        upload_id=upload_id if isinstance(upload_id, str) else None,
        scope=str(payload.get("scope") or SCOPE),
        api_base=str(payload.get("api_base") or DEFAULT_API_BASE),
    )


def clear_credentials(path: Path | None = None) -> bool:
    destination = path or credentials_path()
    if not destination.is_file():
        return False
    destination.unlink()
    return True


def login(
    *,
    api_base: str | None = None,
    app_base: str | None = None,
    open_browser: Callable[[str], None] | None = None,
    credentials_file: Path | None = None,
    timeout_seconds: float = LOGIN_TIMEOUT_SECONDS,
    stdout: Callable[[str], None] = print,
) -> PumpCredentials:
    """Run the browser login and store the Pump token."""
    api = (api_base or os.environ.get("PUMP_API_BASE") or DEFAULT_API_BASE).rstrip("/")
    app = (app_base or os.environ.get("PUMP_APP_BASE") or DEFAULT_APP_BASE).rstrip("/")
    opener = open_browser or (lambda url: webbrowser.open(url))

    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(32)
    server, thread = _start_callback_server()
    try:
        port = server.server_address[1]
        redirect_uri = f"http://127.0.0.1:{port}/callback"
        authorize_url = build_authorize_url(
            app_base=app,
            redirect_uri=redirect_uri,
            code_challenge=challenge,
            state=state,
        )
        stdout("Opening your browser to log in to Pump.")
        stdout("If it does not open, visit:")
        stdout(f"  {authorize_url}")
        opener(authorize_url)
        if not server.done.wait(timeout_seconds):
            raise LoginError("Timed out waiting for the browser login.")

        query = server.query
        if "error" in query:
            description = (
                query.get("error_description") or query.get("error") or ["login was denied"]
            )[0]
            raise LoginError(description)
        returned_state = (query.get("state") or [""])[0]
        if not secrets.compare_digest(returned_state, state):
            raise LoginError(
                "Login response failed the state check. Run `openai-radar login` again."
            )
        code = (query.get("code") or [""])[0]
        if not code:
            raise LoginError("Login response did not include an authorization code.")

        creds = exchange_code(
            api_base=api,
            code=code,
            redirect_uri=redirect_uri,
            code_verifier=verifier,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    destination = save_credentials(creds, credentials_file)
    stdout(f"Logged in. Token expires {creds.expires_at}.")
    stdout(f"Saved credentials to {destination}")
    return creds


def main(argv: list[str] | None = None) -> int:
    """``python pump_login.py login|logout|status`` for shells that invoke this module directly."""
    import argparse

    parser = argparse.ArgumentParser(prog="openai-radar")
    sub = parser.add_subparsers(dest="command", required=True)

    login_parser = sub.add_parser("login", help="Log in with Pump and store the upload token.")
    login_parser.add_argument("--api-base", default=None)
    login_parser.add_argument("--app-base", default=None)
    sub.add_parser("logout", help="Forget the stored Pump token.")
    sub.add_parser("status", help="Show whether a Pump token is stored.")

    args = parser.parse_args(argv)
    try:
        if args.command == "login":
            login(api_base=args.api_base, app_base=args.app_base)
        elif args.command == "logout":
            print("Logged out." if clear_credentials() else "Not logged in.")
        else:
            creds = load_credentials()
            if creds is None:
                print("Not logged in. Run `openai-radar login`.")
                return 1
            upload = f" upload {creds.upload_id}" if creds.upload_id else ""
            print(f"Logged in to {creds.api_base}.{upload} Token expires {creds.expires_at}.")
    except LoginError as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
