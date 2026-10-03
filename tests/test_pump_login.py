"""Browser login against a fake Pump: PKCE, state, and the token exchange."""

from __future__ import annotations

import json
import os
import stat
import threading
import unittest
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pump_login
from pump_login import LoginError, code_challenge_s256, login


class _PumpHandler(BaseHTTPRequestHandler):
    server: "_PumpServer"

    def do_GET(self) -> None:  # noqa: N802
        parts = urllib.parse.urlsplit(self.path)
        if parts.path != "/api/oauth/cli":
            self.send_error(404)
            return
        query = urllib.parse.parse_qs(parts.query)
        self.server.challenge = query["code_challenge"][0]
        self.server.redirect_uri = query["redirect_uri"][0]
        self.server.seen_state = query["state"][0]
        state = "tampered" if self.server.tamper_state else self.server.seen_state
        target = urllib.parse.urlparse(self.server.redirect_uri)
        location = urllib.parse.urlunparse(
            target._replace(query=urllib.parse.urlencode({"code": "auth-code", "state": state}))
        )
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or "0")
        body = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
        verifier = body["code_verifier"][0]
        if code_challenge_s256(verifier) != self.server.challenge:
            self._json(
                400, {"error": "invalid_grant", "error_description": "PKCE verification failed."}
            )
            return
        if body["redirect_uri"][0] != self.server.redirect_uri or body["code"][0] != "auth-code":
            self._json(400, {"error": "invalid_grant", "error_description": "redirect mismatch"})
            return
        self._json(
            200,
            {
                "access_token": "upload-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "radar",
                "upload_id": "upload-9",
            },
        )

    def _json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: object) -> None:
        return


class _PumpServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _PumpHandler)
        self.challenge = ""
        self.redirect_uri = ""
        self.seen_state = ""
        self.tamper_state = False


class LoginFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pump = _PumpServer()
        self.thread = threading.Thread(target=self.pump.serve_forever, daemon=True)
        self.thread.start()
        port = self.pump.server_address[1]
        self.base = f"http://127.0.0.1:{port}"
        self.config_dir = Path(self.id().replace(".", "_"))
        # Keep credentials out of the real home directory.
        self._tmpdir = Path("/tmp") / "openai-radar-login-tests" / self.config_dir.name
        self._tmpdir.mkdir(parents=True, exist_ok=True)
        self.creds = self._tmpdir / "credentials.json"

    def tearDown(self) -> None:
        self.pump.shutdown()
        self.pump.server_close()
        self.thread.join(timeout=2)
        if self.creds.exists():
            self.creds.unlink()

    def _open(self, url: str) -> None:
        with urllib.request.urlopen(url, timeout=5) as response:
            response.read()

    def test_login_stores_the_token_from_pump(self) -> None:
        lines: list[str] = []
        creds = login(
            api_base=self.base,
            app_base=self.base,
            open_browser=self._open,
            credentials_file=self.creds,
            stdout=lines.append,
        )
        self.assertEqual(creds.access_token, "upload-token")
        self.assertEqual(creds.upload_id, "upload-9")
        stored = json.loads(self.creds.read_text(encoding="utf-8"))
        self.assertEqual(stored["access_token"], "upload-token")
        mode = stat.S_IMODE(os.stat(self.creds).st_mode)
        self.assertEqual(mode, 0o600)
        loaded = pump_login.load_credentials(self.creds)
        assert loaded is not None
        self.assertEqual(loaded.access_token, "upload-token")
        joined = "\n".join(lines)
        self.assertNotIn("upload-token", joined)
        self.assertIn("code_challenge_method=S256", joined)

    def test_state_mismatch_does_not_store_a_token(self) -> None:
        self.pump.tamper_state = True
        with self.assertRaises(LoginError):
            login(
                api_base=self.base,
                app_base=self.base,
                open_browser=self._open,
                credentials_file=self.creds,
                stdout=lambda _line: None,
            )
        self.assertFalse(self.creds.exists())


if __name__ == "__main__":
    unittest.main()
