"""Tests for the Pump push (openai_radar.upload). No network: urlopen is stubbed."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from openai_radar import upload


class _FakeResp:
    def __init__(self, body: bytes = b"", status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def test_upload_csvs_exchanges_then_puts(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    billing = tmp_path / "report.csv"
    inventory = tmp_path / "usage.csv"
    billing.write_text("Date,ProjectID,LineItem,Amount,Currency\n2026-01-01,proj,gpt-4o,1.0,USD\n")
    inventory.write_text("model,input_tokens,output_tokens\ngpt-4o,3,4\n")

    calls = []

    def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        calls.append(req)
        if req.method == "POST":
            role = json.loads(req.data.decode())["role"]
            return _FakeResp(json.dumps({"upload_url": f"https://s3/{role}"}).encode())
        return _FakeResp(status=200)

    monkeypatch.setattr(upload.urllib.request, "urlopen", fake_urlopen)

    upload.upload_csvs(
        api_base="http://localhost:8001/",
        token="tok",
        files={"billing": str(billing), "inventory": str(inventory)},
    )

    posts = [c for c in calls if c.method == "POST"]
    puts = [c for c in calls if c.method == "PUT"]
    assert len(posts) == 2 and len(puts) == 2
    assert posts[0].full_url == "http://localhost:8001/api/v1/estimate/radar/urls"
    assert json.loads(posts[0].data.decode()) == {
        "token": "tok",
        "role": "billing",
        "provider": "openai",
    }
    assert json.loads(posts[1].data.decode()) == {
        "token": "tok",
        "role": "inventory",
        "provider": "openai",
    }
    assert puts[0].full_url == "https://s3/billing"
    assert puts[1].full_url == "https://s3/inventory"
    assert puts[0].headers["Content-type"] == "text/csv"
    assert puts[0].data == billing.read_bytes()
    assert puts[1].data == inventory.read_bytes()
    # Both requests send a real User-Agent (Cloudflare blocks the urllib default).
    assert posts[0].headers["User-agent"] == upload._USER_AGENT
    assert puts[0].headers["User-agent"] == upload._USER_AGENT
    assert upload._USER_AGENT.startswith("openai-radar/")


def test_expired_token_raises_clear_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    report = tmp_path / "report.csv"
    report.write_text("x\n")

    def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"expired"))

    monkeypatch.setattr(upload.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(upload.UploadError, match="expired"):
        upload.upload_csvs("http://x", "tok", {"billing": str(report)})


def test_exchange_without_upload_url_raises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    report = tmp_path / "report.csv"
    report.write_text("x\n")

    def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        return _FakeResp(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(upload.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(upload.UploadError, match="upload_url"):
        upload.upload_csvs("http://x", "tok", {"billing": str(report)})


def test_unknown_role_rejected(tmp_path) -> None:
    with pytest.raises(upload.UploadError, match="Unknown upload role"):
        upload.upload_csvs("http://x", "tok", {"report": "nope.csv"})
