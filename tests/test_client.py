"""RadarClient credentials and the admin-key HTTP path. No live API calls."""

from __future__ import annotations

import asyncio

import pytest

from openai_radar.client import ORG_API_BASE, RadarClient, RadarError
from tests.fakes import FakeHTTP, FakeResponse


def test_missing_credentials_raise() -> None:
    with pytest.raises(RadarError, match="Missing OpenAI credentials"):
        RadarClient()


def test_reads_credentials_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-env")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_env")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")

    client = RadarClient()

    assert client.api_key == "sk-env"
    assert client.admin_key == "sk-admin-env"
    assert client.project_id == "proj_env"
    assert client.base_url == "https://example.test/v1"
    assert client.has_admin_key


def test_explicit_arguments_override_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-env")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_env")

    client = RadarClient(api_key="sk-explicit", project_id="proj_explicit")

    assert client.api_key == "sk-explicit"
    assert client.admin_key == "sk-admin-env"
    assert client.project_id == "proj_explicit"
    assert client.base_url == ORG_API_BASE
    assert client.has_admin_key


def test_admin_key_alone_is_enough() -> None:
    client = RadarClient(admin_key="sk-admin")

    assert client.api_key is None
    assert client.has_admin_key
    assert "with admin key" in repr(client)
    assert "all projects" in repr(client)


def test_repr_includes_project_scope() -> None:
    client = RadarClient(api_key="sk-test", project_id="proj_a")
    assert repr(client) == "<RadarClient proj_a, no admin key>"


def test_for_project_reuses_the_shared_client() -> None:
    client = RadarClient(api_key="sk-test", project_id="proj_a")

    assert client.for_project(None) is client.openai
    assert client.for_project("proj_a") is client.openai

    other = client.for_project("proj_b")
    assert other is not client.openai
    assert other.project == "proj_b"


def test_get_org_requires_an_admin_key() -> None:
    client = RadarClient(api_key="sk-test")

    with pytest.raises(RadarError, match="admin key"):
        asyncio.run(client.get_org("/organization/usage/completions"))


def test_get_org_returns_json(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RadarClient(
        api_key="sk-test", admin_key="sk-admin", base_url="https://example.test/v1/"
    )
    http = FakeHTTP(FakeResponse(200, {"data": [{"id": "bucket"}]}))

    def factory(**kwargs: object) -> FakeHTTP:
        http.client_kwargs = kwargs
        return http

    monkeypatch.setattr("openai_radar.client.httpx.AsyncClient", factory)

    payload = asyncio.run(client.get_org("/organization/usage/completions", {"limit": 1}))

    assert payload == {"data": [{"id": "bucket"}]}
    assert http.client_kwargs["timeout"] == client.timeout
    assert http.calls == [
        {
            "url": "https://example.test/v1/organization/usage/completions",
            "params": {"limit": 1},
            "headers": {
                "Authorization": "Bearer sk-admin",
                "User-Agent": "openai-radar",
            },
        }
    ]


def test_get_org_empty_body_is_an_empty_object(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RadarClient(api_key="sk-test", admin_key="sk-admin")
    monkeypatch.setattr(
        "openai_radar.client.httpx.AsyncClient",
        lambda **kwargs: FakeHTTP(FakeResponse(204, content=b"")),
    )

    assert asyncio.run(client.get_org("/organization/usage/completions")) == {}


@pytest.mark.parametrize(
    ("status", "match"),
    [
        (401, "401 Unauthorized"),
        (403, "403 Forbidden"),
        (500, "OpenAI 500"),
    ],
)
def test_get_org_http_errors(monkeypatch: pytest.MonkeyPatch, status: int, match: str) -> None:
    client = RadarClient(api_key="sk-test", admin_key="sk-admin")
    monkeypatch.setattr(
        "openai_radar.client.httpx.AsyncClient",
        lambda **kwargs: FakeHTTP(FakeResponse(status, text="nope")),
    )

    with pytest.raises(RadarError, match=match):
        asyncio.run(client.get_org("/organization/usage/completions"))


def test_async_context_manager_closes_the_sdk_client() -> None:
    async def _use() -> RadarClient:
        client = RadarClient(api_key="sk-test")
        async with client as entered:
            assert entered is client
        return client

    client = asyncio.run(_use())
    assert client.openai.is_closed()
