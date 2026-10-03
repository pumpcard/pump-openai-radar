"""Keep tests from seeing a developer's real OpenAI credentials."""

from __future__ import annotations

import pytest

_OPENAI_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_ADMIN_KEY",
    "OPENAI_PROJECT_ID",
    "OPENAI_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _OPENAI_ENV:
        monkeypatch.delenv(name, raising=False)
