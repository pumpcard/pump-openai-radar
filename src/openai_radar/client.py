"""
RadarClient
-----------
Auth wrapper over the OpenAI SDK.

Two credentials are in play, and they unlock different things:

  * **project key** (``OPENAI_API_KEY``, ``sk-proj-…``) — lists assistants,
    vector stores, fine-tunes and batch jobs for one project.
  * **admin key** (``OPENAI_ADMIN_KEY``, ``sk-admin-…``) — required for the
    ``/v1/organization/usage/*`` endpoints, which are org-scoped and are not
    exposed on the regular SDK surface.

The admin key is optional; without it the usage scanner degrades to empty
rather than failing the whole run.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from openai import AsyncOpenAI

ORG_API_BASE = "https://api.openai.com/v1"


class RadarError(Exception):
    """Raised for unrecoverable client/configuration problems."""


class RadarClient:
    """Thin wrapper holding both credentials and the async OpenAI client."""

    def __init__(
        self,
        api_key: str | None = None,
        admin_key: str | None = None,
        project_id: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.admin_key = admin_key or os.getenv("OPENAI_ADMIN_KEY")
        self.project_id = project_id or os.getenv("OPENAI_PROJECT_ID")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or ORG_API_BASE
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key and not self.admin_key:
            raise RadarError(
                "Missing OpenAI credentials. Set OPENAI_API_KEY (or pass api_key=), "
                "and optionally OPENAI_ADMIN_KEY for org-wide usage data."
            )

        self._client = AsyncOpenAI(
            api_key=self.api_key or self.admin_key,
            project=self.project_id,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    # ------------------------------------------------------------------
    # SDK surface
    # ------------------------------------------------------------------

    @property
    def openai(self) -> AsyncOpenAI:
        """The underlying AsyncOpenAI client."""
        return self._client

    def for_project(self, project_id: str | None) -> AsyncOpenAI:
        """
        Return a client scoped to ``project_id``.

        Returns the shared client when the scope already matches, so the common
        path allocates nothing.
        """
        if not project_id or project_id == self.project_id:
            return self._client
        return AsyncOpenAI(
            api_key=self.api_key or self.admin_key,
            project=project_id,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @property
    def has_admin_key(self) -> bool:
        return bool(self.admin_key)

    # ------------------------------------------------------------------
    # Organization endpoints (admin key only)
    # ------------------------------------------------------------------

    async def get_org(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        GET an ``/organization/*`` endpoint with the admin key.

        These are not on the SDK surface, so this goes over raw httpx.
        """
        if not self.admin_key:
            raise RadarError(
                f"{path} requires an admin key. "
                "Set OPENAI_ADMIN_KEY or pass admin_key= to RadarClient."
            )

        url = f"{self.base_url.rstrip('/')}{path}"
        headers = {
            "Authorization": f"Bearer {self.admin_key}",
            "User-Agent": "pump-openai-radar",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as http:
            resp = await http.get(url, params=params, headers=headers)

            if resp.status_code == 401:
                raise RadarError(
                    f"401 Unauthorized on {path} — the admin key is invalid or expired."
                )
            if resp.status_code == 403:
                raise RadarError(
                    f"403 Forbidden on {path} — this key lacks org-level read scope. "
                    "Usage endpoints require an admin key, not a project key."
                )
            if not resp.is_success:
                raise RadarError(f"OpenAI {resp.status_code} on {path}: {resp.text[:300]}")

            return resp.json() if resp.content else {}

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> RadarClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        scope = self.project_id or "all projects"
        admin = "with admin key" if self.admin_key else "no admin key"
        return f"<RadarClient {scope}, {admin}>"
