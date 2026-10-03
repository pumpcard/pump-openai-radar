"""
Scanner base
------------
Every scanner exposes the same shape::

    scanner = SomeScanner(client)
    items   = await scanner.scan(project_id)

``Runner`` gathers all scanners concurrently and tolerates individual
failures, so a scanner that cannot reach its endpoint should raise — the
Runner downgrades it to a warning rather than aborting the run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, TypeVar

from openai import AsyncOpenAI

from openai_radar.client import RadarClient

T = TypeVar("T")

#: Hard ceiling on items pulled per resource type, so a pathological org
#: cannot spin the scanner forever. Surfaced via ``Scanner.truncated``.
DEFAULT_MAX_ITEMS = 10_000

#: Page size requested from the API. 100 is the documented maximum.
PAGE_SIZE = 100


class Scanner:
    """Base class for all resource scanners."""

    #: Human-readable name, used in warnings and progress output.
    name: str = "scanner"

    def __init__(self, client: RadarClient, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        self.client = client
        self.max_items = max_items
        self.truncated = False
        """True when ``max_items`` cut the listing short."""

    def _sdk(self, project_id: str | None) -> AsyncOpenAI:
        return self.client.for_project(project_id)

    async def _paginate(self, pager: Any) -> AsyncIterator[Any]:
        """
        Iterate an SDK pager, stopping at ``max_items``.

        The OpenAI SDK's cursor pages auto-paginate under ``async for``; this
        only adds the ceiling and the truncation flag.
        """
        count = 0
        async for item in pager:
            yield item
            count += 1
            if count >= self.max_items:
                self.truncated = True
                return

    async def scan(self, project_id: str | None = None) -> list[Any]:
        raise NotImplementedError


def resource_group(sdk: AsyncOpenAI, name: str) -> Any:
    """
    Resolve a resource group that moved namespaces across SDK versions.

    ``vector_stores`` sat under ``client.beta`` in openai<1.9x and was
    promoted to the top level afterwards. Preferring the top level keeps
    newer SDKs off the deprecation path while older ones still resolve.
    """
    top = getattr(sdk, name, None)
    if top is not None:
        return top
    beta = getattr(sdk, "beta", None)
    nested = getattr(beta, name, None) if beta is not None else None
    if nested is None:
        raise AttributeError(
            f"This openai SDK exposes neither client.{name} nor client.beta.{name}."
        )
    return nested
