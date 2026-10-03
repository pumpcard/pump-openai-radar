"""Scanners against fake SDK pagers and a fake admin Usage API."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from openai_radar.client import RadarClient, RadarError
from openai_radar.models.base import ServiceKind, ServiceRelationship
from openai_radar.scanners import SCANNERS
from openai_radar.scanners.assistants import AssistantScanner
from openai_radar.scanners.base import Scanner, resource_group
from openai_radar.scanners.batch_jobs import BatchJobScanner
from openai_radar.scanners.fine_tunes import FineTuneScanner
from openai_radar.scanners.usage import USAGE_PATH, UsageScanner
from openai_radar.scanners.vector_stores import VectorStoreScanner
from tests.fakes import AsyncPage


class _SDKClient:
    def __init__(self, sdk: object) -> None:
        self.sdk = sdk
        self.projects: list[str | None] = []

    def for_project(self, project_id: str | None) -> object:
        self.projects.append(project_id)
        return self.sdk


def test_scanner_registry_matches_the_runner_order() -> None:
    assert SCANNERS == ["assistants", "vector_stores", "fine_tunes", "batch_jobs", "usage"]


def test_resource_group_prefers_the_top_level_namespace() -> None:
    beta = SimpleNamespace(vector_stores="from-beta")
    assert (
        resource_group(SimpleNamespace(vector_stores="from-top", beta=beta), "vector_stores")
        == "from-top"
    )
    assert resource_group(SimpleNamespace(beta=beta), "vector_stores") == "from-beta"

    with pytest.raises(AttributeError, match="client.vector_stores"):
        resource_group(SimpleNamespace(beta=SimpleNamespace()), "vector_stores")


def test_paginate_stops_at_max_items() -> None:
    scanner = Scanner(RadarClient(api_key="sk-test"), max_items=2)

    async def _collect(count: int) -> list[int]:
        async def _pages() -> object:
            for item in range(count):
                yield item

        return [item async for item in scanner._paginate(_pages())]

    assert asyncio.run(_collect(2)) == [0, 1]
    assert scanner.truncated is True

    scanner.truncated = False
    assert asyncio.run(_collect(1)) == [0]
    assert scanner.truncated is False


def test_assistant_scanner_lists_and_detects_relationships() -> None:
    raw = SimpleNamespace(
        id="asst_1",
        name="Helper",
        model="gpt-4o",
        tools=[SimpleNamespace(type="code_interpreter")],
        instructions="Store the object in s3 and ping slack.",
        created_at=1_700_000_000,
        metadata={},
        tool_resources=None,
    )
    sdk = SimpleNamespace(assistants=SimpleNamespace(list=lambda limit: AsyncPage([raw])))
    holder = _SDKClient(sdk)
    scanner = AssistantScanner(holder)
    previous = [
        ServiceRelationship(source_id="old", kind=ServiceKind.EMAIL, signal="smtp", evidence="old")
    ]
    scanner.relationships = previous

    found = asyncio.run(scanner.scan("proj_9"))

    assert holder.projects == ["proj_9"]
    assert [item.id for item in found] == ["asst_1"]
    assert found[0].has_code_interpreter is True
    assert {rel.signal for rel in scanner.relationships} == {"s3", "slack"}
    assert scanner.relationships is not previous


def test_assistant_scanner_keeps_the_previous_relationships_when_scan_fails() -> None:
    good = SimpleNamespace(id="asst_1", instructions="", tools=[])
    sdk = SimpleNamespace(
        assistants=SimpleNamespace(list=lambda limit: AsyncPage([good, SimpleNamespace()]))
    )
    scanner = AssistantScanner(_SDKClient(sdk))
    previous = [ServiceRelationship(source_id="old", kind=ServiceKind.SLACK, signal="slack")]
    scanner.relationships = previous

    with pytest.raises(AttributeError):
        asyncio.run(scanner.scan())

    assert scanner.relationships is previous


def test_vector_store_fine_tune_and_batch_scanners() -> None:
    store = SimpleNamespace(
        id="vs_1",
        name="Docs",
        status="completed",
        usage_bytes=10,
        file_counts=SimpleNamespace(total=2, failed=0),
        created_at=None,
        last_active_at=None,
        expires_at=None,
    )
    tune = SimpleNamespace(
        id="ft_1",
        model="gpt-4o-mini",
        status="failed",
        trained_tokens=12,
        error=SimpleNamespace(message="boom"),
        created_at=None,
        finished_at=None,
    )
    batch = SimpleNamespace(
        id="batch_1",
        endpoint="/v1/chat/completions",
        status="completed",
        completion_window="24h",
        request_counts=SimpleNamespace(total=4, completed=4, failed=0),
        created_at=None,
        completed_at=None,
    )
    sdk = SimpleNamespace(
        vector_stores=SimpleNamespace(list=lambda limit: AsyncPage([store])),
        fine_tuning=SimpleNamespace(jobs=SimpleNamespace(list=lambda limit: AsyncPage([tune]))),
        batches=SimpleNamespace(list=lambda limit: AsyncPage([batch])),
    )
    holder = _SDKClient(sdk)

    stores = asyncio.run(VectorStoreScanner(holder).scan("proj_vs"))
    tunes = asyncio.run(FineTuneScanner(holder).scan())
    batches = asyncio.run(BatchJobScanner(holder).scan())

    assert holder.projects == ["proj_vs", None, None]
    assert stores[0].file_count_total == 2
    assert tunes[0].error_message == "boom"
    assert batches[0].request_counts_completed == 4


def test_usage_scanner_without_an_admin_key_returns_nothing() -> None:
    scanner = UsageScanner(RadarClient(api_key="sk-test"), lookback_days=14)

    with pytest.warns(RuntimeWarning, match="no admin key"):
        assert asyncio.run(scanner.scan()) == []


def test_usage_scanner_pages_groups_and_skips_rows_without_a_model() -> None:
    client = RadarClient(api_key="sk-test", admin_key="sk-admin")
    pages = [
        {
            "data": [
                {
                    "start_time": 1_700_000_000,
                    "end_time": 1_700_086_400,
                    "results": [
                        {
                            "model": "gpt-4o",
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "input_cached_tokens": 2,
                            "num_model_requests": 3,
                            "project_id": "proj_1",
                        },
                        {"input_tokens": 99},
                    ],
                }
            ],
            "has_more": True,
            "next_page": "page-2",
        },
        {
            "data": [
                {
                    "start_time": 1_700_086_400,
                    "end_time": 1_700_172_800,
                    "results": [
                        {
                            "model": "gpt-4o-mini",
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "num_model_requests": 1,
                        }
                    ],
                }
            ],
            "has_more": False,
        },
    ]
    seen: list[tuple[str, dict[str, object]]] = []

    async def get_org(path: str, params: dict[str, object] | None = None) -> dict[str, object]:
        seen.append((path, dict(params or {})))
        return pages[len(seen) - 1]

    client.get_org = get_org  # type: ignore[method-assign]
    scanner = UsageScanner(client, lookback_days=0)

    usage = asyncio.run(scanner.scan("proj_9"))

    assert [item.model for item in usage] == ["gpt-4o", "gpt-4o-mini"]
    assert usage[0].total_tokens == 14
    assert usage[0].cached_tokens == 2
    assert usage[0].num_requests == 3
    assert usage[0].project_id == "proj_1"
    assert usage[0].bucket_start is not None
    assert usage[1].cached_tokens == 0
    assert seen[0][0] == USAGE_PATH
    assert seen[0][1]["bucket_width"] == "1d"
    assert seen[0][1]["group_by"] == ["model", "project_id"]
    assert seen[0][1]["project_ids"] == ["proj_9"]
    assert seen[0][1]["limit"] == 1
    assert seen[1][1]["page"] == "page-2"
    assert scanner.truncated is False


def test_usage_scanner_keeps_earlier_pages_when_a_later_page_fails() -> None:
    client = RadarClient(api_key="sk-test", admin_key="sk-admin")
    calls = {"n": 0}

    async def get_org(path: str, params: dict[str, object] | None = None) -> dict[str, object]:
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "data": [{"results": [{"model": "gpt-4o", "input_tokens": 5, "output_tokens": 1}]}],
                "has_more": True,
                "next_page": "page-2",
            }
        raise RadarError("rate limited")

    client.get_org = get_org  # type: ignore[method-assign]

    with pytest.warns(RuntimeWarning, match="pagination stopped"):
        usage = asyncio.run(UsageScanner(client).scan())

    assert [item.model for item in usage] == ["gpt-4o"]
    assert usage[0].total_tokens == 6


def test_usage_scanner_raises_when_the_first_page_fails() -> None:
    client = RadarClient(api_key="sk-test", admin_key="sk-admin")

    async def get_org(path: str, params: dict[str, object] | None = None) -> dict[str, object]:
        raise RadarError("401 Unauthorized")

    client.get_org = get_org  # type: ignore[method-assign]

    with pytest.raises(RadarError, match="401"):
        asyncio.run(UsageScanner(client).scan())


def test_usage_scanner_marks_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openai_radar.scanners.usage.MAX_PAGES", 1)
    client = RadarClient(api_key="sk-test", admin_key="sk-admin")

    async def get_org(path: str, params: dict[str, object] | None = None) -> dict[str, object]:
        return {
            "data": [{"results": [{"model": "gpt-4o", "input_tokens": 1, "output_tokens": 1}]}],
            "has_more": True,
            "next_page": "page-2",
        }

    client.get_org = get_org  # type: ignore[method-assign]
    scanner = UsageScanner(client, max_items=1)

    usage = asyncio.run(scanner.scan())

    assert len(usage) == 1
    assert scanner.truncated is True
