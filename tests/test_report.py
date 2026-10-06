"""Organization cost report rows and CSV writing. No network."""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone

import pytest

from openai_radar.client import RadarClient, RadarError
from openai_radar.report import (
    COSTS_PATH,
    REPORT_FIELDS,
    CostReport,
    ReportError,
    fetch_cost_report,
    write_report_csv,
)

# 2024-11-01 00:00:00 UTC
_DAY = 1_730_419_200


def test_fetch_cost_report_shapes_rows_and_drops_zeros() -> None:
    client = RadarClient(api_key="sk-test", admin_key="sk-admin")
    seen: list[dict[str, object]] = []

    async def get_org(path: str, params: dict[str, object] | None = None) -> dict[str, object]:
        seen.append({"path": path, **dict(params or {})})
        if len(seen) == 1:
            return {
                "data": [
                    {
                        "start_time": _DAY,
                        "end_time": _DAY + 86_400,
                        "results": [
                            {
                                "amount": {"value": 1.5, "currency": "usd"},
                                "line_item": "gpt-4o, input",
                                "project_id": "proj_1",
                            },
                            {
                                "amount": {"value": 0, "currency": "usd"},
                                "line_item": "gpt-4o, output",
                                "project_id": "proj_1",
                            },
                            {"amount": {"value": "0.25", "currency": "usd"}, "project_id": None},
                        ],
                    }
                ],
                "has_more": True,
                "next_page": "page-2",
            }
        return {
            "data": [
                {
                    "start_time": _DAY + 86_400,
                    "results": [
                        {
                            "amount": {"value": 2, "currency": "eur"},
                            "line_item": "embeddings",
                            "project_id": "proj_2",
                        }
                    ],
                }
            ],
            "has_more": False,
        }

    client.get_org = get_org  # type: ignore[method-assign]

    report = asyncio.run(fetch_cost_report(client, lookback_days=7, project_id="proj_9"))

    assert isinstance(report, CostReport)
    assert report.truncated is False
    assert report.rows == [
        {
            "Date": "2024-11-01",
            "ProjectID": "proj_1",
            "LineItem": "gpt-4o, input",
            "Amount": "1.500000",
            "Currency": "USD",
        },
        {
            "Date": "2024-11-01",
            "ProjectID": "-",
            "LineItem": "-",
            "Amount": "0.250000",
            "Currency": "USD",
        },
        {
            "Date": datetime.fromtimestamp(_DAY + 86_400, tz=timezone.utc).date().isoformat(),
            "ProjectID": "proj_2",
            "LineItem": "embeddings",
            "Amount": "2.000000",
            "Currency": "EUR",
        },
    ]
    assert seen[0]["path"] == COSTS_PATH
    assert seen[0]["bucket_width"] == "1d"
    assert seen[0]["group_by"] == ["project_id", "line_item"]
    assert seen[0]["project_ids"] == ["proj_9"]
    assert seen[0]["limit"] == 7
    assert seen[1]["page"] == "page-2"


def test_fetch_requires_an_admin_key() -> None:
    client = RadarClient(api_key="sk-test")
    with pytest.raises(ReportError, match="OPENAI_ADMIN_KEY"):
        asyncio.run(fetch_cost_report(client))


def test_fetch_rejects_an_empty_window() -> None:
    client = RadarClient(api_key="sk-test", admin_key="sk-admin")

    async def get_org(path: str, params: dict[str, object] | None = None) -> dict[str, object]:
        return {"data": [{"start_time": _DAY, "results": [{"amount": {"value": 0}}]}]}

    client.get_org = get_org  # type: ignore[method-assign]
    with pytest.raises(ReportError, match="no cost rows"):
        asyncio.run(fetch_cost_report(client))


def test_fetch_keeps_earlier_pages_when_a_later_page_fails() -> None:
    client = RadarClient(api_key="sk-test", admin_key="sk-admin")
    calls = {"n": 0}

    async def get_org(path: str, params: dict[str, object] | None = None) -> dict[str, object]:
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "data": [
                    {
                        "start_time": _DAY,
                        "results": [
                            {"amount": {"value": 1, "currency": "usd"}, "line_item": "gpt-4o"}
                        ],
                    }
                ],
                "has_more": True,
                "next_page": "page-2",
            }
        raise RadarError("rate limited")

    client.get_org = get_org  # type: ignore[method-assign]

    with pytest.warns(RuntimeWarning, match="pagination stopped"):
        report = asyncio.run(fetch_cost_report(client))

    assert len(report.rows) == 1
    assert report.truncated is True
    assert report.rows[0]["Amount"] == "1.000000"


def test_fetch_raises_when_the_first_page_fails() -> None:
    client = RadarClient(api_key="sk-test", admin_key="sk-admin")

    async def get_org(path: str, params: dict[str, object] | None = None) -> dict[str, object]:
        raise RadarError("401 Unauthorized")

    client.get_org = get_org  # type: ignore[method-assign]
    with pytest.raises(RadarError, match="401"):
        asyncio.run(fetch_cost_report(client))


def test_fetch_marks_truncation_at_the_page_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openai_radar.report.MAX_PAGES", 1)
    client = RadarClient(api_key="sk-test", admin_key="sk-admin")

    async def get_org(path: str, params: dict[str, object] | None = None) -> dict[str, object]:
        return {
            "data": [
                {
                    "start_time": _DAY,
                    "results": [{"amount": {"value": 3, "currency": "usd"}, "line_item": "gpt-4o"}],
                }
            ],
            "has_more": True,
            "next_page": "page-2",
        }

    client.get_org = get_org  # type: ignore[method-assign]

    with pytest.warns(RuntimeWarning, match="page limit"):
        report = asyncio.run(fetch_cost_report(client, lookback_days=0))

    assert report.truncated is True
    assert len(report.rows) == 1


def test_write_report_csv_round_trips(tmp_path) -> None:
    path = tmp_path / "nested" / "report.csv"
    rows = [
        {
            "Date": "2024-11-01",
            "ProjectID": "proj_1",
            "LineItem": "gpt-4o, input",
            "Amount": "1.500000",
            "Currency": "USD",
        }
    ]
    written = write_report_csv(path, rows)

    with written.open(encoding="utf-8", newline="") as fh:
        parsed = list(csv.DictReader(fh))

    assert list(parsed[0]) == REPORT_FIELDS
    assert parsed[0]["LineItem"] == "gpt-4o, input"
    assert parsed[0]["Amount"] == "1.500000"
