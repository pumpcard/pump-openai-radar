"""
Cost report CSV for Pump onboarding.

Pump's estimate flow (the same ``/api/v1/estimate/radar/urls`` exchange
pump-aws-radar uses) takes two OpenAI files. This module builds the cost
file, uploaded as role ``billing``. Token usage, uploaded as role
``inventory``, is written beside :mod:`openai_radar.scanners.usage`.

Cost rows come from the organization Costs API, one per (day, project,
line item), which is the dollar view the admin key can see.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from openai_radar.client import RadarClient, RadarError

COSTS_PATH = "/organization/costs"

#: Column order. Date/Amount/Currency match the billing CSV pump-aws-radar uploads;
#: ProjectID and LineItem are the OpenAI equivalents of account and service.
REPORT_FIELDS = ["Date", "ProjectID", "LineItem", "Amount", "Currency"]

#: The API caps a 1-day-bucket query at 31 buckets per page.
MAX_BUCKETS_PER_PAGE = 31

#: Stop after this many pages, so a wide lookback cannot loop unbounded.
MAX_PAGES = 50


class ReportError(RadarError):
    """The cost report cannot be built or is empty."""


@dataclass(frozen=True)
class CostReport:
    rows: list[dict[str, str]]
    truncated: bool = False


def _ts(dt: datetime) -> int:
    return int(dt.timestamp())


def _amount(raw: Any) -> tuple[str, str] | None:
    """Return ``(amount, currency)`` or None when the bucket is zero or unusable."""
    if not isinstance(raw, dict):
        return None
    try:
        number = float(raw.get("value"))
    except (TypeError, ValueError):
        return None
    if number == 0:
        return None
    currency = str(raw.get("currency") or "usd").upper()
    return f"{number:.6f}", currency


def _row(bucket_start: Any, result: dict[str, Any]) -> dict[str, str] | None:
    if bucket_start is None:
        return None
    parsed = _amount(result.get("amount"))
    if parsed is None:
        return None
    amount, currency = parsed
    try:
        day = datetime.fromtimestamp(int(bucket_start), tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        return None
    project = result.get("project_id") or "-"
    line_item = result.get("line_item") or "-"
    return {
        "Date": day,
        "ProjectID": str(project),
        "LineItem": str(line_item),
        "Amount": amount,
        "Currency": currency,
    }


async def fetch_cost_report(
    client: RadarClient,
    *,
    lookback_days: int = 30,
    project_id: str | None = None,
) -> CostReport:
    """Pull daily org costs and shape them into report rows.

    Requires an admin key. Zero-cost buckets are dropped. An empty window
    raises :class:`ReportError` so the caller does not upload a header-only file.
    """
    if not client.has_admin_key:
        raise ReportError(
            "The Pump report needs org cost data. Pass --admin-key or set OPENAI_ADMIN_KEY."
        )

    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(days=lookback_days)
    params: dict[str, Any] = {
        "start_time": _ts(start),
        "end_time": _ts(now),
        "bucket_width": "1d",
        "limit": min(lookback_days or 1, MAX_BUCKETS_PER_PAGE),
        "group_by": ["project_id", "line_item"],
    }
    if project_id:
        params["project_ids"] = [project_id]

    rows: list[dict[str, str]] = []
    pages = 0
    truncated = False

    while pages < MAX_PAGES:
        try:
            payload = await client.get_org(COSTS_PATH, params)
        except RadarError as exc:
            if pages == 0:
                raise
            warnings.warn(
                f"Cost report pagination stopped after {pages} page(s): {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            truncated = True
            break

        for bucket in payload.get("data") or []:
            bucket_start = bucket.get("start_time")
            for result in bucket.get("results") or []:
                if not isinstance(result, dict):
                    continue
                row = _row(bucket_start, result)
                if row is not None:
                    rows.append(row)

        pages += 1
        if not payload.get("has_more"):
            break
        next_page = payload.get("next_page")
        if not next_page:
            break
        params["page"] = next_page
    else:
        # The loop ran out of pages while the API still had more buckets.
        truncated = True
        warnings.warn(
            "Cost report hit the page limit and may be incomplete.",
            RuntimeWarning,
            stacklevel=2,
        )

    if not rows:
        raise ReportError(
            "OpenAI returned no cost rows for this window, so there is nothing to upload."
        )

    return CostReport(rows=rows, truncated=truncated)


def write_report_csv(path: str | Path, rows: list[dict[str, str]]) -> Path:
    """Write ``report.csv``. Returns the path written."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination
