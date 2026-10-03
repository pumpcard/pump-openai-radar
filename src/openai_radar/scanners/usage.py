"""
UsageScanner
------------
Pulls token usage from the org-scoped Usage API.

This is the one scanner that needs an **admin** key: ``/organization/usage/*``
is not reachable with a project key. Without one it warns and returns empty,
so a scan with only a project key still produces inventory and findings.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone
from typing import Any

from openai_radar.client import RadarError
from openai_radar.models.base import ModelUsage
from openai_radar.scanners.base import Scanner

USAGE_PATH = "/organization/usage/completions"

#: The API caps a 1-day-bucket query at 31 buckets per page.
MAX_BUCKETS_PER_PAGE = 31

#: Stop after this many pages, so a wide lookback cannot loop unbounded.
MAX_PAGES = 50


def _ts(dt: datetime) -> int:
    return int(dt.timestamp())


class UsageScanner(Scanner):
    name = "usage"

    def __init__(
        self,
        client: Any,
        lookback_days: int = 30,
        max_items: int = 10_000,
    ) -> None:
        super().__init__(client, max_items=max_items)
        self.lookback_days = lookback_days

    async def scan(self, project_id: str | None = None) -> list[ModelUsage]:
        if not self.client.has_admin_key:
            warnings.warn(
                "Usage scan skipped: no admin key. Set OPENAI_ADMIN_KEY (or pass "
                "admin_key=) to collect org-wide token usage.",
                RuntimeWarning,
                stacklevel=2,
            )
            return []

        now = datetime.now(tz=timezone.utc)
        start = now - timedelta(days=self.lookback_days)

        params: dict[str, Any] = {
            "start_time": _ts(start),
            "end_time": _ts(now),
            "bucket_width": "1d",
            "limit": min(self.lookback_days or 1, MAX_BUCKETS_PER_PAGE),
            # group_by repeats as a multi-value param; httpx encodes a list as
            # repeated keys, which is what the API expects.
            "group_by": ["model", "project_id"],
        }
        if project_id:
            params["project_ids"] = [project_id]

        results: list[ModelUsage] = []
        pages = 0

        while pages < MAX_PAGES:
            try:
                payload = await self.client.get_org(USAGE_PATH, params)
            except RadarError as exc:
                if pages == 0:
                    raise
                # Partial data beats no data — keep what earlier pages returned.
                warnings.warn(
                    f"Usage pagination stopped after {pages} page(s): {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            for bucket in payload.get("data") or []:
                bucket_start = bucket.get("start_time")
                bucket_end = bucket.get("end_time")
                for row in bucket.get("results") or []:
                    model = row.get("model")
                    if not model:
                        # Ungrouped rows carry no model; they would collapse
                        # every model into one bogus bucket.
                        continue
                    results.append(
                        ModelUsage(
                            model=model,
                            input_tokens=int(row.get("input_tokens") or 0),
                            output_tokens=int(row.get("output_tokens") or 0),
                            cached_tokens=int(row.get("input_cached_tokens") or 0),
                            num_requests=int(row.get("num_model_requests") or 0),
                            project_id=row.get("project_id"),
                            bucket_start=_dt(bucket_start),
                            bucket_end=_dt(bucket_end),
                        )
                    )
                    if len(results) >= self.max_items:
                        self.truncated = True
                        return results

            pages += 1
            if not payload.get("has_more"):
                break
            next_page = payload.get("next_page")
            if not next_page:
                break
            params["page"] = next_page

        if pages >= MAX_PAGES:
            self.truncated = True

        return results


def _dt(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None
