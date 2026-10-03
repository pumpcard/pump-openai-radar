"""FineTuneScanner — fine-tuning jobs and their terminal state."""

from __future__ import annotations

from openai_radar.models.base import FineTuneInfo
from openai_radar.scanners.base import PAGE_SIZE, Scanner


class FineTuneScanner(Scanner):
    name = "fine_tunes"

    async def scan(self, project_id: str | None = None) -> list[FineTuneInfo]:
        sdk = self._sdk(project_id)

        return [
            FineTuneInfo.from_api(raw)
            async for raw in self._paginate(sdk.fine_tuning.jobs.list(limit=PAGE_SIZE))
        ]
