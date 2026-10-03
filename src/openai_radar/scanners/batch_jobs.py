"""BatchJobScanner — batch jobs and their per-request success counts."""

from __future__ import annotations

from openai_radar.models.base import BatchJobInfo
from openai_radar.scanners.base import PAGE_SIZE, Scanner


class BatchJobScanner(Scanner):
    name = "batch_jobs"

    async def scan(self, project_id: str | None = None) -> list[BatchJobInfo]:
        sdk = self._sdk(project_id)

        return [
            BatchJobInfo.from_api(raw)
            async for raw in self._paginate(sdk.batches.list(limit=PAGE_SIZE))
        ]
