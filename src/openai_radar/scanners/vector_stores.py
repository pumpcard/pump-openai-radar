"""VectorStoreScanner — vector stores, their size and expiry."""

from __future__ import annotations

from openai_radar.models.base import VectorStoreInfo
from openai_radar.scanners.base import PAGE_SIZE, Scanner, resource_group


class VectorStoreScanner(Scanner):
    name = "vector_stores"

    async def scan(self, project_id: str | None = None) -> list[VectorStoreInfo]:
        sdk = self._sdk(project_id)
        stores = resource_group(sdk, "vector_stores")

        return [
            VectorStoreInfo.from_api(raw)
            async for raw in self._paginate(stores.list(limit=PAGE_SIZE))
        ]
