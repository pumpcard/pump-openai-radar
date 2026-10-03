"""Small stand-ins shared by the scanner tests."""

from __future__ import annotations

from typing import Any


class AsyncPage:
    """Async iterator, the shape the OpenAI SDK pagers expose to ``async for``."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    def __aiter__(self) -> AsyncPage:
        self._iter = iter(self._items)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any] | None = None,
        text: str = "",
        content: bytes | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        self.text = text
        self.is_success = 200 <= status_code < 300
        if content is None:
            self.content = b"{}" if self.is_success else text.encode()
        else:
            self.content = content

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeHTTP:
    """Async context manager standing in for ``httpx.AsyncClient``."""

    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.client_kwargs: dict[str, Any] = {}

    async def __aenter__(self) -> FakeHTTP:  # noqa: PYI034
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get(
        self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.response
