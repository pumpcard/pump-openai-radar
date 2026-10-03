"""
Resource models
---------------
Pydantic v2 models for every resource type the scanners emit.

Field names here are load-bearing: ``FindingEngine`` reads them directly
(``tool_count``, ``usage_bytes``, ``request_counts_failed`` …), and the CSV
exporter derives its column order from ``csv_fields()``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _ts(value: Any) -> datetime | None:
    """OpenAI returns Unix seconds; normalise to tz-aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


class RadarModel(BaseModel):
    """Base for all resource models."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @classmethod
    def csv_fields(cls) -> list[str]:
        """Column order for the CSV exporter."""
        return list(cls.model_fields)

    def csv_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for name in self.csv_fields():
            value = getattr(self, name, None)
            if isinstance(value, datetime):
                row[name] = value.isoformat()
            elif isinstance(value, Enum):
                row[name] = value.value
            elif isinstance(value, (list, tuple)):
                row[name] = ",".join(str(v) for v in value)
            elif isinstance(value, dict):
                row[name] = ";".join(f"{k}={v}" for k, v in value.items())
            else:
                row[name] = value
        return row


# ---------------------------------------------------------------------------
# Service relationships
# ---------------------------------------------------------------------------


class ServiceKind(str, Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    DATABASE = "database"
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"


class ServiceRelationship(RadarModel):
    """An external service referenced by an assistant's instructions."""

    source_id: str
    """ID of the assistant that referenced the service."""

    source_name: str | None = None
    kind: ServiceKind
    signal: str
    """The token that matched, e.g. ``s3`` or ``bigquery``."""

    evidence: str = ""
    """Surrounding text from the instructions, trimmed for display."""


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


class AssistantInfo(RadarModel):
    id: str
    name: str | None = None
    model: str = ""
    tool_count: int = 0
    tool_types: list[str] = Field(default_factory=list)
    has_code_interpreter: bool = False
    has_file_search: bool = False
    vector_store_ids: list[str] = Field(default_factory=list)
    instructions_chars: int = 0
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_api(cls, obj: Any) -> AssistantInfo:
        tools = list(getattr(obj, "tools", None) or [])
        types = [getattr(t, "type", None) or "" for t in tools]
        resources = getattr(obj, "tool_resources", None)
        store_ids: list[str] = []
        if resources is not None:
            file_search = getattr(resources, "file_search", None)
            store_ids = list(getattr(file_search, "vector_store_ids", None) or [])
        return cls(
            id=obj.id,
            name=getattr(obj, "name", None),
            model=getattr(obj, "model", "") or "",
            tool_count=len(tools),
            tool_types=[t for t in types if t],
            has_code_interpreter="code_interpreter" in types,
            has_file_search="file_search" in types,
            vector_store_ids=store_ids,
            instructions_chars=len(getattr(obj, "instructions", None) or ""),
            created_at=_ts(getattr(obj, "created_at", None)),
            metadata=dict(getattr(obj, "metadata", None) or {}),
        )


class VectorStoreInfo(RadarModel):
    id: str
    name: str | None = None
    status: str = ""
    usage_bytes: int = 0
    file_count_total: int = 0
    file_count_failed: int = 0
    created_at: datetime | None = None
    last_active_at: datetime | None = None
    expires_at: datetime | None = None

    @property
    def usage_gb(self) -> float:
        return self.usage_bytes / 1024**3

    @classmethod
    def from_api(cls, obj: Any) -> VectorStoreInfo:
        counts = getattr(obj, "file_counts", None)
        return cls(
            id=obj.id,
            name=getattr(obj, "name", None),
            status=getattr(obj, "status", "") or "",
            usage_bytes=int(getattr(obj, "usage_bytes", 0) or 0),
            file_count_total=int(getattr(counts, "total", 0) or 0),
            file_count_failed=int(getattr(counts, "failed", 0) or 0),
            created_at=_ts(getattr(obj, "created_at", None)),
            last_active_at=_ts(getattr(obj, "last_active_at", None)),
            expires_at=_ts(getattr(obj, "expires_at", None)),
        )


class FineTuneInfo(RadarModel):
    id: str
    model: str = ""
    fine_tuned_model: str | None = None
    status: str = ""
    trained_tokens: int = 0
    error_message: str | None = None
    created_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def from_api(cls, obj: Any) -> FineTuneInfo:
        error = getattr(obj, "error", None)
        message = getattr(error, "message", None) if error else None
        return cls(
            id=obj.id,
            model=getattr(obj, "model", "") or "",
            fine_tuned_model=getattr(obj, "fine_tuned_model", None),
            status=getattr(obj, "status", "") or "",
            trained_tokens=int(getattr(obj, "trained_tokens", 0) or 0),
            error_message=message or None,
            created_at=_ts(getattr(obj, "created_at", None)),
            finished_at=_ts(getattr(obj, "finished_at", None)),
        )


class BatchJobInfo(RadarModel):
    id: str
    endpoint: str = ""
    status: str = ""
    completion_window: str = ""
    request_counts_total: int = 0
    request_counts_completed: int = 0
    request_counts_failed: int = 0
    created_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def failure_rate(self) -> float:
        if self.request_counts_total <= 0:
            return 0.0
        return self.request_counts_failed / self.request_counts_total * 100

    @classmethod
    def from_api(cls, obj: Any) -> BatchJobInfo:
        counts = getattr(obj, "request_counts", None)
        return cls(
            id=obj.id,
            endpoint=getattr(obj, "endpoint", "") or "",
            status=getattr(obj, "status", "") or "",
            completion_window=getattr(obj, "completion_window", "") or "",
            request_counts_total=int(getattr(counts, "total", 0) or 0),
            request_counts_completed=int(getattr(counts, "completed", 0) or 0),
            request_counts_failed=int(getattr(counts, "failed", 0) or 0),
            created_at=_ts(getattr(obj, "created_at", None)),
            completed_at=_ts(getattr(obj, "completed_at", None)),
        )


class ModelUsage(RadarModel):
    """One usage bucket for one model."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    num_requests: int = 0
    project_id: str | None = None
    bucket_start: datetime | None = None
    bucket_end: datetime | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def csv_fields(cls) -> list[str]:
        # total_tokens is a property, so it is not in model_fields — add it.
        return [*cls.model_fields, "total_tokens"]
