"""Resource models: API normalisation and CSV row shape."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from openai_radar.models.base import (
    AssistantInfo,
    BatchJobInfo,
    FineTuneInfo,
    ModelUsage,
    ServiceKind,
    ServiceRelationship,
    VectorStoreInfo,
)

EPOCH = datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)


def test_assistant_from_api_normalises_tools_and_timestamps() -> None:
    raw = SimpleNamespace(
        id="asst_1",
        name="Helper",
        model="gpt-4o",
        tools=[
            SimpleNamespace(type="code_interpreter"),
            SimpleNamespace(type="file_search"),
            SimpleNamespace(type=""),
        ],
        tool_resources=SimpleNamespace(
            file_search=SimpleNamespace(vector_store_ids=["vs_1", "vs_2"])
        ),
        instructions="hello",
        created_at=1_700_000_000,
        metadata={"team": "finops"},
    )

    info = AssistantInfo.from_api(raw)

    assert info.id == "asst_1"
    assert info.tool_count == 3
    assert info.tool_types == ["code_interpreter", "file_search"]
    assert info.has_code_interpreter is True
    assert info.has_file_search is True
    assert info.vector_store_ids == ["vs_1", "vs_2"]
    assert info.instructions_chars == 5
    assert info.created_at == EPOCH
    assert info.metadata == {"team": "finops"}


def test_assistant_from_api_tolerates_missing_optional_fields() -> None:
    raw = SimpleNamespace(id="asst_2", name=None, model=None, tools=None, instructions=None)
    raw.created_at = "not-a-timestamp"
    raw.metadata = None
    raw.tool_resources = None

    info = AssistantInfo.from_api(raw)

    assert info.model == ""
    assert info.tool_count == 0
    assert info.tool_types == []
    assert info.has_code_interpreter is False
    assert info.vector_store_ids == []
    assert info.instructions_chars == 0
    assert info.created_at is None
    assert info.metadata == {}


def test_naive_datetime_is_treated_as_utc() -> None:
    # The helper under test is what attaches UTC to a naive timestamp.
    naive = datetime(2024, 1, 2, 3, 4, 5)  # noqa: DTZ001
    info = AssistantInfo.from_api(SimpleNamespace(id="asst_3", created_at=naive))

    assert info.created_at == naive.replace(tzinfo=timezone.utc)


def test_vector_store_from_api_and_usage_gb() -> None:
    raw = SimpleNamespace(
        id="vs_1",
        name="Docs",
        status="completed",
        usage_bytes=2 * 1024**3,
        file_counts=SimpleNamespace(total=4, failed=1),
        created_at=1_700_000_000,
        last_active_at=None,
        expires_at=1_700_000_100,
    )

    info = VectorStoreInfo.from_api(raw)

    assert info.usage_gb == 2
    assert info.file_count_total == 4
    assert info.file_count_failed == 1
    assert info.expires_at == datetime.fromtimestamp(1_700_000_100, tz=timezone.utc)
    assert info.last_active_at is None


def test_fine_tune_from_api_reads_the_error_message() -> None:
    raw = SimpleNamespace(
        id="ftjob_1",
        model="gpt-4o-mini-2024-07-18",
        fine_tuned_model=None,
        status="failed",
        trained_tokens=None,
        error=SimpleNamespace(message="out of memory"),
        created_at=1_700_000_000,
        finished_at=1_700_000_050,
    )

    info = FineTuneInfo.from_api(raw)

    assert info.status == "failed"
    assert info.trained_tokens == 0
    assert info.error_message == "out of memory"
    assert info.finished_at == datetime.fromtimestamp(1_700_000_050, tz=timezone.utc)


def test_fine_tune_without_an_error_object() -> None:
    info = FineTuneInfo.from_api(SimpleNamespace(id="ftjob_2", error=None, status="succeeded"))
    assert info.error_message is None
    assert info.model == ""


def test_batch_job_failure_rate() -> None:
    empty = BatchJobInfo(id="batch_empty")
    assert empty.failure_rate == 0

    raw = SimpleNamespace(
        id="batch_1",
        endpoint="/v1/chat/completions",
        status="completed",
        completion_window="24h",
        request_counts=SimpleNamespace(total=8, completed=5, failed=3),
        created_at=None,
        completed_at=None,
    )
    info = BatchJobInfo.from_api(raw)
    assert info.request_counts_failed == 3
    assert info.failure_rate == 37.5


def test_model_usage_totals_and_csv_fields() -> None:
    usage = ModelUsage(model="gpt-4o", input_tokens=3, output_tokens=4, cached_tokens=9)
    assert usage.total_tokens == 7
    assert usage.csv_fields()[-1] == "total_tokens"
    assert usage.csv_row()["total_tokens"] == 7


def test_csv_row_formats_datetimes_enums_lists_and_dicts() -> None:
    created = datetime(2024, 5, 1, tzinfo=timezone.utc)
    assistant = AssistantInfo(
        id="asst_1",
        tool_types=["file_search", "code_interpreter"],
        created_at=created,
        metadata={"team": "finops", "env": "prod"},
    )
    row = assistant.csv_row()

    assert row["created_at"] == created.isoformat()
    assert row["tool_types"] == "file_search,code_interpreter"
    assert row["metadata"] == "team=finops;env=prod"

    relationship = ServiceRelationship(
        source_id="asst_1",
        kind=ServiceKind.AWS,
        signal="s3",
    )
    assert relationship.csv_row()["kind"] == "aws"
