"""FindingEngine rules and the thresholds callers can override."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from openai_radar.findings import Finding, FindingEngine, Severity
from openai_radar.models.base import (
    AssistantInfo,
    BatchJobInfo,
    FineTuneInfo,
    ModelUsage,
    VectorStoreInfo,
)


def _ids(findings: list[Finding]) -> list[str]:
    return [finding.rule_id for finding in findings]


def test_quiet_resources_produce_no_findings() -> None:
    engine = FindingEngine()
    findings = engine.run(
        assistants=[AssistantInfo(id="asst_1", name="Helper", tool_count=1)],
        vector_stores=[VectorStoreInfo(id="vs_1", usage_bytes=10)],
        fine_tunes=[FineTuneInfo(id="ft_1", status="succeeded")],
        batch_jobs=[BatchJobInfo(id="batch_1", request_counts_total=10, request_counts_failed=1)],
        usage=[ModelUsage(model="gpt-4o", input_tokens=100, output_tokens=100)],
    )
    assert findings == []


def test_missing_inputs_are_skipped() -> None:
    assert FindingEngine().run() == []


def test_assistant_rules() -> None:
    findings = FindingEngine().check_assistants(
        [
            AssistantInfo(id="asst_bare", name=None, tool_count=0),
            AssistantInfo(
                id="asst_code",
                name="Coder",
                tool_count=1,
                has_code_interpreter=True,
            ),
            AssistantInfo(id="asst_both", name="Both", tool_count=0, has_code_interpreter=True),
        ]
    )

    assert _ids(findings) == ["ASST_001", "ASST_002", "ASST_001", "ASST_002"]
    bare, code, both_low, both_info = findings
    assert bare.severity is Severity.LOW
    assert bare.resource_id == "asst_bare"
    assert "asst_bare" in bare.detail
    assert code.severity is Severity.INFO
    assert "Coder" in code.detail
    assert both_low.severity is Severity.LOW
    assert both_info.severity is Severity.INFO


def test_large_vector_store_includes_the_byte_count() -> None:
    five_gb = 5 * 1024**3
    findings = FindingEngine().check_vector_stores(
        [
            VectorStoreInfo(id="vs_small", usage_bytes=five_gb - 1),
            VectorStoreInfo(id="vs_big", name="Archive", usage_bytes=five_gb),
        ]
    )

    assert _ids(findings) == ["VS_001"]
    finding = findings[0]
    assert finding.severity is Severity.MEDIUM
    assert finding.resource_id == "vs_big"
    assert finding.title == "Large vector store (5.0 GB)"
    assert "5.00 GB" in finding.detail
    assert finding.as_dict()["usage_bytes"] == five_gb
    assert finding.as_dict()["severity"] == "medium"


def test_vector_store_expiring_soon(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``timedelta.days`` truncates, so pin "now" and the engine sees exact day counts.
    fixed = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    stores = [
        VectorStoreInfo(id="vs_soon", name="Soon", expires_at=fixed + timedelta(days=3)),
        VectorStoreInfo(id="vs_edge", expires_at=fixed + timedelta(days=7)),
        VectorStoreInfo(id="vs_later", expires_at=fixed + timedelta(days=8)),
        VectorStoreInfo(id="vs_past", expires_at=fixed - timedelta(days=1)),
        VectorStoreInfo(id="vs_today", expires_at=fixed + timedelta(hours=12)),
        VectorStoreInfo(id="vs_none"),
    ]

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            return fixed

    monkeypatch.setattr("datetime.datetime", _Frozen)
    findings = FindingEngine().check_vector_stores(stores)

    assert _ids(findings) == ["VS_002", "VS_002"]
    assert {finding.resource_id for finding in findings} == {"vs_soon", "vs_edge"}
    soon = next(finding for finding in findings if finding.resource_id == "vs_soon")
    assert soon.severity is Severity.HIGH
    assert soon.as_dict()["expires_at"] == stores[0].expires_at.isoformat()
    assert "3 day" in soon.title


def test_failed_fine_tune() -> None:
    findings = FindingEngine().check_fine_tunes(
        [
            FineTuneInfo(id="ft_ok", model="gpt-4o-mini", status="succeeded"),
            FineTuneInfo(id="ft_bad", model="gpt-4o-mini", status="failed"),
        ]
    )

    assert _ids(findings) == ["FT_001"]
    assert findings[0].severity is Severity.MEDIUM
    assert "gpt-4o-mini" in findings[0].detail


def test_batch_failure_rate_threshold_is_exclusive() -> None:
    findings = FindingEngine().check_batch_jobs(
        [
            BatchJobInfo(id="batch_empty", request_counts_total=0, request_counts_failed=0),
            BatchJobInfo(id="batch_ok", request_counts_total=100, request_counts_failed=10),
            BatchJobInfo(
                id="batch_bad",
                endpoint="/v1/chat/completions",
                request_counts_total=3,
                request_counts_failed=1,
            ),
        ]
    )

    assert _ids(findings) == ["BATCH_001"]
    finding = findings[0]
    assert finding.severity is Severity.HIGH
    assert finding.title == "High batch failure rate (33%)"
    assert "1/3" in finding.detail
    assert finding.as_dict()["fail_pct"] == 33.3


def test_usage_is_aggregated_per_model() -> None:
    threshold = FindingEngine.HIGH_TOKEN_THRESHOLD
    findings = FindingEngine().check_usage(
        [
            ModelUsage(model="gpt-4o", input_tokens=threshold - 5, output_tokens=0),
            ModelUsage(model="gpt-4o", input_tokens=0, output_tokens=5),
            ModelUsage(model="gpt-4o-mini", input_tokens=threshold - 1, output_tokens=0),
        ]
    )

    assert _ids(findings) == ["USAGE_001"]
    assert findings[0].resource_id == "gpt-4o"
    assert findings[0].severity is Severity.MEDIUM
    assert f"{threshold:,}" in findings[0].detail
    assert findings[0].as_dict()["total_tokens"] == threshold


def test_run_preserves_rule_group_order() -> None:
    now = datetime.now(tz=timezone.utc)
    findings = FindingEngine().run(
        assistants=[AssistantInfo(id="asst_1", tool_count=0)],
        vector_stores=[
            VectorStoreInfo(id="vs_1", usage_bytes=5 * 1024**3, expires_at=now + timedelta(days=2))
        ],
        fine_tunes=[FineTuneInfo(id="ft_1", status="failed")],
        batch_jobs=[BatchJobInfo(id="batch_1", request_counts_total=2, request_counts_failed=2)],
        usage=[ModelUsage(model="gpt-4o", input_tokens=10_000_000, output_tokens=0)],
    )

    assert _ids(findings) == ["ASST_001", "VS_001", "VS_002", "FT_001", "BATCH_001", "USAGE_001"]


def test_constructor_thresholds_override_the_defaults() -> None:
    engine = FindingEngine(
        LARGE_VECTOR_STORE_BYTES=10,
        HIGH_TOKEN_THRESHOLD=5,
        EXPIRING_SOON_DAYS=1,
    )
    soon = datetime.now(tz=timezone.utc) + timedelta(days=3)

    findings = engine.run(
        vector_stores=[VectorStoreInfo(id="vs_1", usage_bytes=10, expires_at=soon)],
        usage=[ModelUsage(model="gpt-4o", input_tokens=5, output_tokens=0)],
    )

    assert _ids(findings) == ["VS_001", "USAGE_001"]


class _OrgEngine(FindingEngine):
    def check_assistants(self, assistants: list[AssistantInfo]) -> list[Finding]:
        findings = super().check_assistants(assistants)
        findings.append(
            Finding(
                rule_id="ORG_001",
                severity=Severity.CRITICAL,
                resource_type="assistant",
                resource_id=assistants[0].id,
                title="Org rule",
                detail="Custom check fired.",
                recommendation="Review the assistant.",
            )
        )
        return findings


def test_subclass_can_extend_a_rule_group() -> None:
    findings = _OrgEngine().run(assistants=[AssistantInfo(id="asst_1", tool_count=2)])
    assert _ids(findings) == ["ORG_001"]
    assert findings[0].severity is Severity.CRITICAL
