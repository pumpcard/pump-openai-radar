"""
FindingEngine
-------------
Analyses scan results and emits structured Findings — the FinOps layer
of the SDK.  Designed to be extended: subclass and override check_*
methods to add org-specific rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openai_radar.models.base import (
    AssistantInfo,
    BatchJobInfo,
    FineTuneInfo,
    ModelUsage,
    VectorStoreInfo,
)


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Finding:
    rule_id: str
    severity: Severity
    resource_type: str
    resource_id: str
    title: str
    detail: str
    recommendation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "title": self.title,
            "detail": self.detail,
            "recommendation": self.recommendation,
            **self.metadata,
        }


class FindingEngine:
    """
    Rules engine.  Call run() after scanners complete.

    Extend via subclassing:
        class MyEngine(FindingEngine):
            def check_assistants(self, assistants):
                findings = super().check_assistants(assistants)
                # add custom checks ...
                return findings
    """

    # Thresholds (override in subclass or via constructor kwargs)
    LARGE_VECTOR_STORE_BYTES: int = 5 * 1024**3  # 5 GB
    HIGH_TOKEN_THRESHOLD: int = 10_000_000  # 10M tokens / 30d
    EXPIRING_SOON_DAYS: int = 7  # vector store expiry warning

    def __init__(self, **thresholds: Any) -> None:
        for k, v in thresholds.items():
            setattr(self, k, v)

    def run(
        self,
        *,
        assistants: list[AssistantInfo] | None = None,
        vector_stores: list[VectorStoreInfo] | None = None,
        fine_tunes: list[FineTuneInfo] | None = None,
        batch_jobs: list[BatchJobInfo] | None = None,
        usage: list[ModelUsage] | None = None,
    ) -> list[Finding]:
        findings: list[Finding] = []
        if assistants:
            findings.extend(self.check_assistants(assistants))
        if vector_stores:
            findings.extend(self.check_vector_stores(vector_stores))
        if fine_tunes:
            findings.extend(self.check_fine_tunes(fine_tunes))
        if batch_jobs:
            findings.extend(self.check_batch_jobs(batch_jobs))
        if usage:
            findings.extend(self.check_usage(usage))
        return findings

    # ------------------------------------------------------------------
    # Rule groups
    # ------------------------------------------------------------------

    def check_assistants(self, assistants: list[AssistantInfo]) -> list[Finding]:
        findings: list[Finding] = []
        for a in assistants:
            # Assistants with no tools — unusual, likely misconfigured
            if a.tool_count == 0:
                findings.append(
                    Finding(
                        rule_id="ASST_001",
                        severity=Severity.LOW,
                        resource_type="assistant",
                        resource_id=a.id,
                        title="Assistant has no tools configured",
                        detail=f"Assistant '{a.name or a.id}' has zero tools attached.",
                        recommendation="Verify this assistant is intentionally tool-free or add appropriate tools.",
                    )
                )
            # Using code_interpreter (can incur storage costs)
            if a.has_code_interpreter:
                findings.append(
                    Finding(
                        rule_id="ASST_002",
                        severity=Severity.INFO,
                        resource_type="assistant",
                        resource_id=a.id,
                        title="Code Interpreter enabled — file storage costs apply",
                        detail=f"Assistant '{a.name or a.id}' has Code Interpreter; uploaded files incur storage charges.",
                        recommendation="Delete unused Code Interpreter files periodically to avoid storage drift.",
                    )
                )
        return findings

    def check_vector_stores(self, stores: list[VectorStoreInfo]) -> list[Finding]:
        from datetime import datetime, timezone

        findings: list[Finding] = []
        now = datetime.now(tz=timezone.utc)

        for vs in stores:
            # Large stores
            if vs.usage_bytes >= self.LARGE_VECTOR_STORE_BYTES:
                gb = vs.usage_bytes / 1024**3
                findings.append(
                    Finding(
                        rule_id="VS_001",
                        severity=Severity.MEDIUM,
                        resource_type="vector_store",
                        resource_id=vs.id,
                        title=f"Large vector store ({gb:.1f} GB)",
                        detail=f"Vector store '{vs.name or vs.id}' is using {gb:.2f} GB.",
                        recommendation="Review for stale files; delete unused chunks to reduce storage cost.",
                        metadata={"usage_bytes": vs.usage_bytes},
                    )
                )
            # Expiring soon
            if vs.expires_at:
                days_left = (vs.expires_at - now).days
                if 0 < days_left <= self.EXPIRING_SOON_DAYS:
                    findings.append(
                        Finding(
                            rule_id="VS_002",
                            severity=Severity.HIGH,
                            resource_type="vector_store",
                            resource_id=vs.id,
                            title=f"Vector store expires in {days_left} day(s)",
                            detail=f"Vector store '{vs.name or vs.id}' expires {vs.expires_at.date()}.",
                            recommendation="Extend or recreate the vector store before expiry to avoid data loss.",
                            metadata={"expires_at": vs.expires_at.isoformat()},
                        )
                    )
        return findings

    def check_fine_tunes(self, jobs: list[FineTuneInfo]) -> list[Finding]:
        findings: list[Finding] = []
        for ft in jobs:
            if ft.status == "failed":
                findings.append(
                    Finding(
                        rule_id="FT_001",
                        severity=Severity.MEDIUM,
                        resource_type="fine_tune",
                        resource_id=ft.id,
                        title="Failed fine-tune job",
                        detail=f"Fine-tune job '{ft.id}' (base: {ft.model}) ended in failure.",
                        recommendation="Check job events for the failure reason and retry or delete this job.",
                    )
                )
        return findings

    def check_batch_jobs(self, jobs: list[BatchJobInfo]) -> list[Finding]:
        findings: list[Finding] = []
        for bj in jobs:
            if bj.request_counts_total > 0:
                fail_pct = bj.request_counts_failed / bj.request_counts_total * 100
                if fail_pct > 10:
                    findings.append(
                        Finding(
                            rule_id="BATCH_001",
                            severity=Severity.HIGH,
                            resource_type="batch_job",
                            resource_id=bj.id,
                            title=f"High batch failure rate ({fail_pct:.0f}%)",
                            detail=(
                                f"Batch job '{bj.id}' on {bj.endpoint}: "
                                f"{bj.request_counts_failed}/{bj.request_counts_total} requests failed."
                            ),
                            recommendation="Inspect the output file for error details and resubmit failed rows.",
                            metadata={"fail_pct": round(fail_pct, 1)},
                        )
                    )
        return findings

    def check_usage(self, usage: list[ModelUsage]) -> list[Finding]:
        findings: list[Finding] = []
        # Aggregate per model
        totals: dict[str, int] = {}
        for u in usage:
            totals[u.model] = totals.get(u.model, 0) + u.total_tokens

        for model, total in totals.items():
            if total >= self.HIGH_TOKEN_THRESHOLD:
                findings.append(
                    Finding(
                        rule_id="USAGE_001",
                        severity=Severity.MEDIUM,
                        resource_type="model_usage",
                        resource_id=model,
                        title=f"High token consumption on {model}",
                        detail=f"{total:,} tokens consumed in the scan period.",
                        recommendation=(
                            "Review which assistants or API callers are driving this volume. "
                            "Consider caching, prompt shortening, or switching to a smaller model."
                        ),
                        metadata={"total_tokens": total},
                    )
                )
        return findings
