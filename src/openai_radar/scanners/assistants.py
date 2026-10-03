"""
AssistantScanner
----------------
Lists assistants and, as a side effect, mines their ``instructions`` for
references to external services.

The relationship pass is deliberately lexical rather than semantic: it reads
instructions text, so it finds what an operator *told* the assistant to talk
to. Treat the output as leads to confirm, not as an authoritative dependency
graph — an assistant told to "never use S3" still matches ``s3``.
"""

from __future__ import annotations

import re

from openai_radar.models.base import AssistantInfo, ServiceKind, ServiceRelationship
from openai_radar.scanners.base import PAGE_SIZE, Scanner, resource_group

#: Signal -> service kind. Keys are matched case-insensitively.
#:
#: Plain alphanumeric signals match on word boundaries so ``aws`` does not fire
#: inside "laws"; signals containing punctuation (``blob.core.windows``,
#: ``http://``) are matched as literal substrings instead, since a word
#: boundary is meaningless mid-URL.
SERVICE_SIGNALS: dict[str, ServiceKind] = {
    # AWS
    "aws": ServiceKind.AWS,
    "s3": ServiceKind.AWS,
    "ec2": ServiceKind.AWS,
    "lambda": ServiceKind.AWS,
    "dynamodb": ServiceKind.AWS,
    "sqs": ServiceKind.AWS,
    # GCP
    "gcp": ServiceKind.GCP,
    "bigquery": ServiceKind.GCP,
    "gcs": ServiceKind.GCP,
    "google cloud": ServiceKind.GCP,
    # Azure
    "azure": ServiceKind.AZURE,
    "blob.core.windows": ServiceKind.AZURE,
    "cosmosdb": ServiceKind.AZURE,
    # Databases
    "postgres": ServiceKind.DATABASE,
    "mysql": ServiceKind.DATABASE,
    "mongo": ServiceKind.DATABASE,
    "redis": ServiceKind.DATABASE,
    "neon": ServiceKind.DATABASE,
    # Messaging / notification
    "slack": ServiceKind.SLACK,
    "sendgrid": ServiceKind.EMAIL,
    "mailgun": ServiceKind.EMAIL,
    "smtp": ServiceKind.EMAIL,
    "webhook": ServiceKind.WEBHOOK,
    "http://": ServiceKind.WEBHOOK,
}

#: Characters of surrounding text kept as evidence for each match.
EVIDENCE_WINDOW = 40


def _compile(signal: str) -> re.Pattern[str]:
    if re.fullmatch(r"[a-z0-9 ]+", signal):
        return re.compile(rf"\b{re.escape(signal)}\b", re.IGNORECASE)
    return re.compile(re.escape(signal), re.IGNORECASE)


_PATTERNS: list[tuple[re.Pattern[str], str, ServiceKind]] = [
    (_compile(signal), signal, kind) for signal, kind in SERVICE_SIGNALS.items()
]


def detect_relationships(
    text: str,
    source_id: str,
    source_name: str | None = None,
) -> list[ServiceRelationship]:
    """
    Find external-service references in ``text``.

    Emits at most one relationship per signal, anchored on the first match.
    """
    if not text:
        return []

    found: list[ServiceRelationship] = []
    for pattern, signal, kind in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        start = max(0, match.start() - EVIDENCE_WINDOW)
        end = min(len(text), match.end() + EVIDENCE_WINDOW)
        evidence = " ".join(text[start:end].split())
        found.append(
            ServiceRelationship(
                source_id=source_id,
                source_name=source_name,
                kind=kind,
                signal=signal,
                evidence=evidence,
            )
        )
    return found


class AssistantScanner(Scanner):
    """Lists assistants and detects their external service relationships."""

    name = "assistants"

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.relationships: list[ServiceRelationship] = []
        """Populated as a side effect of scan(); read by Runner."""

    async def scan(self, project_id: str | None = None) -> list[AssistantInfo]:
        sdk = self._sdk(project_id)
        assistants = resource_group(sdk, "assistants")

        results: list[AssistantInfo] = []
        relationships: list[ServiceRelationship] = []

        async for raw in self._paginate(assistants.list(limit=PAGE_SIZE)):
            info = AssistantInfo.from_api(raw)
            results.append(info)
            relationships.extend(
                detect_relationships(
                    getattr(raw, "instructions", None) or "",
                    source_id=info.id,
                    source_name=info.name,
                )
            )

        # Assign at the end so a mid-scan failure leaves the previous value
        # intact rather than a half-populated list.
        self.relationships = relationships
        return results
