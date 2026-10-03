"""
Runner
------
Orchestrates all scanners and aggregates results into a RunResult.
Mirrors the openai-agents Runner API surface so it's immediately
familiar to users of that SDK.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from openai_radar.client import RadarClient
from openai_radar.findings import Finding, FindingEngine
from openai_radar.models.base import (
    AssistantInfo,
    BatchJobInfo,
    FineTuneInfo,
    ModelUsage,
    ServiceRelationship,
    VectorStoreInfo,
)
from openai_radar.scanners.assistants import AssistantScanner
from openai_radar.scanners.batch_jobs import BatchJobScanner
from openai_radar.scanners.fine_tunes import FineTuneScanner
from openai_radar.scanners.usage import UsageScanner
from openai_radar.scanners.vector_stores import VectorStoreScanner

if TYPE_CHECKING:
    pass


@dataclass
class RunConfig:
    """Control which scanners run and how."""

    project_id: str | None = None
    """Scope all scanners to this project ID."""

    scan_assistants: bool = True
    scan_vector_stores: bool = True
    scan_fine_tunes: bool = True
    scan_batch_jobs: bool = True
    scan_usage: bool = True
    usage_lookback_days: int = 30

    run_findings: bool = True
    """Whether to run the FindingEngine after scanning."""

    finding_engine: FindingEngine = field(default_factory=FindingEngine)
    """Override with a custom FindingEngine subclass."""


@dataclass
class RunResult:
    """
    Container for all scan output.  Passed to exporters.
    """

    config: RunConfig

    assistants: list[AssistantInfo] = field(default_factory=list)
    vector_stores: list[VectorStoreInfo] = field(default_factory=list)
    fine_tunes: list[FineTuneInfo] = field(default_factory=list)
    batch_jobs: list[BatchJobInfo] = field(default_factory=list)
    usage: list[ModelUsage] = field(default_factory=list)
    relationships: list[ServiceRelationship] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_csv(self, output_dir: str | Path = ".") -> list[Path]:
        """Write one CSV per resource type.  Returns list of written paths."""
        from openai_radar.exporters.csv import CsvExporter

        return CsvExporter(self).export(output_dir)

    def export_drawio(self, path: str | Path = "openai_arch.drawio") -> Path:
        """Write a draw.io architecture diagram.  Returns the written path."""
        from openai_radar.exporters.drawio import DrawioExporter

        return DrawioExporter(self).export(path)

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "openai-radar scan summary",
            "─" * 40,
            f"  Assistants:    {len(self.assistants)}",
            f"  Vector stores: {len(self.vector_stores)}",
            f"  Fine-tunes:    {len(self.fine_tunes)}",
            f"  Batch jobs:    {len(self.batch_jobs)}",
            f"  Usage records: {len(self.usage)}",
            f"  Relationships: {len(self.relationships)}",
            f"  Findings:      {len(self.findings)}",
        ]
        return "\n".join(lines)


class Runner:
    """
    Static factory — mirrors openai-agents Runner API.

    Usage::

        result = Runner.run_sync(client)
        result = await Runner.run(client)
        result = Runner.run_sync(client, RunConfig(scan_batch_jobs=False))
    """

    @staticmethod
    async def run(
        client: RadarClient,
        config: RunConfig | None = None,
    ) -> RunResult:
        """Async entry point."""
        cfg = config or RunConfig()
        result = RunResult(config=cfg)

        tasks: list[asyncio.Task] = []  # type: ignore[type-arg]

        async def _run_scanner(coro):  # type: ignore[no-untyped-def]
            return await coro

        assistant_scanner = AssistantScanner(client)

        coros = []
        if cfg.scan_assistants:
            coros.append(("assistants", assistant_scanner.scan(cfg.project_id)))
        if cfg.scan_vector_stores:
            coros.append(("vector_stores", VectorStoreScanner(client).scan(cfg.project_id)))
        if cfg.scan_fine_tunes:
            coros.append(("fine_tunes", FineTuneScanner(client).scan(cfg.project_id)))
        if cfg.scan_batch_jobs:
            coros.append(("batch_jobs", BatchJobScanner(client).scan(cfg.project_id)))
        if cfg.scan_usage:
            coros.append(
                (
                    "usage",
                    UsageScanner(client, lookback_days=cfg.usage_lookback_days).scan(
                        cfg.project_id
                    ),
                )
            )

        # Run all scanners concurrently
        gathered = await asyncio.gather(*[c for _, c in coros], return_exceptions=True)

        for (key, _), outcome in zip(coros, gathered):
            if isinstance(outcome, Exception):
                # Surface non-fatal scanner failures without aborting
                import warnings

                warnings.warn(f"Scanner '{key}' failed: {outcome}", RuntimeWarning, stacklevel=2)
            else:
                setattr(result, key, outcome)

        if cfg.scan_assistants:
            result.relationships = assistant_scanner.relationships

        if cfg.run_findings:
            result.findings = cfg.finding_engine.run(
                assistants=result.assistants,
                vector_stores=result.vector_stores,
                fine_tunes=result.fine_tunes,
                batch_jobs=result.batch_jobs,
                usage=result.usage,
            )

        return result

    @staticmethod
    def run_sync(
        client: RadarClient,
        config: RunConfig | None = None,
    ) -> RunResult:
        """Synchronous convenience wrapper around Runner.run()."""
        return asyncio.run(Runner.run(client, config))
