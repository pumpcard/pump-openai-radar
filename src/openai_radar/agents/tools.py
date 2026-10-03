"""
Radar tools for the openai-agents SDK.

Requires the ``agents`` extra::

    pip install openai-radar[agents]

Tools share a module-level cache so a conversation can scan once and then
export or re-analyse without paying for a second pass over the API. Call
``reset_session()`` between unrelated runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from agents import function_tool
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise ImportError(
        "openai-agents SDK not found. Install with: pip install openai-radar[agents]"
    ) from exc

from openai_radar.client import RadarClient
from openai_radar.findings import FindingEngine
from openai_radar.runner import RunConfig, Runner, RunResult

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_client: RadarClient | None = None
_result: RunResult | None = None


def reset_session() -> None:
    """Drop the cached client and scan result."""
    global _client, _result
    _client = None
    _result = None


def _get_client() -> RadarClient:
    global _client
    if _client is None:
        _client = RadarClient()
    return _client


def _get_result() -> RunResult:
    global _result
    if _result is None:
        _result = RunResult(config=RunConfig())
    return _result


async def _scan_one(field: str, **config_kwargs: Any) -> RunResult:
    """
    Run exactly one scanner and merge it into the cached result.

    Every ``scan_*`` flag defaults off, then the requested one is switched on,
    so asking for assistants does not silently bill a full org scan.
    """
    flags = {
        "scan_assistants": False,
        "scan_vector_stores": False,
        "scan_fine_tunes": False,
        "scan_batch_jobs": False,
        "scan_usage": False,
        "run_findings": False,
    }
    flags[f"scan_{field}"] = True
    config = RunConfig(**{**flags, **config_kwargs})

    partial = await Runner.run(_get_client(), config)

    cached = _get_result()
    setattr(cached, field, getattr(partial, field))
    if field == "assistants":
        cached.relationships = partial.relationships
    return cached


# ---------------------------------------------------------------------------
# Scan tools
# ---------------------------------------------------------------------------


@function_tool
async def scan_assistants() -> str:
    """List all assistants in the org and detect their external service relationships."""
    result = await _scan_one("assistants")
    lines = [f"Found {len(result.assistants)} assistant(s)."]
    for a in result.assistants[:50]:
        lines.append(f"  {a.id} | {a.name or '(unnamed)'} | model={a.model} | tools={a.tool_count}")
    if result.relationships:
        lines.append(f"\nDetected {len(result.relationships)} service relationship(s):")
        for rel in result.relationships[:50]:
            lines.append(f"  {rel.source_name or rel.source_id} -> {rel.kind.value} ({rel.signal})")
    return "\n".join(lines)


@function_tool
async def scan_vector_stores() -> str:
    """List vector stores with their storage usage and expiry."""
    result = await _scan_one("vector_stores")
    lines = [f"Found {len(result.vector_stores)} vector store(s)."]
    for vs in result.vector_stores[:50]:
        expiry = vs.expires_at.date().isoformat() if vs.expires_at else "no expiry"
        lines.append(
            f"  {vs.id} | {vs.name or '(unnamed)'} | {vs.usage_gb:.2f} GB | "
            f"{vs.file_count_total} files | {expiry}"
        )
    return "\n".join(lines)


@function_tool
async def scan_fine_tunes() -> str:
    """List fine-tuning jobs and their status."""
    result = await _scan_one("fine_tunes")
    lines = [f"Found {len(result.fine_tunes)} fine-tune job(s)."]
    for ft in result.fine_tunes[:50]:
        lines.append(
            f"  {ft.id} | base={ft.model} | status={ft.status} | "
            f"trained_tokens={ft.trained_tokens:,}"
        )
    return "\n".join(lines)


@function_tool
async def scan_batch_jobs() -> str:
    """List batch jobs with their per-request success and failure counts."""
    result = await _scan_one("batch_jobs")
    lines = [f"Found {len(result.batch_jobs)} batch job(s)."]
    for bj in result.batch_jobs[:50]:
        lines.append(
            f"  {bj.id} | {bj.endpoint} | status={bj.status} | "
            f"{bj.request_counts_failed}/{bj.request_counts_total} failed "
            f"({bj.failure_rate:.0f}%)"
        )
    return "\n".join(lines)


@function_tool
async def scan_usage(lookback_days: int = 30) -> str:
    """
    Pull org-wide token usage per model. Requires an admin key.

    Args:
        lookback_days: How many days of usage history to pull.
    """
    result = await _scan_one("usage", usage_lookback_days=lookback_days)
    if not result.usage:
        return (
            "No usage data returned. The Usage API needs an admin key — "
            "set OPENAI_ADMIN_KEY and retry."
        )

    totals: dict[str, int] = {}
    requests: dict[str, int] = {}
    for u in result.usage:
        totals[u.model] = totals.get(u.model, 0) + u.total_tokens
        requests[u.model] = requests.get(u.model, 0) + u.num_requests

    lines = [f"Token usage over the last {lookback_days} day(s):"]
    for model, total in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {model}: {total:,} tokens across {requests[model]:,} requests")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis and export tools
# ---------------------------------------------------------------------------


@function_tool
async def run_findings() -> str:
    """Run the findings engine over everything scanned so far in this session."""
    result = _get_result()
    if not any(
        [
            result.assistants,
            result.vector_stores,
            result.fine_tunes,
            result.batch_jobs,
            result.usage,
        ]
    ):
        return "Nothing scanned yet — run one or more scan tools first."

    result.findings = FindingEngine().run(
        assistants=result.assistants,
        vector_stores=result.vector_stores,
        fine_tunes=result.fine_tunes,
        batch_jobs=result.batch_jobs,
        usage=result.usage,
    )
    if not result.findings:
        return "No findings — nothing flagged against the current rule set."

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    ranked = sorted(result.findings, key=lambda f: order.get(f.severity.value, 9))

    lines = [f"{len(ranked)} finding(s), most severe first:"]
    for f in ranked:
        lines.append(f"  [{f.severity.value.upper()}] {f.rule_id} {f.title} ({f.resource_id})")
        lines.append(f"      {f.detail}")
        lines.append(f"      Fix: {f.recommendation}")
    return "\n".join(lines)


@function_tool
async def export_csv(output_dir: str = "./out") -> str:
    """
    Write the scanned inventory to one CSV per resource type.

    Args:
        output_dir: Directory to write the CSV files into.
    """
    result = _get_result()
    written = result.export_csv(output_dir)
    if not written:
        return "Nothing to export — run one or more scan tools first."
    return "Wrote:\n" + "\n".join(f"  {p}" for p in written)


@function_tool
async def export_drawio(path: str = "./out/openai_arch.drawio") -> str:
    """
    Write a draw.io architecture diagram of the scanned resources.

    Args:
        path: Destination .drawio file.
    """
    result = _get_result()
    written = result.export_drawio(Path(path))
    return f"Wrote diagram to {written}"


__all__ = [
    "export_csv",
    "export_drawio",
    "reset_session",
    "run_findings",
    "scan_assistants",
    "scan_batch_jobs",
    "scan_fine_tunes",
    "scan_usage",
    "scan_vector_stores",
]
