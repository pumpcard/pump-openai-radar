"""CSV and draw.io export."""

from __future__ import annotations

import csv
from datetime import datetime, timezone

from lxml import etree

from openai_radar.findings import Finding, Severity
from openai_radar.models.base import (
    AssistantInfo,
    BatchJobInfo,
    FineTuneInfo,
    ModelUsage,
    ServiceKind,
    ServiceRelationship,
    VectorStoreInfo,
)
from openai_radar.runner import RunConfig, RunResult


def _result() -> RunResult:
    created = datetime(2024, 6, 1, tzinfo=timezone.utc)
    return RunResult(
        config=RunConfig(),
        assistants=[
            AssistantInfo(
                id="asst_1",
                name="A" * 40,
                model="gpt-4o",
                tool_count=1,
                vector_store_ids=["vs_1", "vs_missing"],
                created_at=created,
            )
        ],
        vector_stores=[VectorStoreInfo(id="vs_1", name="Docs", usage_bytes=1024**3)],
        fine_tunes=[FineTuneInfo(id="ft_1", status="succeeded")],
        batch_jobs=[
            BatchJobInfo(
                id="batch_1",
                status="completed",
                request_counts_total=4,
                request_counts_failed=1,
            )
        ],
        usage=[ModelUsage(model="gpt-4o", input_tokens=3, output_tokens=4)],
        relationships=[
            ServiceRelationship(source_id="asst_1", kind=ServiceKind.AWS, signal="s3"),
            ServiceRelationship(source_id="asst_1", kind=ServiceKind.AWS, signal="lambda"),
            ServiceRelationship(source_id="asst_1", kind=ServiceKind.SLACK, signal="slack"),
        ],
        findings=[
            Finding(
                rule_id="VS_001",
                severity=Severity.MEDIUM,
                resource_type="vector_store",
                resource_id="vs_1",
                title="Large",
                detail="detail",
                recommendation="trim it",
                metadata={"usage_bytes": 1024},
            ),
            Finding(
                rule_id="BATCH_001",
                severity=Severity.HIGH,
                resource_type="batch_job",
                resource_id="batch_1",
                title="Failures",
                detail="detail",
                recommendation="retry",
                metadata={"fail_pct": 25.0},
            ),
        ],
    )


def test_csv_export_skips_empty_collections(tmp_path) -> None:
    result = RunResult(config=RunConfig(), assistants=[AssistantInfo(id="asst_1", name="Helper")])

    written = result.export_csv(tmp_path)

    assert written == [tmp_path / "assistants.csv"]
    assert not (tmp_path / "findings.csv").exists()
    with (tmp_path / "assistants.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["id"] == "asst_1"
    assert rows[0]["name"] == "Helper"


def test_csv_export_writes_each_populated_collection(tmp_path) -> None:
    written = _result().export_csv(tmp_path / "nested" / "out")

    names = [path.name for path in written]
    assert names == [
        "assistants.csv",
        "vector_stores.csv",
        "fine_tunes.csv",
        "batch_jobs.csv",
        "usage.csv",
        "relationships.csv",
        "findings.csv",
    ]

    out = tmp_path / "nested" / "out"
    with (out / "usage.csv").open(encoding="utf-8") as fh:
        usage = list(csv.DictReader(fh))
    assert usage[0]["total_tokens"] == "7"

    with (out / "relationships.csv").open(encoding="utf-8") as fh:
        relationships = list(csv.DictReader(fh))
    assert [row["kind"] for row in relationships] == ["aws", "aws", "slack"]

    with (out / "findings.csv").open(encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    assert header[:7] == [
        "rule_id",
        "severity",
        "resource_type",
        "resource_id",
        "title",
        "detail",
        "recommendation",
    ]
    assert header[7:] == ["fail_pct", "usage_bytes"]


def test_drawio_export_places_nodes_and_edges(tmp_path) -> None:
    path = _result().export_drawio(tmp_path / "diagrams" / "arch.drawio")

    tree = etree.parse(str(path))
    root = tree.getroot()
    assert root.tag == "mxfile"
    assert root.get("host") == "openai-radar"

    cells = list(root.iter("mxCell"))
    values = [cell.get("value") or "" for cell in cells]
    joined = "\n".join(values)

    assert "Assistants" in joined
    assert "External services" in joined
    assert ("A" * 33 + "…") in joined
    assert "AWS" in values
    assert "Slack" in values
    assert "1.00 GB" in joined
    assert "25% failed" in joined

    attached = [cell for cell in cells if cell.get("value") == "attached"]
    assert len(attached) == 1
    assert attached[0].get("source", "").startswith("assistant-")
    assert attached[0].get("target", "").startswith("vector_store-")

    dashed = [cell for cell in cells if "dashed=1" in (cell.get("style") or "")]
    labels = {cell.get("value") for cell in dashed}
    assert labels == {"s3", "slack"}


def test_drawio_export_of_an_empty_result_is_still_valid_xml(tmp_path) -> None:
    path = RunResult(config=RunConfig()).export_drawio(tmp_path / "empty.drawio")
    root = etree.parse(str(path)).getroot()
    ids = [cell.get("id") for cell in root.iter("mxCell")]
    assert ids[:2] == ["0", "1"]
