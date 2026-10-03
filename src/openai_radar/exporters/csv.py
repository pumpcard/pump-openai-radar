"""
CsvExporter
-----------
One CSV per resource type, matching the Radar suite schema.

Note this module is ``openai_radar.exporters.csv`` — the bare ``import csv``
below still resolves to the standard library, since Python 3 imports are
absolute.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai_radar.models.base import RadarModel

if TYPE_CHECKING:  # pragma: no cover
    from openai_radar.runner import RunResult


def _write(path: Path, fieldnames: Sequence[str], rows: list[dict[str, Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


class CsvExporter:
    """Writes a RunResult to one CSV per populated resource type."""

    def __init__(self, result: RunResult) -> None:
        self.result = result

    def export(self, output_dir: str | Path = ".") -> list[Path]:
        """Write CSVs into ``output_dir``. Returns the paths actually written."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []

        collections: list[tuple[str, Sequence[RadarModel]]] = [
            ("assistants.csv", self.result.assistants),
            ("vector_stores.csv", self.result.vector_stores),
            ("fine_tunes.csv", self.result.fine_tunes),
            ("batch_jobs.csv", self.result.batch_jobs),
            ("usage.csv", self.result.usage),
            ("relationships.csv", self.result.relationships),
        ]

        for filename, items in collections:
            if not items:
                # Skip empty files rather than leaving header-only stubs that
                # look like a successful-but-empty scan of a real resource.
                continue
            fields = type(items[0]).csv_fields()
            rows = [item.csv_row() for item in items]
            written.append(_write(out / filename, fields, rows))

        if self.result.findings:
            findings_rows = [f.as_dict() for f in self.result.findings]
            # Findings carry rule-specific metadata keys, so the union of all
            # keys becomes the header — a fixed list would silently drop them.
            base = [
                "rule_id",
                "severity",
                "resource_type",
                "resource_id",
                "title",
                "detail",
                "recommendation",
            ]
            extra = sorted({k for row in findings_rows for k in row} - set(base))
            written.append(_write(out / "findings.csv", base + extra, findings_rows))

        return written
