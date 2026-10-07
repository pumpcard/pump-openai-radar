"""
DrawioExporter
--------------
Renders a RunResult as a draw.io (diagrams.net) XML file.

Layout is a fixed set of columns — one per resource type, plus a column of
external services on the right. Edges run from assistants to the services
their instructions referenced, and to the vector stores they have attached.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lxml import etree

from openai_radar.models.base import ServiceKind

if TYPE_CHECKING:  # pragma: no cover
    from openai_radar.runner import RunResult

# Geometry
COL_WIDTH = 240
NODE_WIDTH = 180
NODE_HEIGHT = 60
V_GAP = 20
TOP_MARGIN = 40
LEFT_MARGIN = 40

FILL = {
    "assistant": "#d5e8d4;strokeColor=#82b366",
    "vector_store": "#dae8fc;strokeColor=#6c8ebf",
    "fine_tune": "#ffe6cc;strokeColor=#d79b00",
    "batch_job": "#fff2cc;strokeColor=#d6b656",
    "service": "#f8cecc;strokeColor=#b85450",
}

SERVICE_LABEL = {
    ServiceKind.AWS: "AWS",
    ServiceKind.GCP: "Google Cloud",
    ServiceKind.AZURE: "Azure",
    ServiceKind.DATABASE: "Database",
    ServiceKind.SLACK: "Slack",
    ServiceKind.EMAIL: "Email",
    ServiceKind.WEBHOOK: "Webhook",
}


def _style(kind: str) -> str:
    return (
        f"rounded=1;whiteSpace=wrap;html=1;fillColor={FILL[kind]};verticalAlign=middle;fontSize=11;"
    )


def _truncate(text: str, limit: int = 34) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class DrawioExporter:
    """Writes a RunResult to a draw.io XML diagram."""

    def __init__(self, result: RunResult) -> None:
        self.result = result
        self._seq = 0

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq}"

    def export(self, path: str | Path = "openai_arch.drawio") -> Path:
        target = Path(path)
        if target.parent != Path(""):
            target.parent.mkdir(parents=True, exist_ok=True)

        mxfile = etree.Element("mxfile", host="pump-openai-radar", type="device")
        diagram = etree.SubElement(mxfile, "diagram", name="OpenAI Architecture", id="radar")
        model = etree.SubElement(
            diagram,
            "mxGraphModel",
            dx="1200",
            dy="800",
            grid="1",
            gridSize="10",
            page="1",
            pageWidth="1600",
            pageHeight="1200",
        )
        root = etree.SubElement(model, "root")
        etree.SubElement(root, "mxCell", id="0")
        etree.SubElement(root, "mxCell", id="1", parent="0")

        # node id lookups, keyed by the resource's OpenAI id
        assistant_cells: dict[str, str] = {}
        store_cells: dict[str, str] = {}
        service_cells: dict[ServiceKind, str] = {}

        column = 0

        def add_column(title: str, kind: str, labels: list[tuple[str, str]]) -> dict[str, str]:
            """Place one column of nodes. Returns {resource_id: cell_id}."""
            nonlocal column
            if not labels:
                return {}
            x = LEFT_MARGIN + column * COL_WIDTH
            self._add_label(root, title, x)
            placed: dict[str, str] = {}
            for index, (resource_id, text) in enumerate(labels):
                y = TOP_MARGIN + 40 + index * (NODE_HEIGHT + V_GAP)
                cell_id = self._next_id(kind)
                self._add_node(root, cell_id, text, kind, x, y)
                placed[resource_id] = cell_id
            column += 1
            return placed

        assistant_cells = add_column(
            "Assistants",
            "assistant",
            [(a.id, f"{_truncate(a.name or a.id)}\n{a.model}") for a in self.result.assistants],
        )
        store_cells = add_column(
            "Vector stores",
            "vector_store",
            [
                (vs.id, f"{_truncate(vs.name or vs.id)}\n{vs.usage_gb:.2f} GB")
                for vs in self.result.vector_stores
            ],
        )
        add_column(
            "Fine-tunes",
            "fine_tune",
            [(ft.id, f"{_truncate(ft.id)}\n{ft.status}") for ft in self.result.fine_tunes],
        )
        add_column(
            "Batch jobs",
            "batch_job",
            [
                (bj.id, f"{_truncate(bj.id)}\n{bj.status} · {bj.failure_rate:.0f}% failed")
                for bj in self.result.batch_jobs
            ],
        )

        # One node per distinct external service kind.
        kinds = sorted(
            {rel.kind for rel in self.result.relationships},
            key=lambda k: k.value,
        )
        service_cells = add_column(
            "External services",
            "service",
            [(k.value, SERVICE_LABEL.get(k, k.value)) for k in kinds],
        )
        service_by_kind = {ServiceKind(v): cell for v, cell in service_cells.items()}

        # Edges: assistant -> attached vector store
        for assistant in self.result.assistants:
            src = assistant_cells.get(assistant.id)
            if not src:
                continue
            for store_id in assistant.vector_store_ids:
                dst = store_cells.get(store_id)
                if dst:
                    self._add_edge(root, src, dst, "attached")

        # Edges: assistant -> external service (deduped per pair)
        seen: set[tuple[str, str]] = set()
        for rel in self.result.relationships:
            src = assistant_cells.get(rel.source_id)
            dst = service_by_kind.get(rel.kind)
            if not src or not dst or (src, dst) in seen:
                continue
            seen.add((src, dst))
            self._add_edge(root, src, dst, rel.signal, dashed=True)

        tree = etree.ElementTree(mxfile)
        tree.write(str(target), pretty_print=True, xml_declaration=True, encoding="UTF-8")
        return target

    # ------------------------------------------------------------------

    def _add_label(self, root: etree._Element, text: str, x: int) -> None:
        cell = etree.SubElement(
            root,
            "mxCell",
            id=self._next_id("label"),
            value=text,
            style="text;html=1;fontStyle=1;fontSize=13;align=left;verticalAlign=middle;",
            vertex="1",
            parent="1",
        )
        etree.SubElement(
            cell,
            "mxGeometry",
            x=str(x),
            y=str(TOP_MARGIN),
            width=str(NODE_WIDTH),
            height="30",
            **{"as": "geometry"},
        )

    def _add_node(
        self, root: etree._Element, cell_id: str, text: str, kind: str, x: int, y: int
    ) -> None:
        cell = etree.SubElement(
            root,
            "mxCell",
            id=cell_id,
            value=text,
            style=_style(kind),
            vertex="1",
            parent="1",
        )
        etree.SubElement(
            cell,
            "mxGeometry",
            x=str(x),
            y=str(y),
            width=str(NODE_WIDTH),
            height=str(NODE_HEIGHT),
            **{"as": "geometry"},
        )

    def _add_edge(
        self,
        root: etree._Element,
        source: str,
        target: str,
        label: str = "",
        dashed: bool = False,
    ) -> None:
        style = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;fontSize=10;"
        if dashed:
            style += "dashed=1;strokeColor=#b85450;"
        cell = etree.SubElement(
            root,
            "mxCell",
            id=self._next_id("edge"),
            value=label,
            style=style,
            edge="1",
            parent="1",
            source=source,
            target=target,
        )
        etree.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
