"""CLI commands, with the runner stubbed out."""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from openai_radar.cli import app
from openai_radar.client import RadarError
from openai_radar.findings import Finding, Severity
from openai_radar.models.base import AssistantInfo
from openai_radar.runner import RunConfig, RunResult

runner = CliRunner()


def _patch_run(monkeypatch: pytest.MonkeyPatch, result: RunResult | Exception) -> list[RunConfig]:
    seen: list[RunConfig] = []

    async def fake_run(client: object, config: RunConfig) -> RunResult:
        seen.append(config)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("openai_radar.cli.Runner.run", fake_run)
    return seen


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "openai-radar 2.1.0" in result.stdout


def test_run_rejects_an_unknown_output() -> None:
    result = runner.invoke(app, ["run", "--output", "xml"])
    assert result.exit_code == 2
    assert "--output" in result.stdout


def test_run_rejects_missing_credentials() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 2
    assert "Configuration error" in result.stdout


def test_run_json_includes_the_scan_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_run(
        monkeypatch,
        RunResult(
            config=RunConfig(project_id="proj_x", usage_lookback_days=7),
            assistants=[AssistantInfo(id="asst_1", name="Helper", model="gpt-4o")],
        ),
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--api-key",
            "sk-test",
            "--project",
            "proj_x",
            "--lookback",
            "7",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["assistants"][0]["id"] == "asst_1"
    assert payload["findings"] == []
    assert seen[0].project_id == "proj_x"
    assert seen[0].usage_lookback_days == 7


def test_run_json_can_be_written_to_a_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    destination = tmp_path / "scan.json"
    _patch_run(monkeypatch, RunResult(config=RunConfig()))

    result = runner.invoke(
        app,
        ["run", "--api-key", "sk-test", "--output", "json", "--out-file", str(destination)],
    )

    assert result.exit_code == 0, result.stdout
    assert "Wrote" in result.stdout
    assert json.loads(destination.read_text(encoding="utf-8"))["assistants"] == []


def test_run_table_renders_findings_and_exports(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    csv_dir = tmp_path / "csv"
    diagram = tmp_path / "arch.drawio"
    out_file = tmp_path / "scan.json"
    _patch_run(
        monkeypatch,
        RunResult(
            config=RunConfig(),
            assistants=[AssistantInfo(id="asst_1", name="Helper")],
            findings=[
                Finding(
                    rule_id="ASST_001",
                    severity=Severity.LOW,
                    resource_type="assistant",
                    resource_id="asst_1",
                    title="Assistant has no tools configured",
                    detail="zero tools",
                    recommendation="Add a tool.",
                )
            ],
        ),
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--api-key",
            "sk-test",
            "--admin-key",
            "sk-admin",
            "--csv-dir",
            str(csv_dir),
            "--drawio-file",
            str(diagram),
            "--out-file",
            str(out_file),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "ASST_001" in result.stdout
    assert "Helper" in result.stdout or "asst_1" in result.stdout
    assert (csv_dir / "assistants.csv").is_file()
    assert (csv_dir / "findings.csv").is_file()
    assert diagram.is_file()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["findings"][0]["rule_id"] == "ASST_001"


def test_run_reports_scanner_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(monkeypatch, RadarError("admin key rejected"))

    result = runner.invoke(app, ["run", "--api-key", "sk-test"])

    assert result.exit_code == 1
    assert "admin key rejected" in result.stdout


def test_findings_command_prints_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(
        monkeypatch,
        RunResult(
            config=RunConfig(),
            findings=[
                Finding(
                    rule_id="FT_001",
                    severity=Severity.MEDIUM,
                    resource_type="fine_tune",
                    resource_id="ft_1",
                    title="Failed fine-tune job",
                    detail="failed",
                    recommendation="Retry.",
                )
            ],
        ),
    )

    result = runner.invoke(app, ["findings", "--api-key", "sk-test", "--output", "json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload[0]["rule_id"] == "FT_001"
    assert payload[0]["severity"] == "medium"


def test_findings_command_reports_scan_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(client: object, config: RunConfig) -> Any:
        raise RadarError("down")

    monkeypatch.setattr("openai_radar.cli.Runner.run", fake_run)
    result = runner.invoke(app, ["findings", "--api-key", "sk-test"])
    assert result.exit_code == 1
    assert "down" in result.stdout
