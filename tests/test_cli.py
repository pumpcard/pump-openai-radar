"""CLI commands, with the runner stubbed out."""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from openai_radar.cli import app
from openai_radar.client import RadarError
from openai_radar.findings import Finding, Severity
from openai_radar.models.base import AssistantInfo, ModelUsage
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
    assert "pump-openai-radar 0.1.0" in result.stdout


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


def test_run_uploads_the_cost_report(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _patch_run(
        monkeypatch,
        RunResult(
            config=RunConfig(project_id="proj_x", usage_lookback_days=7),
            usage=[
                ModelUsage(model="gpt-4o", input_tokens=3, output_tokens=4, project_id="proj_x")
            ],
        ),
    )
    report_path = tmp_path / "report.csv"
    seen: dict[str, object] = {}

    async def fake_fetch(client: object, *, lookback_days: int, project_id: str | None) -> object:
        from openai_radar.scanners.report import CostReport

        seen["lookback"] = lookback_days
        seen["project"] = project_id
        assert getattr(client, "has_admin_key", False)
        return CostReport(
            rows=[
                {
                    "Date": "2026-01-01",
                    "ProjectID": "proj_x",
                    "LineItem": "gpt-4o, input",
                    "Amount": "1.500000",
                    "Currency": "USD",
                }
            ]
        )

    def fake_upload(api_base: str, token: str, files: dict[str, str]) -> None:
        seen["api_base"] = api_base
        seen["token"] = token
        seen["files"] = files

    monkeypatch.setattr("openai_radar.scanners.report.fetch_cost_report", fake_fetch)
    monkeypatch.setattr("openai_radar.upload.upload_csvs", fake_upload)

    result = runner.invoke(
        app,
        [
            "run",
            "--api-key",
            "sk-test",
            "--admin-key",
            "sk-admin",
            "--project",
            "proj_x",
            "--lookback",
            "7",
            "--upload-token",
            "tok",
            "--api-base",
            "http://localhost:8001",
            "--report-file",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert seen["lookback"] == 7
    assert seen["project"] == "proj_x"
    assert seen["api_base"] == "http://localhost:8001"
    assert seen["token"] == "tok"
    usage_path = report_path.with_name("usage.csv")
    assert seen["files"] == {"billing": str(report_path), "inventory": str(usage_path)}
    assert "gpt-4o, input" in report_path.read_text(encoding="utf-8")
    assert "gpt-4o" in usage_path.read_text(encoding="utf-8")
    assert "on its way to Pump" in result.stdout


def test_run_can_write_the_report_without_uploading(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_run(monkeypatch, RunResult(config=RunConfig()))
    destination = tmp_path / "custom.csv"

    async def fake_fetch(client: object, *, lookback_days: int, project_id: str | None) -> object:
        from openai_radar.scanners.report import CostReport

        return CostReport(
            rows=[
                {
                    "Date": "2026-01-02",
                    "ProjectID": "-",
                    "LineItem": "embeddings",
                    "Amount": "0.250000",
                    "Currency": "USD",
                }
            ]
        )

    def fail_upload(*args: object, **kwargs: object) -> None:
        raise AssertionError("upload should not run without a token")

    monkeypatch.setattr("openai_radar.scanners.report.fetch_cost_report", fake_fetch)
    monkeypatch.setattr("openai_radar.upload.upload_csvs", fail_upload)

    result = runner.invoke(
        app,
        [
            "run",
            "--api-key",
            "sk-test",
            "--admin-key",
            "sk-admin",
            "--report-file",
            str(destination),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert destination.is_file()
    assert "Uploading" not in result.stdout


def test_run_upload_defaults_the_report_into_csv_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_run(
        monkeypatch,
        RunResult(
            config=RunConfig(),
            usage=[ModelUsage(model="gpt-4o", input_tokens=1, output_tokens=2)],
        ),
    )
    csv_dir = tmp_path / "out"
    seen: dict[str, object] = {}

    async def fake_fetch(client: object, *, lookback_days: int, project_id: str | None) -> object:
        from openai_radar.scanners.report import CostReport

        return CostReport(
            rows=[
                {
                    "Date": "2026-01-02",
                    "ProjectID": "proj",
                    "LineItem": "embeddings",
                    "Amount": "0.250000",
                    "Currency": "USD",
                }
            ]
        )

    def fake_upload(api_base: str, token: str, files: dict[str, str]) -> None:
        seen["files"] = files

    monkeypatch.setattr("openai_radar.scanners.report.fetch_cost_report", fake_fetch)
    monkeypatch.setattr("openai_radar.upload.upload_csvs", fake_upload)

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
            "--upload-token",
            "tok",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert seen["files"] == {
        "billing": str(csv_dir / "report.csv"),
        "inventory": str(csv_dir / "usage.csv"),
    }
    assert (csv_dir / "report.csv").is_file()
    assert (csv_dir / "usage.csv").is_file()


def test_run_report_requires_an_admin_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _patch_run(monkeypatch, RunResult(config=RunConfig()))

    result = runner.invoke(
        app,
        [
            "run",
            "--api-key",
            "sk-test",
            "--upload-token",
            "tok",
            "--report-file",
            str(tmp_path / "r.csv"),
        ],
    )

    assert result.exit_code == 1
    assert "OPENAI_ADMIN_KEY" in result.stdout


def test_run_reports_upload_failure(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _patch_run(monkeypatch, RunResult(config=RunConfig()))

    async def fake_fetch(client: object, *, lookback_days: int, project_id: str | None) -> object:
        from openai_radar.scanners.report import CostReport

        return CostReport(
            rows=[
                {
                    "Date": "2026-01-01",
                    "ProjectID": "proj",
                    "LineItem": "gpt-4o",
                    "Amount": "1.000000",
                    "Currency": "USD",
                }
            ]
        )

    def fake_upload(api_base: str, token: str, files: dict[str, str]) -> None:
        from openai_radar.upload import UploadError

        raise UploadError("token was rejected")

    monkeypatch.setattr("openai_radar.scanners.report.fetch_cost_report", fake_fetch)
    monkeypatch.setattr("openai_radar.upload.upload_csvs", fake_upload)

    destination = tmp_path / "report.csv"
    result = runner.invoke(
        app,
        [
            "run",
            "--api-key",
            "sk-test",
            "--admin-key",
            "sk-admin",
            "--upload-token",
            "tok",
            "--report-file",
            str(destination),
        ],
    )

    assert result.exit_code == 1
    assert "token was rejected" in result.stdout
    assert destination.is_file()


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


def _save_login(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    token: str = "stored-token",
    api_base: str = "http://login.example",
    expires_at: str | None = None,
) -> None:
    from datetime import datetime, timedelta, timezone

    from openai_radar.pump_login import PumpCredentials, save_credentials

    monkeypatch.setenv("OPENAI_RADAR_CONFIG_DIR", str(tmp_path))
    if expires_at is None:
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    save_credentials(
        PumpCredentials(
            access_token=token,
            token_type="Bearer",
            expires_at=expires_at,
            upload_id="upload-9",
            scope="radar",
            api_base=api_base,
        ),
        tmp_path / "credentials.json",
    )


def _patch_cost_report(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    seen: dict[str, object] = {}

    async def fake_fetch(client: object, *, lookback_days: int, project_id: str | None) -> object:
        from openai_radar.scanners.report import CostReport

        return CostReport(
            rows=[
                {
                    "Date": "2026-01-01",
                    "ProjectID": "proj",
                    "LineItem": "gpt-4o",
                    "Amount": "1.000000",
                    "Currency": "USD",
                }
            ]
        )

    def fake_upload(api_base: str, token: str, files: dict[str, str]) -> None:
        seen["api_base"] = api_base
        seen["token"] = token
        seen["files"] = files

    monkeypatch.setattr("openai_radar.scanners.report.fetch_cost_report", fake_fetch)
    monkeypatch.setattr("openai_radar.upload.upload_csvs", fake_upload)
    return seen


def test_run_upload_uses_the_login_token(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _patch_run(
        monkeypatch,
        RunResult(
            config=RunConfig(),
            usage=[ModelUsage(model="gpt-4o", input_tokens=1, output_tokens=2)],
        ),
    )
    _save_login(tmp_path, monkeypatch)
    seen = _patch_cost_report(monkeypatch)
    report_path = tmp_path / "report.csv"

    result = runner.invoke(
        app,
        [
            "run",
            "--api-key",
            "sk-test",
            "--admin-key",
            "sk-admin",
            "--upload",
            "--report-file",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert seen["token"] == "stored-token"
    assert seen["api_base"] == "http://login.example"
    assert seen["files"] == {
        "billing": str(report_path),
        "inventory": str(report_path.with_name("usage.csv")),
    }
    assert "stored-token" not in result.stdout


def test_run_upload_token_overrides_the_stored_login(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_run(monkeypatch, RunResult(config=RunConfig()))
    _save_login(tmp_path, monkeypatch)
    seen = _patch_cost_report(monkeypatch)

    result = runner.invoke(
        app,
        [
            "run",
            "--api-key",
            "sk-test",
            "--admin-key",
            "sk-admin",
            "--upload",
            "--upload-token",
            "one-shot",
            "--api-base",
            "http://override.example",
            "--report-file",
            str(tmp_path / "report.csv"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert seen["token"] == "one-shot"
    assert seen["api_base"] == "http://override.example"


def test_run_upload_without_login_stops_before_the_scan(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    seen = _patch_run(monkeypatch, RunResult(config=RunConfig()))
    monkeypatch.setenv("OPENAI_RADAR_CONFIG_DIR", str(tmp_path))

    result = runner.invoke(app, ["run", "--api-key", "sk-test", "--upload"])

    assert result.exit_code == 1
    assert "pump-openai-radar login" in result.stdout
    assert seen == []


def test_run_upload_rejects_an_expired_login(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from datetime import datetime, timedelta, timezone

    seen = _patch_run(monkeypatch, RunResult(config=RunConfig()))
    expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    _save_login(tmp_path, monkeypatch, expires_at=expired)

    result = runner.invoke(app, ["run", "--api-key", "sk-test", "--upload"])

    assert result.exit_code == 1
    assert "expired" in result.stdout.lower()
    assert seen == []


def test_status_and_logout_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_RADAR_CONFIG_DIR", str(tmp_path))

    missing = runner.invoke(app, ["status"])
    assert missing.exit_code == 1
    assert "Not logged in" in missing.stdout

    _save_login(tmp_path, monkeypatch, token="secret-token", api_base="http://login.example")
    present = runner.invoke(app, ["status"])
    assert present.exit_code == 0, present.stdout
    assert "http://login.example" in present.stdout
    assert "upload-9" in present.stdout
    assert "secret-token" not in present.stdout

    gone = runner.invoke(app, ["logout"])
    assert gone.exit_code == 0
    assert "Logged out" in gone.stdout
    assert runner.invoke(app, ["status"]).exit_code == 1


def test_login_command_reports_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from openai_radar.pump_login import LoginError

    def boom(**kwargs: object) -> None:
        raise LoginError("browser denied")

    monkeypatch.setattr("openai_radar.cli.pump_login", boom)
    result = runner.invoke(app, ["login"])
    assert result.exit_code == 1
    assert "browser denied" in result.stdout + result.stderr


def test_findings_command_reports_scan_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(client: object, config: RunConfig) -> Any:
        raise RadarError("down")

    monkeypatch.setattr("openai_radar.cli.Runner.run", fake_run)
    result = runner.invoke(app, ["findings", "--api-key", "sk-test"])
    assert result.exit_code == 1
    assert "down" in result.stdout
