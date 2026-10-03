"""Runner orchestration. Scanners are stubbed so nothing touches the network."""

from __future__ import annotations

from typing import Any

import pytest

from openai_radar.client import RadarClient
from openai_radar.findings import FindingEngine
from openai_radar.models.base import (
    AssistantInfo,
    BatchJobInfo,
    FineTuneInfo,
    ModelUsage,
    ServiceKind,
    ServiceRelationship,
    VectorStoreInfo,
)
from openai_radar.runner import RunConfig, Runner


def _install(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    items: list[Any],
    *,
    relationships: list[ServiceRelationship] | None = None,
    error: Exception | None = None,
) -> list[Any]:
    instances: list[Any] = []

    class _Scanner:
        def __init__(self, client: RadarClient, **kwargs: Any) -> None:
            self.client = client
            self.kwargs = kwargs
            self.relationships = list(relationships or [])
            self.scanned = False
            instances.append(self)

        async def scan(self, project_id: str | None = None) -> list[Any]:
            self.scanned = True
            self.project_id = project_id
            if error is not None:
                raise error
            return list(items)

    monkeypatch.setattr(f"openai_radar.runner.{name}", _Scanner)
    return instances


def _stub_all(monkeypatch: pytest.MonkeyPatch, **overrides: dict[str, Any]) -> dict[str, list[Any]]:
    specs: dict[str, dict[str, Any]] = {
        "AssistantScanner": {"items": []},
        "VectorStoreScanner": {"items": []},
        "FineTuneScanner": {"items": []},
        "BatchJobScanner": {"items": []},
        "UsageScanner": {"items": []},
    }
    for name, spec in overrides.items():
        specs[name].update(spec)
    return {name: _install(monkeypatch, name, **spec) for name, spec in specs.items()}


def test_run_sync_collects_every_scanner_and_its_relationships(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assistants = [AssistantInfo(id="asst_1", name="Helper", tool_count=1)]
    stores = [VectorStoreInfo(id="vs_1")]
    tunes = [FineTuneInfo(id="ft_1", status="succeeded")]
    batches = [BatchJobInfo(id="batch_1")]
    usage = [ModelUsage(model="gpt-4o", input_tokens=1, output_tokens=2)]
    relationships = [ServiceRelationship(source_id="asst_1", kind=ServiceKind.AWS, signal="s3")]
    installed = _stub_all(
        monkeypatch,
        AssistantScanner={"items": assistants, "relationships": relationships},
        VectorStoreScanner={"items": stores},
        FineTuneScanner={"items": tunes},
        BatchJobScanner={"items": batches},
        UsageScanner={"items": usage},
    )

    result = Runner.run_sync(
        RadarClient(api_key="sk-test"),
        RunConfig(project_id="proj_1", usage_lookback_days=7),
    )

    assert result.assistants == assistants
    assert result.vector_stores == stores
    assert result.fine_tunes == tunes
    assert result.batch_jobs == batches
    assert result.usage == usage
    assert result.relationships == relationships
    assert result.findings == []
    assert "Assistants:    1" in result.summary()
    assert "Findings:      0" in result.summary()

    for instances in installed.values():
        assert instances[0].scanned is True
        assert instances[0].project_id == "proj_1"
    assert installed["UsageScanner"][0].kwargs["lookback_days"] == 7


def test_a_failing_scanner_warns_and_leaves_the_others_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assistants = [AssistantInfo(id="asst_1", tool_count=0)]
    _stub_all(
        monkeypatch,
        AssistantScanner={"items": assistants},
        VectorStoreScanner={"items": [], "error": RuntimeError("vector stores down")},
    )

    with pytest.warns(RuntimeWarning, match="Scanner 'vector_stores' failed"):
        result = Runner.run_sync(RadarClient(api_key="sk-test"))

    assert result.vector_stores == []
    assert [item.id for item in result.assistants] == ["asst_1"]
    assert [finding.rule_id for finding in result.findings] == ["ASST_001"]


def test_disabled_scanners_are_not_called_and_findings_can_be_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = _stub_all(
        monkeypatch,
        BatchJobScanner={"items": [BatchJobInfo(id="batch_1")]},
        AssistantScanner={"items": [AssistantInfo(id="asst_1", tool_count=0)]},
    )

    class _Spy(FindingEngine):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def run(self, **kwargs: Any) -> list[Any]:
            self.called = True
            return super().run(**kwargs)

    engine = _Spy()
    result = Runner.run_sync(
        RadarClient(api_key="sk-test"),
        RunConfig(
            scan_assistants=False,
            scan_vector_stores=False,
            scan_fine_tunes=False,
            scan_usage=False,
            run_findings=False,
            finding_engine=engine,
        ),
    )

    assert result.batch_jobs[0].id == "batch_1"
    assert result.assistants == []
    assert result.relationships == []
    assert result.findings == []
    assert engine.called is False
    assert installed["BatchJobScanner"][0].scanned is True
    assert installed["AssistantScanner"][0].scanned is False
    assert installed["UsageScanner"] == []


def test_default_config_runs_findings() -> None:
    config = RunConfig()
    assert config.scan_usage is True
    assert config.usage_lookback_days == 30
    assert config.run_findings is True
    assert isinstance(config.finding_engine, FindingEngine)
