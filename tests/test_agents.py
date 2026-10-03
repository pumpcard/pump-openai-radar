"""Session cache behind the optional openai-agents tools.

The extra is not installed here. ``agents`` is a stub, so these tests cover
the cache and the import error without pulling in the optional SDK.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from typing import Any

import pytest

from openai_radar.models.base import (
    AssistantInfo,
    ModelUsage,
    ServiceKind,
    ServiceRelationship,
    VectorStoreInfo,
)
from openai_radar.runner import RunConfig, RunResult


def _install_agents(
    monkeypatch: pytest.MonkeyPatch, *, with_agent: bool = True
) -> types.ModuleType:
    agents = types.ModuleType("agents")

    def function_tool(fn: Any = None, **kwargs: Any) -> Any:
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    agents.function_tool = function_tool  # type: ignore[attr-defined]
    if with_agent:

        class Agent:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        agents.Agent = Agent  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "agents", agents)
    sys.modules.pop("openai_radar.agents.tools", None)
    return agents


def _load_tools(monkeypatch: pytest.MonkeyPatch) -> Any:
    _install_agents(monkeypatch)
    tools = importlib.import_module("openai_radar.agents.tools")
    tools.reset_session()
    return tools


def test_tools_explain_the_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "agents", None)
    sys.modules.pop("openai_radar.agents.tools", None)

    with pytest.raises(ImportError, match=r"openai-radar\[agents\]"):
        importlib.import_module("openai_radar.agents.tools")


def test_build_radar_agent_requires_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_agents(monkeypatch, with_agent=False)
    from openai_radar.agents import build_radar_agent

    with pytest.raises(ImportError, match=r"openai-radar\[agents\]"):
        build_radar_agent()


def test_build_radar_agent_wires_every_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_agents(monkeypatch)
    importlib.import_module("openai_radar.agents.tools")
    from openai_radar.agents import build_radar_agent

    agent = build_radar_agent(model="gpt-4.1-mini", name="FinOps")

    assert agent.kwargs["name"] == "FinOps"
    assert agent.kwargs["model"] == "gpt-4.1-mini"
    assert len(agent.kwargs["tools"]) == 8
    assert "FinOps" in agent.kwargs["instructions"]


def test_scan_tools_merge_into_one_session(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    tools = _load_tools(monkeypatch)
    configs: list[RunConfig] = []

    async def fake_run(client: object, config: RunConfig) -> RunResult:
        configs.append(config)
        result = RunResult(config=config)
        if config.scan_assistants:
            result.assistants = [
                AssistantInfo(id="asst_1", name="Helper", model="gpt-4o", tool_count=0)
            ]
            result.relationships = [
                ServiceRelationship(
                    source_id="asst_1",
                    source_name="Helper",
                    kind=ServiceKind.SLACK,
                    signal="slack",
                )
            ]
        if config.scan_usage:
            result.usage = [
                ModelUsage(model="gpt-4o", input_tokens=3, output_tokens=1, num_requests=2),
                ModelUsage(model="gpt-4o-mini", input_tokens=1, output_tokens=0, num_requests=1),
            ]
        if config.scan_vector_stores:
            result.vector_stores = [VectorStoreInfo(id="vs_1", name="Docs", usage_bytes=1024**3)]
        return result

    class _Runner:
        @staticmethod
        async def run(client: object, config: RunConfig) -> RunResult:
            return await fake_run(client, config)

    monkeypatch.setattr(tools, "RadarClient", lambda: "client")
    monkeypatch.setattr(tools, "Runner", _Runner)

    assistants = asyncio.run(tools.scan_assistants())
    assert "asst_1" in assistants
    assert "slack" in assistants
    assert configs[0].scan_assistants is True
    assert configs[0].scan_usage is False
    assert configs[0].run_findings is False

    usage = asyncio.run(tools.scan_usage(lookback_days=14))
    assert configs[1].usage_lookback_days == 14
    assert "gpt-4o: 4 tokens" in usage
    assert "gpt-4o-mini: 1 token" in usage

    stores = asyncio.run(tools.scan_vector_stores())
    assert "1.00 GB" in stores

    report = asyncio.run(tools.run_findings())
    assert "ASST_001" in report

    written = asyncio.run(tools.export_csv(str(tmp_path / "out")))
    assert "assistants.csv" in written
    diagram = asyncio.run(tools.export_drawio(str(tmp_path / "arch.drawio")))
    assert str(tmp_path / "arch.drawio") in diagram
    assert (tmp_path / "arch.drawio").is_file()

    tools.reset_session()
    assert asyncio.run(tools.run_findings()) == (
        "Nothing scanned yet — run one or more scan tools first."
    )
    assert asyncio.run(tools.export_csv(str(tmp_path / "empty"))) == (
        "Nothing to export — run one or more scan tools first."
    )


def test_scan_usage_without_rows_mentions_the_admin_key(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = _load_tools(monkeypatch)

    class _Runner:
        @staticmethod
        async def run(client: object, config: RunConfig) -> RunResult:
            return RunResult(config=config)

    monkeypatch.setattr(tools, "RadarClient", lambda: "client")
    monkeypatch.setattr(tools, "Runner", _Runner)

    assert "admin key" in asyncio.run(tools.scan_usage())


def test_run_findings_with_a_clean_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = _load_tools(monkeypatch)
    tools._result = RunResult(
        config=RunConfig(),
        assistants=[AssistantInfo(id="asst_1", name="Helper", tool_count=2)],
    )

    assert (
        asyncio.run(tools.run_findings())
        == "No findings — nothing flagged against the current rule set."
    )
