"""Public package surface."""

from openai_radar import (
    Finding,
    FindingEngine,
    RadarClient,
    RadarError,
    RunConfig,
    Runner,
    RunResult,
    Severity,
    __version__,
)


def test_version() -> None:
    assert __version__ == "0.0.1"


def test_public_names_are_exported() -> None:
    assert RadarClient.__name__ == "RadarClient"
    assert RadarError.__name__ == "RadarError"
    assert Runner.__name__ == "Runner"
    assert RunConfig.__name__ == "RunConfig"
    assert RunResult.__name__ == "RunResult"
    assert Finding.__name__ == "Finding"
    assert FindingEngine.__name__ == "FindingEngine"
    assert Severity.HIGH.value == "high"
