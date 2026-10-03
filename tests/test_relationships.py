"""Lexical service-relationship detection."""

from openai_radar.models.base import ServiceKind
from openai_radar.scanners.assistants import detect_relationships


def test_empty_text_has_no_relationships() -> None:
    assert detect_relationships("", "asst_1") == []
    assert detect_relationships("", "asst_1", "Helper") == []


def test_word_boundaries_skip_incidental_matches() -> None:
    found = detect_relationships("Follow the laws of robotics.", "asst_1")
    assert found == []

    found = detect_relationships("Deploy the artifact to AWS S3.", "asst_1", "Helper")
    signals = {rel.signal: rel for rel in found}
    assert signals["aws"].kind is ServiceKind.AWS
    assert signals["s3"].kind is ServiceKind.AWS
    assert signals["aws"].source_name == "Helper"
    assert "AWS" in signals["aws"].evidence


def test_punctuation_signals_match_as_substrings() -> None:
    text = "Fetch https://blob.core.windows.net/container and POST to http://hooks.example/inbound"
    found = detect_relationships(text, "asst_9")
    signals = {rel.signal for rel in found}

    assert "blob.core.windows" in signals
    assert "http://" in signals
    assert all(rel.kind is ServiceKind.AZURE for rel in found if rel.signal == "blob.core.windows")
    assert all(rel.kind is ServiceKind.WEBHOOK for rel in found if rel.signal == "http://")


def test_each_signal_is_emitted_once_from_the_first_match() -> None:
    text = "use slack, then slack again, then postgres"
    found = detect_relationships(text, "asst_1")

    assert [rel.signal for rel in found] == ["postgres", "slack"]
    assert found[0].kind is ServiceKind.DATABASE
    assert found[1].kind is ServiceKind.SLACK


def test_multi_word_signal_and_case_folding() -> None:
    found = detect_relationships("Query Google Cloud BigQuery.", "asst_1")
    signals = {rel.signal for rel in found}
    assert "google cloud" in signals
    assert "bigquery" in signals


def test_evidence_collapses_whitespace() -> None:
    text = "prefix\n\n   slack   \n\nsuffix"
    found = detect_relationships(text, "asst_1")
    assert found[0].evidence == "prefix slack suffix"
