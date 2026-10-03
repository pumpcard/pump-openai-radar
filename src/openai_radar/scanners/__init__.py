"""Resource scanners for openai-radar."""

from openai_radar.scanners.assistants import AssistantScanner, detect_relationships
from openai_radar.scanners.base import Scanner
from openai_radar.scanners.batch_jobs import BatchJobScanner
from openai_radar.scanners.fine_tunes import FineTuneScanner
from openai_radar.scanners.usage import UsageScanner
from openai_radar.scanners.vector_stores import VectorStoreScanner

#: Scanner keys in the order Runner reports them.
SCANNERS = [
    "assistants",
    "vector_stores",
    "fine_tunes",
    "batch_jobs",
    "usage",
]

__all__ = [
    "SCANNERS",
    "AssistantScanner",
    "BatchJobScanner",
    "FineTuneScanner",
    "Scanner",
    "UsageScanner",
    "VectorStoreScanner",
    "detect_relationships",
]
