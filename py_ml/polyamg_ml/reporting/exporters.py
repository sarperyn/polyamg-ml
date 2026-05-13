from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .result import ReportDocument


class ResultExporter(Protocol):
    def export(self, document: ReportDocument, destination: str | Path) -> Path:
        ...
