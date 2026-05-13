from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableSpec:
    title: str
    rows: list[dict]


@dataclass(frozen=True)
class FigureSpec:
    title: str
    path: str
    caption: str = ""


@dataclass(frozen=True)
class ReportDocument:
    title: str
    summary: dict[str, float | int | str]
    tables: list[TableSpec] = field(default_factory=list)
    figures: list[FigureSpec] = field(default_factory=list)
