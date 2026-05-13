from __future__ import annotations

from polyamg_ml.metrics import MetricsBundle

from .result import ReportDocument, TableSpec


class ReportBuilder:
    def build(self, title: str, metrics: MetricsBundle) -> ReportDocument:
        tables = [TableSpec(title=name.replace("_", " ").title(), rows=rows) for name, rows in metrics.tables.items()]
        return ReportDocument(title=title, summary=metrics.summary, tables=tables)
