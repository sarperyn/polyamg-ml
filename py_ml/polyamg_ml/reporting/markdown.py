from __future__ import annotations

from pathlib import Path

from .result import ReportDocument


class MarkdownExporter:
    def export(self, document: ReportDocument, destination: str | Path) -> Path:
        out = Path(destination)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {document.title}", "", "## Summary", ""]
        for key, value in document.summary.items():
            lines.append(f"- **{key}**: {value}")
        for table in document.tables:
            lines.extend(["", f"## {table.title}", ""])
            if not table.rows:
                lines.append("_No rows._")
                continue
            columns = list(table.rows[0].keys())
            lines.append("| " + " | ".join(columns) + " |")
            lines.append("| " + " | ".join("---" for _ in columns) + " |")
            for row in table.rows:
                lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
        for figure in document.figures:
            lines.extend(["", f"## {figure.title}", "", f"![{figure.caption or figure.title}]({figure.path})"])
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out
