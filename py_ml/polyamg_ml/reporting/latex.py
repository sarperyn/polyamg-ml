from __future__ import annotations

from pathlib import Path

from .result import ReportDocument


def _escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


class LatexExporter:
    def export(self, document: ReportDocument, destination: str | Path) -> Path:
        out = Path(destination)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "\\documentclass{article}",
            "\\usepackage{booktabs}",
            "\\usepackage{graphicx}",
            "\\begin{document}",
            f"\\section*{{{_escape(document.title)}}}",
            "\\subsection*{Summary}",
            "\\begin{itemize}",
        ]
        for key, value in document.summary.items():
            lines.append(f"\\item \\textbf{{{_escape(key)}}}: {_escape(value)}")
        lines.append("\\end{itemize}")

        for table in document.tables:
            lines.append(f"\\subsection*{{{_escape(table.title)}}}")
            if not table.rows:
                lines.append("No rows.")
                continue
            columns = list(table.rows[0].keys())
            lines.append("\\begin{tabular}{" + "l" * len(columns) + "}")
            lines.append("\\toprule")
            lines.append(" & ".join(_escape(c) for c in columns) + " \\\\")
            lines.append("\\midrule")
            for row in table.rows:
                lines.append(" & ".join(_escape(row.get(c, "")) for c in columns) + " \\\\")
            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")

        lines.append("\\end{document}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out
