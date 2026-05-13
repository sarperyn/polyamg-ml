from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .latex import LatexExporter
from .result import ReportDocument


class PdfExporter:
    def __init__(self, latex_exporter: LatexExporter | None = None):
        self.latex_exporter = latex_exporter or LatexExporter()

    def export(self, document: ReportDocument, destination: str | Path) -> Path:
        out = Path(destination)
        out.parent.mkdir(parents=True, exist_ok=True)
        tex_path = out.with_suffix(".tex")
        self.latex_exporter.export(document, tex_path)
        pdflatex = shutil.which("pdflatex")
        if pdflatex is None:
            raise RuntimeError("pdflatex is not available; generated LaTeX report instead: " + str(tex_path))
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        generated = tex_path.with_suffix(".pdf")
        if generated != out:
            generated.replace(out)
        return out
