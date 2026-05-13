from __future__ import annotations

import argparse
from pathlib import Path

from polyamg_ml.data import SampleRecordRepository
from polyamg_ml.metrics import MetricComputer
from polyamg_ml.reporting import LatexExporter, MarkdownExporter, PdfExporter, ReportBuilder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_glob")
    ap.add_argument("--candidate_glob", required=True)
    ap.add_argument("--title", default="PolyAMG-ML Experiment Report")
    ap.add_argument("--out_md")
    ap.add_argument("--out_tex")
    ap.add_argument("--out_pdf")
    args = ap.parse_args()

    candidate = SampleRecordRepository.from_glob(args.candidate_glob)
    computer = MetricComputer()
    if args.baseline_glob:
        baseline = SampleRecordRepository.from_glob(args.baseline_glob)
        metrics = computer.compare(baseline.all(), candidate.all())
    else:
        metrics = computer.summarize_single(candidate.all())
    document = ReportBuilder().build(args.title, metrics)

    if args.out_md:
        MarkdownExporter().export(document, Path(args.out_md))
    if args.out_tex:
        LatexExporter().export(document, Path(args.out_tex))
    if args.out_pdf:
        PdfExporter().export(document, Path(args.out_pdf))


if __name__ == "__main__":
    main()
