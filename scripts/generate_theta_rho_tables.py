#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from polyamg_ml.data import SampleRecordRepository


DEFAULT_THETAS = (0.24, 0.48, 0.72)
DEFAULT_EPSILONS = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.5, 5.0, 7.0, 9.5)
DEFAULT_H_VALUES = (0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625)


@dataclass(frozen=True)
class TableCell:
    rho: float
    iterations: int


def _parse_csv_floats(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not raw:
        return default
    return tuple(float(value.strip()) for value in raw.split(",") if value.strip())


def _key(theta: float, epsilon: float, h: float) -> tuple[float, float, float]:
    return (round(theta, 8), round(epsilon, 8), round(h, 8))


def _records_by_grid(input_glob: str) -> dict[tuple[float, float, float], TableCell]:
    records = SampleRecordRepository.from_glob(input_glob).all()
    cells: dict[tuple[float, float, float], TableCell] = {}
    for record in records:
        meta = record.sample_meta
        metrics = record.metrics
        try:
            theta = float(meta["theta"])
            epsilon = float(meta["epsilon"])
            h = float(meta["h"])
            rho = float(metrics["rho"])
            iterations = int(metrics["iterations"])
        except (KeyError, TypeError, ValueError):
            continue
        cells[_key(theta, epsilon, h)] = TableCell(rho=rho, iterations=iterations)
    return cells


def _load_tables(
    vertical_split_glob: str | None,
    stripes_glob: str | None,
    checker2x2_glob: str | None,
    checker_glob: str | None,
) -> dict[str, dict[tuple[float, float, float], TableCell]]:
    tables: dict[str, dict[tuple[float, float, float], TableCell]] = {}
    if vertical_split_glob:
        tables["vertical_split"] = _records_by_grid(vertical_split_glob)
    if stripes_glob:
        tables["vertical_stripes4"] = _records_by_grid(stripes_glob)
    if checker2x2_glob:
        tables["checker2x2"] = _records_by_grid(checker2x2_glob)
    if checker_glob:
        tables["checker4x4"] = _records_by_grid(checker_glob)
    if not tables:
        raise ValueError("Provide at least one input glob")
    return tables


def _rho_range(
    tables: dict[str, dict[tuple[float, float, float], TableCell]],
    thetas: tuple[float, ...],
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
) -> tuple[float, float]:
    values: list[float] = []
    for cells in tables.values():
        for theta in thetas:
            for epsilon in epsilons:
                for h in h_values:
                    cell = cells.get(_key(theta, epsilon, h))
                    if cell is not None:
                        values.append(cell.rho)
    if not values:
        return 0.0, 1.0
    return min(values), max(values)


def _cell_color(rho: float, rho_min: float, rho_max: float) -> str:
    if rho_max <= rho_min:
        t = 0.0
    else:
        t = (rho - rho_min) / (rho_max - rho_min)
    low = (0.78, 0.82, 1.00)
    high = (1.00, 0.78, 0.78)
    r = low[0] + (high[0] - low[0]) * t
    g = low[1] + (high[1] - low[1]) * t
    b = low[2] + (high[2] - low[2]) * t
    return f"{r:.3f},{g:.3f},{b:.3f}"


def _format_h(h: float) -> str:
    return f"{h:.2e}"


def _format_cell(cell: TableCell | None, rho_min: float, rho_max: float) -> str:
    if cell is None:
        return "--"
    color = _cell_color(cell.rho, rho_min, rho_max)
    return f"\\cellcolor[rgb]{{{color}}}{cell.rho:.3f}({cell.iterations})"


def _write_csv(
    destination: Path,
    cells: dict[tuple[float, float, float], TableCell],
    theta: float,
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epsilon/h", *(_format_h(h) for h in h_values)])
        for epsilon in epsilons:
            row = [epsilon]
            for h in h_values:
                cell = cells.get(_key(theta, epsilon, h))
                row.append("" if cell is None else f"{cell.rho:.6f}({cell.iterations})")
            writer.writerow(row)


def _latex_table(
    title: str,
    label: str,
    pattern_name: str,
    cells: dict[tuple[float, float, float], TableCell],
    theta: float,
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
    rho_min: float,
    rho_max: float,
) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{title}. Pattern: {pattern_name}. Fixed strong threshold $\\theta={theta:.2f}$.}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{c" + "c" * len(h_values) + "}",
        "\\toprule",
        "$\\varepsilon / h$ & " + " & ".join(_format_h(h) for h in h_values) + " \\\\",
        "\\midrule",
    ]
    for epsilon in epsilons:
        values = [_format_cell(cells.get(_key(theta, epsilon, h)), rho_min, rho_max) for h in h_values]
        lines.append(f"{epsilon:.1f} & " + " & ".join(values) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertical_split_glob", default=None)
    parser.add_argument("--stripes_glob", default=None)
    parser.add_argument("--checker2x2_glob", default=None)
    parser.add_argument("--checker_glob", default=None)
    parser.add_argument("--out_dir", default="data/reports/theta_rho_relation")
    parser.add_argument("--thetas", default=",".join(str(v) for v in DEFAULT_THETAS))
    parser.add_argument("--epsilons", default=",".join(str(v) for v in DEFAULT_EPSILONS))
    parser.add_argument("--h_values", default=",".join(str(v) for v in DEFAULT_H_VALUES))
    args = parser.parse_args()

    thetas = _parse_csv_floats(args.thetas, DEFAULT_THETAS)
    epsilons = _parse_csv_floats(args.epsilons, DEFAULT_EPSILONS)
    h_values = _parse_csv_floats(args.h_values, DEFAULT_H_VALUES)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(args.vertical_split_glob, args.stripes_glob, args.checker2x2_glob, args.checker_glob)
    rho_min, rho_max = _rho_range(tables, thetas, epsilons, h_values)

    preamble = [
        "\\documentclass{article}",
        "\\usepackage[table]{xcolor}",
        "\\usepackage{booktabs}",
        "\\usepackage{geometry}",
        "\\geometry{margin=0.65in}",
        "\\begin{document}",
    ]
    latex_sections: list[str] = list(preamble)

    for pattern_name, cells in tables.items():
        for theta in thetas:
            stem = f"{pattern_name}_theta_{theta:.2f}".replace(".", "p")
            latex_sections.append(
                _latex_table(
                    title="Approximate convergence factor $\\rho$ with preconditioned CG iterations in parentheses",
                    label=f"tab:{stem}",
                    pattern_name=pattern_name.replace("_", " "),
                    cells=cells,
                    theta=theta,
                    epsilons=epsilons,
                    h_values=h_values,
                    rho_min=rho_min,
                    rho_max=rho_max,
                )
            )
            _write_csv(out_dir / f"{stem}.csv", cells, theta, epsilons, h_values)

    latex_sections.append("\\end{document}")
    tex_path = out_dir / "theta_rho_tables.tex"
    tex_path.write_text("\n\n".join(latex_sections) + "\n", encoding="utf-8")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
