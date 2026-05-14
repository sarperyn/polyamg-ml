#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polyamg_ml.data import SampleRecordRepository


DEFAULT_H_VALUES = (0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625)


@dataclass(frozen=True)
class ScatterPoint:
    h: float
    rho: float
    elapsed: float


def _parse_csv_floats(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not raw:
        return default
    return tuple(float(value.strip()) for value in raw.split(",") if value.strip())


def _case_key(record_meta: dict) -> tuple[float, float, int]:
    return (
        round(float(record_meta["h"]), 8),
        round(float(record_meta["epsilon"]), 8),
        int(record_meta.get("seed", 0)),
    )


def _load_case_points(input_globs: list[str]) -> dict[tuple[float, float, int], list[ScatterPoint]]:
    grouped: dict[tuple[float, float, int], list[ScatterPoint]] = defaultdict(list)
    for input_glob in input_globs:
        records = SampleRecordRepository.from_glob(input_glob).all()
        for record in records:
            try:
                h = float(record.sample_meta["h"])
                rho = float(record.metrics["rho"])
                elapsed = float(record.metrics["elapsed_sec"])
                key = _case_key(record.sample_meta)
            except (KeyError, TypeError, ValueError):
                continue
            grouped[key].append(ScatterPoint(h=h, rho=rho, elapsed=elapsed))
    if not grouped:
        raise RuntimeError("No valid rho/elapsed_sec records found")
    return dict(grouped)


def _normalize(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if std <= 0.0:
        return np.zeros_like(values)
    return (values - mean) / std


def _normalized_points(grouped: dict[tuple[float, float, int], list[ScatterPoint]]) -> dict[float, tuple[list[float], list[float]]]:
    by_h: dict[float, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for points in grouped.values():
        if len(points) < 2:
            continue
        rhos = np.asarray([point.rho for point in points], dtype=float)
        elapsed = np.asarray([point.elapsed for point in points], dtype=float)
        rho_norm = _normalize(rhos)
        elapsed_norm = _normalize(elapsed)
        for point, x, y in zip(points, rho_norm, elapsed_norm):
            xs, ys = by_h[round(point.h, 8)]
            xs.append(float(x))
            ys.append(float(y))
    return dict(by_h)


def _format_h(h: float) -> str:
    return f"{h:.2e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", action="append", required=True)
    parser.add_argument("--out", default="data/reports/time_vs_rho/time_vs_rho.png")
    parser.add_argument("--h_values", default=",".join(str(v) for v in DEFAULT_H_VALUES))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--point_size", type=float, default=5.5)
    args = parser.parse_args()

    h_values = _parse_csv_floats(args.h_values, DEFAULT_H_VALUES)
    by_h = _normalized_points(_load_case_points(args.input_glob))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, len(h_values)))

    all_x: list[float] = []
    all_y: list[float] = []
    for h, color in zip(h_values, colors):
        xs, ys = by_h.get(round(h, 8), ([], []))
        if not xs:
            continue
        all_x.extend(xs)
        all_y.extend(ys)
        ax.scatter(
            xs,
            ys,
            s=args.point_size,
            color=color,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.9,
            label=f"h={_format_h(h)}",
        )

    if not all_x:
        raise RuntimeError("No normalized scatter points were produced")

    x_arr = np.asarray(all_x, dtype=float)
    y_arr = np.asarray(all_y, dtype=float)
    x_pad = max(0.2, 0.06 * (float(np.max(x_arr)) - float(np.min(x_arr))))
    y_pad = max(0.2, 0.06 * (float(np.max(y_arr)) - float(np.min(y_arr))))

    ax.set_xlabel(r"normalized $\rho$")
    ax.set_ylabel(r"normalized $t$")
    ax.set_xlim(float(np.min(x_arr)) - x_pad, float(np.max(x_arr)) + x_pad)
    ax.set_ylim(float(np.min(y_arr)) - y_pad, float(np.max(y_arr)) + y_pad)
    ax.tick_params(labelsize=8)
    ax.legend(loc="center right", fontsize=7, frameon=True)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
