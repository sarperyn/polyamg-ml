#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polyamg_ml.data import SampleRecordRepository


DEFAULT_EPSILONS = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.5, 5.0, 7.0, 9.5)
DEFAULT_H_VALUES = (0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625)


@dataclass(frozen=True)
class TimingPoint:
    theta: float
    mean: float
    std: float


def _parse_csv_floats(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not raw:
        return default
    return tuple(float(value.strip()) for value in raw.split(",") if value.strip())


def _key(h: float, epsilon: float, theta: float) -> tuple[float, float, float]:
    return (round(h, 8), round(epsilon, 8), round(theta, 8))


def _load_timings(input_globs: list[str]) -> dict[tuple[float, float, float], list[float]]:
    timings: dict[tuple[float, float, float], list[float]] = defaultdict(list)
    for input_glob in input_globs:
        records = SampleRecordRepository.from_glob(input_glob).all()
        for record in records:
            try:
                h = float(record.sample_meta["h"])
                epsilon = float(record.sample_meta["epsilon"])
                theta = float(record.sample_meta["theta"])
                elapsed = float(record.metrics["elapsed_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            timings[_key(h, epsilon, theta)].append(elapsed)
    if not timings:
        raise RuntimeError("No valid elapsed_sec records found")
    return dict(timings)


def _series_for(
    timings: dict[tuple[float, float, float], list[float]],
    h: float,
    epsilon: float,
) -> list[TimingPoint]:
    rows: list[TimingPoint] = []
    prefix = (round(h, 8), round(epsilon, 8))
    for key, values in timings.items():
        if key[:2] != prefix:
            continue
        arr = np.asarray(values, dtype=float)
        rows.append(
            TimingPoint(
                theta=key[2],
                mean=float(np.mean(arr)),
                std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            )
        )
    return sorted(rows, key=lambda point: point.theta)


def _format_h(h: float) -> str:
    return f"{h:.2e}"


def _positive_ylim(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    upper = max(values)
    if upper <= 0:
        return (0.0, 1.0)
    return (0.0, upper * 1.12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", action="append", required=True)
    parser.add_argument("--out", default="data/reports/theta_solver_times/figure3_solver_times.png")
    parser.add_argument("--epsilons", default=",".join(str(v) for v in DEFAULT_EPSILONS))
    parser.add_argument("--h_values", default=",".join(str(v) for v in DEFAULT_H_VALUES))
    parser.add_argument("--theta_xlim", type=float, nargs=2, default=(-0.02, 0.92))
    parser.add_argument("--legend_cols", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    epsilons = _parse_csv_floats(args.epsilons, DEFAULT_EPSILONS)
    h_values = _parse_csv_floats(args.h_values, DEFAULT_H_VALUES)
    timings = _load_timings(args.input_glob)

    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.4), sharex=True)
    flat_axes = axes.ravel()
    cmap = plt.get_cmap("jet")
    colors = [cmap(i / max(len(epsilons) - 1, 1)) for i in range(len(epsilons))]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*"]

    for ax, h in zip(flat_axes, h_values):
        plotted_values: list[float] = []
        for idx, epsilon in enumerate(epsilons):
            series = _series_for(timings, h, epsilon)
            if not series:
                continue
            xs = np.asarray([point.theta for point in series], dtype=float)
            means = np.asarray([point.mean for point in series], dtype=float)
            stds = np.asarray([point.std for point in series], dtype=float)
            plotted_values.extend((means + stds).tolist())
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                color=colors[idx],
                marker=markers[idx % len(markers)],
                markersize=3.0,
                linewidth=1.1,
                elinewidth=0.8,
                capsize=0,
                label=f"e={epsilon:g}",
            )

        ax.set_title(f"h={_format_h(h)}", fontsize=7)
        ax.set_xlabel(r"$\theta$", fontsize=8)
        ax.set_ylabel(r"solver time $[s]$", fontsize=8)
        ax.set_xlim(*args.theta_xlim)
        ax.set_ylim(*_positive_ylim(plotted_values))
        ax.tick_params(labelsize=6)

    for ax in flat_axes[len(h_values) :]:
        ax.set_axis_off()

    handles, labels = flat_axes[min(len(h_values), len(flat_axes)) - 1].get_legend_handles_labels()
    if handles:
        flat_axes[min(len(h_values), len(flat_axes)) - 1].legend(
            handles,
            labels,
            loc="lower right",
            fontsize=5.7,
            ncol=args.legend_cols,
            frameon=True,
        )

    fig.tight_layout(h_pad=1.2, w_pad=1.4)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
