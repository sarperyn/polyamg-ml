from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, t as student_t
from statsmodels.nonparametric.smoothers_lowess import lowess as statsmodels_lowess

from polyamg_ml.data import SampleRecord, SampleRecordRepository, record_join_key


def _as_records(input_dir: str | None, input_glob: str | None) -> list[SampleRecord]:
    if input_glob:
        return SampleRecordRepository.from_glob(input_glob).all()
    if not input_dir:
        raise ValueError("Either --input_dir or --input_glob must be provided")
    return SampleRecordRepository.from_directory(input_dir).all()


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float | None]:
    n = len(x)
    if n < 3:
        return 0.0, 0.0, None
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    sxx = float(np.sum((x - xbar) ** 2))
    if sxx <= 0:
        return 0.0, 0.0, None
    sxy = float(np.sum((x - xbar) * (y - ybar)))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    yhat = slope * x + intercept
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - ybar) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    sigma2 = sse / (n - 2)
    se_slope = math.sqrt(sigma2 / sxx) if sigma2 > 0 else 0.0
    if se_slope <= 0:
        return r2, 0.0, None
    t_stat = slope / se_slope
    p_value = _p_value_t(t_stat, n - 2)
    return r2, t_stat, p_value


def _p_value_t(t_stat: float, df: int) -> float | None:
    if df <= 0:
        return None
    return float(2.0 * student_t.sf(abs(t_stat), df))


def _group_records(records: Iterable[SampleRecord]) -> dict[tuple, list[SampleRecord]]:
    grouped: dict[tuple, list[SampleRecord]] = {}
    for record in records:
        key = record_join_key(record)
        grouped.setdefault(key, []).append(record)
    return grouped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=None)
    ap.add_argument("--input_glob", default=None)
    ap.add_argument("--out", default="data/reports/theta_levels_relation/theta_vs_levels.png")
    ap.add_argument("--lowess_frac", type=float, default=0.4)
    ap.add_argument("--kde_gridsize", type=int, default=70)
    ap.add_argument("--theta_xlim", type=float, nargs=2, default=(-0.1, 1.1))
    ap.add_argument("--levels_ylim", type=float, nargs=2, default=(-0.25, 1.25))
    ap.add_argument("--pvalue_xlim", type=float, nargs=2, default=(1.0e-9, 1.0e-1))
    ap.add_argument("--pvalue_ylim", type=float, nargs=2, default=(0.0, 0.5))
    ap.add_argument("--r2_xlim", type=float, nargs=2, default=(0.15, 0.9))
    ap.add_argument("--r2_ylim", type=float, nargs=2, default=(0.0, 7.5))
    args = ap.parse_args()

    records = _as_records(args.input_dir, args.input_glob)
    grouped = _group_records(records)

    scatter_x: list[float] = []
    scatter_y: list[float] = []
    p_values: list[float] = []
    r2_values: list[float] = []

    for group_records in grouped.values():
        rows = []
        for r in group_records:
            try:
                theta = float(r.sample_meta["theta"])
                levels = float(r.metrics.get("n_levels", -1))
            except Exception:
                continue
            if levels < 0:
                continue
            rows.append((theta, levels))
        if not rows:
            continue
        rows.sort(key=lambda t: t[0])
        thetas = np.array([t for t, _ in rows], dtype=float)
        levels = np.array([l for _, l in rows], dtype=float)
        min_l = float(np.min(levels))
        max_l = float(np.max(levels))
        if max_l > min_l:
            levels_norm = (levels - min_l) / (max_l - min_l)
        else:
            levels_norm = np.zeros_like(levels)
        scatter_x.extend(thetas.tolist())
        scatter_y.extend(levels_norm.tolist())

        r2, t_stat, p_val = _linear_fit(thetas, levels)
        if p_val is not None:
            p_values.append(p_val)
        r2_values.append(r2)

    if not scatter_x:
        raise RuntimeError("No valid records with n_levels found")

    xs = np.array(scatter_x, dtype=float)
    ys = np.array(scatter_y, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Left: scatter + LOWESS + KDE
    axes[0].scatter(xs, ys, s=12, alpha=0.65, label="scatter")
    lowess_result = statsmodels_lowess(ys, xs, frac=args.lowess_frac, return_sorted=True)
    if len(lowess_result) > 0:
        axes[0].plot(lowess_result[:, 0], lowess_result[:, 1], color="#d62728", linewidth=2.0, label="LOWESS")

    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    pad_x = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
    pad_y = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    xs_grid = np.linspace(xmin - pad_x, xmax + pad_x, args.kde_gridsize)
    ys_grid = np.linspace(ymin - pad_y, ymax + pad_y, args.kde_gridsize)
    xx, yy = np.meshgrid(xs_grid, ys_grid)
    kde = gaussian_kde(np.vstack([xs, ys]))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    axes[0].contour(xx, yy, zz, colors="#1f77b4", linewidths=1.0)
    axes[0].set_xlabel("theta")
    axes[0].set_ylabel("normalized # levels")
    axes[0].set_xlim(*args.theta_xlim)
    axes[0].set_ylim(*args.levels_ylim)
    axes[0].grid(True, alpha=0.55)
    axes[0].legend(frameon=False, fontsize=8)

    # Center: p-value histogram
    if p_values:
        p_floor = max(float(args.pvalue_xlim[0]), np.finfo(float).tiny)
        p_ceil = max(float(args.pvalue_xlim[1]), p_floor * 10.0)
        plot_p_values = np.clip(np.asarray(p_values, dtype=float), p_floor, p_ceil)
        p_bins = np.logspace(np.log10(p_floor), np.log10(p_ceil), 13)
        counts, _ = np.histogram(plot_p_values, bins=p_bins)
        log_widths = np.diff(np.log10(p_bins))
        total = max(int(np.sum(counts)), 1)
        log_density = counts / (total * log_widths)
        axes[1].bar(
            p_bins[:-1],
            log_density,
            width=np.diff(p_bins),
            align="edge",
            color="#1f77b4",
            edgecolor="#1f1f1f",
            linewidth=0.8,
            alpha=0.8,
        )
        axes[1].set_xlabel("p-value")
        axes[1].set_ylabel("density")
        axes[1].set_xscale("log")
        axes[1].set_xlim(p_floor, p_ceil)
        axes[1].set_ylim(*args.pvalue_ylim)
        axes[1].grid(True, alpha=0.55)
    else:
        axes[1].text(0.5, 0.5, "p-values unavailable", ha="center", va="center")
        axes[1].set_axis_off()

    # Right: R^2 histogram
    if r2_values:
        axes[2].hist(
            r2_values,
            bins=12,
            density=True,
            color="#1f77b4",
            edgecolor="#1f1f1f",
            linewidth=0.8,
            alpha=0.8,
        )
        axes[2].set_xlabel("R^2")
        axes[2].set_ylabel("density")
        axes[2].set_xlim(*args.r2_xlim)
        axes[2].set_ylim(*args.r2_ylim)
        axes[2].grid(True, alpha=0.55)
    else:
        axes[2].text(0.5, 0.5, "R^2 unavailable", ha="center", va="center")
        axes[2].set_axis_off()

    fig.suptitle("Regression results (theta vs #levels)")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
