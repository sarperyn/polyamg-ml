#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from polyamg_ml.data import SampleRecord, SampleRecordRepository, record_join_key
from polyamg_ml.models import ModelFactory


DEFAULT_THETA_GRID = (
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.12,
    0.14,
    0.16,
    0.18,
    0.2,
    0.22,
    0.24,
    0.25,
    0.28,
    0.3,
    0.36,
    0.42,
    0.48,
    0.54,
    0.6,
    0.66,
    0.72,
    0.78,
    0.84,
    0.9,
)


@dataclass(frozen=True)
class CaseEvaluation:
    join_key: tuple[Any, ...]
    h: float
    epsilon: float
    theta_ann: float
    theta_min: float
    theta_ref: float
    rho_ann: float
    rho_min: float
    rho_ref: float
    p: float
    p_max: float
    p_over_pmax: float | None


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _architecture_kwargs(meta: dict[str, Any]) -> dict[str, Any]:
    architecture = dict(meta.get("architecture", {}))
    architecture.pop("name", None)
    if {"W1", "D1", "P1", "O", "W3", "D3"} <= set(architecture):
        return {
            "conv1_channels": architecture["W1"],
            "conv1_depth": architecture["D1"],
            "conv1_dropout": architecture["P1"],
            "conv2_channels": architecture.get("W2"),
            "conv2_depth": architecture.get("D2", 0),
            "conv2_dropout": architecture.get("P2", 0.0),
            "cnn_out_width": architecture["O"],
            "dense_width": architecture["W3"],
            "dense_depth": architecture["D3"],
        }
    return architecture


def _load_model(model_pt: str | Path, train_meta: str | Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    with Path(train_meta).open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    shape = meta.get("model_shape", {})
    features = meta.get("features", {})
    model = ModelFactory().create(
        in_channels=int(shape.get("in_channels", 1)),
        m=int(shape.get("m", features.get("m", 50))),
        **_architecture_kwargs(meta),
    )
    state = torch.load(model_pt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta


def _theta_grid(args: argparse.Namespace) -> list[float]:
    if args.theta_grid:
        return _parse_csv_floats(args.theta_grid)
    if args.manifest:
        with Path(args.manifest).open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("theta_grid"):
            return [float(theta) for theta in manifest["theta_grid"]]
    return list(DEFAULT_THETA_GRID)


def _feature_tensor(record: SampleRecord) -> torch.Tensor:
    if not record.feature_tensor:
        raise ValueError(f"Record has no feature_tensor: {record.path}")
    ft = record.feature_tensor
    m = int(ft["m"])
    c = int(ft["c"])
    values = np.asarray(ft["values"], dtype=np.float32)
    expected = c * m * m
    if values.size != expected:
        raise ValueError(f"Invalid feature tensor length in {record.path}: got {values.size}, expected {expected}")
    return torch.from_numpy(values.reshape(1, c, m, m))


def _closest_key(values: dict[float, SampleRecord], target: float) -> float:
    return min(values, key=lambda value: abs(value - target))


def _predict_theta(model: torch.nn.Module, record: SampleRecord, theta_grid: list[float]) -> float:
    x = _feature_tensor(record)
    h_value = float(record.sample_meta["h"])
    h = torch.tensor([-np.log2(max(h_value, 1.0e-8))], dtype=torch.float32)
    theta = torch.asarray(theta_grid, dtype=torch.float32)
    x_batch = x.repeat(len(theta_grid), 1, 1, 1)
    h_batch = h.repeat(len(theta_grid))
    with torch.no_grad():
        pred = model(x_batch, h_batch, theta)
    best = int(torch.argmin(pred).item())
    return float(theta_grid[best])


def _evaluate_cases(
    records: list[SampleRecord],
    model: torch.nn.Module,
    theta_grid: list[float],
    reference_theta: float,
) -> list[CaseEvaluation]:
    grouped: dict[tuple[Any, ...], list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record_join_key(record)].append(record)

    rows: list[CaseEvaluation] = []
    for join_key, case_records in grouped.items():
        by_theta = {round(float(record.sample_meta["theta"]), 8): record for record in case_records}
        if len(by_theta) < 2:
            continue

        representative = case_records[0]
        theta_ann = round(_predict_theta(model, representative, theta_grid), 8)
        if theta_ann not in by_theta:
            theta_ann = _closest_key(by_theta, theta_ann)
        theta_ref = _closest_key(by_theta, reference_theta)
        theta_min = min(by_theta, key=lambda theta: float(by_theta[theta].metrics["rho"]))

        rho_ann = float(by_theta[theta_ann].metrics["rho"])
        rho_ref = float(by_theta[theta_ref].metrics["rho"])
        rho_min = float(by_theta[theta_min].metrics["rho"])
        if rho_ref <= 0.0:
            continue

        p = 1.0 - rho_ann / rho_ref
        p_max = 1.0 - rho_min / rho_ref
        p_over_pmax = p / p_max if abs(p_max) > 1.0e-12 else None
        rows.append(
            CaseEvaluation(
                join_key=join_key,
                h=float(representative.sample_meta["h"]),
                epsilon=float(representative.sample_meta["epsilon"]),
                theta_ann=theta_ann,
                theta_min=theta_min,
                theta_ref=theta_ref,
                rho_ann=rho_ann,
                rho_min=rho_min,
                rho_ref=rho_ref,
                p=p,
                p_max=p_max,
                p_over_pmax=p_over_pmax,
            )
        )
    return rows


def _percent(value: float) -> float:
    return 100.0 * value


def _summary(rows: list[CaseEvaluation]) -> dict[str, float | int]:
    ps = [row.p for row in rows]
    pmaxs = [row.p_max for row in rows]
    ratios = [row.p_over_pmax for row in rows if row.p_over_pmax is not None]
    negatives = [row.p for row in rows if row.p < 0.0]
    return {
        "n": len(rows),
        "PB_percent": _percent(sum(1 for p in ps if p >= 0.0) / len(ps)) if ps else 0.0,
        "P_avg_percent": _percent(mean(ps)) if ps else 0.0,
        "P_median_percent": _percent(median(ps)) if ps else 0.0,
        "P_MAX_avg_percent": _percent(mean(pmaxs)) if pmaxs else 0.0,
        "P_MAX_median_percent": _percent(median(pmaxs)) if pmaxs else 0.0,
        "P_over_PMAX_avg_percent": _percent(mean(ratios)) if ratios else 0.0,
        "P_over_PMAX_median_percent": _percent(median(ratios)) if ratios else 0.0,
        "P_lt_0_avg_percent": _percent(mean(negatives)) if negatives else 0.0,
        "P_lt_0_median_percent": _percent(median(negatives)) if negatives else 0.0,
        "negative_count": len(negatives),
    }


def _write_rows(path: Path, rows: list[CaseEvaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "h",
                "epsilon",
                "theta_ann",
                "theta_min",
                "theta_ref",
                "rho_ann",
                "rho_min",
                "rho_ref",
                "P",
                "P_MAX",
                "P_over_PMAX",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.h,
                    row.epsilon,
                    row.theta_ann,
                    row.theta_min,
                    row.theta_ref,
                    row.rho_ann,
                    row.rho_min,
                    row.rho_ref,
                    row.p,
                    row.p_max,
                    "" if row.p_over_pmax is None else row.p_over_pmax,
                ]
            )


def _write_histogram(path: Path, rows: list[CaseEvaluation], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ps = [row.p for row in rows]
    pmaxs = [row.p_max for row in rows]
    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    ax.hist(pmaxs, bins=30, density=True, alpha=0.75, label=r"$P_{MAX}$")
    ax.hist(ps, bins=30, density=True, alpha=0.75, label=r"$P$")
    ax.set_title(title)
    ax.set_xlabel("performance gain")
    ax.set_ylabel("density")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_glob", required=True)
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--train_meta", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--theta_grid", default=None)
    parser.add_argument("--reference_theta", type=float, default=0.25)
    parser.add_argument("--out_dir", default="data/reports/ann_eval/test_case_1")
    parser.add_argument("--title", default="Case 1")
    args = parser.parse_args()

    model, meta = _load_model(args.model_pt, args.train_meta)
    theta_grid = _theta_grid(args)
    records = SampleRecordRepository.from_glob(args.data_glob).all()
    rows = _evaluate_cases(records, model, theta_grid, args.reference_theta)
    if not rows:
        raise RuntimeError("No evaluable cases found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(rows)
    payload = {
        "schema_version": "1.0",
        "title": args.title,
        "data_glob": args.data_glob,
        "model_pt": str(args.model_pt),
        "train_meta": str(args.train_meta),
        "manifest": args.manifest,
        "reference_theta": args.reference_theta,
        "theta_grid": theta_grid,
        "model": {
            "model_id": meta.get("model_id"),
            "features": meta.get("features"),
            "architecture": meta.get("architecture"),
            "test_mse": meta.get("test_mse"),
            "test_mae": meta.get("test_mae"),
        },
        "summary": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_rows(out_dir / "cases.csv", rows)
    _write_histogram(out_dir / "performance_histogram.png", rows, args.title)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
