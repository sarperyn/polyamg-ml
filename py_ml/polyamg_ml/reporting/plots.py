from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from polyamg_ml.data import SampleRecord


def theta_vs_metric(records: list[SampleRecord], metric: str, destination: str | Path) -> Path:
    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    xs = [float(r.sample_meta["theta"]) for r in records]
    ys = [float(r.metrics[metric]) for r in records]
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, s=8, alpha=0.65)
    plt.xlabel("theta")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out
