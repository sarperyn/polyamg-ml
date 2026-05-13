from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median

from polyamg_ml.data import SampleRecord, record_join_key


@dataclass(frozen=True)
class MetricsBundle:
    summary: dict[str, float | int]
    tables: dict[str, list[dict]] = field(default_factory=dict)


class MetricComputer:
    def summarize_single(self, records: list[SampleRecord]) -> MetricsBundle:
        rhos = [float(r.metrics["rho"]) for r in records if "rho" in r.metrics]
        return MetricsBundle(
            summary={
                "n": len(rhos),
                "rho_avg": mean(rhos) if rhos else 0.0,
                "rho_median": median(rhos) if rhos else 0.0,
            }
        )

    def compare(self, baseline: list[SampleRecord], candidate: list[SampleRecord]) -> MetricsBundle:
        baseline_by_key: dict[tuple, SampleRecord] = {}
        optimum_by_key: dict[tuple, SampleRecord] = {}
        for record in baseline:
            key = record_join_key(record)
            if key not in baseline_by_key or abs(float(record.sample_meta["theta"]) - 0.25) < abs(
                float(baseline_by_key[key].sample_meta["theta"]) - 0.25
            ):
                baseline_by_key[key] = record
            if key not in optimum_by_key or float(record.metrics["rho"]) < float(optimum_by_key[key].metrics["rho"]):
                optimum_by_key[key] = record

        improvements: list[float] = []
        ratios: list[float] = []
        rows: list[dict] = []
        missing = 0
        for record in candidate:
            key = record_join_key(record)
            if key not in baseline_by_key or key not in optimum_by_key:
                missing += 1
                continue
            rho0 = float(baseline_by_key[key].metrics["rho"])
            rhomin = float(optimum_by_key[key].metrics["rho"])
            rho = float(record.metrics["rho"])
            if rho0 <= 0:
                continue
            improvement = 1.0 - rho / rho0
            optimum = 1.0 - rhomin / rho0
            improvements.append(improvement)
            if optimum > 1e-12:
                ratios.append(improvement / optimum)
            rows.append({"join_key": key, "rho": rho, "rho_baseline": rho0, "improvement": improvement})

        return MetricsBundle(
            summary={
                "n": len(improvements),
                "missing_join_keys": missing,
                "PB_percent": (sum(1 for x in improvements if x >= 0) / len(improvements) * 100.0)
                if improvements
                else 0.0,
                "P_avg": mean(improvements) if improvements else 0.0,
                "P_median": median(improvements) if improvements else 0.0,
                "P_over_Pmax_avg": mean(ratios) if ratios else 0.0,
                "P_over_Pmax_median": median(ratios) if ratios else 0.0,
            },
            tables={"comparison": rows},
        )
