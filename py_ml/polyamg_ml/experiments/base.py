from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from polyamg_ml.config import ExperimentConfig
from polyamg_ml.data import SampleRecordRepository
from polyamg_ml.metrics import MetricComputer, MetricsBundle


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    run_id: str
    records: list[dict]
    metrics: MetricsBundle | None = None
    artifacts: dict[str, str] = field(default_factory=dict)


class BaseExperiment(ABC):
    def __init__(self, config: ExperimentConfig, executable: str | None = None):
        self.config = config
        self.executable = executable

    @abstractmethod
    def default_executable(self) -> str:
        raise NotImplementedError

    def run(self) -> SampleRecordRepository:
        exe = self.executable or self.default_executable()
        raise NotImplementedError(
            "Write resolved YAML to the run directory and invoke a generated legacy .kv bridge before calling "
            f"{exe}; this base class defines the contract but does not assume local PETSc availability."
        )

    def evaluate(self, baseline: SampleRecordRepository | None = None) -> MetricsBundle:
        records = SampleRecordRepository.from_directory(self.config.output_dir)
        if baseline is None:
            return MetricComputer().summarize_single(records.all())
        return MetricComputer().compare(baseline.all(), records.all())

    def _run_process(self, args: list[str], cwd: Path | None = None) -> None:
        subprocess.run(args, cwd=cwd, check=True)


class BaselineExperiment(BaseExperiment):
    def default_executable(self) -> str:
        return "build/polyamg_baseline"


class AnnAmgExperiment(BaseExperiment):
    def default_executable(self) -> str:
        return "build/polyamg_ann_amg"
