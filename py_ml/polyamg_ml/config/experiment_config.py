from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SolverConfig:
    max_it: int = 500
    rtol: float = 1e-8
    atol: float = 1e-50
    ksp_type: str = "cg"
    pc_type: str = "hypre"
    hypre_type: str = "boomeramg"


@dataclass(frozen=True)
class FeatureConfig:
    m: int = 50
    op: str = "sum"
    normalize: str = "std+id"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    onnx_path: str
    theta_grid: tuple[float, ...] = ()
    preprocessing_sha256: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    output_dir: str
    mesh_path: str = ""
    pde: str = "elliptic"
    seed: int = 1
    theta_values: tuple[float, ...] = (0.25,)
    epsilon_values: tuple[float, ...] = (0.0,)
    epsilon1_values: tuple[float, ...] = ()
    epsilon2_values: tuple[float, ...] = ()
    h_values: tuple[float, ...] = (0.125,)
    diffusion_pattern: str = "checker2x2"
    solver: SolverConfig = field(default_factory=SolverConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig | None = None
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple_float(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(float(x.strip()) for x in value.split(",") if x.strip())
    return tuple(float(x) for x in value)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping")
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")

    solver_data = _section(data, "solver")
    feature_data = _section(data, "features")
    model_data = data.get("model")
    model = None
    if isinstance(model_data, dict) and model_data.get("model_id"):
        model = ModelConfig(
            model_id=str(model_data["model_id"]),
            onnx_path=str(model_data.get("onnx_path", "")),
            theta_grid=_tuple_float(model_data.get("theta_grid"), ()),
            preprocessing_sha256=str(model_data.get("preprocessing_sha256", "")),
        )

    return ExperimentConfig(
        experiment_id=str(data["experiment_id"]),
        output_dir=str(data["output_dir"]),
        mesh_path=str(data.get("mesh_path", "")),
        pde=str(data.get("pde", "elliptic")),
        seed=int(data.get("seed", 1)),
        theta_values=_tuple_float(data.get("theta_values"), (0.25,)),
        epsilon_values=_tuple_float(data.get("epsilon_values"), (0.0,)),
        epsilon1_values=_tuple_float(data.get("epsilon1_values"), ()),
        epsilon2_values=_tuple_float(data.get("epsilon2_values"), ()),
        h_values=_tuple_float(data.get("h_values"), (0.125,)),
        diffusion_pattern=str(data.get("diffusion_pattern", "checker2x2")),
        solver=SolverConfig(**solver_data),
        features=FeatureConfig(**feature_data),
        model=model,
        schema_version=str(data.get("schema_version", "1.0")),
    )


def config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_run_id(config: ExperimentConfig, when: datetime | None = None) -> str:
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{config.experiment_id}-{config_hash(config)[:10]}-{stamp}"


def write_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=True)
