# PolyAMG-ML Architecture

PolyAMG-ML uses a hybrid research architecture. C++ owns PETSc/DMPlex assembly,
feature extraction, AMG policy selection, and solve execution. Python owns model
training, ONNX export, experiment comparison, reproducibility metadata, and
report generation.

## C++ Core

The central orchestration class is `ExperimentRunner`. It depends only on
interfaces: `MeshAdapter`, `DiscretizationAdapter`, `FeatureExtractor`,
`AMGPolicy`, and `ResultSink`.

`ResultSink` decouples the solver workflow from persistence. Production runs use
`JsonResultSink`; interactive bindings and tests can use `InMemoryResultSink`.
PETSc objects created inside the experiment loop are wrapped by RAII handles in
`polyamg/core/petsc_handles.hpp`.

Compatibility headers remain at `polyamg/*.hpp`, while modular include paths are
available under `polyamg/core`, `polyamg/mesh`, `polyamg/discretization`,
`polyamg/features`, `polyamg/solver`, `polyamg/policy`, and
`polyamg/experiment`.

## Python Layer

Python modules are organized by responsibility:

- `config`: typed YAML config loading, hashing, and run IDs.
- `data`: `SampleRecord` DTOs and repositories.
- `models`: PyTorch model factories.
- `training`: `Trainer` and `TrainingResult`.
- `inference`: ONNX export and model manifest generation.
- `experiments`: `BaseExperiment`, `BaselineExperiment`, and `AnnAmgExperiment`.
- `metrics`: baseline/candidate comparison.
- `reporting`: format-neutral documents plus Markdown, LaTeX, and PDF exporters.
- `bindings`: optional `_polyamg_core` pybind11 loader.
- `cli`: thin command entrypoints.

## Interfaces

The primary reproducible interface is artifact based:

1. YAML experiment config is resolved and fingerprinted.
2. The wrapper creates a legacy `.kv` bridge for the existing C++ executables.
3. C++ writes schema-versioned `SampleRecord` JSON.
4. Python trains models, exports ONNX, compares runs, and generates reports.

The optional interactive interface is `_polyamg_core`, enabled with
`-DPOLYAMG_WITH_PYBIND=ON` when pybind11 is installed. It exposes DTO-level APIs
and does not expose PETSc internals.

## Reproducibility

Every wrapper-created run directory contains:

- `config.resolved.yaml`
- `config.legacy.kv`
- `manifest.json`
- `logs/`
- `records/`
- `metrics/`
- `reports/`

The deterministic run ID format is:

```text
<experiment_id>-<config_hash_short>-<utc_timestamp>
```

Schemas live under `shared_schema/` and include `schema_version` fields.
