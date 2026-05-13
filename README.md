# PolyAMG-ML

Learning Algebraic Multigrid parameters for PDE systems with a PETSc+hypre C++
solver stack and Python ANN training/export, designed for direct extension to
polygonal mesh sequences.

## Stack

- C++: PETSc (`Mat/Vec/KSP/PC`) + hypre BoomerAMG, DMPlex mesh backend.
- Python: PyTorch training, ONNX export, comparison, and reporting.
- Config source of truth: YAML under `configs/`.

## Layout

- `cpp_core/`: solver, mesh/discretization adapters, feature extraction, policies, result sinks, and optional bindings.
- `py_ml/`: typed config loading, datasets, training, inference, experiments, metrics, and reporting.
- `shared_schema/`: JSON schema contracts.
- `configs/experiments/`: reproducible experiment YAML files.
- `scripts/`: supported command entrypoints.

## Build

Install or sync the Python environment first:

```bash
uv sync
```

Then build the C++ executables:

```bash
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-...
./scripts/build_cpp.sh
```

### Build With ONNX Runtime

```bash
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-...
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake -S . -B build -DPOLYAMG_WITH_ORT=ON
cmake --build build -j
```

### Optional Interactive Bindings

```bash
cmake -S . -B build -DPOLYAMG_WITH_PYBIND=ON
cmake --build build -j
```

## Baseline Data Generation

Create a reproducible run directory and execute the baseline C++ binary:

```bash
PYTHONPATH=py_ml .venv/bin/python scripts/run_experiment.py \
  --config configs/experiments/baseline.yaml \
  --mode baseline \
  --execute
```

Outputs sample records under the configured output directory, currently
`data/raw/baseline/`.

## Train ANN

```bash
PYTHONPATH=py_ml .venv/bin/python scripts/train_model.py \
  --data_glob 'data/raw/baseline/*.json' \
  --out_dir data/models/paper_like_v1
```

## Export ONNX

```bash
PYTHONPATH=py_ml .venv/bin/python scripts/export_model.py \
  --model_pt data/models/paper_like_v1/model.pt \
  --train_meta data/models/paper_like_v1/train_meta.json \
  --out_onnx data/models/paper_like_v1/model.onnx
```

This also writes `data/models/paper_like_v1/manifest.json`. The ANN run wrapper
automatically uses that manifest when it exists.

## ANN-Driven AMG Run

```bash
PYTHONPATH=py_ml .venv/bin/python scripts/run_experiment.py \
  --config configs/experiments/ann_amg.yaml \
  --mode ann \
  --execute
```

If built without `POLYAMG_WITH_ORT=ON`, ANN mode runs with the existing fallback
policy behavior.

## Generate Reports

```bash
PYTHONPATH=py_ml .venv/bin/python scripts/generate_report.py \
  --baseline_glob 'data/raw/baseline/*.json' \
  --candidate_glob 'data/raw/ann/*.json' \
  --out_md data/reports/comparison/report.md \
  --out_tex data/reports/comparison/report.tex
```

Add `--out_pdf data/reports/comparison/report.pdf` when `pdflatex` is available.

## Notes

- YAML files in `configs/experiments/` are the only committed experiment configs.
- `scripts/run_experiment.py` writes run metadata, a resolved YAML snapshot, logs,
  metrics/report directories, and the transient bridge config needed by the
  current C++ executable parser.
- `GridSearchPolicy` is interface-stable and ready for ONNX Runtime-backed theta
  inference.
- Polygonal descriptors are carried in record schemas to avoid future schema
  migration.
