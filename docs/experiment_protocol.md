# Experiment Protocol

1. Define the experiment in `configs/experiments/*.yaml`.
2. Create a reproducible run directory:

   ```bash
   PYTHONPATH=py_ml .venv/bin/python scripts/run_experiment.py \
     --config configs/experiments/baseline.yaml \
     --mode baseline
   ```

3. Add `--execute` to run the matching C++ executable after the run metadata and
   legacy bridge config are written.

   Test case 2 configs use `epsilon1_values` and `epsilon2_values`:

   ```bash
   PYTHONPATH=py_ml .venv/bin/python scripts/run_experiment.py \
     --config configs/experiments/test_case2_checker2x2.yaml \
     --mode baseline \
     --execute
   ```

   In these runs, the assembled coefficient is `10^epsilon2` on gray cells and
   `10^epsilon1` on white cells. The older `epsilon_values` field remains the
   dataset-1 shorthand for `epsilon1=0` and `epsilon2=epsilon`.
4. Train a model from baseline records:

   ```bash
   PYTHONPATH=py_ml .venv/bin/python scripts/train_model.py \
     --data_glob 'data/raw/baseline/*.json' \
     --out_dir data/models/paper_like_v1
   ```

5. Export ONNX and the model manifest:

   ```bash
   PYTHONPATH=py_ml .venv/bin/python scripts/export_model.py \
     --model_pt data/models/paper_like_v1/model.pt \
     --train_meta data/models/paper_like_v1/train_meta.json \
     --out_onnx data/models/paper_like_v1/model.onnx
   ```

6. Generate comparison reports:

   ```bash
   PYTHONPATH=py_ml .venv/bin/python scripts/generate_report.py \
     --baseline_glob 'data/raw/baseline/*.json' \
     --candidate_glob 'data/raw/ann/*.json' \
     --out_md data/reports/comparison/report.md \
     --out_tex data/reports/comparison/report.tex
   ```
