#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from polyamg_ml.config import config_hash, load_experiment_config, make_run_id, write_resolved_config


def _csv(values: tuple[float, ...]) -> str:
    return ",".join(str(v) for v in values)


def _resolved_model_fields(model: Any) -> dict[str, Any]:
    fields = {
        "model_id": model.model_id,
        "onnx_path": model.onnx_path,
        "theta_grid": model.theta_grid,
        "preprocessing_sha256": model.preprocessing_sha256,
    }
    manifest_path = Path(model.onnx_path).with_name("manifest.json")
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        fields["model_id"] = manifest.get("model_id", fields["model_id"])
        fields["onnx_path"] = manifest.get("onnx_path", fields["onnx_path"])
        fields["theta_grid"] = tuple(float(x) for x in manifest.get("theta_grid", fields["theta_grid"]))
        fields["preprocessing_sha256"] = manifest.get("preprocessing_sha256", fields["preprocessing_sha256"])
    return fields


def write_legacy_kv(config_path: Path, kv_path: Path) -> None:
    cfg = load_experiment_config(config_path)
    lines = [
        f"experiment_id={cfg.experiment_id}",
        f"output_dir={cfg.output_dir}",
        f"mesh_path={cfg.mesh_path}",
        f"pde={cfg.pde}",
        f"seed={cfg.seed}",
        f"diffusion.pattern={cfg.diffusion_pattern}",
        f"theta_values={_csv(cfg.theta_values)}",
        f"epsilon_values={_csv(cfg.epsilon_values)}",
        f"epsilon1_values={_csv(cfg.epsilon1_values)}",
        f"epsilon2_values={_csv(cfg.epsilon2_values)}",
        f"h_values={_csv(cfg.h_values)}",
        f"solver.max_it={cfg.solver.max_it}",
        f"solver.rtol={cfg.solver.rtol}",
        f"solver.atol={cfg.solver.atol}",
        f"solver.ksp_type={cfg.solver.ksp_type}",
        f"solver.pc_type={cfg.solver.pc_type}",
        f"solver.hypre_type={cfg.solver.hypre_type}",
        f"features.m={cfg.features.m}",
        f"features.op={cfg.features.op}",
        f"features.normalize={cfg.features.normalize}",
    ]
    if cfg.model is not None:
        model = _resolved_model_fields(cfg.model)
        lines.extend(
            [
                f"model.model_id={model['model_id']}",
                f"model.onnx_path={model['onnx_path']}",
                f"model.theta_grid={_csv(model['theta_grid'])}",
                f"model.preprocessing_sha256={model['preprocessing_sha256']}",
            ]
        )
    kv_path.parent.mkdir(parents=True, exist_ok=True)
    kv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["baseline", "ann"], required=True)
    ap.add_argument("--run_root", default="data/runs")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = load_experiment_config(config_path)
    run_id = make_run_id(cfg)
    run_dir = Path(args.run_root) / run_id
    write_resolved_config(cfg, run_dir / "config.resolved.yaml")
    kv_path = run_dir / "config.legacy.kv"
    write_legacy_kv(config_path, kv_path)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": cfg.experiment_id,
        "run_id": run_id,
        "config_hash": config_hash(cfg),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command": " ".join(["scripts/run_experiment.py", "--config", str(config_path), "--mode", args.mode]),
    }
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    executable = "build/polyamg_baseline" if args.mode == "baseline" else "build/polyamg_ann_amg"
    if args.execute:
        subprocess.run([executable, str(kv_path)], check=True)
    else:
        print(run_dir)


if __name__ == "__main__":
    main()
