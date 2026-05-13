from __future__ import annotations

import json
from pathlib import Path

import torch

from polyamg_ml.models import ModelFactory


class ModelExporter:
    def __init__(self, model_factory: ModelFactory | None = None):
        self.model_factory = model_factory or ModelFactory()

    def export_onnx(
        self,
        model_pt: str | Path,
        train_meta: str | Path,
        out_onnx: str | Path,
        theta_grid: list[float],
    ) -> Path:
        with Path(train_meta).open("r", encoding="utf-8") as f:
            meta = json.load(f)

        m = int(meta.get("model_shape", {}).get("m", meta.get("features", {}).get("m", 50)))
        in_channels = int(meta.get("model_shape", {}).get("in_channels", 1))
        model = self.model_factory.create(in_channels=in_channels, m=m)
        state = torch.load(model_pt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval()

        out = Path(out_onnx)
        out.parent.mkdir(parents=True, exist_ok=True)
        x = torch.zeros(1, in_channels, m, m)
        h = torch.tensor([3.0])
        theta = torch.tensor([0.25])
        torch.onnx.export(
            model,
            (x, h, theta),
            str(out),
            input_names=["x_img", "x_h", "x_theta"],
            output_names=["rho_pred"],
            opset_version=17,
        )

        manifest = {
            "schema_version": "1.0",
            "model_id": meta["model_id"],
            "onnx_path": str(out),
            "features": meta["features"],
            "theta_grid": theta_grid,
            "preprocessing_sha256": meta["preprocessing_sha256"],
        }
        manifest_path = out.with_name("manifest.json")
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path
