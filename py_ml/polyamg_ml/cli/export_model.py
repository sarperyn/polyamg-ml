from __future__ import annotations

import argparse

from polyamg_ml.inference import ModelExporter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pt", required=True)
    ap.add_argument("--train_meta", required=True)
    ap.add_argument("--out_onnx", required=True)
    ap.add_argument("--theta_grid", default="0.02,0.04,0.06,0.08,0.1,0.25,0.5,0.72,0.9")
    args = ap.parse_args()
    theta_grid = [float(x.strip()) for x in args.theta_grid.split(",") if x.strip()]
    ModelExporter().export_onnx(args.model_pt, args.train_meta, args.out_onnx, theta_grid)


if __name__ == "__main__":
    main()
