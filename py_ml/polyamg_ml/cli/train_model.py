from __future__ import annotations

import argparse

from polyamg_ml.training import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["mse", "mae"], default="mse")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m", type=int, default=50)
    ap.add_argument("--op", default="sum")
    ap.add_argument("--normalize", default="std+id")
    ap.add_argument("--conv_channels", type=int, default=32)
    ap.add_argument("--conv_depth", type=int, default=2)
    ap.add_argument("--dense_width", type=int, default=64)
    ap.add_argument("--dense_depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--conv1_channels", type=int, default=None)
    ap.add_argument("--conv1_depth", type=int, default=None)
    ap.add_argument("--conv1_dropout", type=float, default=None)
    ap.add_argument("--conv2_channels", type=int, default=None)
    ap.add_argument("--conv2_depth", type=int, default=0)
    ap.add_argument("--conv2_dropout", type=float, default=0.0)
    ap.add_argument("--cnn_out_width", type=int, default=128)
    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()
    kwargs = vars(args)
    kwargs["progress"] = not kwargs.pop("no_progress")
    Trainer().train(**kwargs)


if __name__ == "__main__":
    main()
