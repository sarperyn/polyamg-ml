from __future__ import annotations

import argparse

from polyamg_ml.training import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["mse", "mae"], default="mse")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m", type=int, default=50)
    ap.add_argument("--op", default="sum")
    ap.add_argument("--normalize", default="std+id")
    args = ap.parse_args()
    Trainer().train(**vars(args))


if __name__ == "__main__":
    main()
