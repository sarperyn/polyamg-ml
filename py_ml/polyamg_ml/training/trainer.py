from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from polyamg_ml.dataset import JsonSampleDataset
from polyamg_ml.features import FeatureConfig, preprocessing_sha256
from polyamg_ml.models import ModelFactory
from polyamg_ml.training.device import resolve_device


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    metadata_path: Path
    best_val: float


class Trainer:
    def __init__(self, model_factory: ModelFactory | None = None):
        self.model_factory = model_factory or ModelFactory()

    def train(
        self,
        data_glob: str,
        out_dir: str | Path,
        *,
        epochs: int = 500,
        batch_size: int = 32,
        lr: float = 1e-3,
        loss: str = "mse",
        seed: int = 42,
        m: int = 50,
        op: str = "sum",
        normalize: str = "std+id",
        conv_channels: int = 32,
        conv_depth: int = 2,
        dense_width: int = 64,
        dense_depth: int = 2,
        dropout: float = 0.25,
        conv1_channels: int | None = None,
        conv1_depth: int | None = None,
        conv1_dropout: float | None = None,
        conv2_channels: int | None = None,
        conv2_depth: int = 0,
        conv2_dropout: float = 0.0,
        cnn_out_width: int = 128,
        device: str = "auto",
        progress: bool = True,
    ) -> TrainingResult:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_device = resolve_device(device)

        torch.manual_seed(seed)
        ds = JsonSampleDataset(data_glob)
        n = len(ds)
        if n == 0:
            raise RuntimeError("No training samples found. Generate baseline data first.")
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        n_test = n - n_train - n_val
        train_ds, val_ds, test_ds = random_split(
            ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
        )
        pin_memory = train_device.type == "cuda"
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        val_ld = DataLoader(val_ds, batch_size=batch_size, pin_memory=pin_memory)
        test_ld = DataLoader(test_ds, batch_size=batch_size, pin_memory=pin_memory)

        sample0 = ds[0][0]
        in_channels = int(sample0.shape[0])
        m_in = int(sample0.shape[-1])
        conv1_channels = conv_channels if conv1_channels is None else conv1_channels
        conv1_depth = conv_depth if conv1_depth is None else conv1_depth
        conv1_dropout = dropout if conv1_dropout is None else conv1_dropout
        architecture = {
            "name": "cnn_ffn",
            "W1": conv1_channels,
            "D1": conv1_depth,
            "P1": conv1_dropout,
            "W2": conv2_channels,
            "D2": conv2_depth,
            "P2": conv2_dropout,
            "O": cnn_out_width,
            "W3": dense_width,
            "D3": dense_depth,
        }
        model = self.model_factory.create(
            in_channels=in_channels,
            m=m_in,
            conv1_channels=conv1_channels,
            conv1_depth=conv1_depth,
            conv1_dropout=conv1_dropout,
            conv2_channels=conv2_channels,
            conv2_depth=conv2_depth,
            conv2_dropout=conv2_dropout,
            cnn_out_width=cnn_out_width,
            dense_width=dense_width,
            dense_depth=dense_depth,
        ).to(train_device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.MSELoss() if loss == "mse" else nn.L1Loss()

        best_val = float("inf")
        best_state = None
        patience = 30
        stale = 0
        stopped_epoch = epochs
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_total = 0.0
            train_count = 0
            for x, h, theta, y in train_ld:
                x = x.to(train_device, non_blocking=pin_memory)
                h = h.to(train_device, non_blocking=pin_memory)
                theta = theta.to(train_device, non_blocking=pin_memory)
                y = y.to(train_device, non_blocking=pin_memory)
                pred = model(x, -torch.log2(torch.clamp(h, min=1e-8)), theta)
                train_loss = crit(pred, y)
                opt.zero_grad()
                train_loss.backward()
                opt.step()
                batch_n = int(y.numel())
                train_loss_total += float(train_loss.item()) * batch_n
                train_count += batch_n

            model.eval()
            val_loss_total = 0.0
            val_count = 0
            with torch.no_grad():
                for x, h, theta, y in val_ld:
                    x = x.to(train_device, non_blocking=pin_memory)
                    h = h.to(train_device, non_blocking=pin_memory)
                    theta = theta.to(train_device, non_blocking=pin_memory)
                    y = y.to(train_device, non_blocking=pin_memory)
                    pred = model(x, -torch.log2(torch.clamp(h, min=1e-8)), theta)
                    val_loss = crit(pred, y)
                    batch_n = int(y.numel())
                    val_loss_total += float(val_loss.item()) * batch_n
                    val_count += batch_n
            train_epoch_loss = float(train_loss_total / max(train_count, 1))
            val = float(val_loss_total / max(val_count, 1))
            if val < best_val:
                best_val = val
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if progress:
                self._print_progress(
                    epoch=epoch,
                    epochs=epochs,
                    train_loss=train_epoch_loss,
                    val_loss=val,
                    best_val=best_val,
                    stale=stale,
                    patience=patience,
                )
            if stale >= patience:
                stopped_epoch = epoch
                break
        if progress:
            print(file=sys.stderr)

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        test_mse_total = 0.0
        test_mae_total = 0.0
        test_count = 0
        with torch.no_grad():
            for x, h, theta, y in test_ld:
                x = x.to(train_device, non_blocking=pin_memory)
                h = h.to(train_device, non_blocking=pin_memory)
                theta = theta.to(train_device, non_blocking=pin_memory)
                y = y.to(train_device, non_blocking=pin_memory)
                pred = model(x, -torch.log2(torch.clamp(h, min=1e-8)), theta)
                batch_n = int(y.numel())
                test_mse_total += nn.functional.mse_loss(pred, y, reduction="sum").item()
                test_mae_total += nn.functional.l1_loss(pred, y, reduction="sum").item()
                test_count += batch_n
        test_mse = float(test_mse_total / max(test_count, 1))
        test_mae = float(test_mae_total / max(test_count, 1))

        model_path = out / "model.pt"
        torch.save(model.state_dict(), model_path)
        cfg = FeatureConfig(m=m, op=op, normalize=normalize)
        metadata = {
            "schema_version": "1.0",
            "model_id": out.name,
            "features": {"m": cfg.m, "op": cfg.op, "normalize": cfg.normalize},
            "architecture": architecture,
            "model_shape": {"in_channels": in_channels, "m": m_in},
            "preprocessing_sha256": preprocessing_sha256(cfg),
            "best_val": best_val,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "loss": loss,
            "epochs_requested": epochs,
            "epochs_completed": stopped_epoch,
            "split": {
                "train": n_train,
                "validation": n_val,
                "test": n_test,
                "train_fraction": 0.6,
                "validation_fraction": 0.2,
                "test_fraction": 0.2,
            },
            "batch_size": batch_size,
            "learning_rate": lr,
            "device": str(train_device),
            "seed": seed,
        }
        metadata_path = out / "train_meta.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return TrainingResult(model_path=model_path, metadata_path=metadata_path, best_val=best_val)

    @staticmethod
    def _print_progress(
        *,
        epoch: int,
        epochs: int,
        train_loss: float,
        val_loss: float,
        best_val: float,
        stale: int,
        patience: int,
    ) -> None:
        width = 28
        filled = int(width * epoch / max(epochs, 1))
        bar = "#" * filled + "-" * (width - filled)
        message = (
            f"\r[{bar}] {epoch:4d}/{epochs:<4d} "
            f"train={train_loss:.6g} val={val_loss:.6g} "
            f"best={best_val:.6g} patience={stale}/{patience}"
        )
        print(message, end="", file=sys.stderr, flush=True)
