from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from polyamg_ml.dataset import JsonSampleDataset
from polyamg_ml.features import FeatureConfig, preprocessing_sha256
from polyamg_ml.models import ModelFactory


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
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        loss: str = "mse",
        seed: int = 42,
        m: int = 50,
        op: str = "sum",
        normalize: str = "std+id",
    ) -> TrainingResult:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        ds = JsonSampleDataset(data_glob)
        n = len(ds)
        if n == 0:
            raise RuntimeError("No training samples found. Generate baseline data first.")
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        n_test = n - n_train - n_val
        train_ds, val_ds, _ = random_split(
            ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
        )
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=batch_size)

        sample0 = ds[0][0]
        in_channels = int(sample0.shape[0])
        m_in = int(sample0.shape[-1])
        model = self.model_factory.create(in_channels=in_channels, m=m_in)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.MSELoss() if loss == "mse" else nn.L1Loss()

        best_val = float("inf")
        best_state = None
        patience = 30
        stale = 0
        for _ in range(epochs):
            model.train()
            for x, h, theta, y in train_ld:
                pred = model(x, -torch.log2(torch.clamp(h, min=1e-8)), theta)
                train_loss = crit(pred, y)
                opt.zero_grad()
                train_loss.backward()
                opt.step()

            model.eval()
            vals = []
            with torch.no_grad():
                for x, h, theta, y in val_ld:
                    pred = model(x, -torch.log2(torch.clamp(h, min=1e-8)), theta)
                    vals.append(crit(pred, y).item())
            val = float(sum(vals) / max(len(vals), 1))
            if val < best_val:
                best_val = val
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model_path = out / "model.pt"
        torch.save(model.state_dict(), model_path)
        cfg = FeatureConfig(m=m, op=op, normalize=normalize)
        metadata = {
            "schema_version": "1.0",
            "model_id": out.name,
            "features": {"m": cfg.m, "op": cfg.op, "normalize": cfg.normalize},
            "model_shape": {"in_channels": in_channels, "m": m_in},
            "preprocessing_sha256": preprocessing_sha256(cfg),
            "best_val": best_val,
            "loss": loss,
            "epochs_requested": epochs,
            "seed": seed,
        }
        metadata_path = out / "train_meta.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return TrainingResult(model_path=model_path, metadata_path=metadata_path, best_val=best_val)
