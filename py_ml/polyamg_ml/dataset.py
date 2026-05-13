from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    feature: np.ndarray
    h: float
    theta: float
    rho: float


class JsonSampleDataset(Dataset):
    def __init__(self, pattern: str):
        self.paths = sorted(glob.glob(pattern))
        self.samples: List[Sample] = []
        for p in self.paths:
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
            if "feature_tensor" in j:
                ft = j["feature_tensor"]
                m = int(ft["m"])
                c = int(ft["c"])
                vals = np.asarray(ft["values"], dtype=np.float32)
                expected = c * m * m
                if vals.size != expected:
                    raise ValueError(f"Invalid feature tensor length in {p}: got {vals.size}, expected {expected}")
                feat = vals.reshape(c, m, m)
            else:
                # backward-compatible fallback for old records
                feat = np.zeros((1, 50, 50), dtype=np.float32)
            self.samples.append(Sample(
                feature=feat,
                h=float(j["sample_meta"]["h"]),
                theta=float(j["sample_meta"]["theta"]),
                rho=float(j["metrics"]["rho"]),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s.feature),
            torch.tensor(s.h, dtype=torch.float32),
            torch.tensor(s.theta, dtype=torch.float32),
            torch.tensor(s.rho, dtype=torch.float32),
        )
