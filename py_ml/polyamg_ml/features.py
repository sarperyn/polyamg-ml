from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

OpMode = Literal["sum", "max", "pp+np", "pp+np+sum"]
NormMode = Literal["std+id", "std+avg", "scale+id", "scale+avg", "log+id", "log+avg"]


@dataclass(frozen=True)
class FeatureConfig:
    m: int = 50
    op: OpMode = "sum"
    normalize: NormMode = "std+id"


def pooling_from_coo(n: int, row: np.ndarray, col: np.ndarray, val: np.ndarray, m: int, op: OpMode):
    q = n // m
    p = n % m
    t = (q + 1) * p

    V = np.zeros((m, m), dtype=np.float64)
    C = np.zeros((m, m), dtype=np.float64)

    def map_idx(r: np.ndarray) -> np.ndarray:
        out = np.where(r < t, r // (q + 1), (r - t) // max(q, 1) + p)
        return np.clip(out.astype(np.int64), 0, m - 1)

    ii = map_idx(row)
    jj = map_idx(col)

    if op == "sum":
        for i, j, v in zip(ii, jj, val):
            V[i, j] += float(v)
            C[i, j] += 1
        return np.expand_dims(V, 0), np.expand_dims(C, 0)

    if op == "max":
        for i, j, v in zip(ii, jj, val):
            V[i, j] = max(V[i, j], abs(float(v)))
            C[i, j] += 1
        return np.expand_dims(V, 0), np.expand_dims(C, 0)

    def ch_pp(x):
        return max(0.0, x)

    def ch_np(x):
        return max(0.0, -x)

    if op in ("pp+np", "pp+np+sum"):
        channels = 2 if op == "pp+np" else 3
        VV = np.zeros((channels, m, m), dtype=np.float64)
        CC = np.zeros((channels, m, m), dtype=np.float64)
        for i, j, v in zip(ii, jj, val):
            fv = float(v)
            VV[0, i, j] = max(VV[0, i, j], ch_pp(fv))
            VV[1, i, j] = max(VV[1, i, j], ch_np(fv))
            CC[0, i, j] += 1
            CC[1, i, j] += 1
            if channels == 3:
                VV[2, i, j] += fv
                CC[2, i, j] += 1
        return VV, CC

    raise ValueError(f"Unknown op mode {op}")


def _norm(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "std":
        mu = x.mean()
        sigma = max(x.std(), 1e-12)
        return (x - mu) / sigma
    if mode == "scale":
        mx = max(float(np.abs(x).max()), 1e-12)
        return x / mx
    if mode == "log":
        y = np.sign(x) * np.log(np.abs(x) + 1.0)
        mx = max(float(np.abs(y).max()), 1e-12)
        return y / mx
    raise ValueError(mode)


def normalize(V: np.ndarray, C: np.ndarray, normalize_mode: NormMode) -> np.ndarray:
    left, right = normalize_mode.split("+")
    X = V.copy()
    if right == "avg":
        np.divide(X, C, out=X, where=C > 0)
    return _norm(X, left)


def preprocessing_sha256(cfg: FeatureConfig) -> str:
    s = json.dumps({"m": cfg.m, "op": cfg.op, "normalize": cfg.normalize}, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
