from __future__ import annotations

import torch


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def resolve_device(requested: str = "auto") -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if value == "mps" and not _mps_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
    if value == "cpu":
        return torch.device("cpu")
    if value in {"cuda", "mps"}:
        return torch.device(value)
    raise ValueError(f"Unknown device '{requested}'. Use auto, cpu, cuda, or mps.")
