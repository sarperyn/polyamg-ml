from __future__ import annotations

from typing import Any

from polyamg_ml.model import CNNFFN


class ModelFactory:
    def create(self, name: str = "cnn_ffn", **kwargs: Any):
        if name != "cnn_ffn":
            raise ValueError(f"Unknown model architecture: {name}")
        return CNNFFN(**kwargs)
