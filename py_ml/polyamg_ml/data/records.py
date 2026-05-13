from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SampleRecord:
    path: Path | None
    sample_meta: dict[str, Any]
    feature_config: dict[str, Any]
    metrics: dict[str, Any]
    feature_tensor: dict[str, Any] | None = None
    schema_version: str = "1.0"

    @classmethod
    def from_dict(cls, payload: dict[str, Any], path: Path | None = None) -> "SampleRecord":
        for key in ("sample_meta", "feature_config", "metrics"):
            if key not in payload:
                raise ValueError(f"SampleRecord missing required key '{key}'")
        return cls(
            path=path,
            sample_meta=dict(payload["sample_meta"]),
            feature_config=dict(payload["feature_config"]),
            metrics=dict(payload["metrics"]),
            feature_tensor=payload.get("feature_tensor"),
            schema_version=str(payload.get("schema_version", "1.0")),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "SampleRecord":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f), p)


def record_join_key(record: SampleRecord) -> tuple[Any, ...]:
    meta = record.sample_meta
    descriptors = meta.get("polygonal_descriptors", {})
    return (
        meta.get("pde_type"),
        float(meta.get("h", 0.0)),
        None if meta.get("epsilon") is None else float(meta["epsilon"]),
        descriptors.get("diffusion_pattern"),
        meta.get("mesh_id"),
        int(meta.get("seed", 0)),
    )


class SampleRecordRepository:
    def __init__(self, records: Iterable[SampleRecord]):
        self._records = list(records)

    @classmethod
    def from_glob(cls, pattern: str) -> "SampleRecordRepository":
        return cls(SampleRecord.from_path(p) for p in sorted(glob.glob(pattern)))

    @classmethod
    def from_directory(cls, directory: str | Path) -> "SampleRecordRepository":
        return cls.from_glob(str(Path(directory) / "*.json"))

    def all(self) -> list[SampleRecord]:
        return list(self._records)

    def by_join_key(self) -> dict[tuple[Any, ...], list[SampleRecord]]:
        grouped: dict[tuple[Any, ...], list[SampleRecord]] = {}
        for record in self._records:
            grouped.setdefault(record_join_key(record), []).append(record)
        return grouped
