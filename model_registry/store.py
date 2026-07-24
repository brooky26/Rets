"""
Model Registry — storage backends.

Same swappable-persistence pattern as `data/storage.py`'s `TickStore`:
a narrow `Protocol` plus an in-memory implementation for tests/dev and a
JSON-file implementation for simple local persistence. Swapping in a
Postgres/Supabase-backed store later (matching the Railway deployment
notes elsewhere in this project) means writing one new class against
the same Protocol, not touching `ModelRegistry` or any caller.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from model_registry.types import ModelStatus, ModelVersion, PromotionRecord


class ModelRegistryStore(Protocol):
    def save_version(self, version: ModelVersion) -> None: ...
    def get_version(self, model_id: str) -> ModelVersion | None: ...
    def list_versions(self, model_type: str) -> list[ModelVersion]: ...
    def get_champion(self, model_type: str) -> ModelVersion | None: ...
    def append_promotion_record(self, record: PromotionRecord) -> None: ...
    def get_promotion_history(self, model_type: str) -> list[PromotionRecord]: ...


class InMemoryModelRegistryStore:
    def __init__(self) -> None:
        self._versions: dict[str, ModelVersion] = {}
        self._promotion_history: list[PromotionRecord] = []

    def save_version(self, version: ModelVersion) -> None:
        self._versions[version.model_id] = version

    def get_version(self, model_id: str) -> ModelVersion | None:
        return self._versions.get(model_id)

    def list_versions(self, model_type: str) -> list[ModelVersion]:
        return sorted(
            (v for v in self._versions.values() if v.model_type == model_type),
            key=lambda v: v.version,
        )

    def get_champion(self, model_type: str) -> ModelVersion | None:
        champions = [
            v for v in self._versions.values()
            if v.model_type == model_type and v.status == ModelStatus.CHAMPION
        ]
        return champions[0] if champions else None

    def append_promotion_record(self, record: PromotionRecord) -> None:
        self._promotion_history.append(record)

    def get_promotion_history(self, model_type: str) -> list[PromotionRecord]:
        return [r for r in self._promotion_history if r.model_type == model_type]


class JSONFileModelRegistryStore:
    """
    Simple local persistence: the entire registry state (versions +
    promotion history) round-trips to a single JSON file. Fine for
    development and single-instance deployment; the Railway ephemeral-
    filesystem caveat that applies to `SQLiteTickStore` applies here too
    — mount a volume, or swap in a real database-backed store before
    depending on this surviving a redeploy.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, ModelVersion] = {}
        self._promotion_history: list[PromotionRecord] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._path, "r") as f:
            raw = json.load(f)
        for v in raw.get("versions", []):
            v = dict(v)
            v["status"] = ModelStatus(v["status"])
            self._versions[v["model_id"]] = ModelVersion(**v)
        for r in raw.get("promotion_history", []):
            self._promotion_history.append(PromotionRecord(**r))

    def _persist(self) -> None:
        raw = {
            "versions": [
                {**asdict(v), "status": v.status.value} for v in self._versions.values()
            ],
            "promotion_history": [asdict(r) for r in self._promotion_history],
        }
        with open(self._path, "w") as f:
            json.dump(raw, f, indent=2, default=str)

    def save_version(self, version: ModelVersion) -> None:
        self._versions[version.model_id] = version
        self._persist()

    def get_version(self, model_id: str) -> ModelVersion | None:
        return self._versions.get(model_id)

    def list_versions(self, model_type: str) -> list[ModelVersion]:
        return sorted(
            (v for v in self._versions.values() if v.model_type == model_type),
            key=lambda v: v.version,
        )

    def get_champion(self, model_type: str) -> ModelVersion | None:
        champions = [
            v for v in self._versions.values()
            if v.model_type == model_type and v.status == ModelStatus.CHAMPION
        ]
        return champions[0] if champions else None

    def append_promotion_record(self, record: PromotionRecord) -> None:
        self._promotion_history.append(record)
        self._persist()

    def get_promotion_history(self, model_type: str) -> list[PromotionRecord]:
        return [r for r in self._promotion_history if r.model_type == model_type]
