from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from labelone.errors import ModelRuntimeError

from ..artifacts import ArtifactStore
from ..catalog import ModelRecord
from ..types import FeatureLayer, InferenceResult


class ModelAdapter(ABC):
    def __init__(self, record: ModelRecord, artifact_store: ArtifactStore) -> None:
        self.record = record
        self.artifact_store = artifact_store
        self.loaded = False

    def resolve_local_weight(self, suffix: str) -> Path:
        for location in self.record.descriptor.weight_locations:
            if location.startswith(("http://", "https://")):
                continue
            path = Path(location).expanduser()
            resolved = (path if path.is_absolute() else self.record.descriptor.config_path.parent / path).resolve()
            if resolved.is_file() and resolved.suffix.lower() == suffix:
                return resolved
        raise ModelRuntimeError(
            f"No local {suffix} weight is available",
            details={"model_id": self.record.descriptor.id},
        )

    @abstractmethod
    def load(self, providers: list[str]) -> list[FeatureLayer]:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_layers(self) -> list[FeatureLayer]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image_path: Path, capture_layers: list[str], parameters: dict[str, object]) -> InferenceResult:
        raise NotImplementedError

