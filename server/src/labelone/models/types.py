from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AvailabilityState(str, Enum):
    AVAILABLE = "available"
    MISSING_WEIGHTS = "missing_weights"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


class FeatureCaptureMode(str, Enum):
    NONE = "none"
    EXPORTED_OUTPUTS = "exported_outputs"
    EAGER_HOOKS = "eager_hooks"
    GRAPH_REWRITE = "graph_rewrite"
    REMOTE = "remote"


class Availability(BaseModel):
    state: AvailabilityState
    reason: str | None = None


class FeatureLayer(BaseModel):
    id: str
    group: str = "Outputs"
    name: str
    shape: list[int | str | None] = Field(default_factory=list)
    axes: list[str] = Field(default_factory=list)
    dtype: str | None = None
    spatial: bool = False
    captureable: bool = True
    reason: str | None = None


class FeatureCapture(BaseModel):
    mode: FeatureCaptureMode = FeatureCaptureMode.NONE
    enumerable: bool = False
    layers: list[FeatureLayer] = Field(default_factory=list)


class ModelCapabilities(BaseModel):
    predict: bool = False
    unload: bool = True
    result_kinds: list[Literal["annotations", "classifications", "tensors", "rasters"]] = Field(default_factory=list)
    feature_capture: FeatureCapture = Field(default_factory=FeatureCapture)
    parameters_schema: dict[str, Any] = Field(default_factory=dict)


class ModelDescriptor(BaseModel):
    id: str
    name: str
    display_name: str
    model_type: str
    provider: str = "Unknown"
    task: str
    family: str
    adapter: str
    source_url: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    usage_notice: str | None = None
    runtime: list[str] = Field(default_factory=list)
    config_path: Path
    weight_locations: list[str] = Field(default_factory=list)
    availability: Availability
    capabilities: ModelCapabilities


class CatalogWarning(BaseModel):
    path: Path
    code: str
    message: str


class ModelCatalogStatus(BaseModel):
    runtime_state: Literal["unloaded", "loading", "loaded", "failed"] = "unloaded"
    usage_count: int = Field(default=0, ge=0)
    last_used_at: str | None = None


class ModelCatalogResponse(BaseModel):
    models: list[ModelDescriptor]
    warnings: list[CatalogWarning] = Field(default_factory=list)
    status_by_model: dict[str, ModelCatalogStatus] = Field(default_factory=dict)


class ImportCatalogRequest(BaseModel):
    root_dir: Path


class ModelLoadRequest(BaseModel):
    providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])


class ModelRuntimeState(BaseModel):
    model_id: str
    state: Literal["unloaded", "loading", "loaded", "failed"]
    layers: list[FeatureLayer] = Field(default_factory=list)
    capture_mode: FeatureCaptureMode = FeatureCaptureMode.NONE
    capture_warning: str | None = None
    error: str | None = None


class InferenceRequest(BaseModel):
    model_id: str
    image_path: Path
    capture_layers: list[str] = Field(default_factory=list, max_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("capture_layers")
    @classmethod
    def validate_capture_layers(cls, value: list[str]) -> list[str]:
        if any(not layer or len(layer) > 1024 for layer in value) or len(set(value)) != len(value):
            raise ValueError("capture_layers must contain one unique bounded layer id")
        return value


class AnnotationResult(BaseModel):
    label: str
    score: float
    shape_type: Literal["rectangle", "rotation", "polygon", "point"] = "rectangle"
    points: list[list[float]]


class ClassificationResult(BaseModel):
    label: str
    score: float
    rank: int


class TensorArtifact(BaseModel):
    id: str
    layer_id: str
    path: Path
    shape: list[int]
    dtype: str
    size_bytes: int
    statistics: dict[str, float]
    source_shape: list[int] = Field(default_factory=list)
    transform: dict[str, Any] = Field(default_factory=dict)
    preview_available: bool = False
    preview_width: int | None = None
    preview_height: int | None = None


class RasterArtifact(BaseModel):
    id: str
    role: str
    path: Path
    media_type: str
    width: int
    height: int
    size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceResult(BaseModel):
    model_id: str
    image_path: Path
    annotations: list[AnnotationResult] = Field(default_factory=list)
    classifications: list[ClassificationResult] = Field(default_factory=list)
    artifacts: list[TensorArtifact] = Field(default_factory=list)
    rasters: list[RasterArtifact] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
