from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PipelineNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: str
    enabled: bool = Field(default=True, strict=True)
    parameters: dict[str, Any] = Field(default_factory=dict)


class PipelinePreviewRequest(BaseModel):
    dataset_id: str
    asset_id: str
    nodes: list[PipelineNode]
    output_format: Literal["webp", "png", "jpeg"] = "webp"
    priority: Literal["interactive", "background"] = "interactive"


class PipelineValidationRequest(BaseModel):
    nodes: list[PipelineNode]
    mode: Literal["preview", "derived_dataset"] = "preview"
    width: int | None = None
    height: int | None = None


class PipelineValidationResult(BaseModel):
    valid: bool = True
    registry_hash: str
    normalized_nodes: list[dict[str, Any]]
    transform_count: int
    visualization_count: int
    output_width: int | None = None
    output_height: int | None = None


class PipelineCoordinateMapping(BaseModel):
    kind: Literal["identity", "affine", "unavailable"]
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    source_to_output: tuple[float, float, float, float, float, float] | None = None
    output_to_source: tuple[float, float, float, float, float, float] | None = None
    coordinate_space_id: str
    topology_safe: bool = True
    reason: str | None = None


class PipelineVisualizationResult(BaseModel):
    visualization_id: str
    label: str
    artifact_id: str
    width: int
    height: int
    media_type: str
    annotation_document: dict[str, Any]
    operator_timings_ms: dict[str, float]
    content_kind: Literal["image", "model_feature", "frequency_spectrum", "wavelet_coefficients"] = "image"
    overlay_compatible: bool = True
    coordinate_mapping: PipelineCoordinateMapping


class PipelinePreviewResult(BaseModel):
    dataset_id: str
    asset_id: str
    artifact_id: str
    width: int
    height: int
    media_type: str
    annotation_document: dict[str, Any]
    operator_timings_ms: dict[str, float]
    operator_average_timings_ms: dict[str, float]
    timing_sample_count: dict[str, int]
    visualizations: list[PipelineVisualizationResult] = Field(default_factory=list)
    cache_hit: bool = False


class PipelineOutputPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["preview", "derived_dataset"] = "preview"
    output_root: Path | None = None
    image_format: Literal["png", "webp", "jpeg"] = "png"
    conflict: Literal["reuse", "error"] = "reuse"


class DerivedOutput(BaseModel):
    image_relative_path: str
    annotation_relative_path: str
    width: int
    height: int
    tile: dict[str, int] | None = None
    annotation_count: int


class PipelineDerivedItemResult(BaseModel):
    dataset_id: str
    asset_id: str
    output_root: Path
    item_fingerprint: str
    outputs: list[DerivedOutput]
    cache_hit: bool = False


class DerivedDatasetPublishResult(BaseModel):
    output_root: Path
    dataset_fingerprint: str
    item_count: int
    output_count: int
    reused: bool = False
