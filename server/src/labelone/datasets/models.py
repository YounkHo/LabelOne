from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AssetStatus(str, Enum):
    VALID = "valid"
    DUPLICATE_MATCH = "duplicate_match"
    ORPHAN_ANNOTATION = "orphan_annotation"
    CORRUPT_IMAGE = "corrupt_image"
    CORRUPT_ANNOTATION = "corrupt_annotation"


class DatasetScanRequest(BaseModel):
    dataset_id: str | None = None
    root_dir: Path
    image_dir: Path | None = None
    annotation_dir: Path | None = None
    annotation_storage_root: Path | None = None
    layout: Literal["auto", "same_directory", "parallel", "custom"] = "auto"
    match_strategy: Literal["relative_stem", "same_directory", "image_path", "basename"] = "relative_stem"
    recursive: bool = True
    validate_images: bool = True
    validate_annotations: bool = True
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    annotation_extension: str = ".json"

    @field_validator("image_extensions")
    @classmethod
    def normalize_image_extensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({item.lower() if item.startswith(".") else f".{item.lower()}" for item in value}))

    @field_validator("annotation_extension")
    @classmethod
    def normalize_annotation_extension(cls, value: str) -> str:
        value = value.lower()
        return value if value.startswith(".") else f".{value}"


class DatasetAsset(BaseModel):
    asset_id: str
    match_key: str
    display_path: str
    image_path: Path | None = None
    annotation_paths: list[Path] = Field(default_factory=list)
    status: AssetStatus
    selectable: bool
    reason: str | None = None
    issues: list[str] = Field(default_factory=list)
    width: int | None = None
    height: int | None = None
    annotation_count: int | None = None
    annotation_file_exists: bool = False
    labels: list[str] = Field(default_factory=list)
    shape_types: list[str] = Field(default_factory=list)


class DatasetScanSummary(BaseModel):
    valid: int = 0
    duplicate_match: int = 0
    orphan_annotation: int = 0
    corrupt_image: int = 0
    corrupt_annotation: int = 0
    hidden_image_only: int = 0


class DatasetScanResult(BaseModel):
    dataset_id: str
    root_dir: Path
    image_root: Path
    annotation_roots: list[Path]
    items: list[DatasetAsset]
    summary: DatasetScanSummary


class RegisteredDataset(BaseModel):
    dataset_id: str
    name: str
    root_dir: Path
    image_root: Path
    summary: DatasetScanSummary
    created_at: str
    updated_at: str
    index_revision: int = 1
    source_available: bool = True
    source_error: Literal["root_missing", "image_root_missing"] | None = None


class DatasetListResponse(BaseModel):
    datasets: list[RegisteredDataset]


class AssetListResponse(BaseModel):
    items: list[DatasetAsset]
    total: int
    next_offset: int | None = None


class AssetCursorPage(BaseModel):
    items: list[DatasetAsset]
    total: int
    next_cursor: str | None = None
    index_revision: int


ScanSessionState = Literal["queued", "running", "succeeded", "failed", "interrupted"]


class DatasetScanSession(BaseModel):
    session_id: str
    state: ScanSessionState
    request: DatasetScanRequest
    dataset_id: str | None = None
    root_dir: Path | None = None
    image_root: Path | None = None
    annotation_roots: list[Path] = Field(default_factory=list)
    summary: DatasetScanSummary | None = None
    persisted_items: int = 0
    run_generation: int = 0
    error: str | None = None
    registration_name: str | None = None
    registered_dataset_id: str | None = None
    registered_index_revision: int | None = None
    registered_items: int = 0
    registered_at: str | None = None
    interrupted_at: str | None = None
    interruption_reason: str | None = None
    created_at: str
    updated_at: str


class DatasetScanSessionList(BaseModel):
    sessions: list[DatasetScanSession]


class DatasetScanItemPage(BaseModel):
    items: list[DatasetAsset]
    total: int
    next_after: int | None = None
    state: ScanSessionState
