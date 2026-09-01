from __future__ import annotations

from typing import Any, Literal
from unicodedata import normalize

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from labelone.pipelines.models import PipelineNode, PipelineOutputPolicy


JobKind = Literal["pipeline", "inference", "model_download", "category_rename"]
JobState = Literal[
    "queued", "running", "pausing", "paused", "canceling", "canceled",
    "succeeded", "succeeded_with_errors", "failed", "interrupted",
]
ItemState = Literal["queued", "running", "succeeded", "failed", "canceled"]
JobEventType = Literal[
    "job.created",
    "job.state",
    "item.state",
    "job.progress",
    "item.progress",
    "job.recovered",
    "job.terminal",
]
TERMINAL_JOB_STATES = frozenset({"succeeded", "succeeded_with_errors", "failed", "canceled"})


def is_terminal_job_state(state: str) -> bool:
    return state in TERMINAL_JOB_STATES


class PipelinePrecomputeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_index_revision: int = Field(ge=1)
    registry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_format: Literal["png", "webp", "jpeg"]


class BatchJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: JobKind
    dataset_id: str = ""
    asset_ids: list[str] = Field(default_factory=list)
    preferred_asset_ids: list[str] = Field(default_factory=list, max_length=64)
    concurrency: int = Field(default=2, ge=1, le=4)
    priority: Literal["user_batch", "background"] = "user_batch"
    pipeline_nodes: list[PipelineNode] = Field(default_factory=list)
    output_policy: PipelineOutputPolicy = Field(default_factory=PipelineOutputPolicy)
    model_id: str | None = None
    capture_layers: list[str] = Field(default_factory=list, max_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    weight_url_indices: list[StrictInt] = Field(default_factory=list)
    expected_sha256: dict[StrictInt, str] = Field(default_factory=dict)
    source_category: str | None = Field(default=None, min_length=1, max_length=128)
    target_category: str | None = Field(default=None, min_length=1, max_length=128)
    pipeline_context: PipelinePrecomputeContext | None = None

    @field_validator("asset_ids")
    @classmethod
    def validate_asset_ids(cls, value: list[str]) -> list[str]:
        if any(not asset_id for asset_id in value) or len(set(value)) != len(value):
            raise ValueError("asset_ids must contain unique non-empty values")
        return value

    @field_validator("preferred_asset_ids")
    @classmethod
    def validate_preferred_asset_ids(cls, value: list[str]) -> list[str]:
        if any(not asset_id for asset_id in value) or len(set(value)) != len(value):
            raise ValueError("preferred_asset_ids must contain unique non-empty values")
        return value

    @field_validator("capture_layers")
    @classmethod
    def validate_capture_layers(cls, value: list[str]) -> list[str]:
        if any(not layer or len(layer) > 1024 for layer in value) or len(set(value)) != len(value):
            raise ValueError("capture_layers must contain one unique bounded layer id")
        return value

    @field_validator("source_category", "target_category")
    @classmethod
    def normalize_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize("NFC", value.strip())
        if not normalized:
            raise ValueError("Category names must be non-empty")
        return normalized

    @model_validator(mode="after")
    def validate_kind(self) -> "BatchJobRequest":
        if self.kind == "pipeline" and not self.pipeline_nodes:
            raise ValueError("Pipeline jobs require pipeline_nodes")
        if self.kind == "pipeline":
            has_tile = any(node.enabled and node.kind == "tile" for node in self.pipeline_nodes)
            if self.output_policy.mode == "derived_dataset":
                if self.output_policy.output_root is None or not self.output_policy.output_root.is_absolute():
                    raise ValueError("Derived dataset output_root must be an explicit absolute path")
            elif self.output_policy.output_root is not None:
                raise ValueError("Preview output policy cannot specify output_root")
            if has_tile and self.output_policy.mode != "derived_dataset":
                raise ValueError("Tile pipeline jobs require derived_dataset output policy")
        elif self.pipeline_context is not None:
            raise ValueError("Only pipeline jobs may contain pipeline_context")
        if self.pipeline_context is not None and self.output_policy.mode != "preview":
            raise ValueError("Only preview pipeline jobs may contain pipeline_context")
        if self.kind == "inference" and not self.model_id:
            raise ValueError("Inference jobs require model_id")
        if self.kind == "inference" and self.capture_layers and (not self.asset_ids or len(self.asset_ids) > 16):
            raise ValueError("Batch feature capture requires an explicit scope of at most 16 assets")
        if self.kind in {"pipeline", "inference", "category_rename"} and not self.dataset_id:
            raise ValueError("Dataset jobs require dataset_id")
        if self.asset_ids and set(self.preferred_asset_ids) - set(self.asset_ids):
            raise ValueError("preferred_asset_ids must be contained in asset_ids")
        if self.kind != "model_download" and (self.weight_url_indices or self.expected_sha256):
            raise ValueError("Only model_download jobs may specify weight URLs or hashes")
        if self.kind != "pipeline" and self.output_policy != PipelineOutputPolicy():
            raise ValueError("Only pipeline jobs may configure output_policy")
        if self.kind == "category_rename":
            if self.source_category is None or self.target_category is None:
                raise ValueError("Category rename jobs require source_category and target_category")
            if self.source_category == self.target_category:
                raise ValueError("Category rename source and target must differ")
            if self.model_id is not None or self.capture_layers or self.parameters or self.pipeline_nodes:
                raise ValueError("Category rename jobs cannot include model or pipeline fields")
            if self.asset_ids or self.preferred_asset_ids:
                raise ValueError("Category rename jobs always target the complete dataset category scope")
        elif self.source_category is not None or self.target_category is not None:
            raise ValueError("Only category_rename jobs may specify category names")
        if self.kind == "model_download":
            if not self.model_id:
                raise ValueError("Model download jobs require model_id")
            if self.dataset_id not in {"", "__models__"}:
                raise ValueError("Model download jobs cannot target a dataset")
            self.dataset_id = "__models__"
            if not self.weight_url_indices or len(set(self.weight_url_indices)) != len(self.weight_url_indices):
                raise ValueError("Model download jobs require unique weight_url_indices")
            if any(index < 0 for index in self.weight_url_indices):
                raise ValueError("Model weight URL indices must be non-negative")
            if not 1 <= self.concurrency <= 4:
                raise ValueError("Model download concurrency must be between 1 and 4")
            if self.pipeline_nodes or self.asset_ids or self.preferred_asset_ids or self.capture_layers:
                raise ValueError("Model download jobs cannot include dataset work fields")
            if set(self.expected_sha256) - set(self.weight_url_indices):
                raise ValueError("Expected hashes must refer to requested weight URL indices")
            for checksum in self.expected_sha256.values():
                if len(checksum) != 64 or any(character not in "0123456789abcdefABCDEF" for character in checksum):
                    raise ValueError("Expected model weight SHA-256 is invalid")
        return self


class JobItem(BaseModel):
    asset_id: str
    position: int
    state: ItemState
    attempts: int
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict[str, Any] | None = None


class JobItemLookupRequest(BaseModel):
    asset_ids: list[str] = Field(min_length=1, max_length=200)

    @field_validator("asset_ids")
    @classmethod
    def validate_asset_ids(cls, value: list[str]) -> list[str]:
        if any(not asset_id for asset_id in value) or len(set(value)) != len(value):
            raise ValueError("asset_ids must contain unique non-empty values")
        return value


class JobRecord(BaseModel):
    job_id: str
    kind: JobKind
    dataset_id: str
    state: JobState
    desired_state: Literal["run", "pause", "cancel"] = "run"
    generation: int = 0
    request: BatchJobRequest
    total: int
    completed: int
    failed: int
    canceled: int
    created_at: str
    updated_at: str
    error: str | None = None
    items: list[JobItem] = Field(default_factory=list)


class JobListResponse(BaseModel):
    jobs: list[JobRecord]


class PipelinePrecomputeEnsureResponse(BaseModel):
    job: JobRecord
    reused: bool
    resumed: bool = False
    canceled_job_ids: list[str] = Field(default_factory=list)


class JobItemListResponse(BaseModel):
    items: list[JobItem]
    total: int
    next_offset: int | None = None


class JobPriorityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_ids: list[str] = Field(min_length=1, max_length=64)

    @field_validator("asset_ids")
    @classmethod
    def validate_priority_asset_ids(cls, value: list[str]) -> list[str]:
        if any(not asset_id for asset_id in value) or len(set(value)) != len(value):
            raise ValueError("asset_ids must contain unique non-empty values")
        return value


class JobEvent(BaseModel):
    event_id: int
    job_id: str
    event_type: JobEventType
    created_at: str
    payload: dict[str, Any] = Field(default_factory=dict)
