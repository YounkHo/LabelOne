from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictWorkspaceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspacePipelineNode(StrictWorkspaceModel):
    id: str = Field(min_length=1, max_length=160)
    kind: str = Field(min_length=1, max_length=160)
    name: str = Field(min_length=1, max_length=240)
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)
    operator_version: str | None = Field(default=None, max_length=160)


class WorkspaceVisualizationNode(WorkspacePipelineNode):
    kind: Literal["visualize"] = "visualize"
    tap_after_node_id: str = Field(min_length=1, max_length=160)


class WorkspacePipelineLayerState(StrictWorkspaceModel):
    visible: bool = True
    opacity: int = Field(default=100, ge=0, le=100)


class WorkspacePipelineSettings(StrictWorkspaceModel):
    enabled: bool = True
    scope: Literal["current", "all"] = "current"
    nodes: list[WorkspacePipelineNode] = Field(default_factory=list, max_length=64)
    visualizations: list[WorkspaceVisualizationNode] = Field(default_factory=list, max_length=4)
    display_mode: Literal["source", "split", "overlay"] = "split"
    single_source: str = Field(default="source", min_length=1, max_length=160)
    layer_state: dict[str, WorkspacePipelineLayerState] = Field(default_factory=dict)

    @field_validator("layer_state")
    @classmethod
    def bounded_layer_state(
        cls,
        value: dict[str, WorkspacePipelineLayerState],
    ) -> dict[str, WorkspacePipelineLayerState]:
        if len(value) > 4 or any(not key or len(key) > 160 for key in value):
            raise ValueError("pipeline layer state must contain at most four bounded ids")
        return value

    @model_validator(mode="after")
    def bounded_payload(self) -> "WorkspacePipelineSettings":
        if len(json.dumps(self.model_dump(mode="json"), ensure_ascii=False)) > 256 * 1024:
            raise ValueError("pipeline settings exceed the 256 KiB limit")
        return self


class WorkspaceInferenceSettings(StrictWorkspaceModel):
    model_id: str | None = Field(default=None, max_length=512)
    provider: str = Field(default="CPUExecutionProvider", min_length=1, max_length=160)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def bounded_parameters(self) -> "WorkspaceInferenceSettings":
        if len(json.dumps(self.parameters, ensure_ascii=False)) > 64 * 1024:
            raise ValueError("inference settings exceed the 64 KiB limit")
        return self


class GlobalWorkspaceSettings(StrictWorkspaceModel):
    schema_version: Literal[1] = 1
    pipeline: WorkspacePipelineSettings | None = None
    inference: WorkspaceInferenceSettings = Field(default_factory=WorkspaceInferenceSettings)


class DatasetWorkspaceSettings(StrictWorkspaceModel):
    schema_version: Literal[1] = 1
    last_asset_id: str | None = Field(default=None, max_length=512)
    pipeline: WorkspacePipelineSettings | None = None


class DatasetWorkspaceSettingsResponse(DatasetWorkspaceSettings):
    revision: int = Field(default=0, ge=0)
    updated_at: str | None = None


class DatasetWorkspaceSettingsUpdate(DatasetWorkspaceSettings):
    expected_revision: int = Field(ge=0)


class ModelUsageRecord(StrictWorkspaceModel):
    count: int = Field(default=0, ge=0)
    last_used_at: str | None = None
