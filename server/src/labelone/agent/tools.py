from __future__ import annotations

from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from labelone.pipelines import PipelineNode

from .models import AgentToolCall


class _ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetStatsArguments(_ToolArguments):
    pass


class DatasetSearchArguments(_ToolArguments):
    query: str = Field(default="", max_length=1024)
    mode: Literal["smart", "text", "regex", "condition"] = "smart"
    limit: int = Field(default=20, ge=1, le=100)
    status: str | None = Field(default=None, max_length=64)
    annotated: bool | None = None


class AnnotationQaArguments(_ToolArguments):
    duplicate_precision: int = Field(default=3, ge=0, le=8)


class DatasetDistributionArguments(_ToolArguments):
    max_assets: int = Field(default=10_000, ge=1, le=10_000)
    top_n: int = Field(default=30, ge=1, le=50)


class PipelineJobArguments(_ToolArguments):
    scope: Literal["current", "all"] = "current"
    nodes: list[PipelineNode] = Field(min_length=1, max_length=32)
    concurrency: int = Field(default=2, ge=1, le=4)


class PipelineDraftArguments(_ToolArguments):
    nodes: list[PipelineNode] = Field(min_length=1, max_length=32)


class UiActionArguments(_ToolArguments):
    pass


class InferenceJobArguments(_ToolArguments):
    model_id: str = Field(min_length=1, max_length=256)
    scope: Literal["current", "all"] = "current"
    capture_layers: list[str] = Field(default_factory=list, max_length=8)
    parameters: dict[str, object] = Field(default_factory=dict)


TOOL_ARGUMENT_MODELS = {
    "dataset.stats": DatasetStatsArguments,
    "dataset.search": DatasetSearchArguments,
    "annotation.qa": AnnotationQaArguments,
    "dataset.distribution": DatasetDistributionArguments,
    "ui.open_dataset": UiActionArguments,
    "ui.import_operator": UiActionArguments,
    "ui.open_models": UiActionArguments,
    "pipeline.draft": PipelineDraftArguments,
    "pipeline.create_job": PipelineJobArguments,
    "inference.create_job": InferenceJobArguments,
}


def parse_tool_arguments(call: AgentToolCall) -> _ToolArguments:
    model = TOOL_ARGUMENT_MODELS[call.tool]
    return model.model_validate(call.arguments)


def legacy_tool_call(message: str, *, has_asset: bool) -> AgentToolCall | None:
    normalized = " ".join(message.strip().casefold().split())
    if normalized in {"定位所有异常文件", "检查数据集异常", "dataset stats", "dataset statistics"}:
        return AgentToolCall(tool="dataset.stats")
    if normalized in {"解释当前标注", "检查当前标注", "annotation qa", "qa current annotation"} and has_asset:
        return AgentToolCall(tool="annotation.qa")
    if normalized in {"给当前图设计增强流程", "为当前图设计增强流程", "为当前图创建增强任务"} and has_asset:
        return AgentToolCall(tool="pipeline.create_job", arguments={
            "scope": "current",
            "nodes": [
                {"id": "agent-crop", "kind": "crop", "parameters": {"margin_ratio": 0.05}},
                {"id": "agent-color", "kind": "color", "parameters": {"brightness": 1.05, "contrast": 1.1}},
            ],
        })
    if normalized in {"为全部图像创建增强任务", "给全部图像设计增强流程"}:
        return AgentToolCall(tool="pipeline.create_job", arguments={
            "scope": "all",
            "nodes": [
                {"id": "agent-crop", "kind": "crop", "parameters": {"margin_ratio": 0.05}},
                {"id": "agent-color", "kind": "color", "parameters": {"brightness": 1.05, "contrast": 1.1}},
            ],
        })
    return None


def validate_json_budget(value: object, *, depth: int = 0) -> None:
    if depth > 4:
        raise ValueError("Tool parameters exceed the nesting budget")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Tool parameter number must be finite")
        return
    if isinstance(value, str):
        if len(value) > 2048:
            raise ValueError("Tool parameter string is too long")
        return
    if isinstance(value, list):
        if len(value) > 128:
            raise ValueError("Tool parameter list is too large")
        for item in value:
            validate_json_budget(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 64:
            raise ValueError("Tool parameter object is too large")
        forbidden = {"path", "url", "shell", "command", "code", "script", "exec", "eval"}
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise ValueError("Tool parameter key is invalid")
            if key.casefold() in forbidden:
                raise ValueError(f"Tool parameter is forbidden: {key}")
            validate_json_budget(item, depth=depth + 1)
        return
    raise ValueError("Tool parameters must contain only JSON values")
