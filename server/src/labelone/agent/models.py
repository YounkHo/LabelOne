from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AgentToolName = Literal[
    "dataset.stats",
    "dataset.search",
    "annotation.qa",
    "dataset.distribution",
    "ui.open_dataset",
    "ui.import_operator",
    "ui.open_models",
    "pipeline.draft",
    "pipeline.create_job",
    "inference.create_job",
]

AgentCapabilityGroup = Literal["inspect", "prepare", "run"]
AgentBackendReason = Literal["ready", "disabled", "missing_credential", "invalid_configuration"]


class AgentToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: AgentToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    asset_id: str | None = None
    message: str = Field(min_length=1, max_length=4_000)
    tool_call: AgentToolCall | None = None


class AgentToolResult(BaseModel):
    tool: AgentToolName
    data: dict[str, Any] = Field(default_factory=dict)


class AgentCapability(BaseModel):
    tool: AgentToolName
    group: AgentCapabilityGroup
    title: str
    description: str
    risk: Literal["read", "write"]
    requires_confirmation: bool
    requires_dataset: bool = True
    requires_asset: bool = False


class AgentStatus(BaseModel):
    state: Literal["ready", "unconfigured"]
    reason_code: AgentBackendReason
    message: str
    provider: Literal["openai_compatible"] = "openai_compatible"
    model: str | None = None
    credential_env: str | None = None
    capabilities: list[AgentCapability] = Field(default_factory=list)


class AgentAuditRecord(BaseModel):
    audit_id: int
    run_id: str
    tool: AgentToolName
    risk: Literal["read", "write"]
    status: Literal["completed", "proposed", "executed", "failed", "idempotent"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    created_at: str


class AgentProposal(BaseModel):
    id: str
    tool: str
    title: str
    description: str
    risk: Literal["read", "write"]
    requires_confirmation: bool
    executed: bool = False
    result: dict[str, Any] | None = None


class AgentRun(BaseModel):
    run_id: str
    dataset_id: str
    asset_id: str | None = None
    message: str
    reply: str
    state: Literal["proposed", "completed", "failed"]
    proposals: list[AgentProposal] = Field(default_factory=list)
    tool_results: list[AgentToolResult] = Field(default_factory=list)
    created_at: str
    updated_at: str
