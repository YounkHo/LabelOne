from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from labelone.workspace_settings import GlobalWorkspaceSettings, ModelUsageRecord


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    service: str = "labelone-local"
    version: str
    api_version: Literal["v1"] = "v1"
    model_registry: dict[str, int]
    runtimes: dict[str, Literal["available", "unavailable"]]


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class DirectoryPickerRequest(BaseModel):
    title: str = Field(default="选择文件夹", min_length=1, max_length=160)
    initial_dir: Path | None = None


class DirectoryPickerResponse(BaseModel):
    path: Path | None = None
    canceled: bool = False


class CloudAiSettings(BaseModel):
    enabled: bool = False
    provider: Literal["openai_compatible"] = "openai_compatible"
    endpoint: str = Field(default="", max_length=2048)
    model: str = Field(default="", max_length=200)
    api_key_env: str = Field(default="OPENAI_API_KEY", min_length=1, max_length=128)
    timeout_seconds: int = Field(default=30, ge=5, le=120)
    max_output_tokens: int = Field(default=800, ge=128, le=4096)


class CloudAiSettingsResponse(CloudAiSettings):
    credential_configured: bool = False
    credential_source: Literal["environment", "missing"] = "missing"


class ModelDownloadSourceOption(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=100)


class ModelWeightDownloadRequest(BaseModel):
    url_indices: list[int] = Field(min_length=1, max_length=64)
    expected_sha256: dict[int, str] = Field(default_factory=dict)


class NetworkProxySettings(BaseModel):
    mode: Literal["system", "direct", "manual"] = "system"
    url: str = Field(default="", max_length=2048)
    bypass: str = Field(default="localhost,127.0.0.1,::1", max_length=2048)


class ApplicationSettingsResponse(BaseModel):
    data_dir: Path
    model_source_dir: Path | None = None
    model_weights_dir: Path
    effective_model_weights_dir: Path
    model_weights_managed_by: Literal["default", "persisted", "environment"]
    restart_required: bool = False
    model_download_concurrency: int = 1
    checksum_verification: bool = True
    model_download_source: str = "auto"
    model_download_sources: list[ModelDownloadSourceOption] = Field(default_factory=list)
    network_proxy: NetworkProxySettings = Field(default_factory=NetworkProxySettings)
    network_proxy_restart_required: bool = False
    cloud_ai: CloudAiSettingsResponse
    workspace: GlobalWorkspaceSettings = Field(default_factory=GlobalWorkspaceSettings)
    model_usage: dict[str, ModelUsageRecord] = Field(default_factory=dict)


class ApplicationSettingsUpdate(BaseModel):
    model_weights_dir: Path | None = None
    model_download_source: str | None = Field(default=None, min_length=1, max_length=64)
    network_proxy: NetworkProxySettings | None = None
    cloud_ai: CloudAiSettings | None = None
    workspace: GlobalWorkspaceSettings | None = None
