from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from threading import RLock
from urllib.parse import urlsplit

from labelone.errors import InvalidPathError
from labelone.workspace_settings import GlobalWorkspaceSettings, ModelUsageRecord


MODEL_DOWNLOAD_SOURCES = (
    {"id": "auto", "label": "自动选择"},
    {"id": "github", "label": "GitHub"},
    {"id": "modelscope", "label": "ModelScope"},
    {"id": "huggingface", "label": "Hugging Face"},
)
_MODEL_DOWNLOAD_SOURCE_IDS = frozenset(item["id"] for item in MODEL_DOWNLOAD_SOURCES)
_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
_NO_PROXY_KEYS = ("NO_PROXY", "no_proxy")
_DEFAULT_PROXY_BYPASS = "localhost,127.0.0.1,::1"


def apply_network_proxy_environment(config: dict[str, object]) -> None:
    """Apply persisted proxy settings once, before outbound clients are created."""
    mode = str(config.get("mode") or "system")
    if mode == "system":
        return
    for key in _PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    if mode == "manual":
        proxy_url = str(config.get("url") or "")
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[key] = proxy_url
        bypass = str(config.get("bypass") or _DEFAULT_PROXY_BYPASS)
    else:
        bypass = "*"
    for key in _NO_PROXY_KEYS:
        os.environ[key] = bypass


class ApplicationSettingsStore:
    """Crash-safe persisted application settings owned by the local service."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _read(self) -> dict[str, object]:
        with self._lock:
            if not self.path.is_file():
                return {}
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return {}
        return payload if isinstance(payload, dict) else {}

    def _write(self, payload: dict[str, object]) -> None:
        encoded = json.dumps({"schema_version": 1, **payload}, ensure_ascii=False, indent=2)
        with self._lock:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    def model_weights_dir(self) -> Path | None:
        raw = self._read().get("model_weights_dir")
        if not isinstance(raw, str) or not raw:
            return None
        candidate = Path(raw).expanduser()
        return candidate.resolve() if candidate.is_absolute() else None

    def set_model_weights_dir(self, root: Path) -> Path:
        candidate = root.expanduser()
        if not candidate.is_absolute():
            raise InvalidPathError(
                "Model download directory must be an absolute path",
                details={"path": str(root)},
            )
        resolved = candidate.resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise InvalidPathError(
                "Model download directory must be an existing directory",
                details={"path": str(resolved)},
            )
        if not os.access(resolved, os.W_OK | os.X_OK):
            raise InvalidPathError(
                "Model download directory must be writable",
                details={"path": str(resolved)},
            )
        with self._lock:
            payload = self._read()
            payload["model_weights_dir"] = str(resolved)
            self._write(payload)
        return resolved

    def model_download_source(self) -> str:
        source = self._read().get("model_download_source")
        return str(source) if source in _MODEL_DOWNLOAD_SOURCE_IDS else "auto"

    def set_model_download_source(self, source: str) -> str:
        normalized = source.strip().casefold()
        if normalized not in _MODEL_DOWNLOAD_SOURCE_IDS:
            raise InvalidPathError("Model download source is not supported", details={"source": source})
        with self._lock:
            payload = self._read()
            payload["model_download_source"] = normalized
            self._write(payload)
        return normalized

    def network_proxy(self) -> dict[str, object]:
        raw = self._read().get("network_proxy")
        source = raw if isinstance(raw, dict) else {}
        mode = source.get("mode") if source.get("mode") in {"system", "direct", "manual"} else "system"
        return {
            "mode": mode,
            "url": str(source.get("url") or "") if mode == "manual" else "",
            "bypass": str(source.get("bypass") or _DEFAULT_PROXY_BYPASS),
        }

    def set_network_proxy(self, *, mode: str, url: str, bypass: str) -> dict[str, object]:
        normalized_mode = mode.strip().casefold()
        if normalized_mode not in {"system", "direct", "manual"}:
            raise InvalidPathError("Network proxy mode is not supported", details={"mode": mode})
        normalized_url = url.strip().rstrip("/") if normalized_mode == "manual" else ""
        if normalized_mode == "manual":
            parsed = urlsplit(normalized_url)
            if (
                parsed.scheme.casefold() not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise InvalidPathError("Proxy URL must be a credential-free HTTP(S) endpoint without a path")
            try:
                port = parsed.port
            except ValueError as exc:
                raise InvalidPathError("Proxy URL port is invalid") from exc
            if port is not None and not 1 <= port <= 65535:
                raise InvalidPathError("Proxy URL port is invalid")
        normalized_bypass = ",".join(part.strip() for part in bypass.split(",") if part.strip())
        if len(normalized_bypass) > 2048 or any(not re.fullmatch(r"[A-Za-z0-9.*:_-]+", part) for part in normalized_bypass.split(",") if part):
            raise InvalidPathError("Proxy bypass list contains an invalid hostname or address")
        network_proxy = {
            "mode": normalized_mode,
            "url": normalized_url,
            "bypass": normalized_bypass or _DEFAULT_PROXY_BYPASS,
        }
        with self._lock:
            payload = self._read()
            payload["network_proxy"] = network_proxy
            self._write(payload)
        return network_proxy

    def cloud_ai(self) -> dict[str, object]:
        raw = self._read().get("cloud_ai")
        source = raw if isinstance(raw, dict) else {}
        def integer(name: str, default: int, minimum: int, maximum: int) -> int:
            value = source.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                return default
            return value
        return {
            "enabled": source.get("enabled") is True,
            "provider": "openai_compatible",
            "endpoint": str(source.get("endpoint") or ""),
            "model": str(source.get("model") or ""),
            "api_key_env": str(source.get("api_key_env") or "OPENAI_API_KEY"),
            "timeout_seconds": integer("timeout_seconds", 30, 5, 120),
            "max_output_tokens": integer("max_output_tokens", 800, 128, 4096),
        }

    def set_cloud_ai(
        self,
        *,
        enabled: bool,
        provider: str,
        endpoint: str,
        model: str,
        api_key_env: str,
        timeout_seconds: int,
        max_output_tokens: int,
    ) -> dict[str, object]:
        if provider != "openai_compatible":
            raise InvalidPathError("Cloud AI provider is not supported", details={"provider": provider})
        normalized_endpoint = endpoint.strip().rstrip("/")
        parsed = urlsplit(normalized_endpoint) if normalized_endpoint else None
        if enabled and (
            parsed is None
            or parsed.scheme.casefold() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise InvalidPathError("Cloud AI endpoint must be a credential-free HTTPS URL")
        normalized_model = model.strip()
        if enabled and not 1 <= len(normalized_model) <= 200:
            raise InvalidPathError("Cloud AI model id is required and must be bounded")
        normalized_key_env = api_key_env.strip()
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]{0,127}", normalized_key_env):
            raise InvalidPathError("Cloud AI API key environment variable name is invalid")
        if not 5 <= timeout_seconds <= 120:
            raise InvalidPathError("Cloud AI timeout must be between 5 and 120 seconds")
        if not 128 <= max_output_tokens <= 4096:
            raise InvalidPathError("Cloud AI output token limit must be between 128 and 4096")
        cloud_ai = {
            "enabled": bool(enabled),
            "provider": provider,
            "endpoint": normalized_endpoint,
            "model": normalized_model,
            "api_key_env": normalized_key_env,
            "timeout_seconds": int(timeout_seconds),
            "max_output_tokens": int(max_output_tokens),
        }
        with self._lock:
            payload = self._read()
            payload["cloud_ai"] = cloud_ai
            self._write(payload)
        return cloud_ai

    def workspace(self) -> GlobalWorkspaceSettings:
        raw = self._read().get("workspace")
        try:
            return GlobalWorkspaceSettings.model_validate(raw if isinstance(raw, dict) else {})
        except ValueError:
            return GlobalWorkspaceSettings()

    def set_workspace(self, settings: GlobalWorkspaceSettings) -> GlobalWorkspaceSettings:
        with self._lock:
            payload = self._read()
            payload["workspace"] = settings.model_dump(mode="json")
            self._write(payload)
        return settings

    def model_usage(self) -> dict[str, ModelUsageRecord]:
        raw = self._read().get("model_usage")
        if not isinstance(raw, dict):
            return {}
        usage: dict[str, ModelUsageRecord] = {}
        for model_id, record in raw.items():
            if not isinstance(model_id, str) or not model_id or len(model_id) > 512:
                continue
            try:
                usage[model_id] = ModelUsageRecord.model_validate(record)
            except ValueError:
                continue
        return usage

    def record_model_usage(self, model_id: str) -> ModelUsageRecord:
        normalized = model_id.strip()
        if not normalized or len(normalized) > 512:
            raise InvalidPathError("Model id is required and must be bounded")
        with self._lock:
            payload = self._read()
            raw_usage = payload.get("model_usage")
            usage = raw_usage if isinstance(raw_usage, dict) else {}
            try:
                current = ModelUsageRecord.model_validate(usage.get(normalized, {}))
            except ValueError:
                current = ModelUsageRecord()
            updated = ModelUsageRecord(
                count=min(current.count + 1, 9_223_372_036_854_775_807),
                last_used_at=datetime.now(timezone.utc).isoformat(),
            )
            usage[normalized] = updated.model_dump(mode="json")
            payload["model_usage"] = usage
            self._write(payload)
        return updated
