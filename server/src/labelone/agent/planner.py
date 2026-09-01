from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from labelone.application_settings import ApplicationSettingsStore
from labelone.errors import InvalidPathError

from .models import AgentBackendReason, AgentToolCall


@dataclass(frozen=True, slots=True)
class AgentPlan:
    reply: str
    tool_call: AgentToolCall | None = None


@dataclass(frozen=True, slots=True)
class CloudAgentReadiness:
    ready: bool
    reason_code: AgentBackendReason
    message: str
    model: str | None = None
    credential_env: str | None = None


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class CloudAgentPlanner:
    """OpenAI-compatible planner that can only emit an allowlisted AgentToolCall."""

    def __init__(
        self,
        settings: ApplicationSettingsStore,
        *,
        opener: Callable[..., object] | None = None,
    ) -> None:
        self.settings = settings
        self._open = opener or build_opener(_NoRedirects()).open

    def enabled(self) -> bool:
        return self.readiness().ready

    def readiness(self) -> CloudAgentReadiness:
        config = self.settings.cloud_ai()
        model = str(config.get("model") or "").strip() or None
        key_env = str(config.get("api_key_env") or "").strip() or None
        if config.get("enabled") is not True:
            return CloudAgentReadiness(
                ready=False,
                reason_code="disabled",
                message="尚未配置 Agent 后端。请先在 AI 服务中启用工具规划模型。",
                model=model,
                credential_env=key_env,
            )
        endpoint = str(config.get("endpoint") or "").strip()
        if not endpoint or not model or not key_env:
            return CloudAgentReadiness(
                ready=False,
                reason_code="invalid_configuration",
                message="Agent 后端配置不完整，请检查服务地址、模型和凭据环境变量。",
                model=model,
                credential_env=key_env,
            )
        credential = os.getenv(key_env)
        if not credential or not credential.strip():
            return CloudAgentReadiness(
                ready=False,
                reason_code="missing_credential",
                message=f"Agent 后端缺少凭据。请在本地服务环境中配置 {key_env}。",
                model=model,
                credential_env=key_env,
            )
        return CloudAgentReadiness(
            ready=True,
            reason_code="ready",
            message="Agent 后端已配置，可以规划并执行受控工具。",
            model=model,
            credential_env=key_env,
        )

    def plan(
        self,
        message: str,
        *,
        has_asset: bool,
        history: list[tuple[str, str]] | None = None,
        operator_kinds: list[str] | None = None,
    ) -> AgentPlan:
        config = self.settings.cloud_ai()
        readiness = self.readiness()
        if not readiness.ready:
            raise InvalidPathError(readiness.message, details={"reason_code": readiness.reason_code})
        endpoint = str(config["endpoint"])
        model = str(config["model"])
        key_env = str(config["api_key_env"])
        credential = os.getenv(key_env)
        assert credential is not None
        tools = [
            "dataset.stats {}",
            "dataset.search {query, mode?, limit?, status?, annotated?}",
            "annotation.qa {duplicate_precision?}" if has_asset else "annotation.qa unavailable without current image",
            "dataset.distribution {top_n?, max_assets?}",
            "ui.open_dataset {}",
            "ui.import_operator {}",
            "ui.open_models {}",
            f"pipeline.draft {{nodes; allowed transform kinds: {', '.join((operator_kinds or [])[:64])}}}",
            "pipeline.create_job {scope, concurrency, nodes}",
            "inference.create_job {scope, model_id, capture_layers?, parameters?}",
        ]
        system = (
            "You are a controlled workflow assistant inside a local image annotation application, not a general chatbot. "
            "Help only with dataset inspection, annotation QA, pipelines, models, and inference. "
            "Reply briefly in the user's language and optionally propose exactly one allowlisted app tool. "
            "Return one JSON object and no markdown using "
            '{"reply":"helpful conversational response","tool":"<allowed tool or null>","arguments":{}}. '
            "You are not a coding agent. Never read, write, patch, generate, or discuss application source code; "
            "never use a terminal, shell, Python execution, arbitrary filesystem paths, URLs, secrets, or undeclared tools. "
            "Opening/importing UI and applying a pipeline draft always require a user-confirmed proposal. "
            "Never claim an action already ran. If the request is outside these capabilities, state the boundary briefly. "
            "Pipeline drafts may only use the listed registered transform kinds supplied by the tool schema. "
            "Available tools: " + "; ".join(tools)
        )
        history_messages = [
            {"role": role, "content": content}
            for role, content in (history or [])[-12:]
            if role in {"user", "assistant"} and isinstance(content, str) and content
        ]
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                *history_messages,
                {"role": "user", "content": message},
            ],
            "temperature": 0,
            "max_tokens": int(config["max_output_tokens"]),
            "response_format": {"type": "json_object"},
        }, ensure_ascii=False).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {credential}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "LabelOne-Agent/1",
            },
        )
        try:
            response = self._open(request, timeout=int(config["timeout_seconds"]))
            payload_bytes = response.read(256 * 1024 + 1)
            response.close()
        except HTTPError as exc:
            raise InvalidPathError(
                "Cloud AI request was rejected",
                details={"status": exc.code},
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise InvalidPathError("Cloud AI request failed") from exc
        if len(payload_bytes) > 256 * 1024:
            raise InvalidPathError("Cloud AI response exceeded the size limit")
        try:
            response_payload = json.loads(payload_bytes)
            content = response_payload["choices"][0]["message"]["content"]
            planned = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise InvalidPathError("Cloud AI response did not contain a valid tool plan") from exc
        if not isinstance(planned, dict) or set(planned) - {"reply", "tool", "arguments"}:
            raise InvalidPathError("Cloud AI tool plan fields are invalid")
        reply = planned.get("reply")
        if not isinstance(reply, str) or not reply.strip() or len(reply) > 4_000:
            raise InvalidPathError("Cloud AI response did not contain a valid conversational reply")
        if planned.get("tool") is None:
            return AgentPlan(reply=reply.strip())
        return AgentPlan(
            reply=reply.strip(),
            tool_call=AgentToolCall.model_validate({
                "tool": planned.get("tool"),
                "arguments": planned.get("arguments", {}),
            }),
        )
