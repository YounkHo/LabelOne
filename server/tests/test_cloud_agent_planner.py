from __future__ import annotations

import json
from pathlib import Path

import pytest

from labelone.agent.planner import CloudAgentPlanner
from labelone.application_settings import ApplicationSettingsStore
from labelone.errors import InvalidPathError


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self, maximum: int) -> bytes:
        return self.payload[:maximum]

    def close(self) -> None:
        return None


def _configured_store(tmp_path: Path) -> ApplicationSettingsStore:
    store = ApplicationSettingsStore(tmp_path / "settings.json")
    store.set_cloud_ai(
        enabled=True,
        provider="openai_compatible",
        endpoint="https://llm.example.test/v1/chat/completions",
        model="planner-model",
        api_key_env="LABELONE_TEST_CLOUD_KEY",
        timeout_seconds=20,
        max_output_tokens=600,
    )
    return store


def test_cloud_planner_returns_only_validated_allowlisted_tool_call(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("LABELONE_TEST_CLOUD_KEY", "secret-token")

    def open_request(request, *, timeout: int):
        captured["authorization"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data)
        return _Response({"choices": [{"message": {"content": json.dumps({
            "reply": "我会先查找带有 scratch 的图片。",
            "tool": "dataset.search",
            "arguments": {"query": "scratch", "limit": 20},
        })}}]})

    planner = CloudAgentPlanner(_configured_store(tmp_path), opener=open_request)
    plan = planner.plan(
        "找出 scratch 图片",
        has_asset=True,
        history=[("user", "你好"), ("assistant", "你好，我可以帮助查看数据集。")],
        operator_kinds=["crop", "resize", "color"],
    )

    assert plan.reply == "我会先查找带有 scratch 的图片。"
    assert plan.tool_call is not None
    assert plan.tool_call.tool == "dataset.search"
    assert plan.tool_call.arguments == {"query": "scratch", "limit": 20}
    assert captured["authorization"] == "Bearer secret-token"
    assert captured["timeout"] == 20
    assert "secret-token" not in json.dumps(captured["body"])
    messages = captured["body"]["messages"]
    assert messages[-3:] == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我可以帮助查看数据集。"},
        {"role": "user", "content": "找出 scratch 图片"},
    ]
    assert "crop, resize, color" in messages[0]["content"]
    assert "Never read, write, patch" in messages[0]["content"]


def test_cloud_planner_can_return_conversation_without_a_tool(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LABELONE_TEST_CLOUD_KEY", "secret-token")
    planner = CloudAgentPlanner(_configured_store(tmp_path), opener=lambda *_args, **_kwargs: _Response({
        "choices": [{"message": {"content": '{"reply":"可以，我们先确认目标。","tool":null,"arguments":{}}'}}],
    }))

    plan = planner.plan("你好", has_asset=False)

    assert plan.reply == "可以，我们先确认目标。"
    assert plan.tool_call is None


def test_cloud_planner_requires_environment_credential_and_rejects_unknown_tools(tmp_path: Path, monkeypatch) -> None:
    store = _configured_store(tmp_path)
    monkeypatch.delenv("LABELONE_TEST_CLOUD_KEY", raising=False)
    planner = CloudAgentPlanner(store, opener=lambda *_args, **_kwargs: None)
    with pytest.raises(InvalidPathError, match="凭据"):
        planner.plan("统计数据集", has_asset=False)

    monkeypatch.setenv("LABELONE_TEST_CLOUD_KEY", "secret-token")
    planner = CloudAgentPlanner(store, opener=lambda *_args, **_kwargs: _Response({
        "choices": [{"message": {"content": '{"reply":"不允许执行 shell。","tool":"shell.exec","arguments":{}}'}}],
    }))
    with pytest.raises(Exception):
        planner.plan("运行 shell", has_asset=False)


def test_cloud_planner_readiness_requires_enabled_config_and_nonblank_credential(tmp_path: Path, monkeypatch) -> None:
    disabled = CloudAgentPlanner(ApplicationSettingsStore(tmp_path / "disabled.json"))
    assert disabled.readiness().reason_code == "disabled"
    assert disabled.enabled() is False

    store = _configured_store(tmp_path)
    planner = CloudAgentPlanner(store)
    monkeypatch.setenv("LABELONE_TEST_CLOUD_KEY", "   ")
    assert planner.readiness().reason_code == "missing_credential"
    assert planner.enabled() is False

    monkeypatch.setenv("LABELONE_TEST_CLOUD_KEY", "secret-token")
    assert planner.readiness().reason_code == "ready"
    assert planner.readiness().model == "planner-model"
    assert planner.enabled() is True
