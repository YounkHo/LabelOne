from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
import asyncio
from threading import Event, Timer
from time import monotonic, sleep
import zipfile

from fastapi.testclient import TestClient
import httpx
import onnx
from onnx import TensorProto, helper
from PIL import Image

from labelone.config import Settings
from labelone.application_settings import apply_network_proxy_environment
from labelone.jobs.repository import JobRepository
from labelone.main import create_app
from labelone.models.weights import DownloadedWeight, ModelWeightStore


def _operator_zip(operator_id: str = "api.invert") -> bytes:
    manifest = f"""\
api_version: labelone.operator/v1
id: {operator_id}
name: API invert
description: Inverts every image channel for the API fixture.
version: 1.0.0
entrypoint: operator.py:process
size_behavior: preserve
spatial_behavior: none
annotation_policy: preserve
parameters_schema:
  type: object
  properties: {{}}
"""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("operator.yaml", manifest)
        archive.writestr(
            "operator.py",
            "import numpy as np\n"
            "def process(image, parameters):\n"
            "    return (255 - image).astype(np.uint8)\n",
        )
    return buffer.getvalue()


def test_system_directory_picker_api(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, Path | None]] = []

    def selected(title: str, initial_dir: Path | None) -> Path:
        calls.append((title, initial_dir))
        return tmp_path

    monkeypatch.setattr("labelone.main.pick_directory", selected)
    app = create_app(Settings(data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        response = client.post("/api/v1/system/pick-directory", json={
            "title": "选择数据集文件夹",
            "initial_dir": str(tmp_path),
        })

    assert response.status_code == 200
    assert response.json() == {"path": str(tmp_path), "canceled": False}
    assert calls == [("选择数据集文件夹", tmp_path)]


def test_application_settings_persist_model_weight_directory_for_restart(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    model_weights_dir = tmp_path / "downloaded-models"
    model_weights_dir.mkdir()
    app = create_app(Settings(data_dir=data_dir))

    with TestClient(app) as client:
        initial = client.get("/api/v1/settings")
        updated = client.patch("/api/v1/settings", json={"model_weights_dir": str(model_weights_dir)})

    assert initial.status_code == 200
    assert initial.json()["model_weights_dir"] == str(data_dir / "model-weights")
    assert initial.json()["restart_required"] is False
    assert updated.status_code == 200
    assert updated.json()["model_weights_dir"] == str(model_weights_dir)
    assert updated.json()["effective_model_weights_dir"] == str(data_dir / "model-weights")
    assert updated.json()["model_weights_managed_by"] == "persisted"
    assert updated.json()["restart_required"] is True

    restarted = create_app(Settings(data_dir=data_dir))
    with TestClient(restarted) as client:
        effective = client.get("/api/v1/settings")

    assert effective.status_code == 200
    assert effective.json()["effective_model_weights_dir"] == str(model_weights_dir)
    assert effective.json()["restart_required"] is False


def test_environment_managed_model_weight_directory_cannot_be_overridden(tmp_path: Path) -> None:
    environment_dir = tmp_path / "environment-models"
    replacement_dir = tmp_path / "replacement-models"
    environment_dir.mkdir()
    replacement_dir.mkdir()
    app = create_app(Settings(data_dir=tmp_path / "data", model_weights_dir=environment_dir))

    with TestClient(app) as client:
        current = client.get("/api/v1/settings")
        rejected = client.patch("/api/v1/settings", json={"model_weights_dir": str(replacement_dir)})

    assert current.json()["model_weights_managed_by"] == "environment"
    assert current.json()["effective_model_weights_dir"] == str(environment_dir)
    assert rejected.status_code == 409


def test_cloud_ai_settings_persist_without_storing_or_returning_credentials(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("LABELONE_TEST_CLOUD_KEY", "super-secret-value")
    app = create_app(Settings(data_dir=data_dir))
    cloud_ai = {
        "enabled": True,
        "provider": "openai_compatible",
        "endpoint": "https://llm.example.test/v1/chat/completions",
        "model": "planner-model",
        "api_key_env": "LABELONE_TEST_CLOUD_KEY",
        "timeout_seconds": 25,
        "max_output_tokens": 600,
    }

    with TestClient(app) as client:
        updated = client.patch("/api/v1/settings", json={"cloud_ai": cloud_ai})
        current = client.get("/api/v1/settings")

    assert updated.status_code == 200
    assert current.status_code == 200
    assert current.json()["cloud_ai"] == {
        **cloud_ai,
        "credential_configured": True,
        "credential_source": "environment",
    }
    serialized = (data_dir / "application-settings.json").read_text(encoding="utf-8")
    assert "super-secret-value" not in serialized
    assert '"api_key":' not in current.text.casefold()


def test_cloud_ai_settings_reject_insecure_endpoint(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        response = client.patch("/api/v1/settings", json={"cloud_ai": {
            "enabled": True,
            "provider": "openai_compatible",
            "endpoint": "http://llm.example.test/v1/chat/completions",
            "model": "planner-model",
            "api_key_env": "OPENAI_API_KEY",
            "timeout_seconds": 30,
            "max_output_tokens": 800,
        }})

    assert response.status_code == 400


def test_model_download_source_and_network_proxy_settings_are_persisted_safely(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    app = create_app(Settings(data_dir=data_dir))
    proxy = {
        "mode": "manual",
        "url": "http://127.0.0.1:7890",
        "bypass": "localhost, 127.0.0.1, ::1",
    }

    with TestClient(app) as client:
        initial = client.get("/api/v1/settings")
        updated = client.patch("/api/v1/settings", json={
            "model_download_source": "modelscope",
            "network_proxy": proxy,
        })
        rejected = client.patch("/api/v1/settings", json={
            "network_proxy": {**proxy, "url": "http://user:secret@127.0.0.1:7890"},
        })
        rejected_source = client.patch("/api/v1/settings", json={"model_download_source": "imaginary-mirror"})

    assert initial.status_code == 200
    assert [item["id"] for item in initial.json()["model_download_sources"]] == ["auto", "github", "modelscope", "huggingface"]
    assert updated.status_code == 200
    assert updated.json()["model_download_source"] == "modelscope"
    assert updated.json()["network_proxy"] == {
        "mode": "manual",
        "url": "http://127.0.0.1:7890",
        "bypass": "localhost,127.0.0.1,::1",
    }
    assert updated.json()["network_proxy_restart_required"] is True
    assert rejected.status_code == 400
    assert rejected_source.status_code == 400
    settings_path = data_dir / "application-settings.json"
    assert settings_path.stat().st_mode & 0o777 == 0o600
    serialized = settings_path.read_text(encoding="utf-8")
    assert "secret" not in serialized


def test_network_proxy_modes_control_process_outbound_environment(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://system-proxy.test:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://system-proxy.test:8080")

    apply_network_proxy_environment({"mode": "system", "url": "", "bypass": "localhost"})
    assert os.environ["HTTP_PROXY"] == "http://system-proxy.test:8080"

    apply_network_proxy_environment({"mode": "manual", "url": "http://127.0.0.1:7890", "bypass": "localhost,127.0.0.1"})
    assert os.environ["HTTP_PROXY"] == os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"
    assert os.environ["NO_PROXY"] == "localhost,127.0.0.1"

    apply_network_proxy_environment({"mode": "direct", "url": "", "bypass": "localhost"})
    assert "HTTP_PROXY" not in os.environ and "HTTPS_PROXY" not in os.environ
    assert os.environ["NO_PROXY"] == "*"



def _identity_model(path: Path) -> None:
    input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 8, 8])
    output_info = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 3, 8, 8])
    graph = helper.make_graph([
        helper.make_node("Relu", ["image"], ["backbone.hidden"], name="backbone.relu"),
        helper.make_node("Identity", ["backbone.hidden"], ["features"]),
    ], "identity", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, path)


def _depth_model(path: Path) -> None:
    input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 4, 4])
    output_info = helper.make_tensor_value_info("depth", TensorProto.FLOAT, [1, 4, 4])
    values = [float(index) for index in range(16)]
    node = helper.make_node(
        "Constant",
        [],
        ["depth"],
        value=helper.make_tensor("depth_values", TensorProto.FLOAT, [1, 4, 4], values),
    )
    graph = helper.make_graph([node], "depth", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, path)


def test_health_and_real_dataset_scan(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (16, 10), "white").save(root / "image.png")
    (root / "image.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    Image.new("RGB", (4, 4), "black").save(root / "hidden.png")

    app = create_app(Settings(data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["api_version"] == "v1"

        response = client.post("/api/v1/datasets/scan", json={
            "root_dir": str(root),
            "layout": "same_directory",
        })

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["valid"] == 1
    assert payload["summary"]["hidden_image_only"] == 1
    assert len(payload["items"]) == 1
    assert payload["items"][0]["selectable"] is True


def test_dataset_unregister_removes_only_index_and_preserves_source_files(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    image_path = root / "image.png"
    annotation_path = root / "image.json"
    Image.new("RGB", (16, 10), "white").save(image_path)
    annotation_path.write_text(json.dumps({"shapes": []}), encoding="utf-8")
    app = create_app(Settings(data_dir=tmp_path / "data"))

    with TestClient(app) as client:
        registered = client.post("/api/v1/datasets/register", json={
            "dataset_id": "removable",
            "root_dir": str(root),
            "layout": "same_directory",
        })
        removed = client.delete("/api/v1/datasets/removable")
        listed = client.get("/api/v1/datasets")
        missing = client.get("/api/v1/datasets/removable/assets-cursor")

    assert registered.status_code == 200
    assert removed.status_code == 204
    assert listed.json()["datasets"] == []
    assert missing.status_code == 400
    assert image_path.is_file() and annotation_path.is_file()


def test_persistent_scan_session_registers_without_large_scan_response_and_supports_cursors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    for index in range(3):
        Image.new("RGB", (16 + index, 10), "white").save(root / f"image-{index}.png")
        (root / f"image-{index}.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")

    app = create_app(Settings(data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/dataset-scan-sessions",
            json={"root_dir": str(root), "layout": "same_directory"},
        )
        assert created.status_code == 202
        assert "items" not in created.json()
        session_id = created.json()["session_id"]
        deadline = monotonic() + 5
        session = created.json()
        while monotonic() < deadline and session["state"] not in {"succeeded", "failed"}:
            sleep(0.01)
            session = client.get(f"/api/v1/dataset-scan-sessions/{session_id}").json()
        items = client.get(f"/api/v1/dataset-scan-sessions/{session_id}/items?limit=2").json()
        registered = client.post(f"/api/v1/dataset-scan-sessions/{session_id}/register?name=fixture")
        dataset_id = registered.json()["dataset_id"]
        first = client.get(f"/api/v1/datasets/{dataset_id}/assets-cursor?limit=2").json()
        second = client.get(
            f"/api/v1/datasets/{dataset_id}/assets-cursor",
            params={"limit": 2, "cursor": first["next_cursor"]},
        ).json()
        searched = client.get(
            f"/api/v1/datasets/{dataset_id}/search-cursor",
            params={"q": "image", "mode": "text", "limit": 2},
        ).json()

    assert session["state"] == "succeeded"
    assert items["total"] == 3
    assert len(items["items"]) == 2
    assert items["next_after"] is not None
    assert registered.status_code == 200
    assert registered.json()["name"] == "fixture"
    assert first["total"] == 3
    assert len(first["items"]) == 2
    assert first["next_cursor"]
    assert len(second["items"]) == 1
    assert second["next_cursor"] is None
    assert searched["total"] == 3


def test_catalog_import_api(tmp_path: Path) -> None:
    configs = tmp_path / "x" / "anylabeling" / "configs"
    auto = configs / "auto_labeling"
    auto.mkdir(parents=True)
    (configs / "models.yaml").write_text("- model_name: remote\n  config_file: :/remote.yaml\n", encoding="utf-8")
    (auto / "remote.yaml").write_text("type: remote_server\nname: remote\ndisplay_name: Remote Server\n", encoding="utf-8")

    app = create_app(Settings(data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        imported = client.post("/api/v1/model-sources/x-anylabeling/import", json={"root_dir": str(tmp_path / "x")})
        listed = client.get("/api/v1/models")

    assert imported.status_code == 200
    assert [model["id"] for model in listed.json()["models"]] == ["remote"]
    restarted = create_app(Settings(data_dir=tmp_path / "data"))
    with TestClient(restarted) as client:
        restored = client.get("/api/v1/models")
    assert [model["id"] for model in restored.json()["models"]] == ["remote"]


def test_model_weight_listing_never_downloads_implicitly(tmp_path: Path) -> None:
    configs = tmp_path / "x" / "anylabeling" / "configs"
    auto = configs / "auto_labeling"
    auto.mkdir(parents=True)
    (configs / "models.yaml").write_text("- model_name: depth\n  config_file: :/depth.yaml\n", encoding="utf-8")
    (auto / "depth.yaml").write_text(
        "type: depth_anything_v2\nname: depth\nmodel_path: https://github.com/org/depth.onnx\n",
        encoding="utf-8",
    )
    app = create_app(Settings(data_dir=tmp_path / "data"))

    with TestClient(app) as client:
        imported = client.post("/api/v1/model-sources/x-anylabeling/import", json={"root_dir": str(tmp_path / "x")})
        weights = client.get("/api/v1/models/depth/weights")

    assert imported.json()["models"][0]["availability"]["state"] == "missing_weights"
    assert weights.status_code == 200
    assert weights.json()[0]["filename"] == "depth.onnx"
    assert weights.json()[0]["downloaded"] is False
    assert not (tmp_path / "data" / "model-weights").exists()


def test_model_weight_download_requires_idempotency_and_creates_persistent_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configs = tmp_path / "x" / "anylabeling" / "configs"
    auto = configs / "auto_labeling"
    auto.mkdir(parents=True)
    (configs / "models.yaml").write_text("- model_name: depth\n  config_file: :/depth.yaml\n", encoding="utf-8")
    (auto / "depth.yaml").write_text(
        "type: depth_anything_v2\nname: depth\nmodel_path: https://github.com/org/depth.onnx\n",
        encoding="utf-8",
    )

    def fake_download(self, model_id, record, url_index, **kwargs):  # noqa: ANN001
        del record, kwargs
        target = self.root / model_id / "depth.onnx"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
        return DownloadedWeight(
            model_id=model_id,
            url_index=url_index,
            source_url="https://github.com/org/depth.onnx",
            final_url="https://github.com/org/depth.onnx",
            local_path=target,
            size_bytes=7,
            sha256="0" * 64,
            cache_hit=False,
        )

    monkeypatch.setattr(ModelWeightStore, "download", fake_download)
    app = create_app(Settings(data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        client.post("/api/v1/model-sources/x-anylabeling/import", json={"root_dir": str(tmp_path / "x")})
        missing_key = client.post("/api/v1/models/depth/weights/0/download")
        created = client.post(
            "/api/v1/models/depth/weights/0/download",
            headers={"Idempotency-Key": "depth-weight"},
        )
        duplicate = client.post(
            "/api/v1/models/depth/weights/0/download",
            headers={"Idempotency-Key": "depth-weight"},
        )
        deadline = monotonic() + 5
        job = created.json()
        while monotonic() < deadline and job["state"] not in {"succeeded", "succeeded_with_errors", "failed"}:
            sleep(0.01)
            job = client.get(f"/api/v1/jobs/{created.json()['job_id']}").json()

    assert missing_key.status_code == 428
    assert created.status_code == 202
    assert created.headers["location"] == f"/api/v1/jobs/{created.json()['job_id']}"
    assert created.json()["kind"] == "model_download"
    assert duplicate.json()["job_id"] == created.json()["job_id"]
    assert job["state"] == "succeeded"
    assert job["completed"] == 1


def test_agent_api_requires_confirmation_before_creating_job(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (32, 24), "white").save(root / "image.png")
    (root / "image.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    app = create_app(Settings(data_dir=tmp_path / "data"))
    monkeypatch.setenv("LABELONE_TEST_AGENT_KEY", "secret-token")

    with TestClient(app) as client:
        configured = client.patch("/api/v1/settings", json={"cloud_ai": {
            "enabled": True,
            "provider": "openai_compatible",
            "endpoint": "https://llm.example.test/v1/chat/completions",
            "model": "planner-model",
            "api_key_env": "LABELONE_TEST_AGENT_KEY",
            "timeout_seconds": 20,
            "max_output_tokens": 600,
        }})
        assert configured.status_code == 200
        registered = client.post("/api/v1/datasets/register", json={
            "dataset_id": "dataset",
            "root_dir": str(root),
            "layout": "same_directory",
        })
        asset_id = registered.json()["items"][0]["asset_id"]
        proposed = client.post("/api/v1/agent/runs", json={
            "dataset_id": "dataset",
            "asset_id": asset_id,
            "message": "给当前图设计增强流程",
        })
        assert proposed.status_code == 200
        assert proposed.json()["state"] == "proposed"
        assert client.get("/api/v1/jobs").json()["jobs"] == []
        payload = proposed.json()
        executed = client.post(
            f"/api/v1/agent/runs/{payload['run_id']}/proposals/{payload['proposals'][0]['id']}/execute"
        )
        audit = client.get(f"/api/v1/agent/runs/{payload['run_id']}/audit")

    assert executed.status_code == 200
    assert executed.json()["state"] == "completed"
    assert executed.json()["proposals"][0]["result"]["total"] == 1
    assert [entry["status"] for entry in audit.json()] == ["proposed", "executed"]


def test_agent_api_is_unavailable_until_backend_is_configured(tmp_path: Path, monkeypatch) -> None:
    app = create_app(Settings(data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        status = client.get("/api/v1/agent/status")
        blocked = client.post("/api/v1/agent/runs", json={
            "dataset_id": "missing",
            "message": "检查数据集概况",
            "tool_call": {"tool": "dataset.stats", "arguments": {}},
        })
        client.patch("/api/v1/settings", json={"cloud_ai": {
            "enabled": True,
            "provider": "openai_compatible",
            "endpoint": "https://llm.example.test/v1/chat/completions",
            "model": "planner-model",
            "api_key_env": "LABELONE_MISSING_AGENT_KEY",
            "timeout_seconds": 20,
            "max_output_tokens": 600,
        }})
        missing_credential = client.get("/api/v1/agent/status")

    assert status.status_code == 200
    assert status.json()["state"] == "unconfigured"
    assert status.json()["reason_code"] == "disabled"
    assert {item["tool"] for item in status.json()["capabilities"]} >= {"dataset.stats", "annotation.qa", "pipeline.create_job", "inference.create_job"}
    assert blocked.status_code == 503
    assert blocked.json()["code"] == "agent_backend_unavailable"
    assert missing_credential.json()["state"] == "unconfigured"
    assert missing_credential.json()["reason_code"] == "missing_credential"


def test_model_load_layers_and_inference_api(tmp_path: Path) -> None:
    configs = tmp_path / "x" / "anylabeling" / "configs"
    auto = configs / "auto_labeling"
    auto.mkdir(parents=True)
    (configs / "models.yaml").write_text("- model_name: identity\n  config_file: :/identity.yaml\n", encoding="utf-8")
    (auto / "identity.yaml").write_text("type: fixture\nname: identity\ndisplay_name: Identity\nmodel_path: identity.onnx\n", encoding="utf-8")
    _identity_model(auto / "identity.onnx")
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)

    app = create_app(Settings(data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        imported = client.post("/api/v1/model-sources/x-anylabeling/import", json={"root_dir": str(tmp_path / "x")})
        loaded = client.post("/api/v1/models/identity/load", json={"providers": ["CPUExecutionProvider"]})
        layers = client.get("/api/v1/models/identity/layers")
        inferred = client.post("/api/v1/inference-runs", json={
            "model_id": "identity",
            "image_path": str(image_path),
            "capture_layers": ["backbone.hidden"],
            "parameters": {"feature_transform": {"projection": "mean", "normalization": "minmax"}},
        })
        artifact_id = inferred.json()["artifacts"][0]["id"]
        artifact_content = client.get(f"/api/v1/artifacts/{artifact_id}/content")
        artifact_preview = client.get(f"/api/v1/artifacts/{artifact_id}/preview")
        too_many_layers = client.post("/api/v1/inference-runs", json={
            "model_id": "identity",
            "image_path": str(image_path),
            "capture_layers": ["features", "backbone.hidden"],
        })

    assert imported.status_code == 200
    assert loaded.status_code == 200
    assert loaded.json()["state"] == "loaded"
    assert loaded.json()["capture_mode"] == "graph_rewrite"
    assert {layer["id"] for layer in layers.json()["layers"]} == {"features", "backbone.hidden"}
    assert inferred.status_code == 200
    assert inferred.json()["artifacts"][0]["shape"] == [1, 1, 8, 8]
    assert inferred.json()["artifacts"][0]["layer_id"] == "backbone.hidden"
    assert inferred.json()["artifacts"][0]["preview_available"] is True
    assert artifact_content.status_code == 200
    assert artifact_content.headers["content-type"].startswith("application/x-npy")
    assert artifact_preview.status_code == 200
    assert artifact_preview.headers["content-type"].startswith("image/png")
    assert too_many_layers.status_code == 422


def test_depth_adapter_returns_viewable_raster_without_capture_layers(tmp_path: Path) -> None:
    configs = tmp_path / "x" / "anylabeling" / "configs"
    auto = configs / "auto_labeling"
    auto.mkdir(parents=True)
    (configs / "models.yaml").write_text("- model_name: depth\n  config_file: :/depth.yaml\n", encoding="utf-8")
    (auto / "depth.yaml").write_text(
        "type: depth_anything_v2\nname: depth\nmodel_path: depth.onnx\n",
        encoding="utf-8",
    )
    _depth_model(auto / "depth.onnx")
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 6), (10, 20, 30)).save(image_path)
    app = create_app(Settings(data_dir=tmp_path / "data"))

    with TestClient(app) as client:
        client.post("/api/v1/model-sources/x-anylabeling/import", json={"root_dir": str(tmp_path / "x")})
        loaded = client.post("/api/v1/models/depth/load", json={"providers": ["CPUExecutionProvider"]})
        inferred = client.post("/api/v1/inference-runs", json={
            "model_id": "depth",
            "image_path": str(image_path),
            "parameters": {"color_map": "grayscale", "percentile_low": 0, "percentile_high": 100},
        })
        raster_id = inferred.json()["rasters"][0]["id"]
        raster = client.get(f"/api/v1/artifacts/{raster_id}/content")

    assert loaded.status_code == 200
    assert inferred.status_code == 200
    assert inferred.json()["artifacts"] == []
    assert inferred.json()["rasters"][0]["role"] == "depth-map"
    assert Image.open(BytesIO(raster.content)).size == (8, 6)


def test_registered_dataset_annotation_and_image_api(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (120, 60), (30, 60, 90)).save(root / "image.jpg")
    (root / "image.json").write_text(json.dumps({
        "custom": "preserve",
        "shapes": [{
            "label": "box",
            "shape_type": "rectangle",
            "points": [[10, 10], [50, 40]],
        }],
    }), encoding="utf-8")

    app = create_app(Settings(data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        registered = client.post("/api/v1/datasets/register", json={
            "dataset_id": "dataset",
            "root_dir": str(root),
            "layout": "same_directory",
        })
        assert registered.status_code == 200
        asset_id = registered.json()["items"][0]["asset_id"]
        listed = client.get("/api/v1/datasets/dataset/assets")
        annotation = client.get(f"/api/v1/datasets/dataset/assets/{asset_id}/annotation")
        revision = annotation.headers["etag"]
        document = annotation.json()["document"]
        document["shapes"][0]["label"] = "changed"
        saved = client.put(
            f"/api/v1/datasets/dataset/assets/{asset_id}/annotation",
            headers={"If-Match": revision},
            json={"document": document},
        )
        conflict = client.put(
            f"/api/v1/datasets/dataset/assets/{asset_id}/annotation",
            headers={"If-Match": revision},
            json={"document": document},
        )
        thumbnail = client.get(f"/api/v1/datasets/dataset/assets/{asset_id}/thumbnail?max_size=48")
        not_modified = client.get(
            f"/api/v1/datasets/dataset/assets/{asset_id}/thumbnail?max_size=48",
            headers={"If-None-Match": thumbnail.headers["etag"]},
        )
        region = client.get(f"/api/v1/datasets/dataset/assets/{asset_id}/region?x=10&y=5&width=20&height=10&scale=2")
        tile_metadata = client.get(f"/api/v1/datasets/dataset/assets/{asset_id}/tiles/metadata")
        max_level = tile_metadata.json()["max_level"]
        tile = client.get(f"/api/v1/datasets/dataset/assets/{asset_id}/tiles/{max_level}/0/0")
        tile_not_modified = client.get(
            f"/api/v1/datasets/dataset/assets/{asset_id}/tiles/{max_level}/0/0",
            headers={"If-None-Match": tile.headers["etag"]},
        )
        searched = client.get("/api/v1/datasets/dataset/search", params={"q": "class:changed type:rectangle", "mode": "condition"})
        invalid_regex = client.get("/api/v1/datasets/dataset/search", params={"q": "(a+)+$", "mode": "regex"})

    assert listed.json()["total"] == 1
    assert annotation.json()["document"]["custom"] == "preserve"
    assert saved.status_code == 200
    assert saved.headers["etag"] != revision
    assert conflict.status_code == 412
    assert thumbnail.status_code == 200
    assert not_modified.status_code == 304
    assert Image.open(BytesIO(region.content)).size == (40, 20)
    assert tile_metadata.json()["width"] == 120
    assert tile.headers["x-labelone-tile-backend"] == "pillow"
    assert tile_not_modified.status_code == 304
    assert searched.json()["total"] == 1
    assert invalid_regex.status_code == 400


def test_managed_annotation_path_loads_empty_document_and_creates_json(tmp_path: Path) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (32, 18), "white").save(images / "fresh.png")
    app = create_app(Settings(data_dir=tmp_path / "data"))

    with TestClient(app) as client:
        registered = client.post("/api/v1/datasets/register", json={
            "dataset_id": "managed",
            "root_dir": str(images),
            "annotation_storage_root": str(annotations),
            "layout": "auto",
        })
        asset_id = registered.json()["items"][0]["asset_id"]
        loaded = client.get(f"/api/v1/datasets/managed/assets/{asset_id}/annotation")
        document = loaded.json()["document"]
        document["shapes"] = [{
            "label": "created",
            "shape_type": "rectangle",
            "points": [[1, 1], [8, 1], [8, 6], [1, 6]],
        }]
        saved = client.put(
            f"/api/v1/datasets/managed/assets/{asset_id}/annotation",
            headers={"If-Match": loaded.headers["etag"]},
            json={"document": document},
        )

    assert registered.status_code == 200
    assert loaded.status_code == 200
    assert loaded.json()["document"]["shapes"] == []
    assert saved.status_code == 200
    assert (annotations / "fresh.json").is_file()


def test_batch_pipeline_job_api_and_idempotency(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    for index in range(4):
        Image.new("RGB", (80, 40), (index * 20, 40, 60)).save(root / f"image-{index}.png")
        (root / f"image-{index}.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    app = create_app(Settings(data_dir=tmp_path / "data"))
    request = {
        "kind": "pipeline",
        "dataset_id": "dataset",
        "concurrency": 2,
        "pipeline_nodes": [{"id": "color", "kind": "color", "parameters": {"contrast": 1.1}}],
        "output_policy": {"mode": "preview", "image_format": "jpeg", "conflict": "reuse"},
    }
    with TestClient(app) as client:
        registered = client.post("/api/v1/datasets/register", json={"dataset_id": "dataset", "root_dir": str(root), "layout": "same_directory"})
        missing_key = client.post("/api/v1/jobs", json=request)
        created = client.post("/api/v1/jobs", headers={"Idempotency-Key": "pipeline-job"}, json=request)
        duplicate = client.post("/api/v1/jobs", headers={"Idempotency-Key": "pipeline-job"}, json=request)
        job_id = created.json()["job_id"]
        deadline = monotonic() + 5
        while monotonic() < deadline:
            snapshot = client.get(f"/api/v1/jobs/{job_id}").json()
            if snapshot["state"] in {"succeeded", "succeeded_with_errors", "failed"}:
                break
            sleep(0.01)
        items = client.get(f"/api/v1/jobs/{job_id}/items?limit=2")
        asset_id = registered.json()["items"][0]["asset_id"]
        lookup = client.post(f"/api/v1/jobs/{job_id}/items/lookup", json={"asset_ids": [asset_id]})
        item_result = lookup.json()["items"][0]["result"]
        artifact = client.get(f"/api/v1/pipeline-artifacts/{item_result['artifact_id']}")
        scheduler = client.get("/api/v1/jobs-scheduler")

    assert missing_key.status_code == 428
    assert created.status_code == 202
    assert created.headers["location"].endswith(job_id)
    assert duplicate.json()["job_id"] == job_id
    assert created.json()["request"]["pipeline_context"]["output_format"] == "jpeg"
    assert snapshot["state"] == "succeeded"
    assert snapshot["completed"] == 4
    assert items.json()["total"] == 4
    assert len(items.json()["items"]) == 2
    assert lookup.status_code == 200
    assert item_result["asset_id"] == asset_id
    assert item_result["media_type"] == "image/jpeg"
    assert item_result["visualizations"][-1]["artifact_id"] == item_result["artifact_id"]
    assert artifact.status_code == 200
    assert artifact.headers["content-type"] == "image/jpeg"
    assert scheduler.json()["global_capacity"] == 8
    assert scheduler.json()["lane_capacities"]["cpu_pipeline"] == 4
    assert scheduler.json()["lane_interactive_reserves"]["cpu_pipeline"] == 1


def test_pipeline_precompute_ensure_reuses_exact_context_and_explicitly_replaces_background_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from labelone.pipelines import PipelineEngine

    root = tmp_path / "dataset"
    root.mkdir()
    for index in range(3):
        Image.new("RGB", (80, 40), (index * 20, 40, 60)).save(root / f"image-{index}.png")
        (root / f"image-{index}.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    started = Event()
    release = Event()
    original_preview = PipelineEngine.preview

    def blocked_preview(self, request, progress=None, canceled=None):
        started.set()
        assert release.wait(timeout=4)
        return original_preview(self, request, progress=progress, canceled=canceled)

    monkeypatch.setattr(PipelineEngine, "preview", blocked_preview)
    app = create_app(Settings(data_dir=tmp_path / "data"))
    base_request = {
        "kind": "pipeline",
        "dataset_id": "dataset",
        "priority": "background",
        "concurrency": 2,
        "pipeline_nodes": [{"id": "color", "kind": "color"}],
        "output_policy": {"mode": "preview", "image_format": "webp", "conflict": "reuse"},
    }
    try:
        with TestClient(app) as client:
            registered = client.post("/api/v1/datasets/register", json={
                "dataset_id": "dataset",
                "root_dir": str(root),
                "layout": "same_directory",
            }).json()
            preferred = [item["asset_id"] for item in registered["items"][:2]]
            first = client.post(
                "/api/v1/pipelines/precompute/ensure",
                json={**base_request, "preferred_asset_ids": preferred},
            )
            assert started.wait(timeout=2)
            reused = client.post(
                "/api/v1/pipelines/precompute/ensure",
                json={**base_request, "concurrency": 4, "preferred_asset_ids": list(reversed(preferred))},
            )
            replacement = client.post(
                "/api/v1/pipelines/precompute/ensure",
                json={
                    **base_request,
                    "pipeline_nodes": [{"id": "color", "kind": "color", "parameters": {"contrast": 1.2}}],
                },
            )
            invalid_priority = client.post(
                "/api/v1/pipelines/precompute/ensure",
                json={**base_request, "priority": "user_batch"},
            )
            invalid_subset = client.post(
                "/api/v1/pipelines/precompute/ensure",
                json={**base_request, "asset_ids": preferred[:1]},
            )
            invalid_derived = client.post(
                "/api/v1/pipelines/precompute/ensure",
                json={
                    **base_request,
                    "output_policy": {
                        "mode": "derived_dataset",
                        "output_root": str((tmp_path / "derived").resolve()),
                        "image_format": "png",
                        "conflict": "reuse",
                    },
                },
            )
            release.set()
    finally:
        release.set()

    assert first.status_code == reused.status_code == replacement.status_code == 200
    assert first.json()["reused"] is False
    assert reused.json()["reused"] is True
    assert reused.json()["job"]["job_id"] == first.json()["job"]["job_id"]
    assert reused.json()["canceled_job_ids"] == []
    assert replacement.json()["reused"] is False
    assert replacement.json()["job"]["job_id"] != first.json()["job"]["job_id"]
    assert replacement.json()["canceled_job_ids"] == [first.json()["job"]["job_id"]]
    assert invalid_priority.status_code == invalid_subset.status_code == invalid_derived.status_code == 400


def test_concurrent_pipeline_precompute_ensure_creates_exactly_one_job(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    for index in range(2):
        Image.new("RGB", (32, 24), (index * 20, 40, 60)).save(root / f"image-{index}.png")
        (root / f"image-{index}.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    app = create_app(Settings(data_dir=tmp_path / "data"))
    request = {
        "kind": "pipeline",
        "dataset_id": "dataset",
        "priority": "background",
        "concurrency": 2,
        "pipeline_nodes": [{"id": "color", "kind": "color"}],
        "output_policy": {"mode": "preview", "image_format": "png", "conflict": "reuse"},
    }

    async def exercise() -> tuple[list[httpx.Response], httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as client:
                registered = await client.post("/datasets/register", json={
                    "dataset_id": "dataset",
                    "root_dir": str(root),
                    "layout": "same_directory",
                })
                assert registered.status_code == 200
                responses = await asyncio.gather(
                    client.post("/pipelines/precompute/ensure", json=request),
                    client.post("/pipelines/precompute/ensure", json=request),
                )
                jobs = await client.get("/jobs?limit=20")
                return list(responses), jobs

    responses, jobs = asyncio.run(exercise())

    assert [response.status_code for response in responses] == [200, 200]
    payloads = [response.json() for response in responses]
    assert {payload["job"]["job_id"] for payload in payloads} == {payloads[0]["job"]["job_id"]}
    assert sorted(payload["reused"] for payload in payloads) == [False, True]
    matching = [
        job for job in jobs.json()["jobs"]
        if job["request"].get("pipeline_context", {}).get("signature")
        == payloads[0]["job"]["request"]["pipeline_context"]["signature"]
    ]
    assert len(matching) == 1


def test_pipeline_preview_api_returns_multiple_visualization_artifacts_and_final_primary(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (80, 40), (20, 40, 60)).save(root / "image.png")
    (root / "image.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    app = create_app(Settings(data_dir=tmp_path / "data"))

    with TestClient(app) as client:
        registered = client.post("/api/v1/datasets/register", json={
            "dataset_id": "dataset",
            "root_dir": str(root),
            "layout": "same_directory",
        })
        asset_id = registered.json()["items"][0]["asset_id"]
        preview_request = {
            "dataset_id": "dataset",
            "asset_id": asset_id,
            "nodes": [
                {"id": "source", "kind": "source"},
                {"id": "original-size", "kind": "visualize", "parameters": {"label": "Original size"}},
                {"id": "resize", "kind": "resize", "parameters": {"width": 20, "height": 10}},
                {"id": "small", "kind": "visualize", "parameters": {"label": "Small"}},
            ],
        }
        preview = client.post("/api/v1/pipelines/preview", json=preview_request)
        payload = preview.json()
        background = client.post(
            "/api/v1/pipelines/preview",
            json={**preview_request, "priority": "background"},
        ).json()
        contents = [
            client.get(f"/api/v1/pipeline-artifacts/{item['artifact_id']}")
            for item in payload["visualizations"]
        ]

    assert preview.status_code == 200
    assert [(item["visualization_id"], item["width"], item["height"]) for item in payload["visualizations"]] == [
        ("original-size", 80, 40),
        ("small", 20, 10),
    ]
    assert payload["artifact_id"] == payload["visualizations"][-1]["artifact_id"]
    assert background["artifact_id"] == payload["artifact_id"]
    assert (payload["width"], payload["height"]) == (20, 10)
    assert background["timing_sample_count"] == {
        "source": 1,
        "original-size": 1,
        "resize": 1,
        "small": 1,
    }
    assert set(background["operator_average_timings_ms"]) == {"source", "original-size", "resize", "small"}
    assert all(response.status_code == 200 and response.headers["content-type"] == "image/webp" for response in contents)
    assert all(response.headers["cache-control"] == "private, max-age=31536000, immutable" for response in contents)


def test_pipeline_validate_api_normalizes_legacy_counts_nodes_and_checks_dimensions(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path / "data"))
    request = {
        "width": 100,
        "height": 80,
        "nodes": [
            {"id": "crop", "kind": "crop", "parameters": {"x": 10, "y": 5, "width": 60, "height": 40}},
            {"id": "resize", "kind": "resize", "parameters": {"width": 30, "height": 20}},
            {"id": "legacy", "kind": "output"},
        ],
    }

    with TestClient(app) as client:
        valid = client.post("/api/v1/pipelines/validate", json=request)
        invalid_crop = client.post("/api/v1/pipelines/validate", json={
            **request,
            "nodes": [
                {"id": "crop", "kind": "crop", "parameters": {"x": 90, "y": 0, "width": 20, "height": 10}},
                {"id": "display", "kind": "visualize"},
            ],
        })
        preview_tile = client.post("/api/v1/pipelines/validate", json={
            "nodes": [
                {"id": "tile", "kind": "tile"},
                {"id": "display", "kind": "visualize"},
            ],
        })
        derived_tile = client.post("/api/v1/pipelines/validate", json={
            "mode": "derived_dataset",
            "nodes": [
                {"id": "tile", "kind": "tile"},
                {"id": "display", "kind": "visualize"},
            ],
        })

    payload = valid.json()
    assert valid.status_code == 200
    assert payload["valid"] is True
    assert len(payload["registry_hash"]) == 64
    assert [node["kind"] for node in payload["normalized_nodes"]] == [
        "source", "crop", "resize", "visualize",
    ]
    assert payload["transform_count"] == 2
    assert payload["visualization_count"] == 1
    assert (payload["output_width"], payload["output_height"]) == (30, 20)
    assert invalid_crop.status_code == 400
    assert invalid_crop.json()["code"] == "pipeline_validation_error"
    assert preview_tile.status_code == 400
    assert preview_tile.json()["code"] == "pipeline_validation_error"
    assert derived_tile.status_code == 200


def test_operator_zip_import_registers_and_executes_without_service_restart(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (12, 8), (20, 40, 60)).save(root / "image.png")
    (root / "image.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    app = create_app(Settings(data_dir=tmp_path / "data"))

    with TestClient(app) as client:
        inspected = client.post(
            "/api/v1/pipelines/operators/inspect?filename=api-invert.zip",
            content=_operator_zip(),
            headers={"Content-Type": "application/zip"},
        )
        imported = client.post(
            "/api/v1/pipelines/operators/import?filename=api-invert.zip",
            content=_operator_zip(),
            headers={"Content-Type": "application/zip"},
        )
        catalog = client.get("/api/v1/pipelines/operators")
        registered = client.post("/api/v1/datasets/register", json={
            "dataset_id": "dataset",
            "root_dir": str(root),
            "layout": "same_directory",
        })
        asset_id = registered.json()["items"][0]["asset_id"]
        preview = client.post("/api/v1/pipelines/preview", json={
            "dataset_id": "dataset",
            "asset_id": asset_id,
            "output_format": "png",
            "nodes": [
                {"id": "source", "kind": "source"},
                {"id": "invert", "kind": "api.invert"},
                {"id": "display", "kind": "visualize", "parameters": {"label": "Custom"}},
            ],
        })
        content = client.get(f"/api/v1/pipeline-artifacts/{preview.json()['artifact_id']}")

    assert inspected.status_code == 200
    assert inspected.json()["operator"]["kind"] == "api.invert"
    assert inspected.json()["operator"]["title"] == "API invert"
    assert inspected.json()["operator"]["description"] == "Inverts every image channel for the API fixture."
    assert inspected.json()["will_execute_local_code"] is True
    assert inspected.json()["is_os_sandboxed"] is False
    assert imported.status_code == 201
    assert imported.json()["operator"]["kind"] == "api.invert"
    assert imported.json()["trusted_local_code"] is True
    assert imported.json()["is_os_sandboxed"] is False
    assert any(item["kind"] == "api.invert" for item in catalog.json()["operators"])
    assert catalog.json()["installed_packages"][0]["kind"] == "api.invert"
    assert preview.status_code == 200
    assert Image.open(BytesIO(content.content)).getpixel((0, 0)) == (235, 215, 195)


def test_slow_job_repository_read_does_not_block_async_api_event_loop(tmp_path: Path, monkeypatch) -> None:
    app = create_app(Settings(data_dir=tmp_path / "data"))
    entered = Event()
    release = Event()
    original_list = JobRepository.list

    def blocked_list(self, limit=100):
        entered.set()
        assert release.wait(timeout=2)
        return original_list(self, limit)

    monkeypatch.setattr(JobRepository, "list", blocked_list)

    async def exercise() -> tuple[httpx.Response, httpx.Response, float]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as client:
                jobs_task = asyncio.create_task(client.get("/jobs"))
                assert await asyncio.to_thread(entered.wait, 1)
                started_at = monotonic()
                health = await client.get("/health")
                elapsed = monotonic() - started_at
                release.set()
                return await jobs_task, health, elapsed

    safety_release = Timer(1.0, release.set)
    safety_release.start()
    try:
        jobs_response, health_response, elapsed = asyncio.run(exercise())
    finally:
        release.set()
        safety_release.cancel()

    assert jobs_response.status_code == 200
    assert health_response.status_code == 200
    assert elapsed < 0.5


def test_pipeline_operator_catalog_and_persisted_safe_composite_api(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    app = create_app(settings)
    definition = {
        "id": "web-enhance",
        "name": "Web enhance",
        "steps": [
            {"kind": "resize", "parameters": {"width": 320, "height": 160}},
            {"kind": "color", "parameters": {"contrast": 1.2}},
        ],
    }
    with TestClient(app) as client:
        catalog = client.get("/api/v1/pipelines/operators")
        created = client.post("/api/v1/pipelines/composites", json=definition)
        expanded = client.get("/api/v1/pipelines/composites/web-enhance/expand?width=640&height=480")
        rejected = client.post("/api/v1/pipelines/composites", json={
            "id": "unsafe",
            "name": "Unsafe",
            "python": "import os",
            "steps": [{"kind": "flip"}],
        })

    assert catalog.status_code == 200
    assert len(catalog.json()["registry_hash"]) == 64
    display = next(item for item in catalog.json()["operators"] if item["kind"] == "visualize")
    assert display["title"] == "显示"
    assert display["description"]
    assert display["parameters_schema"]["properties"]["label"]["description"]
    assert created.status_code == 201
    assert [node["kind"] for node in expanded.json()["nodes"]] == ["resize", "color"]
    assert expanded.json()["output_width"] == 320
    assert rejected.status_code == 400

    restarted = create_app(settings)
    with TestClient(restarted) as client:
        restored = client.get("/api/v1/pipelines/operators")
    assert restored.json()["composites"][0]["id"] == "web-enhance"


def test_corrupt_composite_store_does_not_prevent_service_start(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "pipeline-composites.json").write_text("{broken", encoding="utf-8")

    app = create_app(Settings(data_dir=data_dir))
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        operators = client.get("/api/v1/pipelines/operators")

    assert health.status_code == 200
    assert operators.status_code == 200
    assert operators.json()["composites"] == []
    assert operators.json()["warnings"]
