from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
import onnx
from onnx import TensorProto, helper
from PIL import Image

from labelone.application_settings import ApplicationSettingsStore
from labelone.config import Settings
from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.errors import RevisionConflictError
from labelone.main import create_app
from labelone.workspace_settings import (
    DatasetWorkspaceSettings,
    GlobalWorkspaceSettings,
    WorkspaceInferenceSettings,
    WorkspacePipelineNode,
    WorkspacePipelineSettings,
    WorkspaceVisualizationNode,
)


def _dataset(root: Path, *, count: int = 2) -> None:
    root.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (16, 10), (index * 30, 20, 40)).save(root / f"image-{index}.png")
        (root / f"image-{index}.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")


def _pipeline() -> WorkspacePipelineSettings:
    return WorkspacePipelineSettings(
        nodes=[WorkspacePipelineNode(id="source", kind="source", name="原图像", parameters={})],
        visualizations=[WorkspaceVisualizationNode(
            id="visualize-1",
            name="显示",
            parameters={"label": "显示"},
            tap_after_node_id="source",
        )],
    )


def _identity_model(path: Path) -> None:
    input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 8, 8])
    output_info = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 3, 8, 8])
    graph = helper.make_graph([helper.make_node("Identity", ["image"], ["features"])], "identity", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, path)


def test_dataset_workspace_round_trip_uses_independent_revision(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _dataset(root)
    database = tmp_path / "index.sqlite3"
    repository = DatasetRepository(database)
    registered = repository.register(scan_dataset(DatasetScanRequest(
        root_dir=root,
        layout="same_directory",
        dataset_id="workspace",
    )))
    asset_id = repository.list_assets("workspace").items[1].asset_id
    initial_index_revision = registered.index_revision

    initial = repository.get_workspace_settings("workspace")
    saved = repository.set_workspace_settings(
        "workspace",
        DatasetWorkspaceSettings(last_asset_id=asset_id, pipeline=_pipeline()),
        expected_revision=initial.revision,
    )

    assert initial.revision == 0
    assert saved.revision == 1
    assert saved.last_asset_id == asset_id
    assert repository.get_dataset("workspace").index_revision == initial_index_revision
    with pytest.raises(RevisionConflictError):
        repository.set_workspace_settings(
            "workspace",
            DatasetWorkspaceSettings(last_asset_id=asset_id, pipeline=_pipeline()),
            expected_revision=0,
        )
    repository.close()

    reopened = DatasetRepository(database)
    restored = reopened.get_workspace_settings("workspace")
    assert restored.revision == 1
    assert restored.pipeline is not None
    assert restored.pipeline.visualizations[0].tap_after_node_id == "source"
    reopened.delete_dataset("workspace")
    assert reopened._connection.execute(  # noqa: SLF001 - verifies the FK cascade contract
        "SELECT COUNT(*) FROM dataset_workspace_settings WHERE dataset_id='workspace'"
    ).fetchone()[0] == 0
    reopened.close()


def test_global_workspace_and_model_usage_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "application-settings.json"
    store = ApplicationSettingsStore(path)
    workspace = GlobalWorkspaceSettings(
        pipeline=_pipeline(),
        inference=WorkspaceInferenceSettings(model_id="detector", parameters={"conf": 0.4}),
    )
    store.set_workspace(workspace)
    first = store.record_model_usage("detector")
    second = store.record_model_usage("detector")

    reopened = ApplicationSettingsStore(path)
    assert reopened.workspace() == workspace
    assert first.count == 1
    assert second.count == 2
    assert reopened.model_usage()["detector"].count == 2


def test_workspace_api_restores_last_asset_and_rejects_stale_or_invalid_updates(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _dataset(root)
    data_dir = tmp_path / "data"
    app = create_app(Settings(data_dir=data_dir))
    pipeline = _pipeline().model_dump(mode="json")

    with TestClient(app) as client:
        registered = client.post("/api/v1/datasets/register", json={
            "dataset_id": "workspace-api",
            "root_dir": str(root),
            "layout": "same_directory",
        })
        assets = client.get("/api/v1/datasets/workspace-api/assets").json()["items"]
        initial = client.get("/api/v1/datasets/workspace-api/settings")
        saved = client.put("/api/v1/datasets/workspace-api/settings", json={
            "schema_version": 1,
            "last_asset_id": assets[1]["asset_id"],
            "pipeline": pipeline,
            "expected_revision": 0,
        })
        stale = client.put("/api/v1/datasets/workspace-api/settings", json={
            "schema_version": 1,
            "last_asset_id": assets[0]["asset_id"],
            "pipeline": pipeline,
            "expected_revision": 0,
        })
        invalid = client.put("/api/v1/datasets/workspace-api/settings", json={
            "schema_version": 1,
            "last_asset_id": assets[0]["asset_id"],
            "pipeline": {
                **pipeline,
                "visualizations": [{**pipeline["visualizations"][0], "tap_after_node_id": "missing"}],
            },
            "expected_revision": 1,
        })
        global_saved = client.patch("/api/v1/settings", json={
            "workspace": {
                "schema_version": 1,
                "pipeline": pipeline,
                "inference": {"model_id": None, "provider": "CPUExecutionProvider", "parameters": {}},
            },
        })

    assert registered.status_code == 200
    assert initial.json()["revision"] == 0
    assert saved.status_code == 200
    assert saved.json()["revision"] == 1
    assert saved.json()["last_asset_id"] == assets[1]["asset_id"]
    assert stale.status_code == 412
    assert invalid.status_code == 400
    assert global_saved.status_code == 200

    restarted = create_app(Settings(data_dir=data_dir))
    with TestClient(restarted) as client:
        restored_dataset = client.get("/api/v1/datasets/workspace-api/settings")
        restored_global = client.get("/api/v1/settings")
    assert restored_dataset.json()["last_asset_id"] == assets[1]["asset_id"]
    assert restored_global.json()["workspace"]["pipeline"]["visualizations"][0]["id"] == "visualize-1"


def test_model_catalog_prioritizes_loaded_runtime_then_global_usage(tmp_path: Path) -> None:
    configs = tmp_path / "x" / "anylabeling" / "configs"
    auto = configs / "auto_labeling"
    auto.mkdir(parents=True)
    (configs / "models.yaml").write_text(
        "- model_name: frequent\n  config_file: :/frequent.yaml\n"
        "- model_name: loaded\n  config_file: :/loaded.yaml\n",
        encoding="utf-8",
    )
    for model_id, display_name in (("frequent", "Frequent"), ("loaded", "Loaded")):
        (auto / f"{model_id}.yaml").write_text(
            f"type: fixture\nname: {model_id}\ndisplay_name: {display_name}\nmodel_path: shared.onnx\n",
            encoding="utf-8",
        )
    _identity_model(auto / "shared.onnx")
    data_dir = tmp_path / "data"
    app = create_app(Settings(data_dir=data_dir))

    with TestClient(app) as client:
        imported = client.post("/api/v1/model-sources/x-anylabeling/import", json={"root_dir": str(tmp_path / "x")})
        for _ in range(3):
            used = client.post("/api/v1/models/frequent/usage")
            assert used.status_code == 200
        usage_ranked = client.get("/api/v1/models")
        loaded = client.post("/api/v1/models/loaded/load", json={"providers": ["CPUExecutionProvider"]})
        runtime_ranked = client.get("/api/v1/models")

    assert imported.status_code == 200
    assert [model["id"] for model in usage_ranked.json()["models"]] == ["frequent", "loaded", "hypir-sd2"]
    assert usage_ranked.json()["status_by_model"]["frequent"]["usage_count"] == 3
    assert loaded.status_code == 200
    assert [model["id"] for model in runtime_ranked.json()["models"]] == ["loaded", "frequent", "hypir-sd2"]
    assert runtime_ranked.json()["status_by_model"]["loaded"]["runtime_state"] == "loaded"

    restarted = create_app(Settings(data_dir=data_dir))
    with TestClient(restarted) as client:
        persisted = client.get("/api/v1/models")
    assert [model["id"] for model in persisted.json()["models"]] == ["frequent", "loaded", "hypir-sd2"]
    assert persisted.json()["status_by_model"]["frequent"]["usage_count"] == 3
    assert persisted.json()["status_by_model"]["loaded"]["runtime_state"] == "unloaded"
