from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from PIL import Image
import pytest

import labelone.models.manager as manager_module
from labelone.errors import ModelRuntimeError
from labelone.models import ModelCatalog, ModelManager
from labelone.models.artifacts import ArtifactStore
from labelone.models.worker_supervisor import ModelWorkerSupervisor


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _identity_model(path: Path) -> None:
    input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 8, 8])
    output_info = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 3, 8, 8])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["image"], ["features"])],
        "identity",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, path)


def _catalog(tmp_path: Path, *, with_models_yaml: bool = True) -> tuple[ModelCatalog, Path, Path]:
    root = tmp_path / "source"
    auto = root / "anylabeling" / "configs" / "auto_labeling"
    if with_models_yaml:
        _write(auto.parent / "models.yaml", "- model_name: identity\n  config_file: :/identity.yaml\n")
    _write(
        auto / "identity.yaml",
        "type: fixture\nname: identity\ndisplay_name: Identity\nmodel_path: identity.onnx\n",
    )
    _identity_model(auto / "identity.onnx")
    catalog = ModelCatalog()
    catalog.import_x_anylabeling(auto if not with_models_yaml else root)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (64, 128, 192)).save(image_path)
    return catalog, tmp_path / "data", image_path


def _assert_process_gone(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_manager_isolation_load_layers_predict_unload_and_close_real_worker(tmp_path: Path) -> None:
    catalog, data_dir, image_path = _catalog(tmp_path)
    manager = ModelManager(
        catalog,
        ArtifactStore(data_dir / "artifacts"),
        isolate_processes=True,
        data_dir=data_dir,
        worker_startup_timeout=10,
        worker_request_timeout=10,
    )

    loaded = manager.load("identity", ["CPUExecutionProvider"])
    supervisor = manager._supervisors["identity"]
    pid = supervisor.pid
    layers = manager.layers("identity")
    result = manager.predict("identity", image_path, ["features"], {
        "feature_transform": {"projection": "mean", "normalization": "minmax"}
    })
    unloaded = manager.unload("identity")
    assert manager._supervisors == {}
    assert pid is not None
    _assert_process_gone(pid)
    manager.close_all()

    assert loaded.state == layers.state == "loaded"
    assert layers.layers[0].id == "features"
    assert unloaded.state == "unloaded"
    assert result.artifacts[0].shape == [1, 1, 8, 8]
    assert result.artifacts[0].path.is_file()
    assert manager.state("identity").state == "unloaded"


def test_concurrent_load_same_model_creates_one_supervisor_and_one_process(tmp_path: Path) -> None:
    catalog, data_dir, _ = _catalog(tmp_path)
    manager = ModelManager(
        catalog,
        ArtifactStore(data_dir / "artifacts"),
        isolate_processes=True,
        data_dir=data_dir,
        worker_startup_timeout=10,
        worker_request_timeout=10,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        states = list(executor.map(
            lambda _: manager.load("identity", ["CPUExecutionProvider"]),
            range(2),
        ))

    assert [state.state for state in states] == ["loaded", "loaded"]
    assert list(manager._supervisors) == ["identity"]
    assert manager._supervisors["identity"].is_alive
    manager.close_all()


def test_manager_does_not_fallback_when_catalog_source_is_not_recoverable(tmp_path: Path) -> None:
    catalog, data_dir, _ = _catalog(tmp_path, with_models_yaml=False)
    manager = ModelManager(
        catalog,
        ArtifactStore(data_dir / "artifacts"),
        isolate_processes=True,
        data_dir=data_dir,
    )

    with pytest.raises(ModelRuntimeError, match="recoverable models.yaml"):
        manager.load("identity", ["CPUExecutionProvider"])

    assert manager.state("identity").state == "failed"
    assert manager._supervisors == {}
    manager.close_all()


def test_manager_supervisor_crash_restart_and_close_are_preserved_through_pydantic_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    catalog, data_dir, image_path = _catalog(tmp_path)
    marker = tmp_path / "worker-crashed"
    real_supervisor = ModelWorkerSupervisor

    def fixture_supervisor(model_id: str, **kwargs):
        return real_supervisor(
            model_id,
            worker_module="labelone.models.worker_fixture",
            worker_options={
                "crash_op": "predict",
                "crash_once_file": str(marker),
                "crash_exit_code": 137,
            },
            **kwargs,
        )

    monkeypatch.setattr(manager_module, "ModelWorkerSupervisor", fixture_supervisor)
    manager = ModelManager(
        catalog,
        ArtifactStore(data_dir / "artifacts"),
        isolate_processes=True,
        data_dir=data_dir,
        worker_startup_timeout=5,
        worker_request_timeout=5,
    )
    manager.load("identity", [])
    supervisor = manager._supervisors["identity"]
    first_pid = supervisor.pid

    result = manager.predict("identity", image_path, [], {})
    current_pid = supervisor.pid
    manager.close_all()

    assert result.model_id == "identity"
    assert supervisor.restart_count == 1
    assert supervisor.last_crash is not None and supervisor.last_crash["suspected_oom"] is True
    assert first_pid is not None and current_pid is not None and first_pid != current_pid
    _assert_process_gone(first_pid)
    _assert_process_gone(current_pid)
    with pytest.raises(ModelRuntimeError, match="closed"):
        manager.load("identity", [])


def test_isolation_constructor_requires_explicit_data_directory(tmp_path: Path) -> None:
    catalog, _, _ = _catalog(tmp_path)

    with pytest.raises(ValueError, match="data_dir"):
        ModelManager(
            catalog,
            ArtifactStore(tmp_path / "artifacts"),
            isolate_processes=True,
        )
