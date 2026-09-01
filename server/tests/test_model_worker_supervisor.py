from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import os
from pathlib import Path
from time import monotonic

import onnx
from onnx import TensorProto, helper
from PIL import Image
import pytest

from labelone.models.worker_supervisor import (
    ModelWorkerBudgetExceeded,
    ModelWorkerClosed,
    ModelWorkerCrashed,
    ModelWorkerInvalidRequest,
    ModelWorkerRequestBudgetExceeded,
    ModelWorkerSupervisor,
    ModelWorkerTimeout,
)
from labelone.models.worker_process import _initialize, _write_response
from labelone.models.worker_protocol import decode_message


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


def _catalog_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "source"
    configs = root / "anylabeling" / "configs"
    auto = configs / "auto_labeling"
    _write(configs / "models.yaml", "- model_name: identity\n  config_file: :/identity.yaml\n")
    _write(
        auto / "identity.yaml",
        "type: fixture\nname: identity\ndisplay_name: Identity\nmodel_path: identity.onnx\n",
    )
    _identity_model(auto / "identity.onnx")
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (64, 128, 192)).save(image_path)
    return root, tmp_path / "data", image_path


def _fixture_supervisor(tmp_path: Path, **options) -> ModelWorkerSupervisor:
    return ModelWorkerSupervisor(
        "fixture",
        catalog_root=tmp_path / "unused-source",
        data_dir=tmp_path / "data",
        worker_module="labelone.models.worker_fixture",
        worker_options=options,
        startup_timeout=5,
        request_timeout=2,
    )


def _assert_process_gone(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_real_worker_restores_catalog_loads_predicts_and_returns_artifact_metadata(tmp_path: Path) -> None:
    catalog_root, data_dir, image_path = _catalog_fixture(tmp_path)
    supervisor = ModelWorkerSupervisor(
        "identity",
        catalog_root=catalog_root,
        data_dir=data_dir,
        startup_timeout=10,
        request_timeout=10,
    )

    loaded = supervisor.load(["CPUExecutionProvider"])
    pid = supervisor.pid
    layers = supervisor.layers()
    result = supervisor.predict(image_path, ["features"], {
        "feature_transform": {"projection": "mean", "normalization": "minmax"}
    })
    unloaded = supervisor.unload()
    supervisor.close()

    assert loaded["state"] == "loaded"
    assert layers["layers"][0]["id"] == "features"
    assert unloaded["state"] == "unloaded"
    assert result["artifacts"][0]["shape"] == [1, 1, 8, 8]
    assert Path(result["artifacts"][0]["path"]).is_file()
    assert "tensor" not in result["artifacts"][0]
    assert supervisor.is_alive is False
    assert pid is not None
    _assert_process_gone(pid)


def test_worker_initialization_uses_the_configured_model_weight_directory(tmp_path: Path) -> None:
    catalog_root, data_dir, _ = _catalog_fixture(tmp_path)
    model_weights_dir = tmp_path / "external-model-weights"
    manager, model_id, _, _ = _initialize({
        "type": "init",
        "protocol": 1,
        "model_id": "identity",
        "catalog_root": str(catalog_root),
        "data_dir": str(data_dir),
        "model_weights_dir": str(model_weights_dir),
        "max_request_bytes": 2 * 1024 * 1024,
        "max_response_bytes": 8 * 1024 * 1024,
        "options": {},
    })

    assert model_id == "identity"
    assert manager.weight_store is not None
    assert manager.weight_store.root == model_weights_dir.resolve()
    manager.close_all()


def test_concurrent_calls_are_serialized_on_one_model_worker(tmp_path: Path) -> None:
    supervisor = _fixture_supervisor(tmp_path, delay_op="predict", delay_seconds=0.06)
    supervisor.load(["CPUExecutionProvider"])
    started = monotonic()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda index: supervisor.predict(tmp_path / f"image-{index}.png", [], {}),
            range(4),
        ))
    elapsed = monotonic() - started

    assert sorted(result["sequence"] for result in results) == [1, 2, 3, 4]
    assert elapsed >= 0.20
    assert supervisor.restart_count == 0
    supervisor.close()


def test_timeout_terminates_worker_and_leaves_no_orphan(tmp_path: Path) -> None:
    supervisor = _fixture_supervisor(tmp_path, delay_op="predict", delay_seconds=2.0)
    supervisor.load([])
    pid = supervisor.pid

    with pytest.raises(ModelWorkerTimeout):
        supervisor.predict(tmp_path / "image.png", [], {}, timeout=0.1)

    assert supervisor.is_alive is False
    assert pid is not None
    _assert_process_gone(pid)
    supervisor.close()


def test_nonzero_oom_like_exit_restarts_once_restores_load_and_replays(tmp_path: Path) -> None:
    marker = tmp_path / "crashed-once"
    supervisor = _fixture_supervisor(
        tmp_path,
        crash_op="predict",
        crash_once_file=str(marker),
        crash_exit_code=137,
    )
    supervisor.load(["CPUExecutionProvider"])
    first_pid = supervisor.pid

    result = supervisor.predict(tmp_path / "image.png", [], {})

    assert result["sequence"] == 1
    assert marker.is_file()
    assert supervisor.restart_count == 1
    assert supervisor.pid is not None and supervisor.pid != first_pid
    assert supervisor.last_crash is not None
    assert supervisor.last_crash["exit_code"] == 137
    assert supervisor.last_crash["suspected_oom"] is True
    supervisor.close()
    assert first_pid is not None
    _assert_process_gone(first_pid)


def test_repeated_crash_exhausts_exactly_one_automatic_restart(tmp_path: Path) -> None:
    supervisor = _fixture_supervisor(
        tmp_path,
        crash_op="predict",
        crash_always=True,
        crash_exit_code=19,
    )
    supervisor.load([])

    with pytest.raises(ModelWorkerCrashed):
        supervisor.predict(tmp_path / "image.png", [], {})

    assert supervisor.restart_count == 1
    assert supervisor.is_alive is False
    supervisor.close()


def test_request_and_response_budgets_reject_large_or_non_json_ipc(tmp_path: Path) -> None:
    supervisor = ModelWorkerSupervisor(
        "fixture",
        catalog_root=tmp_path / "source",
        data_dir=tmp_path / "data",
        worker_module="labelone.models.worker_fixture",
        maximum_request_bytes=1024,
        maximum_response_bytes=1024,
        request_timeout=2,
    )
    supervisor.load([])
    pid = supervisor.pid

    with pytest.raises(ModelWorkerRequestBudgetExceeded):
        supervisor.predict(tmp_path / "image.png", [], {"text": "x" * 2000})
    assert supervisor.pid == pid
    with pytest.raises(ModelWorkerInvalidRequest):
        supervisor.predict(tmp_path / "image.png", [], {"callback": lambda: None})
    assert supervisor.pid == pid
    supervisor.close()

    oversized = ModelWorkerSupervisor(
        "fixture",
        catalog_root=tmp_path / "source",
        data_dir=tmp_path / "data-oversized",
        worker_module="labelone.models.worker_fixture",
        worker_options={"large_result_bytes": 3000},
        maximum_response_bytes=1024,
        request_timeout=2,
    )
    oversized.load([])
    oversized_pid = oversized.pid
    with pytest.raises(ModelWorkerBudgetExceeded):
        oversized.predict(tmp_path / "image.png", [], {})
    assert oversized.is_alive is False
    assert oversized_pid is not None
    _assert_process_gone(oversized_pid)
    oversized.close()


def test_worker_response_writer_distinguishes_budget_and_finite_json_errors() -> None:
    oversized = BytesIO()
    _write_response(
        oversized,
        {"id": 7, "ok": True, "result": {"large": "x" * 3_000}},
        maximum_bytes=1_024,
    )
    oversized_response = decode_message(oversized.getvalue(), maximum_bytes=1_024)
    assert oversized_response["error"]["code"] == "worker_response_budget_exceeded"

    non_finite = BytesIO()
    _write_response(
        non_finite,
        {"id": 8, "ok": True, "result": {"mean": float("nan")}},
        maximum_bytes=1_024,
    )
    protocol_response = decode_message(non_finite.getvalue(), maximum_bytes=1_024)
    assert protocol_response["error"]["code"] == "worker_response_protocol_error"


def test_close_is_idempotent_and_rejects_future_work(tmp_path: Path) -> None:
    supervisor = _fixture_supervisor(tmp_path)
    supervisor.load([])
    pid = supervisor.pid

    supervisor.close()
    supervisor.close()

    assert pid is not None
    _assert_process_gone(pid)
    with pytest.raises(ModelWorkerClosed, match="closed"):
        supervisor.layers()
