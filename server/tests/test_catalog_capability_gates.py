from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image

from labelone.models import ModelCatalog, ModelManager
from labelone.models.artifacts import ArtifactStore
from labelone.models.types import AvailabilityState, FeatureCaptureMode


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _catalog(tmp_path: Path, name: str, body: str) -> ModelCatalog:
    config_root = tmp_path / "anylabeling" / "configs"
    auto = config_root / "auto_labeling"
    _write(config_root / "models.yaml", f"- model_name: {name}\n  config_file: :/{name}.yaml\n")
    _write(auto / f"{name}.yaml", body)
    catalog = ModelCatalog()
    catalog.import_x_anylabeling(tmp_path)
    return catalog


def test_remote_catalog_requires_endpoint_host_protocol_and_secret(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LABELONE_REMOTE_FIXTURE_TOKEN", raising=False)
    body = """
type: remote_server
name: remote
display_name: Remote
remote_endpoint: https://trusted.example/v1/predict
trusted_hosts: [trusted.example]
remote_protocol: labelone_v1
credential_env: LABELONE_REMOTE_FIXTURE_TOKEN
remote_model_id: detector
"""
    blocked = _catalog(tmp_path / "blocked", "remote", body).get("remote").descriptor

    assert blocked.adapter == "remote_catalog"
    assert blocked.availability.state is AvailabilityState.UNSUPPORTED
    assert "not set" in (blocked.availability.reason or "")
    assert blocked.capabilities.predict is False

    monkeypatch.setenv("LABELONE_REMOTE_FIXTURE_TOKEN", "secret")
    enabled = _catalog(tmp_path / "enabled", "remote", body).get("remote").descriptor
    assert enabled.adapter == "trusted_remote_http"
    assert enabled.availability.state is AvailabilityState.AVAILABLE
    assert enabled.capabilities.predict is True
    assert enabled.capabilities.result_kinds == ["annotations"]
    assert enabled.capabilities.feature_capture.mode is FeatureCaptureMode.NONE
    assert enabled.capabilities.feature_capture.layers == []


def _yoloe_fixture(path: Path) -> None:
    input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 8, 8])
    output_info = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 2])
    rows = np.array([[4, 4, 4, 4, 0.9], [2, 2, 2, 2, 0.1]], dtype=np.float32)
    tensor = helper.make_tensor("predictions", TensorProto.FLOAT, [1, 5, 2], rows.T[None].ravel())
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output0"], value=tensor)],
        "yoloe",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, path)


def test_explicit_fixed_class_yoloe_onnx_contract_loads_and_predicts(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path, "yoloe", """
type: yoloe
name: yoloe
display_name: YOLOE ONNX
model_path: model.onnx
onnx_contract: ultralytics_detection_v1
with_mask: false
classes: [object]
""")
    model_path = tmp_path / "anylabeling" / "configs" / "auto_labeling" / "model.onnx"
    _yoloe_fixture(model_path)
    catalog.import_x_anylabeling(tmp_path)
    descriptor = catalog.get("yoloe").descriptor
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    manager = ModelManager(catalog, ArtifactStore(tmp_path / "artifacts"))

    state = manager.load("yoloe", ["CPUExecutionProvider"])
    result = manager.predict("yoloe", image_path, [], {"conf_threshold": 0.25})

    assert descriptor.adapter == "yolo_detection_onnx"
    assert descriptor.capabilities.predict is True
    assert state.state == "loaded"
    assert len(result.annotations) == 1
    assert result.annotations[0].label == "object"
    assert result.annotations[0].points == [[2.0, 2.0], [6.0, 6.0]]


def test_specialized_runtime_types_remain_gated_with_specific_reasons(tmp_path: Path) -> None:
    entries = {
        "upn": "type: upn\nname: upn\nmodel_path: upn.pth\n",
        "yoloe": "type: yoloe\nname: yoloe\nmodel_path: yoloe.pt\n",
        "florence": "type: florence2\nname: florence\nmodel_path: microsoft/Florence-2\ntrust_remote_code: true\n",
        "sam2video": "type: segment_anything_2_video\nname: sam2video\nmodel_path: sam2.pt\nmodel_cfg: sam2.yaml\n",
        "tensorrt": "type: yolo26\nname: tensorrt\nmodel_path: yolo.engine\nengine: trt\n",
        "grounding": "type: grounding_dino_api\nname: grounding\nconf_threshold: 0.25\niou_threshold: 0.8\n",
    }
    config_root = tmp_path / "anylabeling" / "configs"
    auto = config_root / "auto_labeling"
    _write(
        config_root / "models.yaml",
        "".join(f"- model_name: {name}\n  config_file: :/{name}.yaml\n" for name in entries),
    )
    for name, body in entries.items():
        _write(auto / f"{name}.yaml", body)
    catalog = ModelCatalog()
    catalog.import_x_anylabeling(tmp_path)

    for name in entries:
        descriptor = catalog.get(name).descriptor
        assert descriptor.adapter in {"catalog_only", "remote_catalog"}
        assert descriptor.capabilities.predict is False
        assert descriptor.availability.state is AvailabilityState.UNSUPPORTED
        assert descriptor.availability.reason
    trt_reason = catalog.get("tensorrt").descriptor.availability.reason or ""
    if importlib.util.find_spec("tensorrt") is None:
        assert "not installed" in trt_reason
    else:
        assert "not been implemented" in trt_reason
    assert "trust_remote_code is not executed" in (catalog.get("florence").descriptor.availability.reason or "")
    assert "single-image predict" in (catalog.get("sam2video").descriptor.availability.reason or "")
