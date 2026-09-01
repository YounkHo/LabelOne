from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper
from PIL import Image

from labelone.models.adapters.onnx import OnnxRuntimeAdapter
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.onnx_graph import instrument_onnx_outputs
from labelone.models.types import Availability, AvailabilityState, FeatureCaptureMode, ModelCapabilities, ModelDescriptor


def _identity_model(path: Path) -> None:
    input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 8, 8])
    output_info = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 3, 8, 8])
    graph = helper.make_graph([helper.make_node("Identity", ["image"], ["features"])], "identity", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, path)


def _intermediate_model(path: Path) -> None:
    input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 8, 8])
    output_info = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 3, 8, 8])
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["image"], ["backbone.hidden"], name="backbone.relu"),
            helper.make_node("Identity", ["backbone.hidden"], ["features"], name="head.output"),
        ],
        "intermediate",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, path)


def test_real_onnx_load_layer_enumeration_inference_and_capture(tmp_path: Path) -> None:
    model_path = tmp_path / "identity.onnx"
    config_path = tmp_path / "identity.yaml"
    image_path = tmp_path / "image.png"
    _identity_model(model_path)
    config_path.write_text("type: fixture\nname: identity\nmodel_path: identity.onnx\n", encoding="utf-8")
    Image.new("RGB", (8, 8), (64, 128, 192)).save(image_path)
    descriptor = ModelDescriptor(
        id="identity",
        name="identity",
        display_name="Identity",
        model_type="fixture",
        task="feature",
        family="fixture",
        adapter="onnx_raw",
        runtime=["ONNX Runtime"],
        config_path=config_path,
        weight_locations=["identity.onnx"],
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=False),
    )
    adapter = OnnxRuntimeAdapter(ModelRecord(descriptor=descriptor, config={"model_path": "identity.onnx"}), ArtifactStore(tmp_path / "artifacts"))

    layers = adapter.load(["CPUExecutionProvider"])
    result = adapter.predict(image_path, ["features"], {
        "feature_transform": {
            "projection": "mean",
            "normalization": "minmax",
            "spatial_scale": 2,
            "interpolation": "nearest",
        }
    })

    assert [layer.id for layer in layers] == ["features"]
    assert layers[0].shape == [1, 3, 8, 8]
    assert layers[0].spatial is True
    assert result.annotations == []
    assert len(result.artifacts) == 1
    assert result.artifacts[0].source_shape == [1, 3, 8, 8]
    assert result.artifacts[0].shape == [1, 1, 16, 16]
    assert result.artifacts[0].transform["projection"] == "mean"
    assert result.artifacts[0].path.is_file()
    assert result.timings_ms["inference"] >= 0


def test_real_onnx_graph_rewrite_enumerates_and_captures_an_intermediate_tensor(tmp_path: Path) -> None:
    model_path = tmp_path / "intermediate.onnx"
    config_path = tmp_path / "intermediate.yaml"
    image_path = tmp_path / "image.png"
    _intermediate_model(model_path)
    config_path.write_text("type: fixture\nname: intermediate\nmodel_path: intermediate.onnx\n", encoding="utf-8")
    Image.new("RGB", (8, 8), (64, 128, 192)).save(image_path)
    descriptor = ModelDescriptor(
        id="intermediate",
        name="intermediate",
        display_name="Intermediate",
        model_type="fixture",
        task="feature",
        family="fixture",
        adapter="onnx_raw",
        runtime=["ONNX Runtime"],
        config_path=config_path,
        weight_locations=["intermediate.onnx"],
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True),
    )
    adapter = OnnxRuntimeAdapter(ModelRecord(descriptor=descriptor, config={"model_path": "intermediate.onnx"}), ArtifactStore(tmp_path / "artifacts"))

    rewritten = onnx.load_from_string(instrument_onnx_outputs(model_path, ["backbone.hidden"]))
    assert [output.name for output in rewritten.graph.output] == ["features", "backbone.hidden"]

    layers = adapter.load(["CPUExecutionProvider"])
    result = adapter.predict(image_path, ["backbone.hidden"], {
        "feature_transform": {"projection": "mean", "normalization": "minmax"}
    })

    by_id = {layer.id: layer for layer in layers}
    assert set(by_id) == {"features", "backbone.hidden"}
    assert by_id["backbone.hidden"].group == "中间层 · Relu"
    assert by_id["backbone.hidden"].name == "backbone.relu"
    assert by_id["backbone.hidden"].shape == [1, 3, 8, 8]
    assert adapter.runtime_capture_mode is FeatureCaptureMode.GRAPH_REWRITE
    assert result.artifacts[0].layer_id == "backbone.hidden"
    assert result.artifacts[0].source_shape == [1, 3, 8, 8]
    assert result.artifacts[0].shape == [1, 1, 8, 8]
    assert result.artifacts[0].preview_available is True
