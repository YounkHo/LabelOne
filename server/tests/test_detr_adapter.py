from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.adapters.detr import DetrDetectionOnnxAdapter
from labelone.models.adapters.onnx import _ImageTransform
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


def _adapter(tmp_path: Path, config: dict[str, object] | None = None) -> DetrDetectionOnnxAdapter:
    descriptor = ModelDescriptor(
        id="rtdetr",
        name="rtdetr",
        display_name="RT-DETR",
        model_type="rtdetrv2",
        task="detection",
        family="rtdetr",
        adapter="detr_detection_onnx",
        runtime=["ONNX Runtime"],
        config_path=tmp_path / "rtdetr.yaml",
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True),
    )
    values: dict[str, object] = {"classes": ["scratch", "particle"], "conf_threshold": 0.25}
    values.update(config or {})
    return DetrDetectionOnnxAdapter(
        ModelRecord(descriptor=descriptor, config=values),
        ArtifactStore(tmp_path / "artifacts"),
    )


def _transform() -> _ImageTransform:
    return _ImageTransform(
        original_width=1280,
        original_height=640,
        input_width=640,
        input_height=640,
        scale=0.5,
        pad_x=0,
        pad_y=160,
    )


def _two_input_detr_model(path: Path) -> None:
    image = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])
    sizes = helper.make_tensor_value_info("orig_target_sizes", TensorProto.INT64, [1, 2])
    labels = helper.make_tensor_value_info("labels", TensorProto.INT64, [1, 1])
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 1, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1])
    nodes = [
        helper.make_node("Constant", [], ["labels"], value=helper.make_tensor("label_value", TensorProto.INT64, [1, 1], [0])),
        helper.make_node("Constant", [], ["boxes"], value=helper.make_tensor("box_value", TensorProto.FLOAT, [1, 1, 4], [1, 2, 6, 7])),
        helper.make_node("Constant", [], ["scores"], value=helper.make_tensor("score_value", TensorProto.FLOAT, [1, 1], [0.9])),
    ]
    graph = helper.make_graph(nodes, "two-input-detr", [image, sizes], [labels, boxes, scores])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, path)


def test_postprocessed_labels_boxes_scores_are_filtered_sorted_and_labeled(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    outputs = {
        "labels": np.array([[1, 0, 1]], dtype=np.int64),
        "boxes": np.array([[[100, 100, 300, 300], [10, 20, 50, 80], [0, 0, 20, 20]]], dtype=np.float32),
        "scores": np.array([[0.80, 0.95, 0.10]], dtype=np.float32),
    }

    results = adapter._annotations(outputs, _transform(), {"top_k": 2})

    assert [result.label for result in results] == ["scratch", "particle"]
    assert [result.score for result in results] == pytest.approx([0.95, 0.80])
    assert results[0].points == [[10.0, 20.0], [50.0, 80.0]]


def test_normalized_model_cxcywh_boxes_are_restored_through_letterbox(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    outputs = {
        "labels": np.array([0]),
        "boxes": np.array([[0.5, 0.5, 0.3125, 0.15625]], dtype=np.float32),
        "scores": np.array([0.9], dtype=np.float32),
    }

    results = adapter._annotations(
        outputs,
        _transform(),
        {"box_format": "cxcywh", "coordinate_space": "normalized_model"},
    )

    assert results[0].points == [[440.0, 220.0], [840.0, 420.0]]


def test_absolute_model_xyxy_boxes_are_restored_through_letterbox(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    outputs = {
        "labels": np.array([0]),
        "boxes": np.array([[220, 270, 420, 370]], dtype=np.float32),
        "scores": np.array([0.9], dtype=np.float32),
    }

    results = adapter._annotations(
        outputs,
        _transform(),
        {"box_format": "xyxy", "coordinate_space": "model"},
    )

    assert results[0].points == [[440.0, 220.0], [840.0, 420.0]]


def test_softmax_logits_exclude_background_and_use_query_best(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    outputs = {
        "pred_boxes": np.array(
            [[[0.25, 0.5, 0.2, 0.2], [0.75, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]],
            dtype=np.float32,
        ),
        "pred_logits": np.array(
            [[[4.0, 1.0, -1.0], [1.0, 5.0, -1.0], [0.0, 0.0, 5.0]]],
            dtype=np.float32,
        ),
    }

    results = adapter._annotations(
        outputs,
        _transform(),
        {"logits_activation": "softmax", "background_class": "last", "top_k": 2},
    )

    assert [result.label for result in results] == ["particle", "scratch"]
    assert results[0].score > results[1].score > 0.9
    np.testing.assert_allclose(results[0].points, [[832.0, 192.0], [1088.0, 448.0]], atol=1e-5)


def test_sigmoid_probabilities_support_global_top_k_across_query_and_class(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, {"coordinate_space": "original", "box_format": "xyxy"})
    outputs = {
        "boxes": np.array([[10, 10, 30, 30], [50, 50, 80, 80]], dtype=np.float32),
        "logits": np.array([[0.90, 0.80], [0.70, 0.60]], dtype=np.float32),
    }

    results = adapter._annotations(
        outputs,
        _transform(),
        {"logits_activation": "none", "selection_mode": "global_topk", "top_k": 2},
    )

    assert [result.label for result in results] == ["scratch", "particle"]
    assert results[0].points == results[1].points == [[10.0, 10.0], [30.0, 30.0]]


def test_explicit_output_name_mapping_allows_nonstandard_export_names(tmp_path: Path) -> None:
    adapter = _adapter(
        tmp_path,
        {
            "output_names": {
                "boxes": "detection_boxes",
                "labels": "detection_classes",
                "scores": "detection_scores",
            },
        },
    )
    outputs = {
        "detection_boxes": np.array([[1, 2, 20, 30]], dtype=np.float32),
        "detection_classes": np.array([1]),
        "detection_scores": np.array([0.75]),
    }

    results = adapter._annotations(outputs, _transform(), {})

    assert results[0].label == "particle"
    assert results[0].points == [[1.0, 2.0], [20.0, 30.0]]


@pytest.mark.parametrize(
    "outputs, parameters, message",
    [
        ({}, {}, "returned no outputs"),
        (
            {"output0": np.zeros((1, 10, 4)), "output1": np.zeros((1, 10, 2))},
            {},
            "identify the DETR boxes",
        ),
        (
            {"boxes": np.zeros((1, 10, 4)), "labels": np.zeros((1, 10))},
            {},
            "requires both labels and scores",
        ),
        (
            {
                "boxes": np.zeros((1, 10, 4)),
                "pred_boxes": np.zeros((1, 10, 4)),
                "logits": np.zeros((1, 10, 2)),
            },
            {},
            "Multiple DETR outputs match boxes",
        ),
        (
            {"boxes": np.zeros((1, 3, 4)), "logits": np.zeros((1, 2, 2))},
            {},
            "query count does not match",
        ),
        (
            {"boxes": np.full((1, 2, 4), np.nan), "logits": np.zeros((1, 2, 2))},
            {},
            "finite numeric",
        ),
        (
            {"labels": np.array([0]), "boxes": np.array([[0, 0, 1, 1]]), "scores": np.array([0.9])},
            {"coordinate_space": "mystery"},
            "coordinate_space",
        ),
    ],
)
def test_invalid_or_ambiguous_outputs_raise_clear_errors(
    tmp_path: Path,
    outputs: dict[str, np.ndarray],
    parameters: dict[str, object],
    message: str,
) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ModelRuntimeError, match=message):
        adapter._annotations(outputs, _transform(), parameters)


def test_explicit_class_ids_must_be_integral_and_in_range(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ModelRuntimeError, match="integer class IDs"):
        adapter._annotations(
            {"labels": np.array([0.5]), "boxes": np.array([[0, 0, 10, 10]]), "scores": np.array([0.9])},
            _transform(),
            {},
        )
    with pytest.raises(ModelRuntimeError, match="unknown class ID"):
        adapter._annotations(
            {"labels": np.array([4]), "boxes": np.array([[0, 0, 10, 10]]), "scores": np.array([0.9])},
            _transform(),
            {},
        )


def test_detr_two_input_contract_and_model_family_size_semantics(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    image_input = SimpleNamespace(name="images", shape=[1, 3, 640, 640], type="tensor(float)")
    size_input = SimpleNamespace(name="orig_target_sizes", shape=[1, 2], type="tensor(int64)")
    adapter._configure_inputs([image_input, size_input])
    transform = _transform()
    tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)

    rtdetr_feed = adapter._input_feed(tensor, transform)
    np.testing.assert_array_equal(rtdetr_feed["orig_target_sizes"], [[1280, 640]])
    assert adapter._image_resize_mode() == "stretch"

    adapter.record.descriptor.model_type = "dfine"
    dfine_feed = adapter._input_feed(tensor, transform)
    np.testing.assert_array_equal(dfine_feed["orig_target_sizes"], [[640, 640]])
    assert adapter._image_resize_mode() == "letterbox"


def test_real_two_input_detr_onnx_load_and_predict(tmp_path: Path) -> None:
    model_path = tmp_path / "rtdetr.onnx"
    image_path = tmp_path / "image.png"
    _two_input_detr_model(model_path)
    Image.new("RGB", (8, 8), "white").save(image_path)
    adapter = _adapter(tmp_path)
    adapter.record.descriptor.weight_locations = [str(model_path)]

    layers = adapter.load(["CPUExecutionProvider"])
    result = adapter.predict(image_path, ["boxes"], {})

    assert [layer.id for layer in layers] == ["labels", "boxes", "scores"]
    assert [layer.captureable for layer in layers] == [False, True, True]
    assert result.annotations[0].points == [[1.0, 2.0], [6.0, 7.0]]
    assert result.artifacts[0].source_shape == [1, 1, 4]
