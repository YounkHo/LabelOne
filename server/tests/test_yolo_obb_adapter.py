from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.adapters.onnx import _ImageTransform
from labelone.models.adapters.yolo_obb import YoloObbOnnxAdapter
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


def _adapter(tmp_path: Path, *, classes: list[str] | None = None) -> YoloObbOnnxAdapter:
    descriptor = ModelDescriptor(
        id="yolo-obb",
        name="yolo-obb",
        display_name="YOLO OBB",
        model_type="yolov11_obb",
        task="rotated_detection",
        family="yolo",
        adapter="yolo_obb_onnx",
        runtime=["ONNX Runtime"],
        config_path=tmp_path / "yolo-obb.yaml",
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True),
    )
    record = ModelRecord(
        descriptor=descriptor,
        config={
            "classes": classes or ["scratch", "particle"],
            "conf_threshold": 0.25,
            "iou_threshold": 0.3,
            "max_det": 100,
        },
    )
    return YoloObbOnnxAdapter(record, ArtifactStore(tmp_path / "artifacts"))


def _letterbox_transform() -> _ImageTransform:
    return _ImageTransform(
        original_width=1280,
        original_height=640,
        input_width=640,
        input_height=640,
        scale=0.5,
        pad_x=0,
        pad_y=160,
    )


def test_ultralytics_v8_v11_raw_channels_first_maps_rotation_and_class_nms(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    # Raw Ultralytics OBB rows are cx, cy, w, h, class scores..., angle (radians).
    rows = np.array(
        [
            [320, 320, 200, 100, 0.90, 0.10, 0.0],
            [322, 320, 200, 100, 0.80, 0.20, 0.0],
            [320, 320, 200, 100, 0.10, 0.85, 0.0],
            [100, 100, 40, 20, 0.10, 0.10, 0.0],
        ],
        dtype=np.float32,
    )

    results = adapter._annotations({"output0": rows.T[None, ...]}, _letterbox_transform(), {})

    assert [result.label for result in results] == ["scratch", "particle"]
    assert all(result.shape_type == "rotation" for result in results)
    assert results[0].score == pytest.approx(0.90)
    np.testing.assert_allclose(
        results[0].points,
        [[440.0, 220.0], [840.0, 220.0], [840.0, 420.0], [440.0, 420.0]],
        atol=1e-5,
    )


def test_ultralytics_end_to_end_xywhr_conf_class_uses_rotated_iou(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, classes=["a", "b", "c"])
    transform = _ImageTransform(
        original_width=640,
        original_height=640,
        input_width=640,
        input_height=640,
        scale=1.0,
        pad_x=0,
        pad_y=0,
    )
    # End-to-end Ultralytics OBB rows are cx, cy, w, h, angle, confidence, class.
    # The first two thin boxes have identical AABBs but low rotated IoU, so both survive.
    rows = np.array(
        [
            [320, 320, 120, 20, pi / 4, 0.92, 1],
            [320, 320, 120, 20, -pi / 4, 0.90, 1],
            [321, 320, 120, 20, pi / 4, 0.80, 1],
        ],
        dtype=np.float32,
    )

    results = adapter._annotations({"output0": rows[None, ...]}, transform, {})

    assert len(results) == 2
    assert [result.label for result in results] == ["b", "b"]
    assert [result.score for result in results] == pytest.approx([0.92, 0.90])
    assert all(len(result.points) == 4 for result in results)


def test_angle_last_end_to_end_and_legacy_objectness_layouts_are_supported(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    transform = _ImageTransform(
        original_width=640,
        original_height=640,
        input_width=640,
        input_height=640,
        scale=1.0,
        pad_x=0,
        pad_y=0,
    )
    angle_last = np.array([[[100, 80, 40, 20, 0.88, 0, pi / 2]]], dtype=np.float32)
    end_to_end = adapter._annotations(
        {"output0": angle_last},
        transform,
        {"output_layout": "end_to_end_angle_last"},
    )

    objectness = np.array(
        [[[100], [80], [40], [20], [0.5], [0.8], [0.2], [0.0]]],
        dtype=np.float32,
    )
    legacy = adapter._annotations({"output0": objectness}, transform, {})

    assert end_to_end[0].score == pytest.approx(0.88)
    np.testing.assert_allclose(
        end_to_end[0].points,
        [[110.0, 60.0], [110.0, 100.0], [90.0, 100.0], [90.0, 60.0]],
        atol=1e-5,
    )
    assert legacy[0].score == pytest.approx(0.4)
    assert legacy[0].label == "scratch"


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((1, 2, 3, 4), dtype=np.float32),
        np.zeros((1, 4, 9), dtype=np.float32),
        np.zeros((2, 7, 10), dtype=np.float32),
    ],
)
def test_malformed_obb_shapes_raise_clear_runtime_errors(tmp_path: Path, output: np.ndarray) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ModelRuntimeError) as caught:
        adapter._annotations({"output0": output}, _letterbox_transform(), {})

    assert caught.value.message == "No supported YOLO OBB detection output was found"
    assert caught.value.details["outputs"]["output0"] == list(output.shape)
    assert "output0" in caught.value.details["failures"]


def test_ambiguous_or_malformed_end_to_end_rows_require_an_explicit_layout(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, classes=["a", "b", "c"])
    malformed = np.array(
        [[[100, 100, 20, 10, 0.1, 0.8, 1.25], [120, 120, 20, 10, 0.2, 0.7, 1.50]]],
        dtype=np.float32,
    )

    with pytest.raises(ModelRuntimeError, match="Could not infer YOLO OBB output layout"):
        adapter._annotations({"output0": malformed}, _letterbox_transform(), {})
