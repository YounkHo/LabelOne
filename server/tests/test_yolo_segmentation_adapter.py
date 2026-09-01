from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.adapters.onnx import _ImageTransform
from labelone.models.adapters.yolo_segmentation import YoloSegmentationOnnxAdapter
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


def _adapter(tmp_path: Path) -> YoloSegmentationOnnxAdapter:
    descriptor = ModelDescriptor(
        id="yolo-seg",
        name="yolo-seg",
        display_name="YOLO Segmentation",
        model_type="yolov11_seg",
        task="segmentation",
        family="yolo",
        adapter="yolo_segmentation_onnx",
        runtime=["ONNX Runtime"],
        config_path=tmp_path / "yolo-seg.yaml",
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True),
    )
    record = ModelRecord(
        descriptor=descriptor,
        config={
            "classes": ["scratch", "particle"],
            "conf_threshold": 0.25,
            "iou_threshold": 0.4,
            "mask_threshold": 0.5,
        },
    )
    return YoloSegmentationOnnxAdapter(record, ArtifactStore(tmp_path / "artifacts"))


def _identity_transform(size: int = 12) -> _ImageTransform:
    return _ImageTransform(
        original_width=size,
        original_height=size,
        input_width=size,
        input_height=size,
        scale=1.0,
        pad_x=0,
        pad_y=0,
    )


def _polygon_area(points: list[list[float]]) -> float:
    array = np.asarray(points)
    return abs(float(np.dot(array[:, 0], np.roll(array[:, 1], -1)) - np.dot(array[:, 1], np.roll(array[:, 0], -1)))) * 0.5


def test_channels_first_detection_restores_letterbox_and_emits_concave_polygon(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prototype = np.full((1, 1, 8, 8), -10.0, dtype=np.float32)
    prototype[0, 0, 2:6, 2:4] = 10.0
    prototype[0, 0, 4:6, 2:7] = 10.0
    # cx, cy, w, h, two class scores, one mask coefficient.
    detection = np.array([[[4.5], [4.0], [5.0], [4.0], [0.9], [0.1], [1.0]]], dtype=np.float32)
    transform = _ImageTransform(
        original_width=16,
        original_height=8,
        input_width=8,
        input_height=8,
        scale=0.5,
        pad_x=0,
        pad_y=2,
    )

    results = adapter._annotations(
        {"detections": detection, "prototypes": prototype},
        transform,
        {"polygon_simplify": 0, "min_mask_area": 1},
    )

    assert len(results) == 1
    result = results[0]
    assert result.label == "scratch"
    assert result.shape_type == "polygon"
    assert len(result.points) >= 6
    points = np.asarray(result.points)
    assert points[:, 0].min() == pytest.approx(4.0)
    assert points[:, 0].max() == pytest.approx(14.0)
    assert points[:, 1].min() == pytest.approx(0.0)
    assert points[:, 1].max() == pytest.approx(8.0)
    bounding_area = float(np.ptp(points[:, 0]) * np.ptp(points[:, 1]))
    assert _polygon_area(result.points) < bounding_area * 0.8


def test_channels_last_detection_applies_class_nms_and_preserves_components(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prototypes = np.full((1, 2, 12, 12), -10.0, dtype=np.float32)
    prototypes[0, 0, 1:4, 1:4] = 10.0
    prototypes[0, 0, 7:10, 7:10] = 10.0
    prototypes[0, 1, 2:9, 2:9] = 10.0
    rows = np.array(
        [
            [6, 6, 12, 12, 0.90, 0.10, 1.0, 0.0],
            [6.1, 6, 12, 12, 0.80, 0.20, 1.0, 0.0],
            [6, 6, 12, 12, 0.10, 0.85, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    results = adapter._annotations(
        {"output0": rows[None, ...], "output1": prototypes},
        _identity_transform(),
        {"polygon_simplify": 0, "min_mask_area": 1},
    )

    assert [result.label for result in results].count("scratch") == 2
    assert [result.label for result in results].count("particle") == 1
    assert all(result.shape_type == "polygon" for result in results)
    assert all(result.score != pytest.approx(0.80) for result in results)


def test_component_and_point_budgets_bound_noisy_masks(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prototype = np.full((1, 1, 16, 16), -10.0, dtype=np.float32)
    prototype[0, 0, ::2, ::2] = 10.0
    detection = np.array([[[8], [8], [16], [16], [0.9], [0.1], [1.0]]], dtype=np.float32)

    results = adapter._annotations(
        {"detections": detection, "prototypes": prototype},
        _identity_transform(16),
        {
            "max_mask_components": 2,
            "max_polygon_points": 4,
            "max_total_polygon_points": 8,
            "min_mask_area": 0,
            "polygon_simplify": 0,
        },
    )

    assert len(results) == 2
    assert sum(len(result.points) for result in results) == 8


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        (
            {"detections": np.zeros((1, 8, 1), dtype=np.float32)},
            "No supported YOLO segmentation prototype output was found",
        ),
        (
            {
                "detections": np.zeros((1, 7, 1), dtype=np.float32),
                "prototypes": np.zeros((1, 2, 8, 8), dtype=np.float32),
            },
            "No supported YOLO segmentation detection output was found",
        ),
        (
            {
                "detections": np.zeros((1, 7, 1), dtype=np.float32),
                "prototypes": np.zeros((2, 1, 8, 8), dtype=np.float32),
            },
            "No supported YOLO segmentation prototype output was found",
        ),
        (
            {
                "detections": np.zeros((1, 7, 1), dtype=np.float32),
                "prototype_a": np.zeros((1, 1, 8, 8), dtype=np.float32),
                "prototype_b": np.zeros((1, 1, 4, 4), dtype=np.float32),
            },
            "Multiple YOLO segmentation prototype outputs are ambiguous",
        ),
    ],
)
def test_invalid_output_shapes_raise_clear_runtime_errors(
    tmp_path: Path,
    outputs: dict[str, np.ndarray],
    message: str,
) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ModelRuntimeError) as caught:
        adapter._annotations(outputs, _identity_transform(), {})

    assert caught.value.message == message


def test_prototype_value_budget_and_explicit_output_names_are_enforced(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prototypes = np.zeros((1, 2, 32, 32), dtype=np.float32)
    detections = np.zeros((1, 8, 1), dtype=np.float32)

    with pytest.raises(ModelRuntimeError, match="prototype output exceeds the value budget"):
        adapter._annotations(
            {"det": detections, "proto": prototypes},
            _identity_transform(),
            {
                "detection_output_name": "det",
                "prototype_output_name": "proto",
                "prototype_max_values": 1024,
            },
        )
