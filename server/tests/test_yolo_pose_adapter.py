from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.adapters.onnx import _ImageTransform
from labelone.models.adapters.yolo_pose import YoloPoseOnnxAdapter
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


def _adapter(tmp_path: Path, *, include_shape: bool = True) -> YoloPoseOnnxAdapter:
    descriptor = ModelDescriptor(
        id="yolo-pose",
        name="yolo-pose",
        display_name="YOLO Pose",
        model_type="yolov8_pose",
        task="pose",
        family="yolo",
        adapter="yolo_pose_onnx",
        runtime=["ONNX Runtime"],
        config_path=tmp_path / "yolo-pose.yaml",
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True),
    )
    config: dict[str, object] = {
        "classes": ["person"],
        "conf_threshold": 0.25,
        "iou_threshold": 0.5,
        "kpt_threshold": 0.25,
        "keypoint_names": ["nose", "left_eye"] + [f"joint_{index}" for index in range(2, 17)],
    }
    if include_shape:
        config["kpt_shape"] = [17, 3]
    return YoloPoseOnnxAdapter(ModelRecord(descriptor=descriptor, config=config), ArtifactStore(tmp_path / "artifacts"))


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


def _raw_pose_row(box: list[float], score: float, visible: list[tuple[int, float, float, float]]) -> np.ndarray:
    row = np.zeros(56, dtype=np.float32)
    row[:4] = box
    row[4] = score
    keypoints = row[5:].reshape(17, 3)
    for index, x, y, confidence in visible:
        keypoints[index] = [x, y, confidence]
    return row


def test_pose_channel_first_output_is_nms_filtered_and_returns_box_and_points(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    rows = np.stack([
        _raw_pose_row([320, 320, 200, 100], 0.90, [(0, 320, 320, 0.90), (1, 300, 300, 0.50)]),
        _raw_pose_row([322, 320, 200, 100], 0.80, [(0, 322, 320, 0.80)]),
        _raw_pose_row([100, 250, 50, 60], 0.85, [(0, 100, 250, 0.80)]),
    ])

    results = adapter._annotations({"output0": rows.T[np.newaxis, ...]}, _transform(), {})

    rectangles = [result for result in results if result.shape_type == "rectangle"]
    points = [result for result in results if result.shape_type == "point"]
    assert len(rectangles) == 2
    assert len(points) == 3
    assert rectangles[0].label == "person"
    assert rectangles[0].points == [[440.0, 220.0], [840.0, 420.0]]
    assert points[0].label == "person:nose"
    assert points[0].points == [[640.0, 320.0]]
    assert points[1].label == "person:left_eye"


def test_pose_shape_can_be_inferred_for_common_three_value_keypoints(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, include_shape=False)
    row = _raw_pose_row([320, 320, 100, 100], 0.9, [(0, 320, 320, 0.7)])

    results = adapter._annotations({"output0": row[np.newaxis, :]}, _transform(), {})

    assert [result.shape_type for result in results] == ["rectangle", "point"]


def test_x_anylabeling_class_to_keypoint_mapping_is_supported(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter.record.config["classes"] = {
        "person": ["nose", "left_eye"] + [f"joint_{index}" for index in range(2, 17)]
    }
    adapter.record.config.pop("keypoint_names", None)
    row = _raw_pose_row([320, 320, 100, 100], 0.9, [(0, 320, 320, 0.7)])

    results = adapter._annotations({"output0": row[np.newaxis, :]}, _transform(), {})

    assert results[0].label == "person"
    assert results[1].label == "person:nose"


def test_pose_objectness_output_multiplies_objectness_and_class_score(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    row = np.zeros(57, dtype=np.float32)
    row[:6] = [320, 320, 100, 100, 0.5, 0.8]
    row[6:].reshape(17, 3)[0] = [320, 320, 0.9]

    results = adapter._annotations({"output0": row[np.newaxis, :]}, _transform(), {})

    assert results[0].score == pytest.approx(0.4)
    assert results[1].shape_type == "point"


@pytest.mark.parametrize(
    "outputs, message",
    [
        ({}, "returned no outputs"),
        ({"output0": np.zeros((1, 10, 10, 2), dtype=np.float32)}, "output rank"),
        ({"output0": np.full((1, 56, 2), np.nan, dtype=np.float32)}, "finite numeric"),
        ({"output0": np.zeros((1, 55, 2), dtype=np.float32)}, "does not match"),
    ],
)
def test_pose_invalid_outputs_raise_clear_errors(
    tmp_path: Path,
    outputs: dict[str, np.ndarray],
    message: str,
) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ModelRuntimeError, match=message):
        adapter._annotations(outputs, _transform(), {})
