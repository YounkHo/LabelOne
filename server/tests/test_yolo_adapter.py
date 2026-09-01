from __future__ import annotations

from pathlib import Path

import numpy as np

from labelone.models.adapters.onnx import YoloDetectionOnnxAdapter, _ImageTransform
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


def _adapter(tmp_path: Path) -> YoloDetectionOnnxAdapter:
    descriptor = ModelDescriptor(
        id="yolo",
        name="yolo",
        display_name="YOLO",
        model_type="yolov8",
        task="detection",
        family="yolo",
        adapter="yolo_detection_onnx",
        runtime=["ONNX Runtime"],
        config_path=tmp_path / "yolo.yaml",
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True),
    )
    record = ModelRecord(descriptor=descriptor, config={"classes": ["scratch", "particle"], "conf_threshold": 0.25, "iou_threshold": 0.5})
    return YoloDetectionOnnxAdapter(record, ArtifactStore(tmp_path / "artifacts"))


def test_yolov8_output_is_filtered_nms_and_mapped_to_original_image(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    output = np.array([
        [320, 320, 200, 100, 0.90, 0.10],
        [322, 320, 200, 100, 0.80, 0.20],
        [100, 100, 50, 60, 0.05, 0.92],
    ], dtype=np.float32)
    transform = _ImageTransform(
        original_width=1280,
        original_height=640,
        input_width=640,
        input_height=640,
        scale=0.5,
        pad_x=0,
        pad_y=160,
    )

    results = adapter._annotations({"output0": output}, transform, {})

    assert len(results) == 2
    assert results[0].label == "scratch"
    assert results[0].score > 0.89
    assert results[0].points[0][0] == 440.0
    assert results[0].points[0][1] == 220.0
    assert results[1].label == "particle"

