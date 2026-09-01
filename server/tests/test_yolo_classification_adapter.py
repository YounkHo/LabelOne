from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.adapters.yolo_classification import YoloClassificationOnnxAdapter
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


def _adapter(tmp_path: Path, classes: list[str] | None = None) -> YoloClassificationOnnxAdapter:
    descriptor = ModelDescriptor(
        id="yolo-classification",
        name="yolo-classification",
        display_name="YOLO Classification",
        model_type="yolov8_cls",
        task="classification",
        family="yolo",
        adapter="yolo_classification_onnx",
        runtime=["ONNX Runtime"],
        config_path=tmp_path / "yolo-classification.yaml",
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True),
    )
    config = {"classes": classes if classes is not None else ["cat", "dog", "bird"]}
    return YoloClassificationOnnxAdapter(
        ModelRecord(descriptor=descriptor, config=config),
        ArtifactStore(tmp_path / "artifacts"),
    )


def test_classification_probability_output_returns_ranked_top_k(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    results = adapter._classifications(
        {"probabilities": np.array([[0.1, 0.7, 0.2]], dtype=np.float32)},
        {"top_k": 2},
    )

    assert [result.label for result in results] == ["dog", "bird"]
    assert [result.score for result in results] == pytest.approx([0.7, 0.2])
    assert [result.rank for result in results] == [1, 2]


def test_classification_logits_are_softmaxed_and_numeric_labels_are_available(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, classes=[])

    results = adapter._classifications(
        {"scores": np.array([1.0, 3.0, 2.0], dtype=np.float32)},
        {"top_k": 3},
    )

    assert [result.label for result in results] == ["1", "2", "0"]
    assert sum(result.score for result in results) == pytest.approx(1.0)


def test_classification_can_select_one_output_by_name(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    results = adapter._classifications(
        {
            "embedding": np.zeros((1, 4, 4), dtype=np.float32),
            "probabilities": np.array([[0.1, 0.7, 0.2]], dtype=np.float32),
        },
        {"output_name": "probabilities", "top_k": 1},
    )

    assert [result.label for result in results] == ["dog"]


@pytest.mark.parametrize(
    "outputs, parameters, message",
    [
        ({}, {}, "returned no outputs"),
        ({"scores": np.zeros((1, 2, 3), dtype=np.float32)}, {}, "must have shape"),
        ({"scores": np.array([0.2, np.nan, 0.8])}, {}, "finite numeric"),
        (
            {"first": np.array([0.2, 0.3, 0.5]), "second": np.array([0.1, 0.7, 0.2])},
            {},
            "multiple score vectors",
        ),
        ({"scores": np.array([0.2, 0.3, 0.5])}, {"top_k": 0}, "positive integer"),
        ({"scores": np.array([0.2, 0.3, 0.5])}, {"top_k": 1.5}, "positive integer"),
        ({"scores": np.array([1.0, 2.0, 3.0])}, {"apply_softmax": False}, "not probabilities"),
    ],
)
def test_classification_invalid_outputs_and_parameters_raise_clear_errors(
    tmp_path: Path,
    outputs: dict[str, np.ndarray],
    parameters: dict[str, object],
    message: str,
) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ModelRuntimeError, match=message):
        adapter._classifications(outputs, parameters)


def test_classification_rejects_class_count_mismatch(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, classes=["cat", "dog"])

    with pytest.raises(ModelRuntimeError, match="class count"):
        adapter._classifications({"scores": np.array([0.2, 0.3, 0.5])}, {})
