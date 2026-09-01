from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.adapters.ppocr import PpOcrOnnxAdapter, _db_quads, _decode_ctc, _perspective_crop
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


class _Meta:
    def __init__(self, name: str, shape: list[int | str], type_name: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = type_name


class _Session:
    def __init__(self, input_shape: list[int | str], output_name: str, output_factory) -> None:
        self.input = _Meta("image", input_shape)
        self.output = _Meta(output_name, [])
        self.output_factory = output_factory
        self.feeds: list[np.ndarray] = []

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return [self.output]

    def run(self, _outputs, feed):
        tensor = feed["image"]
        self.feeds.append(tensor.copy())
        return [self.output_factory(tensor)]


def _record(tmp_path: Path, config: dict[str, object] | None = None) -> ModelRecord:
    descriptor = ModelDescriptor(
        id="ppocr",
        name="ppocr",
        display_name="PP-OCR",
        model_type="ppocr_v6",
        task="ocr",
        family="ppocr",
        adapter="ppocr_onnx",
        runtime=["ONNX Runtime"],
        config_path=tmp_path / "ppocr.yaml",
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True),
    )
    values: dict[str, object] = {
        "use_angle_cls": True,
        "use_space_char": False,
        "drop_score": 0.5,
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.5,
        "det_db_unclip_ratio": 0.0,
        "det_min_component_pixels": 3,
    }
    values.update(config or {})
    return ModelRecord(descriptor=descriptor, config=values)


def _adapter_with_fake_stages(tmp_path: Path) -> tuple[PpOcrOnnxAdapter, _Session, _Session, _Session, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("a\nb\n", encoding="utf-8")
    record = _record(tmp_path, {"rec_char_dict_path": "dict.txt"})
    adapter = PpOcrOnnxAdapter(record, ArtifactStore(tmp_path / "artifacts"))

    def detection(_tensor: np.ndarray) -> np.ndarray:
        result = np.zeros((1, 1, 32, 64), dtype=np.float32)
        result[:, :, 8:24, 10:50] = 0.95
        return result

    def classification(tensor: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[0.02, 0.98]], dtype=np.float32), (len(tensor), 1))

    def recognition(tensor: np.ndarray) -> np.ndarray:
        result = np.full((len(tensor), 5, 3), 0.02, dtype=np.float32)
        sequence = [0, 1, 1, 0, 2]
        for time, class_id in enumerate(sequence):
            result[:, time, class_id] = 0.96
        result /= result.sum(axis=2, keepdims=True)
        return result

    det = _Session([1, 3, 32, 64], "maps", detection)
    cls = _Session(["N", 3, 48, 192], "prob", classification)
    rec = _Session(["N", 3, 48, 320], "logits", recognition)
    from labelone.models.adapters.ppocr import _Stage
    adapter.detector = _Stage("det", det, det.input, [det.output])
    adapter.classifier = _Stage("cls", cls, cls.input, [cls.output])
    adapter.recognizer = _Stage("rec", rec, rec.input, [rec.output])
    adapter.characters = ["a", "b"]
    adapter.loaded = True
    image_path = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (220, 220, 220)).save(image_path)
    return adapter, det, cls, rec, image_path


def test_predict_runs_three_stages_emits_text_polygon_and_captures_outputs(tmp_path: Path) -> None:
    adapter, det, cls, rec, image_path = _adapter_with_fake_stages(tmp_path)

    result = adapter.predict(image_path, ["det:maps", "cls:prob", "rec:logits"], {})

    assert len(result.annotations) == 1
    annotation = result.annotations[0]
    assert annotation.label == "ab"
    assert annotation.shape_type == "polygon"
    assert len(annotation.points) == 4
    assert annotation.score > 0.9
    assert det.feeds[0].shape == (1, 3, 32, 64)
    assert cls.feeds[0].shape == (1, 3, 48, 192)
    assert rec.feeds[0].shape == (1, 3, 48, 320)
    assert np.all(np.isfinite(det.feeds[0]))
    assert -3 < float(det.feeds[0].min()) <= float(det.feeds[0].max()) < 3
    assert -1 <= float(cls.feeds[0].min()) <= float(cls.feeds[0].max()) <= 1
    assert -1 <= float(rec.feeds[0].min()) <= float(rec.feeds[0].max()) <= 1
    assert np.all(rec.feeds[0][:, :, :, -1] == 0)
    assert [artifact.layer_id for artifact in result.artifacts] == ["det:maps", "cls:prob", "rec:logits"]
    assert all(artifact.path.is_file() for artifact in result.artifacts)


def test_skip_angle_classifier_parameter_bypasses_optional_stage(tmp_path: Path) -> None:
    adapter, _, cls, rec, image_path = _adapter_with_fake_stages(tmp_path)

    result = adapter.predict(image_path, ["rec:logits"], {"skip_angle_cls": True})

    assert result.annotations[0].label == "ab"
    assert cls.feeds == []
    assert len(rec.feeds) == 1


def test_ctc_decode_removes_blank_and_duplicate_and_enforces_character_limit() -> None:
    probabilities = np.full((1, 7, 4), 0.01, dtype=np.float32)
    for time, class_id in enumerate([0, 1, 1, 0, 2, 3, 3]):
        probabilities[0, time, class_id] = 0.97
    probabilities /= probabilities.sum(axis=2, keepdims=True)

    assert _decode_ctc(probabilities, ["a", "b", "c"], 2)[0][0] == "ab"


def test_db_postprocess_returns_score_sorted_reading_order_quadrilaterals() -> None:
    probability = np.zeros((32, 64), dtype=np.float32)
    probability[3:10, 30:55] = 0.8
    probability[18:28, 5:35] = 0.95

    results = _db_quads(
        probability,
        original_width=128,
        original_height=64,
        threshold=0.3,
        box_threshold=0.5,
        unclip_ratio=0,
        maximum_candidates=10,
        maximum_boxes=10,
        minimum_pixels=3,
    )

    assert len(results) == 2
    assert results[0].points[0][1] < results[1].points[0][1]
    assert all(result.points.shape == (4, 2) for result in results)
    assert [result.score for result in results] == pytest.approx([0.8, 0.95])


def test_perspective_crop_rectifies_quad_and_rotates_tall_text() -> None:
    image = Image.new("RGB", (30, 50), "white")
    quad = np.array([[10, 2], [20, 2], [20, 45], [10, 45]], dtype=np.float64)

    crop = _perspective_crop(image, quad)

    assert crop.width > crop.height


def _identity_onnx(path: Path, input_shape: list[int], output_shape: list[int], output_name: str) -> None:
    input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, input_shape)
    output_info = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
    if input_shape == output_shape:
        nodes = [helper.make_node("Identity", ["image"], [output_name])]
    else:
        value = np.zeros(output_shape, dtype=np.float32)
        tensor = helper.make_tensor("constant", TensorProto.FLOAT, output_shape, value.ravel())
        nodes = [helper.make_node("Constant", [], [output_name], value=tensor)]
    graph = helper.make_graph(nodes, "fixture", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, path)


def test_load_resolves_three_models_and_relative_dictionary_and_lists_stage_layers(tmp_path: Path) -> None:
    _identity_onnx(tmp_path / "det.onnx", [1, 3, 32, 32], [1, 1, 32, 32], "maps")
    _identity_onnx(tmp_path / "cls.onnx", [1, 3, 48, 192], [1, 2], "prob")
    _identity_onnx(tmp_path / "rec.onnx", [1, 3, 48, 320], [1, 5, 3], "logits")
    (tmp_path / "dict.txt").write_text("a\nb\n", encoding="utf-8")
    record = _record(tmp_path, {
        "det_model_path": "det.onnx",
        "cls_model_path": "cls.onnx",
        "rec_model_path": "rec.onnx",
        "rec_char_dict_path": "dict.txt",
    })
    adapter = PpOcrOnnxAdapter(record, ArtifactStore(tmp_path / "artifacts"))

    layers = adapter.load(["CPUExecutionProvider"])

    assert [layer.id for layer in layers] == ["det:maps", "cls:prob", "rec:logits"]
    assert adapter.characters == ["a", "b"]
    adapter.unload()
    assert adapter.loaded is False


@pytest.mark.parametrize(
    "action, message",
    [
        (lambda adapter, image: adapter.predict(image, ["missing"], {}), "capture layer"),
        (lambda adapter, image: adapter.predict(image, ["cls:prob"], {"skip_angle_cls": True}), "Cannot capture"),
        (lambda adapter, image: adapter.predict(image, [], {"max_image_pixels": 10}), "pixel budget"),
        (lambda adapter, image: adapter.predict(image, [], {"max_crop_pixels": 2}), "text crops"),
    ],
)
def test_predict_budgets_and_capture_errors_are_explicit(tmp_path: Path, action, message: str) -> None:
    adapter, _, _, _, image_path = _adapter_with_fake_stages(tmp_path)

    with pytest.raises(ModelRuntimeError, match=message):
        action(adapter, image_path)


def test_ambiguous_detection_outputs_and_dictionary_mismatch_are_rejected(tmp_path: Path) -> None:
    adapter, det, _, rec, image_path = _adapter_with_fake_stages(tmp_path)
    second = _Meta("other", [1, 1, 32, 64])
    det.output_meta = [det.output, second]
    adapter.detector.output_meta = [det.output, second]
    original_run = det.output_factory
    det.run = lambda _outputs, feed: [original_run(feed["image"]), original_run(feed["image"])]  # type: ignore[method-assign]

    with pytest.raises(ModelRuntimeError, match="ambiguous"):
        adapter.predict(image_path, [], {})

    adapter, _, _, rec, image_path = _adapter_with_fake_stages(tmp_path / "other")
    rec.output_factory = lambda tensor: np.zeros((len(tensor), 4, 4), dtype=np.float32)
    with pytest.raises(ModelRuntimeError, match="dictionary"):
        adapter.predict(image_path, [], {"skip_angle_cls": True})


def test_dictionary_path_cannot_escape_imported_model_source(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    record = _record(model_dir, {"rec_char_dict_path": str(outside)})
    adapter = PpOcrOnnxAdapter(record, ArtifactStore(tmp_path / "artifacts"))

    with pytest.raises(ModelRuntimeError, match="dictionary is missing"):
        adapter._dictionary_path()
