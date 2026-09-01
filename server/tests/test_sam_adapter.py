from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.adapters.sam import SegmentAnythingOnnxAdapter, _prepare_prompts
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


def _meta(name: str, shape: list[int | str], data_type: str = "tensor(float)") -> SimpleNamespace:
    return SimpleNamespace(name=name, shape=shape, type=data_type)


def _adapter(tmp_path: Path, *, model_type: str = "segment_anything") -> SegmentAnythingOnnxAdapter:
    descriptor = ModelDescriptor(
        id="sam",
        name="sam",
        display_name="Segment Anything",
        model_type=model_type,
        task="interactive_segmentation",
        family="sam",
        adapter="segment_anything_onnx",
        runtime=["ONNX Runtime"],
        config_path=tmp_path / "sam.yaml",
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True),
    )
    record = ModelRecord(
        descriptor=descriptor,
        config={"encoder_model_path": "encoder.onnx", "decoder_model_path": "decoder.onnx"},
    )
    return SegmentAnythingOnnxAdapter(record, ArtifactStore(tmp_path / "artifacts"))


def _decoder_inputs(*, hq: bool = False) -> list[SimpleNamespace]:
    inputs = [
        _meta("image_embeddings", [1, 4, 2, 2]),
        _meta("point_coords", [1, "P", 2]),
        _meta("point_labels", [1, "P"]),
        _meta("mask_input", [1, 1, 4, 4]),
        _meta("has_mask_input", [1]),
        _meta("orig_im_size", [2]),
    ]
    if hq:
        inputs.insert(1, _meta("interm_embeddings", [2, 1, 1, 2, 2]))
    return inputs


class _FakeSession:
    def __init__(self, inputs, outputs, values) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self.values = values
        self.feed: dict[str, np.ndarray] | None = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, _requested, feed):
        self.feed = feed
        return self.values


def _attach_standard_sessions(adapter: SegmentAnythingOnnxAdapter, *, size: int = 8) -> tuple[_FakeSession, _FakeSession]:
    embedding = np.ones((1, 4, 2, 2), dtype=np.float32)
    masks = np.full((1, 3, size, size), -4.0, dtype=np.float32)
    masks[0, 1, 1:7, 1:3] = 4.0
    masks[0, 1, 5:7, 1:7] = 4.0
    scores = np.asarray([[0.2, 0.9, 0.4]], dtype=np.float32)
    low_res = np.zeros((1, 3, 2, 2), dtype=np.float32)
    encoder = _FakeSession(
        [_meta("input_image", [1, 3, size, size])],
        [_meta("image_embeddings", [1, 4, 2, 2])],
        [embedding],
    )
    decoder = _FakeSession(
        _decoder_inputs(),
        [
            _meta("masks", [1, 3, size, size]),
            _meta("iou_predictions", [1, 3]),
            _meta("low_res_masks", [1, 3, 2, 2]),
        ],
        [masks, scores, low_res],
    )
    adapter.encoder_session = encoder
    adapter.decoder_session = decoder
    adapter.encoder_input_meta = encoder.get_inputs()[0]
    adapter.encoder_output_meta = encoder.get_outputs()
    adapter.decoder_input_meta = decoder.get_inputs()
    adapter.decoder_output_meta = decoder.get_outputs()
    adapter.loaded = True
    return encoder, decoder


def _polygon_area(points: list[list[float]]) -> float:
    array = np.asarray(points)
    return abs(float(np.dot(array[:, 0], np.roll(array[:, 1], -1)) - np.dot(array[:, 1], np.roll(array[:, 0], -1)))) * 0.5


def test_loads_two_sessions_lists_namespaced_outputs_and_unloads(tmp_path: Path, monkeypatch) -> None:
    adapter = _adapter(tmp_path)
    (tmp_path / "encoder.onnx").write_bytes(b"encoder")
    (tmp_path / "decoder.onnx").write_bytes(b"decoder")
    encoder = _FakeSession(
        [_meta("input_image", [1, 3, 8, 8])],
        [_meta("image_embeddings", [1, 4, 2, 2])],
        [np.ones((1, 4, 2, 2), dtype=np.float32)],
    )
    decoder = _FakeSession(
        _decoder_inputs(),
        [_meta("masks", [1, 3, 8, 8]), _meta("iou_predictions", [1, 3])],
        [],
    )

    class _FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(path, providers):
            assert providers == ["CPUExecutionProvider"]
            return encoder if Path(path).name == "encoder.onnx" else decoder

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt)

    layers = adapter.load(["MissingProvider"])

    assert [layer.id for layer in layers] == [
        "encoder:image_embeddings",
        "decoder:masks",
        "decoder:iou_predictions",
    ]
    assert adapter.loaded is True
    adapter.unload()
    assert adapter.loaded is False
    assert adapter.encoder_session is None
    assert adapter.decoder_session is None


def test_predict_maps_prompts_selects_best_mask_and_emits_concave_polygon_raster_and_captures(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    encoder, decoder = _attach_standard_sessions(adapter)
    image_path = tmp_path / "source.png"
    Image.new("RGB", (8, 8), (123, 116, 104)).save(image_path)

    result = adapter.predict(
        image_path,
        ["encoder:image_embeddings", "decoder:masks"],
        {
            "points": [{"x": 4, "y": 3, "label": 1}],
            "boxes": [[1, 1, 7, 7]],
            "label": "object",
            "polygon_simplify": 0,
        },
    )

    assert len(result.rasters) == 1
    assert result.rasters[0].role == "sam-mask"
    assert result.rasters[0].metadata["selected_mask"] == 1
    assert result.rasters[0].metadata["predicted_iou"] == pytest.approx(0.9)
    assert Image.open(result.rasters[0].path).mode == "L"
    assert len(result.annotations) == 1
    polygon = result.annotations[0]
    assert polygon.label == "object"
    assert polygon.shape_type == "polygon"
    assert len(polygon.points) >= 6
    points = np.asarray(polygon.points)
    bounding_area = float(np.ptp(points[:, 0]) * np.ptp(points[:, 1]))
    assert _polygon_area(polygon.points) < bounding_area * 0.8
    assert len(result.artifacts) == 2
    assert all(artifact.path.is_file() for artifact in result.artifacts)
    assert encoder.feed is not None and encoder.feed["input_image"].shape == (1, 3, 8, 8)
    assert decoder.feed is not None
    np.testing.assert_allclose(decoder.feed["point_coords"], [[[4, 3], [1, 1], [7, 7], [0, 0]]])
    np.testing.assert_allclose(decoder.feed["point_labels"], [[1, 2, 3, -1]])
    np.testing.assert_allclose(decoder.feed["orig_im_size"], [8, 8])


def test_standard_longest_side_resize_normalization_and_bottom_right_padding(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    encoder, _ = _attach_standard_sessions(adapter)
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (4, 2), (123, 116, 104)).save(image_path)

    tensor, transform = adapter._preprocess_image(image_path, max_input_pixels=64)

    assert tensor.shape == (1, 3, 8, 8)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert (transform.resized_width, transform.resized_height) == (8, 4)
    assert transform.scale == 2.0
    expected = np.asarray(
        [(123 - 123.675) / 58.395, (116 - 116.28) / 57.12, (104 - 103.53) / 57.375],
        dtype=np.float32,
    )
    np.testing.assert_allclose(tensor[0, :, 0, 0], expected, atol=1e-6)
    assert np.all(tensor[0, :, 4:, :] == 0)
    assert encoder.feed is None


def test_prompt_coordinates_use_the_rounded_longest_side_dimensions(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    _attach_standard_sessions(adapter)
    image_path = tmp_path / "uneven.png"
    Image.new("RGB", (7, 5), "white").save(image_path)
    _, transform = adapter._preprocess_image(image_path, max_input_pixels=64)

    coords, labels, _ = _prepare_prompts(
        {"points": [{"x": 3.5, "y": 2.5, "label": 1}]},
        transform,
        maximum=8,
    )

    assert (transform.resized_width, transform.resized_height) == (8, 6)
    np.testing.assert_allclose(coords[0, 0], [3.5 * 8 / 7, 2.5 * 6 / 5])
    np.testing.assert_allclose(labels, [[1, -1]])


def test_low_resolution_decoder_mask_is_unpadded_and_restored_to_original_size(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    embedding = np.ones((1, 4, 2, 2), dtype=np.float32)
    low_mask = np.full((1, 1, 2, 2), 4.0, dtype=np.float32)
    encoder = _FakeSession([_meta("input_image", [1, 3, 8, 8])], [_meta("image_embeddings", [1, 4, 2, 2])], [embedding])
    decoder = _FakeSession(_decoder_inputs(), [_meta("masks", [1, 1, 2, 2])], [low_mask])
    adapter.encoder_session = encoder
    adapter.decoder_session = decoder
    adapter.encoder_input_meta = encoder.get_inputs()[0]
    adapter.encoder_output_meta = encoder.get_outputs()
    adapter.decoder_input_meta = decoder.get_inputs()
    adapter.decoder_output_meta = decoder.get_outputs()
    adapter.loaded = True
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (8, 4), "white").save(image_path)

    result = adapter.predict(image_path, [], {"points": [{"x": 2, "y": 1, "label": 1}]})

    raster = Image.open(result.rasters[0].path)
    assert raster.size == (8, 4)
    assert np.all(np.asarray(raster) == 255)
    assert result.rasters[0].metadata["resized_size"] == [8, 4]


def test_sam_hq_stacks_encoder_intermediate_outputs_for_decoder(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, model_type="sam_hq")
    embedding = np.ones((1, 4, 2, 2), dtype=np.float32)
    intermediate_a = np.ones((1, 1, 2, 2), dtype=np.float32)
    intermediate_b = np.full((1, 1, 2, 2), 2.0, dtype=np.float32)
    mask = np.ones((1, 1, 4, 4), dtype=np.float32)
    encoder = _FakeSession(
        [_meta("input_image", [1, 3, 4, 4])],
        [
            _meta("image_embeddings", [1, 4, 2, 2]),
            _meta("intermediate_0", [1, 1, 2, 2]),
            _meta("intermediate_1", [1, 1, 2, 2]),
        ],
        [embedding, intermediate_a, intermediate_b],
    )
    decoder = _FakeSession(_decoder_inputs(hq=True), [_meta("masks", [1, 1, 4, 4])], [mask])
    adapter.encoder_session = encoder
    adapter.decoder_session = decoder
    adapter.encoder_input_meta = encoder.get_inputs()[0]
    adapter.encoder_output_meta = encoder.get_outputs()
    adapter.decoder_input_meta = decoder.get_inputs()
    adapter.decoder_output_meta = decoder.get_outputs()
    adapter.loaded = True
    image_path = tmp_path / "source.png"
    Image.new("RGB", (4, 4), "white").save(image_path)

    adapter.predict(image_path, [], {"boxes": [[0, 0, 4, 4]]})

    assert decoder.feed is not None
    assert decoder.feed["interm_embeddings"].shape == (2, 1, 1, 2, 2)
    np.testing.assert_allclose(decoder.feed["interm_embeddings"][0], intermediate_a)
    np.testing.assert_allclose(decoder.feed["interm_embeddings"][1], intermediate_b)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({}, "at least one point or box"),
        ({"points": [{"x": 1, "y": 1, "label": 2}]}, "label must be 0 or 1"),
        ({"points": [{"x": 20, "y": 1, "label": 1}]}, "outside the image"),
        ({"boxes": [[2, 2, 1, 3]]}, "box is invalid"),
        ({"boxes": [[0, 0, 8]]}, "must contain"),
    ],
)
def test_prompt_validation_is_explicit(tmp_path: Path, parameters: dict[str, object], message: str) -> None:
    adapter = _adapter(tmp_path)
    _attach_standard_sessions(adapter)
    image_path = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    with pytest.raises(ModelRuntimeError, match=message):
        adapter.predict(image_path, [], parameters)


def test_capture_prompt_and_output_budgets_and_unknown_layers_are_rejected(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    _attach_standard_sessions(adapter)
    image_path = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    prompt = {"points": [{"x": 1, "y": 1, "label": 1}]}

    with pytest.raises(ModelRuntimeError, match="capture output is unavailable"):
        adapter.predict(image_path, ["encoder:missing"], prompt)
    with pytest.raises(ModelRuntimeError, match="input exceeds the pixel budget"):
        adapter.predict(image_path, [], {**prompt, "max_input_pixels": 16})
    with pytest.raises(ModelRuntimeError, match="prompt count exceeds"):
        adapter.predict(
            image_path,
            [],
            {
                "points": [
                    {"x": 1, "y": 1, "label": 1},
                    {"x": 2, "y": 2, "label": 0},
                ],
                "max_prompt_elements": 1,
            },
        )
    with pytest.raises(ModelRuntimeError, match="output pixel budget"):
        adapter.predict(image_path, [], {**prompt, "max_output_pixels": 16})
    with pytest.raises(ModelRuntimeError, match="embedding exceeds the value budget"):
        adapter.predict(image_path, [], {**prompt, "max_capture_values": 4})


def test_unsupported_family_missing_weights_and_decoder_signature_fail_cleanly(tmp_path: Path, monkeypatch) -> None:
    unsupported = _adapter(tmp_path, model_type="edge_sam")
    with pytest.raises(ModelRuntimeError, match="supports only"):
        unsupported.load(["CPUExecutionProvider"])

    standard = _adapter(tmp_path)
    with pytest.raises(ModelRuntimeError, match="does not resolve"):
        standard.load(["CPUExecutionProvider"])

    (tmp_path / "encoder.onnx").write_bytes(b"encoder")
    (tmp_path / "decoder.onnx").write_bytes(b"decoder")
    encoder = _FakeSession([_meta("input_image", [1, 3, 8, 8])], [_meta("image_embeddings", [1, 4, 2, 2])], [])
    bad_decoder = _FakeSession([_meta("mystery", [1])], [_meta("masks", [1, 1, 8, 8])], [])

    class _FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(path, providers):
            return encoder if Path(path).name == "encoder.onnx" else bad_decoder

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt)
    with pytest.raises(ModelRuntimeError, match="input signature is unsupported"):
        standard.load(["CPUExecutionProvider"])
