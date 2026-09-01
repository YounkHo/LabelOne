from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.adapters.depth import DepthAnythingOnnxAdapter
from labelone.models.adapters.onnx import _ImageTransform
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


def _adapter(tmp_path: Path, *, model_type: str = "depth_anything_v2") -> DepthAnythingOnnxAdapter:
    descriptor = ModelDescriptor(
        id="depth",
        name="depth",
        display_name="Depth Anything",
        model_type=model_type,
        task="depth",
        family="depth_anything",
        adapter="depth_anything_onnx",
        runtime=["ONNX Runtime"],
        config_path=tmp_path / "depth.yaml",
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True),
    )
    return DepthAnythingOnnxAdapter(
        ModelRecord(descriptor=descriptor, config={}),
        ArtifactStore(tmp_path / "artifacts"),
    )


def _transform(*, original: tuple[int, int] = (8, 6), model: tuple[int, int] = (4, 3)) -> _ImageTransform:
    return _ImageTransform(
        original_width=original[0],
        original_height=original[1],
        input_width=model[0],
        input_height=model[1],
        scale=model[0] / original[0],
        pad_x=0,
        pad_y=0,
    )


@pytest.mark.parametrize(
    "wrapped",
    [
        lambda value: value,
        lambda value: value[None, ...],
        lambda value: value[None, None, ...],
    ],
    ids=["hw", "one-hw", "one-one-hw"],
)
def test_common_depth_shapes_create_default_original_size_raster_with_raw_metadata(
    tmp_path: Path,
    wrapped,
) -> None:
    adapter = _adapter(tmp_path)
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)

    rasters = adapter._rasters(
        {"depth": wrapped(raw)},
        _transform(),
        tmp_path / "source.png",
        {"color_map": "grayscale", "percentile_low": 0, "percentile_high": 100},
    )

    assert len(rasters) == 1
    raster = rasters[0]
    assert raster.role == "depth-map"
    assert (raster.width, raster.height) == (8, 6)
    assert raster.media_type == "image/png"
    assert Image.open(raster.path).mode == "L"
    assert raster.metadata["output_name"] == "depth"
    assert raster.metadata["source_shape"] == [3, 4]
    assert raster.metadata["raw_min"] == 0.0
    assert raster.metadata["raw_max"] == 11.0
    assert raster.metadata["raw_percentiles"]["p50"] == pytest.approx(5.5)
    assert raster.metadata["clip_values"] == pytest.approx([0.0, 11.0])


def test_inverse_grayscale_and_turbo_like_color_maps(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    raw = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    transform = _transform(original=(2, 2), model=(2, 2))
    common = {"percentile_low": 0, "percentile_high": 100, "color_map": "grayscale"}

    normal = adapter._rasters({"depth": raw}, transform, tmp_path / "source.png", common)[0]
    inverse = adapter._rasters({"depth": raw}, transform, tmp_path / "source.png", {**common, "inverse": True})[0]
    turbo = adapter._rasters(
        {"depth": raw},
        transform,
        tmp_path / "source.png",
        {"percentile_low": 0, "percentile_high": 100, "colormap": "turbo-like"},
    )[0]

    normal_pixels = np.asarray(Image.open(normal.path), dtype=np.int16)
    inverse_pixels = np.asarray(Image.open(inverse.path), dtype=np.int16)
    assert np.all(normal_pixels + inverse_pixels == 255)
    assert Image.open(turbo.path).mode == "RGB"
    assert turbo.metadata["color_map"] == "turbo"
    assert inverse.metadata["inverse"] is True


def test_multiple_outputs_require_explicit_selection(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    outputs = {
        "coarse": np.zeros((1, 2, 2), dtype=np.float32),
        "fine": np.ones((1, 2, 2), dtype=np.float32),
    }

    with pytest.raises(ModelRuntimeError, match="Multiple depth outputs are ambiguous"):
        adapter._rasters(outputs, _transform(original=(2, 2), model=(2, 2)), tmp_path / "source.png", {})

    selected = adapter._rasters(
        outputs,
        _transform(original=(2, 2), model=(2, 2)),
        tmp_path / "source.png",
        {"depth_output_name": "fine", "color_map": "grayscale"},
    )[0]
    assert selected.metadata["output_name"] == "fine"
    assert selected.metadata["raw_min"] == 1.0


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.zeros((1, 2, 3, 4), dtype=np.float32), "single batch and channel"),
        (np.zeros((2, 3, 4), dtype=np.float32), "single leading dimension"),
        (np.zeros((4,), dtype=np.float32), "Unsupported depth output rank"),
        (np.asarray([[0.0, np.nan]], dtype=np.float32), "non-finite values"),
        (np.asarray([[0.0, np.inf]], dtype=np.float32), "non-finite values"),
    ],
)
def test_invalid_shapes_and_non_finite_values_are_reported(
    tmp_path: Path,
    output: np.ndarray,
    message: str,
) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ModelRuntimeError, match=message):
        adapter._rasters(
            {"depth": output},
            _transform(original=(2, 2), model=(2, 2)),
            tmp_path / "source.png",
            {"depth_output_name": "depth"},
        )


def test_raw_and_output_pixel_budgets_are_enforced(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    raw = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ModelRuntimeError, match="raw value budget"):
        adapter._rasters(
            {"depth": raw},
            _transform(original=(2, 2), model=(2, 2)),
            tmp_path / "source.png",
            {"max_depth_values": 3},
        )
    with pytest.raises(ModelRuntimeError, match="output pixel budget"):
        adapter._rasters(
            {"depth": raw},
            _transform(original=(100, 100), model=(2, 2)),
            tmp_path / "source.png",
            {"max_raster_pixels": 100},
        )


class _FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.feed: dict[str, np.ndarray] | None = None

    def run(self, _outputs, feed):
        self.feed = feed
        return [self.output]


def test_v2_preprocessing_is_direct_rgb_bicubic_imagenet_nchw_and_predict_needs_no_capture(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path, model_type="depth_anything_v2")
    image_path = tmp_path / "source.png"
    Image.new("RGB", (1, 1), (255, 127, 0)).save(image_path)
    adapter.input_meta = SimpleNamespace(name="image", shape=[1, 3, 2, 3], type="tensor(float)")
    adapter.output_meta = [SimpleNamespace(name="depth", shape=[1, 2, 3], type="tensor(float)")]
    adapter.session = _FakeSession(np.arange(6, dtype=np.float32).reshape(1, 2, 3))
    adapter.loaded = True

    tensor, transform = adapter._prepare_image(image_path)
    result = adapter.predict(
        image_path,
        [],
        {"color_map": "grayscale", "percentile_low": 0, "percentile_high": 100},
    )

    expected = np.asarray(
        [
            (1.0 - 0.485) / 0.229,
            (127.0 / 255.0 - 0.456) / 0.224,
            (0.0 - 0.406) / 0.225,
        ],
        dtype=np.float32,
    )
    assert tensor.shape == (1, 3, 2, 3)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    np.testing.assert_allclose(tensor[0, :, 0, 0], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tensor[0, :, -1, -1], expected, rtol=1e-6, atol=1e-6)
    assert (transform.input_width, transform.input_height) == (3, 2)
    assert (transform.pad_x, transform.pad_y) == (0, 0)
    assert result.artifacts == []
    assert len(result.rasters) == 1
    assert result.rasters[0].role == "depth-map"
    assert result.rasters[0].width == 1
    assert adapter.session.feed is not None
    np.testing.assert_allclose(adapter.session.feed["image"], tensor)


def test_v1_preprocessing_uses_fixed_518_square_and_same_channel_normalization(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, model_type="depth_anything")
    image_path = tmp_path / "source.png"
    Image.new("RGB", (2, 1), (0, 255, 127)).save(image_path)
    adapter.input_meta = SimpleNamespace(name="image", shape=[1, 3, 518, 518], type="tensor(float)")

    tensor, transform = adapter._prepare_image(image_path)

    expected = np.asarray(
        [
            (0.0 - 0.485) / 0.229,
            (1.0 - 0.456) / 0.224,
            (127.0 / 255.0 - 0.406) / 0.225,
        ],
        dtype=np.float32,
    )
    assert tensor.shape == (1, 3, 518, 518)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, :, 0, 0], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tensor[0, :, 259, 259], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tensor[0, :, -1, -1], expected, rtol=1e-6, atol=1e-6)
    assert (transform.input_width, transform.input_height) == (518, 518)
    assert (transform.original_width, transform.original_height) == (2, 1)
    assert (transform.pad_x, transform.pad_y) == (0, 0)
