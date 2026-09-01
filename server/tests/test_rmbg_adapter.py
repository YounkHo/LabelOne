from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.adapters.onnx import _ImageTransform
from labelone.models.adapters.rmbg import RmbgMattingOnnxAdapter
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


def _adapter(tmp_path: Path, config: dict[str, object] | None = None) -> RmbgMattingOnnxAdapter:
    descriptor = ModelDescriptor(
        id="rmbg",
        name="rmbg",
        display_name="RMBG",
        model_type="rmbg",
        task="segmentation",
        family="rmbg",
        adapter="rmbg_matting_onnx",
        runtime=["ONNX Runtime"],
        config_path=tmp_path / "rmbg.yaml",
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True),
    )
    values: dict[str, object] = {"resize_mode": "letterbox", "mask_activation": "probability"}
    values.update(config or {})
    return RmbgMattingOnnxAdapter(
        ModelRecord(descriptor=descriptor, config=values),
        ArtifactStore(tmp_path / "artifacts"),
    )


class _InputMeta:
    name = "image"
    type = "tensor(float)"

    def __init__(self, shape: list[int | str | None]) -> None:
        self.shape = shape


def _image(path: Path, size: tuple[int, int] = (8, 4), *, alpha: int | None = None) -> None:
    if alpha is None:
        Image.new("RGB", size, (180, 60, 30)).save(path)
    else:
        Image.new("RGBA", size, (180, 60, 30, alpha)).save(path)


def _letterbox_transform() -> _ImageTransform:
    return _ImageTransform(
        original_width=8,
        original_height=4,
        input_width=8,
        input_height=8,
        scale=1.0,
        pad_x=0,
        pad_y=2,
    )


def test_rmbg_14_preprocess_uses_model_shape_stretch_and_centered_half_range(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (0, 0, 0))
    image.putpixel((1, 0), (255, 255, 255))
    image.save(image_path)
    adapter = _adapter(tmp_path, {"version": 1.4, "resize_mode": "stretch"})
    adapter.input_meta = _InputMeta([1, 3, 2, 4])

    tensor, transform = adapter._prepare_image(image_path)

    assert tensor.shape == (1, 3, 2, 4)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) == pytest.approx(-0.5)
    assert float(tensor.max()) == pytest.approx(0.5)
    assert (transform.input_width, transform.input_height) == (4, 2)
    assert (transform.pad_x, transform.pad_y) == (0, 0)


def test_rmbg_20_preprocess_uses_fixed_1024_and_imagenet_normalization(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (3, 2), (255, 0, 0)).save(image_path)
    adapter = _adapter(tmp_path, {"version": 2.0, "resize_mode": "stretch"})
    adapter.input_meta = _InputMeta([1, 3, 1024, 1024])

    tensor, transform = adapter._prepare_image(image_path)

    assert tensor.shape == (1, 3, 1024, 1024)
    assert tensor.dtype == np.float32
    assert tensor[0, :, 0, 0] == pytest.approx([
        (1.0 - 0.485) / 0.229,
        (0.0 - 0.456) / 0.224,
        (0.0 - 0.406) / 0.225,
    ])
    assert float(tensor.min()) >= -2.12
    assert float(tensor.max()) <= 2.65
    assert (transform.input_width, transform.input_height) == (1024, 1024)


def test_rmbg_20_rejects_an_incompatible_static_onnx_input(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    _image(image_path)
    adapter = _adapter(tmp_path, {"version": 2.0, "resize_mode": "stretch"})
    adapter.input_meta = _InputMeta([1, 3, 512, 512])

    with pytest.raises(ModelRuntimeError, match="1024 by 1024"):
        adapter._prepare_image(image_path)


@pytest.mark.parametrize("shape", [(1, 1, 8, 8), (1, 8, 8), (8, 8)])
def test_supported_single_channel_layouts_restore_letterbox_to_original_png(
    tmp_path: Path,
    shape: tuple[int, ...],
) -> None:
    image_path = tmp_path / "source.png"
    _image(image_path)
    values = np.zeros((8, 8), dtype=np.float32)
    values[2:6] = 0.75
    output = values.reshape(shape)
    adapter = _adapter(tmp_path)

    artifacts = adapter._rasters({"alpha": output}, _letterbox_transform(), image_path, {})

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.role == "alpha-mask"
    assert artifact.media_type == "image/png"
    assert (artifact.width, artifact.height) == (8, 4)
    mask = np.asarray(Image.open(artifact.path))
    assert mask.shape == (4, 8)
    assert np.all(mask == pytest.approx(191, abs=1))
    assert artifact.metadata["source_shape"] == list(shape)
    assert artifact.metadata["activation"] == "probability"
    assert artifact.metadata["letterbox"]["pad"] == [0, 2]


def test_auto_activation_sigmoids_logits_and_threshold_suppresses_background(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    _image(image_path)
    logits = np.full((1, 1, 8, 8), -4.0, dtype=np.float32)
    logits[:, :, 2:6, :4] = 4.0
    adapter = _adapter(tmp_path, {"mask_activation": "auto"})

    artifact = adapter._rasters(
        {"matte": logits},
        _letterbox_transform(),
        image_path,
        {"mask_threshold": 0.5},
    )[0]

    mask = np.asarray(Image.open(artifact.path))
    assert np.all(mask[:, :4] > 245)
    assert np.all(mask[:, 4:] == 0)
    assert artifact.metadata["activation"] == "sigmoid"
    assert artifact.metadata["threshold"] == 0.5


def test_x_anylabeling_default_minmax_is_applied_after_stretch_restore(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    _image(image_path, size=(4, 2))
    output = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    transform = _ImageTransform(4, 2, 2, 2, 0.5, 0, 0)
    adapter = _adapter(tmp_path, {"version": 1.4, "resize_mode": "stretch", "mask_activation": "minmax"})

    artifact = adapter._rasters({"alpha": output}, transform, image_path, {})[0]

    mask = np.asarray(Image.open(artifact.path))
    assert mask.min() == 0
    assert mask.max() == 255
    assert artifact.metadata["activation"] == "minmax"
    assert artifact.metadata["preprocessing"] == {
        "version": 1.4,
        "resize_mode": "stretch",
        "normalization": "center_0.5",
    }


def test_optional_rgba_cutout_preserves_source_alpha_and_links_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    _image(image_path, size=(4, 2), alpha=128)
    probability = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    transform = _ImageTransform(4, 2, 4, 2, 1.0, 0, 0)
    adapter = _adapter(tmp_path)

    artifacts = adapter._rasters(
        {"alpha": probability},
        transform,
        image_path,
        {"output_cutout": True},
    )

    assert [artifact.role for artifact in artifacts] == ["alpha-mask", "foreground-cutout"]
    cutout = artifacts[1]
    assert cutout.media_type == "image/webp"
    assert cutout.metadata["mask_artifact_id"] == artifacts[0].id
    alpha = np.asarray(Image.open(cutout.path).convert("RGBA"))[:, :, 3]
    assert np.all(alpha[:, :2] == pytest.approx(128, abs=1))
    assert np.all(alpha[:, 2:] == 0)


def test_explicit_output_name_resolves_one_mask_among_multiple_outputs(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    _image(image_path)
    adapter = _adapter(tmp_path)

    artifacts = adapter._rasters(
        {
            "features": np.zeros((1, 4, 2, 2), dtype=np.float32),
            "alpha": np.ones((1, 1, 8, 8), dtype=np.float32),
        },
        _letterbox_transform(),
        image_path,
        {"mask_output_name": "alpha"},
    )

    assert artifacts[0].metadata["output_name"] == "alpha"


@pytest.mark.parametrize(
    "outputs, parameters, message",
    [
        ({}, {}, "returned no outputs"),
        ({"mask": np.zeros((1, 2, 8, 8), dtype=np.float32)}, {}, "No supported single-channel"),
        ({"mask": np.zeros((2, 8, 8), dtype=np.float32)}, {}, "No supported single-channel"),
        ({"mask": np.zeros((8,), dtype=np.float32)}, {}, "No supported single-channel"),
        ({"mask": np.full((8, 8), np.nan, dtype=np.float32)}, {}, "No supported single-channel"),
        (
            {"mask_a": np.zeros((8, 8), dtype=np.float32), "mask_b": np.zeros((1, 8, 8), dtype=np.float32)},
            {},
            "Multiple RMBG mask outputs",
        ),
        ({"mask": np.full((8, 8), 2.0, dtype=np.float32)}, {"mask_activation": "probability"}, "outside zero and one"),
        ({"mask": np.zeros((8, 8), dtype=np.float32)}, {"mask_activation": "softmax"}, "mask_activation"),
        ({"mask": np.zeros((8, 8), dtype=np.float32)}, {"mask_threshold": 1.5}, "mask_threshold"),
        ({"mask": np.zeros((8, 8), dtype=np.float32)}, {"output_cutout": "yes"}, "output_cutout"),
    ],
)
def test_invalid_outputs_and_parameters_raise_clear_errors(
    tmp_path: Path,
    outputs: dict[str, np.ndarray],
    parameters: dict[str, object],
    message: str,
) -> None:
    image_path = tmp_path / "source.png"
    _image(image_path)
    adapter = _adapter(tmp_path)

    with pytest.raises(ModelRuntimeError, match=message):
        adapter._rasters(outputs, _letterbox_transform(), image_path, parameters)


def test_original_size_output_pixel_budget_is_enforced_before_artifact_write(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    _image(image_path)
    adapter = _adapter(tmp_path)

    with pytest.raises(ModelRuntimeError, match="pixel budget"):
        adapter._rasters(
            {"mask": np.zeros((8, 8), dtype=np.float32)},
            _letterbox_transform(),
            image_path,
            {"max_output_pixels": 31},
        )

    assert not list((tmp_path / "artifacts").rglob("*.png"))


def test_annotations_remain_empty_for_raster_only_models(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    assert adapter._annotations({"mask": np.ones((8, 8))}, _letterbox_transform(), {}) == []
