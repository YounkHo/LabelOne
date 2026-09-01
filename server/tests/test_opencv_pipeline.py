from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from labelone.pipelines.opencv_ops import OPENCV_OPERATORS, apply_opencv_operator
from labelone.pipelines.registry import PipelineValidationError


def _defaults(kind: str) -> dict[str, object]:
    properties = OPENCV_OPERATORS[kind].parameters_schema["properties"]
    assert isinstance(properties, dict)
    return {
        name: schema["default"]
        for name, schema in properties.items()
        if isinstance(schema, dict) and "default" in schema
    }


@pytest.mark.parametrize("kind", sorted(OPENCV_OPERATORS))
def test_allowlisted_opencv_operators_keep_a_single_image_contract(kind: str) -> None:
    pixels = np.zeros((48, 64, 3), dtype=np.uint8)
    pixels[8:40, 12:52] = (220, 80, 40)
    pixels[20:28, 20:44] = (10, 240, 190)
    source = Image.fromarray(pixels, mode="RGB")

    result = apply_opencv_operator(kind, source, _defaults(kind))

    assert isinstance(result, Image.Image)
    assert result.size == source.size
    assert result.mode in {"L", "RGB", "RGBA"}
    contract = OPENCV_OPERATORS[kind]
    assert contract.input_type == contract.output_type == "image"
    assert contract.annotation_policy["coordinates"] == ("unavailable" if kind in {"opencv.fourier_transform", "opencv.haar_wavelet"} else "unchanged")


def test_opencv_result_is_not_a_noop_for_visible_filters() -> None:
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[8:24, 8:24] = (255, 100, 20)
    source = Image.fromarray(pixels, mode="RGB")

    inverted = apply_opencv_operator("opencv.invert", source, {})
    edges = apply_opencv_operator("opencv.canny", source, _defaults("opencv.canny"))

    assert not np.array_equal(np.asarray(inverted), pixels)
    assert np.asarray(edges).ndim == 2
    assert int(np.asarray(edges).max()) == 255


def test_frequency_operators_are_finite_same_size_and_visibly_transform_odd_images() -> None:
    pixels = np.zeros((17, 19, 3), dtype=np.uint8)
    pixels[8, 9] = 255
    source = Image.fromarray(pixels, mode="RGB")

    magnitude = apply_opencv_operator("opencv.fourier_transform", source, _defaults("opencv.fourier_transform"))
    phase = apply_opencv_operator("opencv.fourier_transform", source, {"mode": "phase", "center": True})
    wavelet = apply_opencv_operator("opencv.haar_wavelet", source, {"levels": 3})

    for result in (magnitude, phase, wavelet):
        array = np.asarray(result)
        assert result.size == source.size
        assert np.all(np.isfinite(array))
    assert not np.array_equal(np.asarray(wavelet), np.asarray(source.convert("L")))


@pytest.mark.parametrize(
    "kind, updates, message",
    [
        ("opencv.gaussian_blur", {"kernel_size": 4}, "odd"),
        ("opencv.canny", {"lower": 220.0, "upper": 10.0}, "cannot exceed"),
        ("opencv.sobel", {"dx": 0, "dy": 0}, "cannot both"),
        ("opencv.normalize", {"alpha": 255.0, "beta": 0.0}, "smaller"),
    ],
)
def test_opencv_runtime_constraints_reject_invalid_combinations(
    kind: str,
    updates: dict[str, object],
    message: str,
) -> None:
    parameters = {**_defaults(kind), **updates}
    with pytest.raises(PipelineValidationError, match=message):
        apply_opencv_operator(kind, Image.new("RGB", (16, 16), "white"), parameters)
