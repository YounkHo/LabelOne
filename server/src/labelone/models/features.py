from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, model_validator

from labelone.errors import ModelRuntimeError


HARD_MAX_FEATURE_ELEMENTS = 64_000_000
MAX_FEATURE_PREVIEW_PIXELS = 2_000_000


class FeatureTransformOptions(BaseModel):
    projection: Literal["none", "mean", "max", "pca1", "channel", "token_grid"] = "none"
    channel: int = Field(default=0, ge=0)
    normalization: Literal["none", "minmax", "zscore", "l2"] = "none"
    interpolation: Literal["nearest", "bilinear", "bicubic", "lanczos"] = "bicubic"
    spatial_scale: float = Field(default=1.0, gt=0, le=16)
    gain: float = Field(default=1.0, gt=0, le=100)
    gamma: float = Field(default=1.0, gt=0, le=10)
    clip_percentiles: tuple[float, float] | None = None
    max_output_elements: int = Field(default=HARD_MAX_FEATURE_ELEMENTS, ge=1_000, le=HARD_MAX_FEATURE_ELEMENTS)

    @model_validator(mode="after")
    def validate_percentiles(self) -> "FeatureTransformOptions":
        if self.clip_percentiles is not None:
            low, high = self.clip_percentiles
            if not 0 <= low < high <= 100:
                raise ValueError("clip_percentiles must satisfy 0 <= low < high <= 100")
        return self


def _pca_projection(matrix: np.ndarray, iterations: int = 12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    minimum = float(finite.min())
    maximum = float(finite.max())
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=maximum, neginf=minimum)
    channels, samples = matrix.shape
    if channels == 1:
        return matrix[0].astype(np.float32)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    vector = np.ones(channels, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    for _ in range(iterations):
        projected = centered.T @ vector
        next_vector = centered @ projected
        norm = float(np.linalg.norm(next_vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            return np.zeros(samples, dtype=np.float32)
        vector = next_vector / norm
    nonzero = np.flatnonzero(np.abs(vector) > 1e-8)
    if nonzero.size and vector[int(nonzero[0])] < 0:
        vector = -vector
    result = vector @ centered
    float32_maximum = float(np.finfo(np.float32).max)
    result = np.nan_to_num(result, nan=0.0, posinf=float32_maximum, neginf=-float32_maximum)
    return np.clip(result, -float32_maximum, float32_maximum).astype(np.float32)


def _project(tensor: np.ndarray, options: FeatureTransformOptions) -> np.ndarray:
    value = np.asarray(tensor, dtype=np.float32)
    projection = options.projection
    if value.ndim == 1:
        return value
    if value.ndim == 2:
        if value.shape[0] == 1:
            return value[0]
        if projection == "none":
            return value
        if projection == "channel":
            if options.channel >= value.shape[1]:
                raise ModelRuntimeError(
                    "Feature channel is outside the tensor shape",
                    details={"channel": options.channel, "shape": list(value.shape)},
                )
            scalar = value[:, options.channel]
        elif projection == "max":
            scalar = value.max(axis=1)
        elif projection == "pca1":
            scalar = _pca_projection(value.T)
        else:
            scalar = value.mean(axis=1, dtype=np.float64).astype(np.float32)
        side = int(round(value.shape[0] ** 0.5))
        if projection == "token_grid":
            if side * side != value.shape[0]:
                raise ModelRuntimeError(
                    "Token count cannot be reshaped to a square grid",
                    details={"tokens": value.shape[0], "shape": list(value.shape)},
                )
            return scalar.reshape(side, side)
        return scalar
    if projection == "none":
        return value
    if value.ndim == 4:
        batch, channels, height, width = value.shape
        if projection in {"mean", "token_grid"}:
            return value.mean(axis=1, keepdims=True)
        if projection == "max":
            return value.max(axis=1, keepdims=True)
        if projection == "channel":
            if options.channel >= channels:
                raise ModelRuntimeError(
                    "Feature channel is outside the tensor shape",
                    details={"channel": options.channel, "shape": list(value.shape)},
                )
            return value[:, options.channel : options.channel + 1]
        if projection == "pca1":
            projected = [_pca_projection(item.reshape(channels, -1)).reshape(1, height, width) for item in value]
            return np.stack(projected, axis=0)
    if value.ndim == 3:
        batch, tokens, channels = value.shape
        if projection == "channel":
            if options.channel >= channels:
                raise ModelRuntimeError(
                    "Feature channel is outside the tensor shape",
                    details={"channel": options.channel, "shape": list(value.shape)},
                )
            scalar = value[:, :, options.channel]
        elif projection == "max":
            scalar = value.max(axis=2)
        elif projection == "pca1":
            scalar = np.stack([_pca_projection(item.T) for item in value], axis=0)
        else:
            scalar = value.mean(axis=2)
        side = int(round(tokens ** 0.5))
        if projection == "token_grid" and side * side != tokens:
            raise ModelRuntimeError(
                "Token count cannot be reshaped to a square grid",
                details={"tokens": tokens, "shape": list(value.shape)},
            )
        height, width = (side, side) if side * side == tokens else (1, tokens)
        return scalar.reshape(batch, 1, height, width)
    raise ModelRuntimeError(
        "Feature projection requires an NCHW or NTC tensor",
        details={"projection": projection, "shape": list(value.shape)},
    )


def _normalize(tensor: np.ndarray, options: FeatureTransformOptions) -> np.ndarray:
    value = np.asarray(tensor, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return np.zeros_like(value, dtype=np.float32)
    finite_minimum = float(finite.min())
    finite_maximum = float(finite.max())
    value = np.nan_to_num(
        value,
        copy=True,
        nan=0.0,
        posinf=finite_maximum,
        neginf=finite_minimum,
    )
    if options.clip_percentiles is not None:
        low, high = np.percentile(finite.astype(np.float64, copy=False), options.clip_percentiles)
        value = np.clip(value, low, high)
    if options.normalization == "minmax":
        minimum = float(value.min())
        span = float(value.max()) - minimum
        value = (value - minimum) / max(span, 1e-12)
    elif options.normalization == "zscore":
        value64 = value.astype(np.float64, copy=False)
        value = (value64 - float(value64.mean())) / max(float(value64.std()), 1e-12)
    elif options.normalization == "l2":
        value = value / max(float(np.linalg.norm(value.astype(np.float64, copy=False).ravel())), 1e-12)
    value = np.sign(value) * np.power(np.abs(value), options.gamma)
    value *= options.gain
    return np.nan_to_num(value, copy=False, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _resize_spatial(tensor: np.ndarray, options: FeatureTransformOptions) -> np.ndarray:
    if options.spatial_scale == 1 or tensor.ndim != 4:
        return tensor
    batch, channels, height, width = tensor.shape
    output_height = max(1, round(height * options.spatial_scale))
    output_width = max(1, round(width * options.spatial_scale))
    output_elements = batch * channels * output_height * output_width
    if output_elements > options.max_output_elements:
        raise ModelRuntimeError(
            "Scaled feature tensor exceeds the output budget",
            details={
                "source_shape": list(tensor.shape),
                "output_shape": [batch, channels, output_height, output_width],
                "max_output_elements": options.max_output_elements,
            },
        )
    methods = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    resized = np.empty((batch, channels, output_height, output_width), dtype=np.float32)
    for batch_index in range(batch):
        for channel_index in range(channels):
            image = Image.fromarray(tensor[batch_index, channel_index], mode="F")
            resized[batch_index, channel_index] = np.asarray(
                image.resize((output_width, output_height), methods[options.interpolation]),
                dtype=np.float32,
            )
    return resized


def transform_feature(tensor: np.ndarray, options: FeatureTransformOptions) -> np.ndarray:
    projected = _project(tensor, options)
    normalized = _normalize(projected, options)
    resized = _resize_spatial(normalized, options)
    return np.nan_to_num(resized, copy=False, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def feature_preview_image(tensor: np.ndarray) -> Image.Image | None:
    value = np.asarray(tensor)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim not in {1, 2} or value.size == 0 or not np.issubdtype(value.dtype, np.number):
        return None
    if value.ndim == 1:
        vector = np.asarray(value, dtype=np.float64)
        if vector.size > 2_048:
            indices = np.linspace(0, vector.size - 1, 2_048).astype(np.int64)
            vector = vector[indices]
        finite = vector[np.isfinite(vector)]
        if finite.size == 0:
            normalized = np.zeros_like(vector)
        else:
            minimum = float(finite.min())
            span = max(float(finite.max()) - minimum, 1e-12)
            cleaned = np.nan_to_num(vector, nan=minimum, posinf=float(finite.max()), neginf=minimum)
            normalized = np.clip((cleaned - minimum) / span, 0, 1)
        width, height = 512, 128
        image = Image.new("RGB", (width, height), (8, 13, 20))
        draw = ImageDraw.Draw(image)
        draw.line([(0, height - 1), (width - 1, height - 1)], fill=(48, 63, 82), width=1)
        if normalized.size == 1:
            x_positions = np.array([width // 2], dtype=np.float32)
        else:
            x_positions = np.linspace(0, width - 1, normalized.size)
        points = [
            (float(x), float((height - 8) - sample * (height - 16)))
            for x, sample in zip(x_positions, normalized, strict=True)
        ]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(71, 205, 178))
        else:
            draw.line(points, fill=(71, 205, 178), width=2, joint="curve")
        return image
    scalar = np.asarray(value, dtype=np.float32)
    if scalar.size > MAX_FEATURE_PREVIEW_PIXELS:
        scale = (MAX_FEATURE_PREVIEW_PIXELS / scalar.size) ** 0.5
        scalar = np.asarray(
            Image.fromarray(scalar, mode="F").resize(
                (max(1, round(scalar.shape[1] * scale)), max(1, round(scalar.shape[0] * scale))),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
    finite = scalar[np.isfinite(scalar)]
    if finite.size == 0:
        normalized = np.zeros_like(scalar, dtype=np.float32)
    else:
        minimum = float(finite.min())
        span = max(float(finite.max()) - minimum, 1e-12)
        normalized = np.clip((np.nan_to_num(scalar, nan=minimum, posinf=minimum, neginf=minimum) - minimum) / span, 0, 1)
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    colors = np.array([
        [7, 12, 28],
        [33, 65, 122],
        [36, 151, 155],
        [142, 206, 91],
        [255, 231, 128],
    ], dtype=np.float32)
    flattened = normalized.ravel()
    rgb = np.stack([np.interp(flattened, stops, colors[:, channel]) for channel in range(3)], axis=1)
    image = Image.fromarray(rgb.reshape(*normalized.shape, 3).astype(np.uint8), mode="RGB")
    return image
